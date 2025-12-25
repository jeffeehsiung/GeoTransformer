import gc
import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from geotransformer.utils.pointcloud import (
    apply_transform,
    get_nearest_neighbor,
    get_rotation_translation_from_transform,
    get_transform_from_rotation_translation,
    random_sample_rotation_v3,
)
from geotransformer.utils.registration import get_correspondences
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


def to_tensor(arr, device=None, dtype=torch.float32):
    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(arr)
        if device is not None:
            t = t.to(device)
        return t.to(dtype)
    elif isinstance(arr, torch.Tensor):
        if device is not None:
            arr = arr.to(device)
        return arr.to(dtype)
    else:
        raise TypeError(f"Input must be np.ndarray or torch.Tensor, got {type(arr)}")


def convert_mm_to_m(
    points_mm: Union[np.ndarray, torch.Tensor],
    scale: float = 0.001,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert point coordinates from millimeters to meters. Supports np.ndarray or torch.Tensor."""
    if points_mm is None:
        return None
    if torch.is_tensor(points_mm):
        if device:
            points_mm = points_mm.to(device)
        return points_mm.float() * scale
    # numpy → tensor
    t = torch.from_numpy(points_mm)
    if device:
        t = to_tensor(t, device=device)
    return t * scale


def quat_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to rotation matrix. Supports np.ndarray or torch.Tensor."""
    if torch.is_tensor(q):
        q_np = q.detach().cpu().numpy()
    else:
        q_np = q
    mat_np = R.from_quat([q_np[1], q_np[2], q_np[3], q_np[0]]).as_matrix()
    mat = torch.from_numpy(mat_np)
    return mat.to(q.device if torch.is_tensor(q) else torch.device("cpu"))


def update_preproc_ref_only(
    preproc_R_ref: torch.Tensor,
    preproc_t_ref: torch.Tensor,
    R_op: torch.Tensor,
    t_op: torch.Tensor,
):
    """
    Update net preproc transform for reference cloud only.
    Does NOT change ref_points, rotation, or translation.
    p' = R_op p + t_op was already applied to ref_points by caller.
    Returns updated preproc_R_ref, preproc_t_ref.
    """
    preproc_t_ref = (R_op @ preproc_t_ref.view(3, 1)).view(3,) + t_op
    preproc_R_ref = R_op @ preproc_R_ref
    return preproc_R_ref, preproc_t_ref


def update_preproc_src_only(
    preproc_R_src: torch.Tensor,
    preproc_t_src: torch.Tensor,
    R_op: torch.Tensor,
    t_op: torch.Tensor,
):
    """
    Update net preproc transform for source cloud only.
    Does NOT change src_points, rotation, or translation.
    """
    preproc_t_src = (R_op @ preproc_t_src.view(3, 1)).view(3,) + t_op
    preproc_R_src = R_op @ preproc_R_src
    return preproc_R_src, preproc_t_src


def compose_ref_op(
    R: torch.Tensor, t: torch.Tensor, R_op: torch.Tensor, t_op: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply an operation to the reference cloud and update mapping src -> ref.

        Row/column convention aware, assuming mapping uses column-vector form: ref = R src + t.
        If ref' = R_op ref + t_op was applied to points, then new mapping is:
            R' = R_op R,  t' = R_op t + t_op
        """
    R_new = R_op @ R
    t_new = R_op @ t + t_op
    return R_new, t_new


def compose_src_op(
    R: torch.Tensor, t: torch.Tensor, R_op: torch.Tensor, t_op: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply an operation to the source cloud and update mapping src -> ref.

        If src' = R_op src + t_op was applied to points, then new mapping keeping ref = R src + t is:
            R' = R R_op^{-T} (with rotations, R_op^{-1} = R_op^T)
            t' = t - R' t_op
        Using orthonormal rotations: R' = R @ R_op.T and t' = t - R' @ t_op.
        """
    R_new = R @ R_op.T
    t_new = t - R_new @ t_op
    return R_new, t_new


def preprocess_pair(
    curr_epoch: int,
    ref_points: Union[torch.Tensor, np.ndarray],
    src_points: Union[torch.Tensor, np.ndarray],
    ref_normals: Optional[Union[torch.Tensor, np.ndarray]] = None,
    src_normals: Optional[Union[torch.Tensor, np.ndarray]] = None,
    *,
    voxel_size: Optional[float] = None,
    normalize: bool = False,
    estimate_missing_normals: bool = True,
    return_corr_indices: bool = False,
    matching_radius: Optional[float] = None,
    rotation: Optional[Union[torch.Tensor, np.ndarray]] = None,
    translation: Optional[Union[torch.Tensor, np.ndarray]] = None,
    preproc_R_ref: Optional[Union[torch.Tensor, np.ndarray]] = None,
    preproc_t_ref: Optional[Union[torch.Tensor, np.ndarray]] = None,
    preproc_R_src: Optional[Union[torch.Tensor, np.ndarray]] = None,
    preproc_t_src: Optional[Union[torch.Tensor, np.ndarray]] = None,
    use_augmentation: bool = False,
    so3_augmentation: bool = False,
    so3_curriculum_epochs: int = 50,
    max_so3_rotation_deg: float = 180.0,
    translation_jitter_m: float = 0.005,
    subset: str = "train",
    is_preprocessed: bool = False,
    device: Optional[torch.device] = None,
    debug: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Shared geometry pipeline for Roboeye training + inference.
    All outputs are torch.Tensor on the target device (GPU if available).

    Parameters
    ----------
    curr_epoch : int
        Current training epoch for curriculum learning.
    ref_points, src_points : (N,3), (M,3) np.float32
        Input point clouds in **meters**, row-vector convention.
    ref_normals, src_normals : (N,3), (M,3) or None
    voxel_size : float or None [meters]
    normalize : bool
        If True, apply the same centering logic as in training dataset (center_pair).
    rotation, translation :
        GT transform mapping src -> ref. For inference, pass identity (R=I, t=0).
    use_augmentation :
        If True, apply the same augmentations as training (SO3 + extra random rotations).
        For inference you must pass False.
    subset : "train" or "test"
        If "train" and use_augmentation=True, apply the final random rotations A,B.


    Returns
    -------
    data_dict : dict
        Keys: ref_points, src_points, ref_normals, src_normals, transform, rotation, translation,
              ref_feats, src_feats (+ optional corr_indices and pre_* if requested)
    """
    # Choose device
    if device is None:
        device = (
            ref_points.device
            if torch.is_tensor(ref_points)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    assert (
        rotation is not None and translation is not None
    ), "rotation/translation must be provided (identity if unknown)."
    # Initialize output dict
    data_dict = {
        "raw_gt_rotation": None,
        "raw_gt_translation": None,
        "ref_points": None,
        "src_points": None,
        "ref_feats": None,
        "src_feats": None,
        "transform": None,
        "rotation": None,
        "translation": None,
        "pre_ref_points": None,
        "pre_src_points": None,
        "pre_ref_normals": None,
        "pre_src_normals": None,
        "pre_rotation": None,
        "pre_translation": None,
        "pre_transform": None,
        "preproc_R_ref": None,
        "preproc_t_ref": None,
        "preproc_R_src": None,
        "preproc_t_src": None,
    }
    # A) Convert to Tensor
    ref_points = torch.as_tensor(ref_points, dtype=torch.float32, device=device)
    src_points = torch.as_tensor(src_points, dtype=torch.float32, device=device)
    if ref_normals is not None:
        ref_normals = torch.as_tensor(ref_normals, dtype=torch.float32, device=device)
    if src_normals is not None:
        src_normals = torch.as_tensor(src_normals, dtype=torch.float32, device=device)
    rotation = torch.as_tensor(rotation, dtype=torch.float32, device=device)
    translation = torch.as_tensor(translation, dtype=torch.float32, device=device)
    preproc_scale = 0.001  # mm -> m scaling factor
    # Initialize net preproc transforms (column-vector convention: p' = R p + t)
    preproc_R_ref = (
        torch.eye(3, dtype=torch.float32, device=device)
        if preproc_R_ref is None
        else torch.as_tensor(preproc_R_ref, dtype=torch.float32, device=device)
    )
    preproc_t_ref = (
        torch.zeros((3,), dtype=torch.float32, device=device)
        if preproc_t_ref is None
        else torch.as_tensor(preproc_t_ref, dtype=torch.float32, device=device)
    )
    preproc_R_src = (
        torch.eye(3, dtype=torch.float32, device=device)
        if preproc_R_src is None
        else torch.as_tensor(preproc_R_src, dtype=torch.float32, device=device)
    )
    preproc_t_src = (
        torch.zeros((3,), dtype=torch.float32, device=device)
        if preproc_t_src is None
        else torch.as_tensor(preproc_t_src, dtype=torch.float32, device=device)
    )
    # B) Preprocess
    if not is_preprocessed:
        if debug:
            data_dict["raw_gt_rotation"] = rotation.clone()
            data_dict["raw_gt_translation"] = translation.clone()
        # 1) Centering normalization
        if normalize:
            ref_points, src_points, rotation, translation, c_ref, c_src = center_pair(
                ref_points, src_points, rotation, translation
            )
            I = torch.eye(3, dtype=ref_points.dtype, device=device)
            t_ref_op = (-c_ref).to(device)
            t_src_op = (-c_src).to(device)
            preproc_R_ref, preproc_t_ref = update_preproc_ref_only(
                preproc_R_ref, preproc_t_ref, I, t_ref_op
            )
            preproc_R_src, preproc_t_src = update_preproc_src_only(
                preproc_R_src, preproc_t_src, I, t_src_op
            )
        # clean up
        del I, t_ref_op, t_src_op, c_ref, c_src
        gc.collect()

        # 2) Remove outliers
        if estimate_missing_normals:
            if ref_normals is None:
                ref_normals = estimate_normals(ref_points, device=device)
            if src_normals is None:
                src_normals = estimate_normals(src_points, device=device)
        if ref_normals is not None:
            ref_normals = ref_normals.to(torch.float32)
            ref_normals /= ref_normals.norm(dim=1, keepdim=True) + 1e-8
        if src_normals is not None:
            src_normals = src_normals.to(torch.float32)
            src_normals /= src_normals.norm(dim=1, keepdim=True) + 1e-8

        ref_points, ref_normals = remove_outliers(ref_points, ref_normals, device=device)
        src_points, src_normals = remove_outliers(src_points, src_normals, device=device)

        # 3) Voxel downsample
        if voxel_size is not None:
            TARGET_POINT_COUNT = 30000
            # Always use dynamic voxel size based on actual point cloud density
            voxel_size_ref = calculate_dynamic_voxel_size(ref_points, TARGET_POINT_COUNT)
            voxel_size_src = calculate_dynamic_voxel_size(src_points, TARGET_POINT_COUNT)
            # clean up
            gc.collect()

            ref_points, ref_normals = safe_unpack_func_result(
                voxel_downsample(ref_points, voxel_size_ref, normals=ref_normals, device=device)
            )
            src_points, src_normals = safe_unpack_func_result(
                voxel_downsample(src_points, voxel_size_src, normals=src_normals, device=device)
            )

            # clean up
            del voxel_size_ref, voxel_size_src
            gc.collect()

        # 4) Scaling (to meters) : If no centering -> scale
        ref_points = convert_mm_to_m(points_mm=ref_points, scale=preproc_scale, device=device)
        src_points = convert_mm_to_m(points_mm=src_points, scale=preproc_scale, device=device)

        # Create scaling transformation matrix for preproc tracking
        T_scale = torch.eye(3, dtype=ref_points.dtype, device=ref_points.device)
        T_scale[:3, :3] *= preproc_scale
        t_zero = torch.zeros(3, dtype=ref_points.dtype, device=ref_points.device)

        # Update preproc transforms (these track what was applied to the points)
        preproc_R_ref, preproc_t_ref = update_preproc_ref_only(
            preproc_R_ref, preproc_t_ref, T_scale.to(device), t_zero.to(device)
        )
        preproc_R_src, preproc_t_src = update_preproc_src_only(
            preproc_R_src, preproc_t_src, T_scale.to(device), t_zero.to(device)
        )

        # Update ground truth mapping: when both point clouds scale by same factor,
        # rotation stays same (dimensionless) but translation scales proportionally
        translation = translation * preproc_scale

        # clean up
        del T_scale, t_zero
        gc.collect()

        # 5) snapshot pre-augmentation core for caching
        data_dict["pre_ref_points"] = torch.as_tensor(
            ref_points, dtype=torch.float32, device=device
        )
        data_dict["pre_src_points"] = torch.as_tensor(
            src_points, dtype=torch.float32, device=device
        )
        data_dict["pre_ref_normals"] = (
            torch.as_tensor(ref_normals, dtype=torch.float32, device=device)
            if ref_normals is not None
            else None
        )
        data_dict["pre_src_normals"] = (
            torch.as_tensor(src_normals, dtype=torch.float32, device=device)
            if src_normals is not None
            else None
        )
        data_dict["pre_rotation"] = torch.as_tensor(rotation, dtype=torch.float32, device=device)
        data_dict["pre_translation"] = torch.as_tensor(
            translation, dtype=torch.float32, device=device
        )
        data_dict["pre_transform"] = torch.as_tensor(
            get_transform_from_rotation_translation(rotation, translation),
            dtype=torch.float32,
            device=device,
        )
        data_dict["preproc_R_ref"] = torch.as_tensor(
            preproc_R_ref, dtype=torch.float32, device=device
        )
        data_dict["preproc_t_ref"] = torch.as_tensor(
            preproc_t_ref, dtype=torch.float32, device=device
        )
        data_dict["preproc_R_src"] = torch.as_tensor(
            preproc_R_src, dtype=torch.float32, device=device
        )
        data_dict["preproc_t_src"] = torch.as_tensor(
            preproc_t_src, dtype=torch.float32, device=device
        )

        gc.collect()

    # Post-preprocessing geometry ops
    # Initialize net preproc transforms (column-vector convention: p' = R p + t)
    preproc_R_ref = torch.as_tensor(preproc_R_ref, dtype=torch.float32, device=device)
    preproc_t_ref = torch.as_tensor(preproc_t_ref, dtype=torch.float32, device=device)
    preproc_R_src = torch.as_tensor(preproc_R_src, dtype=torch.float32, device=device)
    preproc_t_src = torch.as_tensor(preproc_t_src, dtype=torch.float32, device=device)

    # 6) (Optional) augmentation (only for training)
    if use_augmentation:
        (
            ref_points,
            src_points,
            ref_normals,
            src_normals,
            rotation,
            translation,
            is_ref_aug,
            R_aug,
            t_aug,
        ) = augment_point_cloud(
            curr_epoch=curr_epoch,
            ref_points=ref_points,
            src_points=src_points,
            rotation=rotation,
            translation=translation,
            ref_normals=ref_normals,
            src_normals=src_normals,
            so3_augmentation=so3_augmentation,
            so3_curriculum_epochs=so3_curriculum_epochs,
            max_so3_rotation_deg=max_so3_rotation_deg,
            translation_jitter_m=translation_jitter_m,
            device=device,
        )
        if is_ref_aug:
            preproc_R_ref, preproc_t_ref = update_preproc_ref_only(
                preproc_R_ref, preproc_t_ref, R_aug, t_aug
            )
        else:
            preproc_R_src, preproc_t_src = update_preproc_src_only(
                preproc_R_src, preproc_t_src, R_aug, t_aug
            )
        # clean up
        del is_ref_aug, R_aug, t_aug
        gc.collect()

    # 7) Final transform with canonicalization
    transform = get_transform_from_rotation_translation(rotation, translation)
    transform, flipped = canonicalize_transform_src_to_ref(ref_points, src_points, transform)
    rotation, translation = get_rotation_translation_from_transform(transform)

    # 8) Build torch tensors and features (ones) on device
    ref_points_t = torch.as_tensor(ref_points, dtype=torch.float32, device=device)
    src_points_t = torch.as_tensor(src_points, dtype=torch.float32, device=device)
    ref_normals_t = (
        torch.as_tensor(ref_normals, dtype=torch.float32, device=device)
        if ref_normals is not None
        else None
    )
    src_normals_t = (
        torch.as_tensor(src_normals, dtype=torch.float32, device=device)
        if src_normals is not None
        else None
    )
    assert ref_points.ndim == 2 and ref_points.shape[1] == 3
    assert src_points.ndim == 2 and src_points.shape[1] == 3
    ref_feats_t = torch.ones((ref_points_t.shape[0], 1), dtype=torch.float32, device=device)
    src_feats_t = torch.ones((src_points_t.shape[0], 1), dtype=torch.float32, device=device)

    # clean up
    del ref_points, src_points, ref_normals, src_normals
    gc.collect()

    data_dict["ref_points"] = ref_points_t
    data_dict["src_points"] = src_points_t
    data_dict["ref_feats"] = ref_feats_t
    data_dict["src_feats"] = src_feats_t
    data_dict["transform"] = torch.as_tensor(transform, dtype=torch.float32, device=device)
    data_dict["rotation"] = torch.as_tensor(rotation, dtype=torch.float32, device=device)
    data_dict["translation"] = torch.as_tensor(translation, dtype=torch.float32, device=device)
    data_dict["preproc_R_ref"] = torch.as_tensor(preproc_R_ref, dtype=torch.float32, device=device)
    data_dict["preproc_t_ref"] = torch.as_tensor(preproc_t_ref, dtype=torch.float32, device=device)
    data_dict["preproc_R_src"] = torch.as_tensor(preproc_R_src, dtype=torch.float32, device=device)
    data_dict["preproc_t_src"] = torch.as_tensor(preproc_t_src, dtype=torch.float32, device=device)
    data_dict["canonical_flipped"] = bool(flipped)
    data_dict["preproc_scale"] = float(preproc_scale)

    # clean up
    del rotation, translation, preproc_R_ref, preproc_t_ref, preproc_R_src, preproc_t_src
    gc.collect()

    if ref_normals_t is not None:
        data_dict["ref_normals"] = ref_normals_t
    if src_normals_t is not None:
        data_dict["src_normals"] = src_normals_t

    # 11) Optional correspondences (training only)
    if return_corr_indices and matching_radius is not None:
        corr_indices = get_correspondences(
            ref_points=ref_points_t,
            src_points=src_points_t,
            transform=transform,
            matching_radius=matching_radius,
            output_type="torch",
            device=device,
        )
        data_dict["corr_indices"] = torch.as_tensor(
            corr_indices, dtype=torch.float32, device=device
        )
        # clean up
        del corr_indices
        gc.collect()

    return data_dict


def center_pair(
    ref_points: Union[torch.Tensor, np.ndarray],
    src_points: Union[torch.Tensor, np.ndarray],
    rotation: Union[torch.Tensor, np.ndarray],
    translation: Union[torch.Tensor, np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Center ref/src clouds around the origin while keeping the rigid
    relation ref = R * src + t valid in the new coordinates.

    ref' = ref - c_ref
    src' = src - c_src
    => ref' = R * src' + t',  where  t' = R * c_src + t - c_ref

    All inputs and outputs are torch.Tensor.
    """
    # Use ref_points device if possible
    device = ref_points.device if isinstance(ref_points, torch.Tensor) else torch.device("cpu")
    dtype = ref_points.dtype if isinstance(ref_points, torch.Tensor) else torch.float32

    ref_points = to_tensor(ref_points, device=device, dtype=dtype)
    src_points = to_tensor(src_points, device=device, dtype=dtype)
    rotation = to_tensor(rotation, device=device, dtype=dtype)
    translation = to_tensor(translation, device=device, dtype=dtype)

    c_ref = (
        ref_points.mean(dim=0)
        if ref_points.shape[0] > 0
        else torch.zeros((3,), dtype=dtype, device=device)
    )
    c_src = (
        src_points.mean(dim=0)
        if src_points.shape[0] > 0
        else torch.zeros((3,), dtype=dtype, device=device)
    )
    ref_c = ref_points - c_ref
    src_c = src_points - c_src
    t_c = rotation @ c_src + translation - c_ref
    return ref_c, src_c, rotation, t_c, c_ref, c_src


def _mean_alignment_error(
    ref_points: Union[np.ndarray, torch.Tensor],
    src_points: Union[np.ndarray, torch.Tensor],
    transform: Union[np.ndarray, torch.Tensor],
) -> float:
    """Mean NN distance between ref and transformed src."""
    is_tensor = torch.is_tensor(ref_points)
    # Use numpy for nearest neighbor (for now)
    if is_tensor:
        ref_np = ref_points.detach().cpu().numpy()
        src_np = src_points.detach().cpu().numpy()
        transform_np = transform.detach().cpu().numpy() if torch.is_tensor(transform) else transform
    else:
        ref_np = ref_points
        src_np = src_points
        transform_np = transform
    aligned_src = apply_transform(src_np, transform_np)  # (M, 3)
    nn_dist = get_nearest_neighbor(ref_np, aligned_src)  # (N,)

    del ref_np, src_np, transform_np, aligned_src
    gc.collect()

    if nn_dist.size == 0:
        return float("inf")
    return float(nn_dist.mean())


def canonicalize_transform_src_to_ref(
    ref_points: Union[np.ndarray, torch.Tensor],
    src_points: Union[np.ndarray, torch.Tensor],
    transform: Union[np.ndarray, torch.Tensor],
    tol_factor: float = 0.9,
) -> Tuple[torch.Tensor, bool]:
    """
    Make sure `transform` maps src to ref.

    Given an unknown 4x4 `transform` (might be src->ref or ref->src),
    we test both directions and pick the one with lower alignment error.

    Returns:
        transform_canon: 4x4 src->ref transform
        flipped: bool, True if we inverted the original transform
    """
    is_tensor = torch.is_tensor(transform)
    if is_tensor:
        transform_np = transform.detach().cpu().numpy()
    else:
        transform_np = transform
    device = transform.device if is_tensor else torch.device("cpu")
    err_src_to_ref = _mean_alignment_error(ref_points, src_points, transform_np)
    transform_inv = np.linalg.inv(transform_np)
    err_ref_to_src = _mean_alignment_error(ref_points, src_points, transform_inv)

    if err_ref_to_src < tol_factor * err_src_to_ref:
        T = to_tensor(transform_inv, device=device)
    else:
        T = to_tensor(transform_np, device=device)

    del transform_np, transform_inv
    gc.collect()

    if is_tensor:
        return T.to(device), err_ref_to_src < tol_factor * err_src_to_ref
    return T, err_ref_to_src < tol_factor * err_src_to_ref


def augment_point_cloud(
    curr_epoch: int,
    ref_points: Union[np.ndarray, torch.Tensor],
    src_points: Union[np.ndarray, torch.Tensor],
    rotation: Union[np.ndarray, torch.Tensor],
    translation: Union[np.ndarray, torch.Tensor],
    ref_normals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    src_normals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    so3_augmentation: bool = False,
    so3_curriculum_epochs: int = 50,
    max_so3_rotation_deg: float = 180.0,
    translation_jitter_m: float = 0.005,
    device: Optional[torch.device] = None,
) -> Tuple[
    Union[np.ndarray, torch.Tensor],
    Union[np.ndarray, torch.Tensor],
    Optional[Union[np.ndarray, torch.Tensor]],
    Optional[Union[np.ndarray, torch.Tensor]],
    Union[np.ndarray, torch.Tensor],
    Union[np.ndarray, torch.Tensor],
    bool,
    Union[np.ndarray, torch.Tensor],
    Union[np.ndarray, torch.Tensor],
]:
    """Augment point clouds (device-aware).
    Returns ref_points, src_points, rotation, translation (all updated).
    """
    if device is None:
        device = (
            ref_points.device
            if torch.is_tensor(ref_points)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    # Convert inputs
    ref_points = torch.as_tensor(ref_points, dtype=torch.float32, device=device)
    src_points = torch.as_tensor(src_points, dtype=torch.float32, device=device)
    rotation = torch.as_tensor(rotation, dtype=torch.float32, device=device)
    translation = torch.as_tensor(translation, dtype=torch.float32, device=device)
    ref_normals = (
        torch.as_tensor(ref_normals, dtype=torch.float32, device=device)
        if ref_normals is not None
        else None
    )
    src_normals = (
        torch.as_tensor(src_normals, dtype=torch.float32, device=device)
        if src_normals is not None
        else None
    )
    # Sample augmentation rotation and translation
    if so3_augmentation:
        frac = min(1.0, curr_epoch / max(1, so3_curriculum_epochs))
        max_angle = np.deg2rad(frac * max_so3_rotation_deg)
        R_aug = random_sample_rotation_v3(max_angle=max_angle, device=device)
    else:
        max_angle = np.deg2rad(180.0)
        R_aug = random_sample_rotation_v3(max_angle=max_angle, device=device)

    t_aug = np.random.normal(scale=translation_jitter_m, size=(3,))
    # Convert R_aug, t_aug to tensor
    R_aug = to_tensor(R_aug, device=device)
    t_aug = to_tensor(t_aug, device=device)

    # clean up
    del max_angle
    gc.collect()

    # Choose whether to augment ref or src
    is_ref_aug = random.random() > 0.5
    if is_ref_aug:
        # Augment reference and update mapping via compose_ref_op
        ref_points = ref_points @ R_aug.T + t_aug.unsqueeze(0)
        rotation, translation = compose_ref_op(rotation, translation, R_aug, t_aug)
        if ref_normals is not None:
            ref_normals = ref_normals @ R_aug.T
            ref_normals = torch.nn.functional.normalize(ref_normals, dim=1, eps=1e-8)
    else:
        # Augment source and update mapping via compose_src_op
        src_points = src_points @ R_aug.T + t_aug.unsqueeze(0)
        rotation, translation = compose_src_op(rotation, translation, R_aug, t_aug)
        if src_normals is not None:
            src_normals = src_normals @ R_aug.T
            src_normals = torch.nn.functional.normalize(src_normals, dim=1, eps=1e-8)
    return (
        ref_points,
        src_points,
        ref_normals,
        src_normals,
        rotation,
        translation,
        is_ref_aug,
        R_aug,
        t_aug,
    )


# misc
def safe_unpack_func_result(
    result: Tuple,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor, None]]:
    """
    Accept either arr or (arr, normals) and return (arr, normals_or_None).
    Example: res = func(); points, norms = safe_unpack_func_result(res)
    Supports np.ndarray or torch.Tensor.
    """
    if isinstance(result, tuple) or isinstance(result, list):
        if len(result) == 2:
            return result[0], result[1]
        return result[0], None
    else:
        return result, None


def chunked_knn_torch(
    points: torch.Tensor, k: int, chunk_size: int = 512, device=None, pad_idx: int = -1
):
    """
    Exact k-NN using torch.cdist computed in chunks to avoid O(N^2) peak memory.
    Returns (indices, dists) where indices shape = (N, k), dists shape = (N, k).
    - pads with pad_idx and large distance if not enough neighbors (e.g. N small).
    """
    if device is None:
        device = points.device if torch.is_tensor(points) else torch.device("cpu")
    points = points.to(device).contiguous().float()
    N, D = points.shape
    if N == 0:
        return (
            torch.empty((0, k), dtype=torch.long, device=device),
            torch.empty((0, k), device=device),
        )
    # Handle trivial cases
    if N == 1:
        idx = torch.full((1, k), pad_idx, dtype=torch.long, device=device)
        d = torch.full((1, k), float("inf"), device=device)
        return idx, d

    # block-by-block: (chunk_size x N)
    all_idx = []
    all_d = []
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        block = points[start:end]  # (B, D)

        # squared distances
        d2 = torch.cdist(block, points, p=2.0)  # (B, N)
        # exclude self (set self dist to +inf)
        self_slice = torch.arange(start, end, device=device)
        d2[:, self_slice] = float("inf")

        k_eff = min(k, N - 1)
        d_block, i_block = torch.topk(d2, k_eff, largest=False, dim=1)

        if k_eff < k:
            pad_count = k - k_eff
            pad_idx_tensor = torch.full(
                (d_block.shape[0], pad_count), pad_idx, dtype=torch.long, device=device
            )
            pad_d_tensor = torch.full((d_block.shape[0], pad_count), float("inf"), device=device)
            i_block = torch.cat([i_block, pad_idx_tensor], dim=1)
            d_block = torch.cat([d_block, pad_d_tensor], dim=1)
            del pad_idx_tensor, pad_d_tensor
        all_idx.append(i_block)
        all_d.append(d_block)

        # clean up
        del block, d2, self_slice, i_block, d_block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    idx = torch.cat(all_idx, dim=0)
    dists = torch.cat(all_d, dim=0)
    # clean up
    del all_idx, all_d
    gc.collect()

    return idx, dists


def knn_cKDTree(points: np.ndarray, k: int, pad_idx: int = -1):
    """
    Exact CPU k-NN using scipy.spatial.cKDTree.
    points: (N,3) numpy array
    Returns (indices, dists) as numpy arrays. Pads with pad_idx / inf if not enough neighbors.
    """
    if cKDTree is None:
        raise RuntimeError("scipy.spatial.cKDTree not available")
    N = points.shape[0]
    if N == 0:
        return np.empty((0, k), dtype=np.int64), np.empty((0, k), dtype=np.float32)
    if N == 1:
        idx = np.full((1, k), pad_idx, dtype=np.int64)
        d = np.full((1, k), np.inf, dtype=np.float32)
        return idx, d

    tree = cKDTree(points)
    # ignore self (distance 0)
    k_eff = min(k + 1, N)
    dists, idx = tree.query(points, k=k_eff)
    # clean up
    del tree
    gc.collect()

    # If k_eff == 1 then shapes differ; normalize
    if idx.ndim == 1:
        idx = idx[:, None]
        dists = dists[:, None]
    # remove self
    for i in range(N):
        # find where idx[i] == i and set that distance to inf
        mask_self = idx[i] == i
        if mask_self.any():
            dists[i][mask_self] = np.inf

    # now select k smallest per row
    order = np.argsort(dists, axis=1)
    chosen = order[:, :k]
    idx_k = np.take_along_axis(idx, chosen, axis=1)
    d_k = np.take_along_axis(dists, chosen, axis=1)

    if idx_k.shape[1] < k:
        pad_width = k - idx_k.shape[1]
        idx_k = np.pad(idx_k, ((0, 0), (0, pad_width)), constant_values=pad_idx)
        d_k = np.pad(d_k, ((0, 0), (0, pad_width)), constant_values=np.inf)

    # clean up
    del order, chosen, dists, idx
    gc.collect()

    return idx_k.astype(np.int64), d_k.astype(np.float32)


def knn_auto(points: torch.Tensor, k: int, prefer_gpu: bool = True, chunk_size: int = 512):
    """
    Automatic selection:
      - If GPU available & prefer_gpu -> use chunked_knn_torch on GPU
      - Else fallback to cKDTree (CPU)
    Returns indices (N,k) torch.long and dists (N,k) torch.float
    """
    target_device = points.device if torch.is_tensor(points) else torch.device("cpu")
    if prefer_gpu and torch.cuda.is_available():
        try:
            idx, dists = chunked_knn_torch(
                points.to(torch.device("cuda")),
                k,
                chunk_size=chunk_size,
                device=torch.device("cuda"),
            )
            result = idx.to(target_device), dists.to(target_device)
            # Clean up GPU tensors
            del idx, dists
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result
        except Exception:
            # fallback to CPU
            points_np = points.cpu().numpy()
            idx_np, d_np = knn_cKDTree(points_np, k)
            result = (
                to_tensor(idx_np, device=target_device, dtype=torch.long),
                to_tensor(d_np, device=target_device, dtype=torch.float32),
            )
            del points_np, idx_np, d_np
            gc.collect()
            return result
    else:
        points_np = points.cpu().numpy()
        idx_np, d_np = knn_cKDTree(points_np, k)
        result = (
            to_tensor(idx_np, device=target_device, dtype=torch.long),
            to_tensor(d_np, device=target_device, dtype=torch.float32),
        )
        # Clean up numpy arrays
        del points_np, idx_np, d_np
        gc.collect()
        return result


def estimate_normals(
    points: Union[np.ndarray, torch.Tensor], k: int = 16, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Estimate normals for a point cloud using PyTorch (GPU if available).
    Supports np.ndarray or torch.Tensor input.
    Returns (N, 3) torch.Tensor of normals on device.
    """
    if device is None:
        device = (
            points.device
            if torch.is_tensor(points)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    if torch.is_tensor(points):
        points = to_tensor(points, device=device, dtype=torch.float32)
    else:
        points = to_tensor(points, device=device, dtype=torch.float32)

    N = points.shape[0]
    if N == 0:
        return torch.empty((0, 3), dtype=points.dtype, device=device)
    if N == 1:
        return torch.zeros((1, 3), dtype=points.dtype, device=device)

    # get neighbor indices and distances
    idx, dists = knn_auto(points, k, prefer_gpu=True, chunk_size=(N // 100) + 1)
    idx_clamped = idx.clamp(min=0)  # (N, k)
    gathered = points[idx_clamped]  # (N, k, 3)
    valid_mask = (idx != -1).to(points.dtype).unsqueeze(-1)  # (N, k, 1) float mask

    # per-point neighbor counts
    counts = valid_mask.sum(dim=1).clamp(min=1.0)  # (N, 1)

    # centroid using only valid neighbors
    sums = (gathered * valid_mask).sum(dim=1)  # (N, 3)
    centroid = sums / counts  # (N, 3)

    # center neighbors
    neigh_centered = (gathered - centroid.unsqueeze(1)) * valid_mask

    # compute covariance matrices robustly: cov = (X^T X) / counts
    # neigh_centered: (N, k, 3) -> permute to (N, 3, k) for matmul
    neigh_T = neigh_centered.permute(0, 2, 1)  # (N, 3, k)
    cov = (neigh_T @ neigh_centered) / counts.unsqueeze(-1)  # (N, 3, 3)

    # For numerical stability, make cov symmetric
    cov = 0.5 * (cov + cov.permute(0, 2, 1))

    # compute smallest eigenvector of cov (normal)
    try:
        e_vals, e_vecs = torch.linalg.eigh(cov)  # e_vecs: (N, 3, 3) columns are eigenvectors
        normals = e_vecs[:, :, 0]  # smallest eigenvalue -> index 0 (N,3)
    except Exception:
        # fallback to SVD per item (slower) but robust
        normals_list = []
        for i in range(N):
            try:
                u, s, v = torch.svd(cov[i])
                normals_list.append(v[:, -1])
            except Exception:
                # fallback: use cross-product of two neighbor vectors if possible
                neigh = gathered[i] if "gathered" in locals() else None
                valid = (
                    (idx[i] != -1).nonzero(as_tuple=False).squeeze(1) if "idx" in locals() else None
                )
                if valid is not None and valid.numel() >= 2:
                    a, b = neigh[valid[0]], neigh[valid[1]]
                    n = torch.cross(a - centroid[i], b - centroid[i])
                else:
                    n = torch.tensor([0.0, 0.0, 0.0], device=device)
                normals_list.append(n)
        normals = torch.stack(normals_list, dim=0)

    # normalize normals (handle zero vectors)
    normals = torch.nn.functional.normalize(normals, dim=1, eps=1e-8)
    # Clean up
    del cov, e_vals, e_vecs, centroid
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return normals


def remove_outliers(
    points: Union[np.ndarray, torch.Tensor],
    norms: Optional[Union[np.ndarray, torch.Tensor]] = None,
    k: int = 16,
    std_ratio: float = 2.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Remove statistical outliers using PyTorch (GPU if available).
    Supports np.ndarray or torch.Tensor input.
    Returns (M, 3) torch.Tensor of filtered points (and normals if provided) on device.
    """
    if device is None:
        device = (
            points.device
            if torch.is_tensor(points)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    if torch.is_tensor(points):
        points = points.float().to(device)
    else:
        points = to_tensor(points, device=device, dtype=torch.float32)

    N = points.shape[0]
    if N == 0:
        return points, None if norms is None else torch.empty((0, 3), device=device)
    if N == 1:
        if norms is not None:
            if torch.is_tensor(norms):
                return points, to_tensor(norms, device=device, dtype=torch.float32)
            else:
                return points, to_tensor(norms, device=device, dtype=torch.float32)
        return points, None

    # get neighbor indices and distances
    idx, dists = knn_auto(points, k, prefer_gpu=True, chunk_size=(N // 100) + 1)  # (N, k), (N, k)
    # mean distance per row
    finite_mask = torch.isfinite(dists)
    finite_counts = finite_mask.sum(dim=1).clamp(min=1).to(dists.dtype)  # (N,)
    # sum finite distances
    sum_d = torch.where(finite_mask, dists, torch.tensor(0.0, device=dists.device)).sum(dim=1)
    mean_dists = sum_d / finite_counts  # (N,)

    # global mean and std
    finite_rows_mask = torch.isfinite(mean_dists)
    if finite_rows_mask.any():
        mean = mean_dists[finite_rows_mask].mean()
        std = mean_dists[finite_rows_mask].std(unbiased=False)
    else:
        del idx, dists, finite_mask, finite_counts, sum_d, mean_dists
        gc.collect()
        return (
            points,
            (
                None
                if norms is None
                else (
                    norms
                    if torch.is_tensor(norms)
                    else to_tensor(norms, device=device, dtype=torch.float32)
                )
            ),
        )

    threshold = mean + std_ratio * (
        std if not torch.isnan(std) else torch.tensor(0.0, device=device)
    )

    keep_mask = mean_dists < threshold
    keep_mask = keep_mask.bool()

    filtered_points = points[keep_mask]
    filtered_norms = None
    if norms is not None:
        if torch.is_tensor(norms):
            norms_t = to_tensor(norms, device=device, dtype=torch.float32)
        else:
            norms_t = to_tensor(norms, device=device, dtype=torch.float32)
        filtered_norms = norms_t[keep_mask]

    # Clean up intermediate tensors
    del (
        idx,
        dists,
        finite_mask,
        finite_counts,
        sum_d,
        mean_dists,
        finite_rows_mask,
        mean,
        std,
        threshold,
        keep_mask,
    )
    if norms_t in locals():
        del norms_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return filtered_points, filtered_norms


def calculate_dynamic_voxel_size(points: torch.Tensor, target_n_points: int) -> float:
    """
    Calculate dynamic voxel size based on point cloud bounding box and target number of points.
    points: (N, 3) torch.Tensor
    target_n_points: desired number of points after downsampling
    Returns voxel size as float.
    """
    min_coords = torch.min(points, dim=0).values
    max_coords = torch.max(points, dim=0).values
    dimensions = max_coords - min_coords
    V_bbox = torch.prod(dimensions)
    actual_n_points = points.shape[0]
    adjusted_target = max(actual_n_points // 2, target_n_points)

    V_voxel = V_bbox / adjusted_target
    l = torch.pow(V_voxel, 1 / 3)

    min_voxel_size = 0.35
    result = max(l.item(), min_voxel_size)

    del min_coords, max_coords, dimensions, V_bbox, V_voxel, l
    gc.collect()

    return result


def voxel_downsample(
    points: Union[np.ndarray, torch.Tensor],
    voxel_size: float,
    normals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Downsample point cloud using voxel grid with PyTorch (GPU if available).
    Supports np.ndarray or torch.Tensor input.
    Returns (M, 3) torch.Tensor of downsampled points (and normals if provided) on device.
    """
    if device is None:
        device = (
            points.device
            if torch.is_tensor(points)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    if torch.is_tensor(points):
        points = points.float().to(device)
    else:
        points = to_tensor(points, device=device, dtype=torch.float32)

    coords = torch.floor(points / voxel_size)
    unique_coords, idx = torch.unique(coords, dim=0, return_inverse=True)
    voxel_means = torch.zeros((idx.max() + 1, 3), device=device)
    voxel_counts = torch.zeros((idx.max() + 1,), device=device)
    voxel_means.index_add_(0, idx, points)
    voxel_counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    voxel_means = voxel_means / voxel_counts.unsqueeze(1)

    del coords, unique_coords, voxel_counts
    gc.collect()

    norms_t = None
    if normals is not None:
        if torch.is_tensor(normals):
            norms_t = normals.float().to(device)
        else:
            norms_t = to_tensor(normals, device=device, dtype=torch.float32)
        voxel_normals = torch.zeros((idx.max() + 1, 3), device=device)
        voxel_normals.index_add_(0, idx, norms_t)

        del idx, norms_t
        gc.collect()

        voxel_normals = torch.nn.functional.normalize(voxel_normals, dim=1)
        return voxel_means, voxel_normals

    del idx
    gc.collect()

    return voxel_means, None


def reprojection_stats(original_ref_points, original_src_points, rotation, translation, note=""):
    """
    original_ref_points, original_src_points: numpy arrays (N,3)/(M,3) *in same units* used in mapping*
    rotation: (3,3) torch or numpy; translation: (3,) torch or numpy
    The mapping assumed: ref = R * src + t  (column-vector)
    """
    T = get_transform_from_rotation_translation(rotation, translation)  # (4,4)
    aligned_src = apply_transform(original_src_points, T)  # (M,3)
    dists = get_nearest_neighbor(original_ref_points, aligned_src)  # (N,)
    if dists.size == 0:
        return {"mean": np.inf}
    stats = {
        "mean": float(dists.cpu().numpy().mean()),
        "median": float(np.median(dists.cpu().numpy())),
        "max": float(dists.cpu().numpy().max()),
    }
    print(
        f"[REPROJ] {note} mean={stats['mean']:.6f} median={stats['median']:.6f} max={stats['max']:.6f}"
    )

    del T, aligned_src, dists
    gc.collect()

    return stats


def invert_affine_R_t(
    affinite_R: torch.Tensor, affinite_t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (R_inv, t_inv) as torch.Tensors (float32).
    Uses column-vector convention: processed = R @ raw + t
    => raw = R_inv @ (processed - t), so t_inv = -R_inv @ t.
    Works if inputs are numpy or torch. Output is on same device as input torch (or cpu if numpy).
    """
    # convert to torch
    if torch.is_tensor(affinite_R):
        device = affinite_R.device
        R_t = affinite_R.float()
        t_t = affinite_t.float()
    else:
        device = torch.device("cpu")
        R_t = to_tensor(np.asarray(affinite_R), device=device).float()
        t_t = to_tensor(np.asarray(affinite_t), device=device).float()

    R_inv = torch.linalg.inv(R_t)
    t_inv = -R_inv @ t_t
    return R_inv, t_inv


def check_reproject_processed_to_raw(
    processed_points: torch.Tensor,
    raw_points: torch.Tensor,
    preproc_R: torch.Tensor,
    preproc_t: torch.Tensor,
    atol: float = 1e-5,
    note="",
) -> Tuple[torch.Tensor, float, float]:

    R_inv, t_inv = invert_affine_R_t(preproc_R, preproc_t)
    reprojected_raw = (processed_points @ R_inv.T) + t_inv.unsqueeze(0)

    dists = get_nearest_neighbor(raw_points, reprojected_raw)
    mean_d = float(dists.mean().cpu().item())
    max_d = float(dists.max().cpu().item())
    print(f"[REPROJ] {note} processed->raw: mean={mean_d:.6e}, max={max_d:.6e}")

    del R_inv, t_inv, dists
    gc.collect()

    return reprojected_raw, mean_d, max_d


def reproject_processed_to_raw(
    processed_points: torch.Tensor,
    preproc_R: torch.Tensor,
    preproc_t: torch.Tensor,
    atol: float = 1e-5,
) -> torch.Tensor:
    # invert the affinite transform to map processed -> original
    R_inv, t_inv = invert_affine_R_t(preproc_R, preproc_t)
    reprojected_raw = (processed_points @ R_inv.T) + t_inv.unsqueeze(0)

    del R_inv, t_inv
    gc.collect()

    return reprojected_raw
