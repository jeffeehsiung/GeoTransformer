import os
import os.path as osp
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict as edict
from geotransformer.utils.data import _to_tensor
from geotransformer.utils.open3d import make_open3d_corr_lines, make_open3d_point_cloud
from geotransformer.utils.visualization import (
    _to_numpy,
    build_node_correspondence_visualization,
    draw_node_correspondences,
    get_colors_with_tsne,
)
from nova.proto.model import model_config_pb2
from sklearn.decomposition import PCA

import iris
from geoTransformer.GeoTransformer.experiments.roboeye.config import (
    make_cfg,
    setup_dataset_config,
    setup_dataset_proto,
)
from geoTransformer.GeoTransformer.experiments.roboeye.preprocess import (
    calculate_dynamic_voxel_size,
    invert_affine_R_t,
    remove_outliers,
    reproject_processed_to_raw,
    safe_unpack_func_result,
    voxel_downsample,
)


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert torch.Tensor to numpy.ndarray."""
    return _to_numpy(x)


def to_tensor(
    x: Union[np.ndarray, torch.Tensor], device: Optional[torch.device] = None
) -> torch.Tensor:
    """Convert numpy.ndarray to torch.Tensor."""
    return _to_tensor(x, device=device)


def recommend_voxel_size(median_inter_point_distance: float, strategy: str = "conservative"):
    """
    Recommend optimal voxel size based on point cloud characteristics

    Args:
        median_inter_point_distance: Median distance between points in meters
        strategy: 'conservative', 'balanced', or 'robust'

    Returns:
        Optimal voxel size and expected neighbor count
    """
    strategies = {
        "conservative": 1.2,  # 20% larger than median → preserve detail (0.009 → 0.0108 ≈ 0.011m)
        "balanced": 1.5,  # 50% larger → good balance (0.009 → 0.0135 ≈ 0.014m)
        "robust": 1.8,  # 80% larger → maximum robustness (0.009 → 0.0162 ≈ 0.016m)
    }

    multiplier = strategies.get(strategy, 1.5)
    target_voxel_size = median_inter_point_distance * multiplier

    # Round to computationally efficient values
    efficient_sizes = [0.008, 0.010, 0.011, 0.012, 0.014, 0.016, 0.018, 0.020, 0.024]
    optimal_voxel_size = min(efficient_sizes, key=lambda x: abs(x - target_voxel_size))

    # Estimate neighbor count (empirical relationship)
    ratio = optimal_voxel_size / median_inter_point_distance
    expected_neighbors = int(8 * ratio ** 2.5)  # More accurate for sparse→dense transition

    return optimal_voxel_size, expected_neighbors


def create_transfer_learning_cfg(
    model_config_proto: model_config_pb2.ModelConfig = None,
    params: Any = None,
    dataset_id: str = None,
    dataset_root: str = None,
    output_dir: str = None,
    log_dir: str = None,
    snapshot_dir: str = None,
    phase: int = None,
) -> edict:
    """
    Create configuration for transfer learning with adapted parameters

    Args:
        model_config_proto: Base model configuration proto
        dataset_id: Dataset ID for RoboEye custom datasets (e.g., "roboeye/yolo_pretrain/2025-08-08-14-41-05")
        dataset_root: Root path for BOP datasets (e.g., "~/repos/roboeye/datasets/ITODD_converted")
        output_dir: Output directory for logs and snapshots
        log_dir: Log directory
        snapshot_dir: Snapshot directory
        phase: Training phase for auto-generated output directory
    """
    # Helper to get param from dict or object
    def get_param(key: str, default=None):
        if params is None:
            return default
        value = (
            params.get(key, default) if isinstance(params, dict) else getattr(params, key, default)
        )
        return value if value is not None else default

    def get_nested_param(obj: Union[Dict, Any], key: str, default=None):
        if obj is None:
            return default
        value = obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
        return value if value is not None else default

    # Create config with integrated daxtaset and directory setup
    cfg = make_cfg(output_dir=output_dir, log_dir=log_dir, snapshot_dir=snapshot_dir, phase=phase,)
    if model_config_proto:
        cfg = setup_dataset_proto(cfg, model_config=model_config_proto)
    else:
        cfg = setup_dataset_config(cfg, dataset_id=dataset_id, dataset_root=dataset_root)
    # Analyze voxel size optimization
    target_voxel_size, expected_neighbors = recommend_voxel_size(
        median_inter_point_distance=get_param("target_voxel_size", 0.009),
        strategy=get_param("voxel_strategy", "conservative"),
    )

    # Scale parameters based on voxel size ratio
    pretrained_voxel_size = get_param("pretrained_voxel_size", cfg.backbone.init_voxel_size)
    scale_factor = target_voxel_size / pretrained_voxel_size

    # Update configuration for new scale
    cfg.backbone.init_voxel_size = cfg.backbone.init_voxel_size * scale_factor

    # Scale radius parameters
    cfg.backbone.init_radius = cfg.backbone.init_radius * scale_factor
    cfg.backbone.init_sigma = cfg.backbone.init_sigma * scale_factor

    # Update GeoTransformer parameters based on new scale
    cfg.geotransformer.sigma_d = 2 * cfg.backbone.init_sigma * scale_factor

    # angle_k controls the number of angular neighbors
    if scale_factor < 0.5:  # If target is less than half the original scale
        cfg.geotransformer.angle_k = max(2, int(cfg.geotransformer.angle_k * (1 / scale_factor)))
    else:
        cfg.geotransformer.angle_k = cfg.geotransformer.angle_k

    # Adjust fine matching parameters for smaller scale
    cfg.fine_matching.acceptance_radius = cfg.fine_matching.acceptance_radius * scale_factor
    cfg.model.ground_truth_matching_radius = cfg.model.ground_truth_matching_radius * scale_factor

    cfg.train.matching_radius = cfg.model.ground_truth_matching_radius
    cfg.test.matching_radius = cfg.model.ground_truth_matching_radius

    cfg.fine_loss.positive_radius = cfg.fine_loss.positive_radius * scale_factor
    cfg.coarse_loss.positive_margin = cfg.coarse_loss.positive_margin * scale_factor

    # Transfer learning specific settings - Enhanced Strategic Framework
    cfg.transfer_learning = {
        # Top-level transfer learning settings
        "cache_bool": get_param("cache_bool", False),
        "pretrained_voxel_size": pretrained_voxel_size,
        "target_voxel_size": target_voxel_size,
        "keep_ratio": get_param("train_split", 0.7),
        "pretrained_weights": get_param(
            "pretrained_weights",
            os.path.join(
                iris.pretrained_weights_dir(), "geotransformer/geotransformer-3dmatch.pth.tar"
            ),
        ),
        "phase": get_param("phase", 0),
        "resume": get_param("resume", False),
        "log_steps": get_param("log_steps", 10),
        "continue_optimizer": get_param("continue_optimizer", True),
        "optim_lr": get_param("optim_lr", 5e-6 if params.phase is not None else 1e-4),
        "optim_max_epoch": get_param("optim_max_epoch", 50),
        "gradient_accumulation_steps": get_param("gradient_accumulation_steps", 4),
        "enable_progressive_unfreezing": get_param("enable_progressive_unfreezing", False),
        "freeze_backbone": get_param("freeze_backbone", True),
        "freeze_epochs": get_param("freeze_epochs", [8, 20, 32, 42]),
        "learning_rates": get_param("learning_rates", [5e-6, 1e-5, 2e-5, 5e-5, 1e-4]),
        "adaptation_phases": get_param("adaptation_phases", 5),
        "scale_factor": scale_factor,
        "scale_validation_epochs": get_param("scale_validation_epochs", 3),
        "architectural_compatibility_check": get_param("architectural_compatibility_check", True),
        "nan_loss_threshold": get_param("nan_loss_threshold", 3),
        "feature_adaptation_strategy": get_param("feature_adaptation_strategy", "hierarchical"),
        "attention_warmup_epochs": get_param("attention_warmup_epochs", 5),
        "cross_modal_learning_focus": get_param("cross_modal_learning_focus", True),
        "backbone_integration_strategy": get_param(
            "backbone_integration_strategy", "decoder_first"
        ),
        "feature_compatibility_monitoring": get_param("feature_compatibility_monitoring", True),
        "adaptive_learning_rate": get_param("adaptive_learning_rate", True),
        # Curriculum learning settings
        "curriculum_learning": {
            "enable_geometric_curriculum": get_nested_param(
                get_param("curriculum_learning", None), "enable_geometric_curriculum", False
            ),
            "enable_so3_curriculum": get_nested_param(
                get_param("curriculum_learning", None), "enable_so3_curriculum", False
            ),
            "rotation_curriculum_epochs": get_nested_param(
                get_param("curriculum_learning", None), "rotation_curriculum_epochs", 12
            ),
            "translation_curriculum_epochs": get_nested_param(
                get_param("curriculum_learning", None), "translation_curriculum_epochs", 8
            ),
            "scale_curriculum_epochs": get_nested_param(
                get_param("curriculum_learning", None), "scale_curriculum_epochs", 6
            ),
            "noise_curriculum_epochs": get_nested_param(
                get_param("curriculum_learning", None), "noise_curriculum_epochs", 10
            ),
            "so3_curriculum_epochs": get_nested_param(
                get_param("curriculum_learning", None), "so3_curriculum_epochs", 50
            ),
            "max_so3_rotation_deg": get_nested_param(
                get_param("curriculum_learning", None), "max_so3_rotation_deg", 180
            ),
        },
        # Multi-scale training strategy
        "multi_scale_strategy": {
            "enable_multi_scale": get_nested_param(
                get_param("multi_scale_strategy", None), "enable_multi_scale", False
            ),
            "voxel_size_variations": get_nested_param(
                get_param("multi_scale_strategy", None),
                "voxel_size_variations",
                [0.008, 0.009, 0.010, 0.011],
            ),
            "scale_scheduling": get_nested_param(
                get_param("multi_scale_strategy", None), "scale_validation_epochs", "progressive"
            ),
            "scale_mixing_ratio": get_nested_param(
                get_param("multi_scale_strategy", None), "scale_mixing_ratio", 0.3
            ),
        },
        # Adaptive loss weighting
        "adaptive_loss_weighting": {
            "enable_adaptive_weighting": get_nested_param(
                get_param("adaptive_loss_weighting", None), "enable_adaptive_weighting", False
            ),
            "coarse_loss_emphasis_epochs": get_nested_param(
                get_param("adaptive_loss_weighting", None), "coarse_loss_emphasis_epochs", 15
            ),
            "fine_loss_emphasis_epochs": get_nested_param(
                get_param("adaptive_loss_weighting", None), "fine_loss_emphasis_epochs", 20
            ),
            "dynamic_loss_balancing": get_nested_param(
                get_param("adaptive_loss_weighting", None), "dynamic_loss_balancing", True
            ),
        },
        # Robustness enhancements
        "robustness_strategy": {
            "gradient_clipping": get_nested_param(
                get_param("robustness_strategy", None), "gradient_clipping", True
            ),
            "gradient_clip_value": get_nested_param(
                get_param("robustness_strategy", None), "gradient_clip_value", 1.0
            ),
            "loss_spike_detection": get_nested_param(
                get_param("robustness_strategy", None), "loss_spike_detection", True
            ),
            "checkpoint_recovery": get_nested_param(
                get_param("robustness_strategy", None), "checkpoint_recovery", True
            ),
            "validation_based_scheduling": get_nested_param(
                get_param("robustness_strategy", None), "validation_based_scheduling", True
            ),
        },
        # Checkpoint strategy
        "checkpoint_strategy": {
            "save_strategy": get_nested_param(
                get_param("checkpoint_strategy", None), "save_strategy", "selective"
            ),
            "save_best_only": get_nested_param(
                get_param("checkpoint_strategy", None), "save_best_only", True
            ),
            "save_phase_transitions": get_nested_param(
                get_param("checkpoint_strategy", None), "save_phase_transitions", True
            ),
            "save_frequency": get_nested_param(
                get_param("checkpoint_strategy", None), "save_frequency", 25
            ),
            "cleanup_old_checkpoints": get_nested_param(
                get_param("checkpoint_strategy", None), "cleanup_old_checkpoints", True
            ),
            "max_checkpoints": get_nested_param(
                get_param("checkpoint_strategy", None), "max_checkpoints", 3
            ),
        },
        # Synthetic→real curriculum (nested attribute check)
        "synthetic_real_curriculum": {
            "enable_curriculum": get_nested_param(
                get_param("synthetic_real_curriculum", None), "enable_curriculum", False
            ),
            "synthetic_epochs": get_nested_param(
                get_param("synthetic_real_curriculum", None), "synthetic_epochs", 20
            ),
            "transition_epochs": get_nested_param(
                get_param("synthetic_real_curriculum", None), "transition_epochs", 10
            ),
            "real_fine_tune_epochs": get_nested_param(
                get_param("synthetic_real_curriculum", None), "real_fine_tune_epochs", 15
            ),
            "domain_adaptation_techniques": get_nested_param(
                get_param("synthetic_real_curriculum", None),
                "domain_adaptation_techniques",
                ["gradual_mixing", "adversarial_adaptation"],
            ),
        },
    }
    cfg.data.keep_ratio = cfg.transfer_learning.keep_ratio
    cfg.train.so3_augmentation = cfg.transfer_learning.curriculum_learning.enable_so3_curriculum
    cfg.train.so3_curriculum_epochs = (
        cfg.transfer_learning.curriculum_learning.so3_curriculum_epochs
    )
    cfg.train.max_so3_rotation_deg = cfg.transfer_learning.curriculum_learning.max_so3_rotation_deg
    cfg.optim.lr = cfg.transfer_learning.optim_lr
    cfg.optim.max_epoch = cfg.transfer_learning.optim_max_epoch
    cfg.optim.grad_acc_steps = cfg.transfer_learning.gradient_accumulation_steps

    return cfg


def freeze_model_components(model: torch.nn.Module, phase: int = 0) -> tuple[int, int]:
    """
    Enhanced Strategic Progressive Unfreezing for Transfer Learning

    Phase 0 (0): Scale Validation + Architectural Compatibility (Epochs 1-8)
    Phase A (1): Coarse Feature Adaptation + Early Attention (Epochs 9-20)
    Phase B (2): Cross-Modal Learning + Mid-Level Features (Epochs 21-32)
    Phase C1 (3): Backbone Integration + Full Transformer (Epochs 33-42)
    Phase C2 (4): Full Model Polish + Domain Adaptation (Epochs 43-50)

    Args:
        model: The GeoTransformer model
        phase: Training phase (0-4) mapping to enhanced strategic phases

    Architecture Context:
        backbone: KPConvFPN (6,009,600 params) - 4 encoder + 3 decoder stages
        transformer: GeometricTransformer (3,819,776 params) - 6 layers ['self', 'cross', 'self', 'cross', 'self', 'cross']
        coarse_target: SuperPointTargetGenerator (0 params) ❌ ALGORITHMIC
        coarse_matching: SuperPointMatching (0 params) ❌ ALGORITHMIC
        fine_matching: LocalGlobalRegistration (0 params) ❌ ALGORITHMIC
        optimal_transport: LearnableLogOptimalTransport (1 params) 🟢 LEARNABLE
    """

    # Check if model is wrspped insde DataParallel
    model = model.module if hasattr(model, "module") else model
    # Enhanced strategic phase mapping
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    if phase == 0:
        # Phase 0: Scale validation + architectural compatibility check
        # Unfreeze optimal transport (critical scale parameter)
        if hasattr(model, "optimal_transport"):
            for param in model.optimal_transport.parameters():
                param.requires_grad = True

        # Unfreeze geometric structure embedding (scale-sensitive components)
        if hasattr(model, "transformer") and hasattr(model.transformer, "embedding"):
            for param in model.transformer.embedding.parameters():
                param.requires_grad = True

    elif phase == 1:
        # Phase A: Coarse feature adaptation + early attention warmup
        # Unfreeze optimal transport
        if hasattr(model, "optimal_transport"):
            for param in model.optimal_transport.parameters():
                param.requires_grad = True

        # Unfreeze geometric structure embedding
        if hasattr(model, "transformer") and hasattr(model.transformer, "embedding"):
            for param in model.transformer.embedding.parameters():
                param.requires_grad = True

        # Unfreeze first 2 transformer self-attention layers for attention warmup
        if hasattr(model, "transformer") and hasattr(model.transformer, "transformer"):
            transformer_layers = model.transformer.transformer.layers
            for layer_idx in [0, 2]:  # First two self-attention layers
                if layer_idx < len(transformer_layers):
                    for param in transformer_layers[layer_idx].parameters():
                        param.requires_grad = True

    elif phase == 2:
        # Phase B: Cross-modal learning + mid-level features
        # Unfreeze optimal transport
        if hasattr(model, "optimal_transport"):
            for param in model.optimal_transport.parameters():
                param.requires_grad = True

        # Unfreeze transformer input/output projections and embedding
        if hasattr(model, "transformer"):
            if hasattr(model.transformer, "embedding"):
                for param in model.transformer.embedding.parameters():
                    param.requires_grad = True
            if hasattr(model.transformer, "in_proj"):
                for param in model.transformer.in_proj.parameters():
                    param.requires_grad = True
            if hasattr(model.transformer, "out_proj"):
                for param in model.transformer.out_proj.parameters():
                    param.requires_grad = True

        # Unfreeze first 4 transformer layers (cross-modal learning focus)
        if hasattr(model, "transformer") and hasattr(model.transformer, "transformer"):
            transformer_layers = model.transformer.transformer.layers
            for layer_idx in range(4):  # Layers 0-3: self, cross, self, cross
                if layer_idx < len(transformer_layers):
                    for param in transformer_layers[layer_idx].parameters():
                        param.requires_grad = True

    elif phase == 3:
        # Phase C1: Backbone integration + full transformer
        # Unfreeze optimal transport
        if hasattr(model, "optimal_transport"):
            for param in model.optimal_transport.parameters():
                param.requires_grad = True

        # Unfreeze full transformer (complete geometric reasoning)
        if hasattr(model, "transformer"):
            for param in model.transformer.parameters():
                param.requires_grad = True

        # Unfreeze backbone decoder stages (feature integration)
        if hasattr(model, "backbone"):
            backbone = model.backbone
            # Strategy: Decoder-first integration while preserving encoder features
            for name, module in backbone.named_children():
                if (
                    "decoder" in name.lower()
                    or any(stage_name in name for stage_name in ["stage4", "stage5", "stage6"])
                    or "up" in name.lower()
                    or "final" in name.lower()
                ):
                    for param in module.parameters():
                        param.requires_grad = True

    elif phase == 4:
        # Phase C2: Full model polish + domain adaptation
        # Unfreeze everything - complete domain adaptation
        for param in model.parameters():
            param.requires_grad = True

    # Count trainable parameters with enhanced context
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return trainable_params, total_params


def load_pretrained_weights(
    model: torch.nn.Module, pretrained_path: str, strict=False
) -> tuple[int, int]:
    """
    Load pretrained weights with size mismatch handling
    """

    if not osp.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

    # Load checkpoint
    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)

    if "model" in checkpoint:
        pretrained_dict = checkpoint["model"]
    else:
        pretrained_dict = checkpoint

    # Get model state dict
    model_dict = model.state_dict()

    # Filter out size mismatches and log info
    compatible_dict = {}
    incompatible_keys = []

    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                compatible_dict[k] = v
            else:
                incompatible_keys.append(f"{k}: {v.shape} -> {model_dict[k].shape}")
        else:
            incompatible_keys.append(f"{k}: not found in model")

    if incompatible_keys:
        raise Warning("Incompatible parameters")

    # Load compatible weights
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict, strict=strict)

    return len(compatible_dict), len(incompatible_keys)


def ensure_tensor(transform: Union[torch.Tensor, np.ndarray, list], device=None) -> torch.Tensor:
    if isinstance(transform, torch.Tensor):
        if device is not None and transform.device != device:
            transform = transform.to(device)
        return transform
    elif isinstance(transform, np.ndarray):
        t = torch.from_numpy(transform).float()
        if device is not None:
            t = t.to(device)
        return t
    elif isinstance(transform, list):
        # If list of tensors, stack
        if all(isinstance(x, torch.Tensor) for x in transform):
            t = torch.stack(transform)
        else:
            t = torch.from_numpy(np.array(transform, dtype=np.float32))
        if device is not None:
            t = t.to(device)
        return t
    else:
        raise TypeError(f"Unsupported type for transform: {type(transform)}")


def remove_outlier_points(
    points: Union[torch.Tensor, np.ndarray],
    norms: Optional[Union[torch.Tensor, np.ndarray]] = None,
    std_ratio: float = 2.0,
    device: Optional[torch.device] = None,
) -> Tuple[Union[torch.Tensor, np.ndarray], Optional[Union[torch.Tensor, np.ndarray]]]:
    """Remove outlier points from point cloud"""
    input_type = "tensor" if torch.is_tensor(points) else "ndarray"
    k = len(points) // 20 if len(points) >= 2000 else 100
    filtered_points, filtered_norms = remove_outliers(
        points, norms=norms, k=k, std_ratio=std_ratio, device=device
    )
    if input_type == "ndarray":
        filtered_points = to_numpy(filtered_points)
        if filtered_norms is not None:
            filtered_norms = to_numpy(filtered_norms)

    return filtered_points, filtered_norms


def dynamic_pair_downsample(
    ref_points: Union[torch.Tensor, np.ndarray],
    src_points: Union[torch.Tensor, np.ndarray],
    ref_normals: Optional[Union[torch.Tensor, np.ndarray]] = None,
    src_normals: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ref_src_density_factor: float = 1.0,
    target_num_points: int = 30000,
    device: Optional[torch.device] = None,
) -> Tuple[
    Union[torch.Tensor, np.ndarray],
    Union[torch.Tensor, np.ndarray],
    Optional[Union[torch.Tensor, np.ndarray]],
    Optional[Union[torch.Tensor, np.ndarray]],
]:
    """Dynamically downsample point cloud to target number of points"""
    # Check input types and convert to numpy for processing
    input_type = "tensor" if torch.is_tensor(ref_points) else "ndarray"
    if not torch.is_tensor(ref_points):
        ref_points = to_tensor(ref_points, device=device)
    if not torch.is_tensor(src_points):
        src_points = to_tensor(src_points, device=device)
    if ref_normals is not None and not torch.is_tensor(ref_normals):
        ref_normals = to_tensor(ref_normals, device=device)
    if src_normals is not None and not torch.is_tensor(src_normals):
        src_normals = to_tensor(src_normals, device=device)

    # Always use dynamic voxel size based on actual point cloud density
    voxel_size_ref = ref_src_density_factor * calculate_dynamic_voxel_size(
        ref_points, target_num_points
    )
    voxel_size_src = calculate_dynamic_voxel_size(src_points, target_num_points)
    ref_points, ref_normals = safe_unpack_func_result(
        voxel_downsample(ref_points, voxel_size_ref, normals=ref_normals, device=device)
    )
    src_points, src_normals = safe_unpack_func_result(
        voxel_downsample(src_points, voxel_size_src, normals=src_normals, device=device)
    )
    # Convert back to original type if needed
    if input_type == "ndarray":
        ref_points = to_numpy(ref_points)
        src_points = to_numpy(src_points)
        if ref_normals is not None:
            ref_normals = to_numpy(ref_normals)
        if src_normals is not None:
            src_normals = to_numpy(src_normals)

    return ref_points, src_points, ref_normals, src_normals


def pca_alignment(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    max_components: int = 3,
    var_threshold: float = 1e-6,
) -> np.ndarray:
    """Source to Target PCA alignment"""
    # Remove outliers using simple statistical filtering
    def remove_outliers(points, std_ratio=2.0):
        if len(points) < 10:
            return points
        distances = np.linalg.norm(points - np.mean(points, axis=0), axis=1)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        mask = distances < (mean_dist + std_ratio * std_dist)
        return points[mask]

    src_clean = remove_outliers(src_points)
    tgt_clean = remove_outliers(tgt_points)
    if len(src_clean) < len(src_points) * 0.3 or len(tgt_clean) < len(tgt_points) * 0.3:
        src_clean = src_points
        tgt_clean = tgt_points

    src_pca = PCA(n_components=min(max_components, src_clean.shape[1]), svd_solver="full")
    tgt_pca = PCA(n_components=min(max_components, tgt_clean.shape[1]), svd_solver="full")
    src_axes = src_pca.fit(src_clean).components_.T
    tgt_axes = tgt_pca.fit(tgt_clean).components_.T
    # Ensure consistent orientation
    if np.dot(src_axes[:, 0], tgt_axes[:, 0]) < 0:
        src_axes[:, 0] *= -1
    R = tgt_axes @ src_axes.T
    src_centroid = np.mean(src_clean, axis=0)
    tgt_centroid = np.mean(tgt_clean, axis=0)
    t = tgt_centroid - R @ src_centroid
    T_np = np.eye(4)
    T_np[:3, :3] = R
    T_np[:3, 3] = t
    return T_np


def print_entry_summary(entry: dict) -> None:
    keys = list(entry.keys())
    print("Entry keys:", keys)
    if "raw_points" in entry:
        print(" raw_points:", entry["raw_points"].shape)
    if "ref_points" in entry:
        print(" ref_points:", entry["ref_points"].shape)
    if "src_points" in entry:
        print(" src_points:", entry["src_points"].shape)
    if "raw_normals" in entry:
        print(" raw_normals:", entry["raw_normals"].shape)
    if "ref_normals" in entry:
        print(" ref_normals:", entry["ref_normals"].shape)
    if "transform" in entry:
        print(" transform:", entry["transform"].shape)
    if "index" in entry:
        print(" index:", entry["index"])


def visualize_pair(
    ref_points: np.ndarray, src_points: np.ndarray, window_name: str = "pair",
) -> None:
    if torch.is_tensor(ref_points):
        ref_points = ref_points.cpu().numpy()
    if torch.is_tensor(src_points):
        src_points = src_points.cpu().numpy()
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(ref_points)
    pcd_ref.paint_uniform_color([0.2, 0.7, 0.2])  # green

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_points)
    pcd_src.paint_uniform_color([0.7, 0.2, 0.2])  # red

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd_ref)
    vis.add_geometry(pcd_src)
    vis.get_render_option().point_size = 2.0
    vis.run()
    vis.destroy_window()


def visualize_features(
    points: np.ndarray, features: np.ndarray = None, window_name: str = "Feature Visualization"
) -> None:
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if features is not None and torch.is_tensor(features):
        features = features.cpu().numpy()
    if features is not None and features.shape[0] == points.shape[0]:
        colors = get_colors_with_tsne(features)
    else:
        colors = np.ones_like(points) * np.array([[0.5, 0.5, 0.5]])
    pcd = make_open3d_point_cloud(points, colors=colors)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def visualize_correspondences(
    ref_corr_points: np.ndarray,
    src_corr_points: np.ndarray,
    label: str = "pos",
    window_name: str = "Correspondences",
) -> None:
    if torch.is_tensor(ref_corr_points):
        ref_corr_points = ref_corr_points.cpu().numpy()
    if torch.is_tensor(src_corr_points):
        src_corr_points = src_corr_points.cpu().numpy()
    ref_pcd = make_open3d_point_cloud(ref_corr_points)
    src_pcd = make_open3d_point_cloud(src_corr_points)
    corr_lines = make_open3d_corr_lines(ref_corr_points, src_corr_points, label=label)
    o3d.visualization.draw_geometries([ref_pcd, src_pcd, corr_lines], window_name=window_name)


def visualize_transformation(
    ref_points: np.ndarray,
    src_points: np.ndarray,
    tfm: np.ndarray,
    window_name: str = "Transformation Visualization",
) -> None:
    if torch.is_tensor(ref_points):
        ref_points = ref_points.cpu().numpy()
    if torch.is_tensor(src_points):
        src_points = src_points.cpu().numpy()
    if torch.is_tensor(tfm):
        tfm = tfm.cpu().numpy()
    src_points_transformed = (tfm[:3, :3] @ src_points.T + tfm[:3, 3:4]).T
    ref_pcd = make_open3d_point_cloud(
        ref_points, colors=np.array([[0, 1, 0]] * ref_points.shape[0])
    )
    src_pcd = make_open3d_point_cloud(
        src_points_transformed, colors=np.array([[1, 0, 0]] * src_points.shape[0])
    )
    o3d.visualization.draw_geometries([ref_pcd, src_pcd], window_name=window_name)


def visualize_point_to_node(
    src_points: np.ndarray,
    knn_idx: np.ndarray,
    knn_masks: np.ndarray,
    window_name: str = "Point to Node Visualization",
) -> None:
    if torch.is_tensor(src_points):
        src_points = src_points.cpu().numpy()
    if torch.is_tensor(knn_idx):
        knn_idx = knn_idx.cpu().numpy()
    if torch.is_tensor(knn_masks):
        knn_masks = knn_masks.cpu().numpy()
    from geotransformer.utils.visualization import (
        build_point_to_node_visualization,
        draw_point_to_node,
    )

    subset_points, point2node, node_centers = build_point_to_node_visualization(
        src_points, knn_idx, knn_masks
    )
    draw_point_to_node(
        subset_points, node_centers, point2node, node_colors=None, window_name=window_name
    )


def visualize_node_correspondences(
    ref_knn_points: np.ndarray,
    ref_knn_masks: np.ndarray,
    src_knn_points: np.ndarray,
    src_knn_masks: np.ndarray,
    window_name: str = "Node Correspondences Visualization",
) -> None:
    if torch.is_tensor(ref_knn_points):
        ref_knn_points = ref_knn_points.cpu().numpy()
    if torch.is_tensor(ref_knn_masks):
        ref_knn_masks = ref_knn_masks.cpu().numpy()
    if torch.is_tensor(src_knn_points):
        src_knn_points = src_knn_points.cpu().numpy()
    if torch.is_tensor(src_knn_masks):
        src_knn_masks = src_knn_masks.cpu().numpy()
    (
        ref_points,
        ref_point2node,
        ref_node_centers,
        src_points,
        src_point2node,
        src_node_centers,
        node_corr_indices,
    ) = build_node_correspondence_visualization(
        ref_knn_points, ref_knn_masks, src_knn_points, src_knn_masks,
    )
    draw_node_correspondences(
        ref_points,
        ref_node_centers,
        ref_point2node,
        src_points,
        src_node_centers,
        src_point2node,
        node_corr_indices,
        window_name=window_name,
    )


def visualize_coarse_alignment(
    ref_points: np.ndarray,
    src_points: np.ndarray,
    T_coarse: np.ndarray,
    window_name: str = "Coarse Alignment Only",
) -> None:
    if torch.is_tensor(ref_points):
        ref_points = ref_points.cpu().numpy()
    if torch.is_tensor(src_points):
        src_points = src_points.cpu().numpy()
    if torch.is_tensor(T_coarse):
        T_coarse = T_coarse.cpu().numpy()
    src_points = (T_coarse[:3, :3] @ src_points.T + T_coarse[:3, 3:4]).T

    ref_pcd = make_open3d_point_cloud(ref_points, colors=[[0, 1, 0]])
    src_pcd = make_open3d_point_cloud(src_points, colors=[[1, 0, 0]])

    o3d.visualization.draw_geometries([ref_pcd, src_pcd], window_name=window_name)


def visualize_normals(
    points: np.ndarray,
    normals: np.ndarray,
    length: float = 0.05,
    window_name: str = "Normals Visualization",
) -> None:
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(normals):
        normals = normals.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pcd], window_name=window_name, point_show_normal=True)


def visualize_neighbor_counts(
    points: np.ndarray,
    neighbor_counts: np.ndarray,
    window_name: str = "Neighbor Count Visualization",
) -> None:
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(neighbor_counts):
        neighbor_counts = neighbor_counts.cpu().numpy()
    # Normalize neighbor counts to [0,1] for colormap
    norm_counts = (neighbor_counts - neighbor_counts.min()) / (neighbor_counts.ptp() + 1e-8)
    cmap = plt.get_cmap("viridis")
    colors = cmap(norm_counts)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def reproject_to_raw(
    processed_points: torch.Tensor,
    preproc_R: torch.Tensor,
    preproc_t: torch.Tensor,
    atol: float = 1e-5,
) -> torch.Tensor:
    return reproject_processed_to_raw(
        processed_points=processed_points, preproc_R=preproc_R, preproc_t=preproc_t, atol=atol,
    )


def compose_final_transform(
    T_model: torch.Tensor,
    preproc_R_ref: torch.Tensor,
    preproc_t_ref: torch.Tensor,
    preproc_R_src: torch.Tensor,
    preproc_t_src: torch.Tensor,
    T_coarse: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute the final transformation from raw source points to raw reference points.

    Args:
        T_model: [4,4] estimated transform by model (preprocessed src -> preprocessed ref)
        preproc_R_ref: [3,3] rotation applied to reference during preprocessing
        preproc_t_ref: [3] translation applied to reference during preprocessing
        preproc_R_src: [3,3] rotation applied to source during preprocessing
        preproc_t_src: [3] translation applied to source during preprocessing
        T_coarse: optional coarse PCA transform (src -> preprocessed src)

    Returns:
        T_final: [4,4] transformation matrix mapping raw_src -> raw_ref
    """
    device = T_model.device

    # Move all inputs to the same device and dtype once
    R_model = T_model[:3, :3].float()
    t_model = T_model[:3, 3].float()
    preproc_R_ref = preproc_R_ref.float().to(device)
    preproc_t_ref = preproc_t_ref.float().to(device)
    preproc_R_src = preproc_R_src.float().to(device)
    preproc_t_src = preproc_t_src.float().to(device)
    print(f"model estimated rotation:\n{R_model}")
    print(f"model estimated translation:\n{t_model}")
    print(f"preproc_R_ref:\n{preproc_R_ref}")
    print(f"preproc_t_ref:\n{preproc_t_ref}")
    print(f"preproc_R_src:\n{preproc_R_src}")
    print(f"preproc_t_src:\n{preproc_t_src}")
    print(f"T_coarse:\n{T_coarse}")
    # Compose optional coarse PCA transform
    if T_coarse is not None:
        T_coarse = T_coarse.to(device)
        R_model = R_model @ T_coarse[:3, :3]
        t_model = R_model @ T_coarse[:3, 3] + t_model

    # Invert reference preprocessing (handles scaling + centering + augmentation)
    # src_raw = preproc_R_src⁻¹ @ (src_processed - preproc_t_src)
    # src_processed = preproc_R_src @ src_raw + preproc_t_src
    # src_coarse = R_coarse @ (preproc_R_src @ src_raw + preproc_t_src) + t_coarse
    # src_coarse = (R_coarse @ preproc_R_src @ src_raw) + R_coarse @ preproc_t_src + t_coarse
    # ref_processed = R_model @ ((R_coarse @ preproc_R_src @ src_raw) + R_coarse @ preproc_t_src + t_coarse) + t_model
    # ref_processed = (R_model @ R_coarse @ preproc_R_src @ src_raw) + (R_model @ R_coarse @ preproc_t_src + R_model @ t_coarse + t_model)
    # ref_processed = preproc_R_ref @ ref_raw + preproc_t_ref
    # ref_raw = preproc_R_ref⁻¹ @ ((R_model @ R_coarse @ preproc_R_src @ src_raw) + (R_model @ R_coarse @ preproc_t_src + R_model @ t_coarse + t_model) - preproc_t_ref)
    # ref_raw = preproc_R_ref⁻¹ @ (R_model @ R_coarse @ preproc_R_src @ src_raw) + preproc_R_ref⁻¹ @ (R_model @ R_coarse @ preproc_t_src + R_model @ t_coarse + t_model - preproc_t_ref)
    # R_final = preproc_R_ref⁻¹ @ R_model @ R_coarse @ preproc_R_src
    # t_final = preproc_R_ref⁻¹ @ (R_model @ R_coarse @ preproc_t_src + R_model @ t_coarse + t_model - preproc_t_ref)
    R_ref_inv, t_ref_inv = invert_affine_R_t(affinite_R=preproc_R_ref, affinite_t=preproc_t_ref)
    print(f"preproc_R_ref inv:\n{R_ref_inv}")
    # Compose final transform: raw_src -> raw_ref
    R_final = R_ref_inv @ R_model @ preproc_R_src
    t_final = R_ref_inv @ (R_model @ preproc_t_src + t_model - preproc_t_ref)
    print(f"final rotation:\n{R_final}")
    print(f"final translation:\n{t_final}")

    # Construct full 4x4 matrix
    T_final = torch.eye(4, device=device, dtype=torch.float32)
    T_final[:3, :3] = R_final
    T_final[:3, 3] = t_final
    print(f"final transformation:\n{T_final}")
    return T_final
