# roboeye_dataset.py
import fcntl
import gc
import os
import os.path as osp
import pickle
import random
from multiprocessing import cpu_count, get_context
from typing import Dict, Optional

import numpy as np
import torch
import torch.utils.data
from geotransformer.transforms.functional import random_sample_points
from nova.datasets import dataset_factory
from nova.proto.model import model_config_pb2

from geoTransformer.GeoTransformer.experiments.roboeye.logger import setup_logger
from geoTransformer.GeoTransformer.experiments.roboeye.preprocess import (
    preprocess_pair,
    safe_unpack_func_result,
)


def _process_single_sample(args):
    dataset_id, idx, subset = args
    sample, templates, d = None, None, None
    try:
        from nova.datasets import dataset_factory as _df

        d = _df.get_dataset(dataset_id, get=True, overwrite=True)
        sample = d[idx]
        sample.project.load_templates(overwrite=True)
        templates = sample.project.templates

        sample_name = getattr(sample, "name", None)
        company = getattr(sample, "company", None)
        project_name = getattr(sample, "project_name", None)
        standard_dataset_name = getattr(sample, "standard_dataset_name", None)
        dataset_id_info = (
            f"{company}/{project_name}/{standard_dataset_name}"
            if all([company, project_name, standard_dataset_name])
            else getattr(sample, "dataset_id", None)
        )

        try:
            sample.prepare_sample(debug=False, headless=True, data_prep_cfg=None)
        except Exception as point_cloud_error:
            result = (
                idx,
                [],
                {
                    "index": idx,
                    "name": sample_name,
                    "dataset_id": dataset_id_info,
                    "error": f"Point cloud loading failed: {str(point_cloud_error)}",
                },
            )
            # clean up
            if d is not None:
                del d
            if sample is not None:
                del sample, templates
            if templates is not None:
                del templates
            gc.collect()
            return result

        if subset == "test" and (not hasattr(sample, "labels") or sample.labels is None):
            result = (
                idx,
                [
                    {
                        "scene_name": sample_name,
                        "ref_frame": 0,
                        "src_frame": -1,
                        "sample_idx": idx,
                        "label_idx": -1,
                        "template": templates[0],
                        "rotation": np.eye(3),
                        "translation": np.zeros(3),
                    }
                ],
                None,
            )
            # clean up
            if d is not None:
                del d
            if sample is not None:
                del sample
            if templates is not None:
                del templates
            gc.collect()
            return result

        if not hasattr(sample, "labels") or sample.labels is None:
            result = (
                idx,
                [],
                {
                    "index": idx,
                    "name": sample_name,
                    "dataset_id": dataset_id_info,
                    "error": "Missing labels/poses",
                },
            )
            # clean up
            if d is not None:
                del d
            if sample is not None:
                del sample
            if templates is not None:
                del templates
            gc.collect()
            return result

        poses = getattr(sample.labels, "poses", None)
        masks = sample.labels.masks.data
        if poses is None or len(masks) != len(poses):
            result = (
                idx,
                [],
                {
                    "index": idx,
                    "name": sample_name,
                    "dataset_id": dataset_id_info,
                    "error": "Missing labels/poses or mismatch masks/poses",
                },
            )
            # clean up
            if d is not None:
                del d
            if sample is not None:
                del sample
            if templates is not None:
                del templates
            if poses is not None:
                del poses
            if masks is not None:
                del masks
            gc.collect()
            return result

        entries = []
        for label_i, pose in enumerate(poses):
            valid_mask = np.sum(masks[label_i] > 0) > 10
            if not valid_mask:
                continue
            R = pose[:3, :3].astype(np.float32)
            t = pose[:3, 3].astype(np.float32)
            entries.append(
                {
                    "scene_name": sample_name,
                    "ref_frame": 0,
                    "src_frame": label_i,
                    "sample_idx": idx,
                    "label_idx": label_i,
                    "template": templates[0],
                    "rotation": R,
                    "translation": t,
                }
            )
            # clean up
            del R, t
            gc.collect()

        if not entries:
            result = (
                idx,
                [],
                {
                    "index": idx,
                    "name": sample_name,
                    "dataset_id": dataset_id_info,
                    "error": "No valid label masks (>10 points) found",
                },
            )
        else:
            result = (idx, entries, None)

        # clean up
        if d is not None:
            del d
        if sample is not None:
            del sample
        if templates is not None:
            del templates
        if "poses" in locals():
            del poses
        if "masks" in locals():
            del masks
        if "entries" in locals():
            del entries
        gc.collect()
        return result

    except Exception as e:
        result = (idx, [], {"index": idx, "name": None, "dataset_id": dataset_id, "error": str(e)})
        if "d" in locals() and d is not None:
            del d
        if "sample" in locals() and sample is not None:
            del sample
        if "templates" in locals() and templates is not None:
            del templates
        gc.collect()
        return result


class RoboeyePairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: str,
        subset: str,
        num_points: int = 1024,
        voxel_size: Optional[float] = 0.009,
        normalize: bool = False,
        deterministic: bool = False,
        use_augmentation: bool = True,
        so3_augmentation: bool = True,
        so3_curriculum_epochs: int = 0,  # if >0, curriculum must be handled externally
        max_so3_rotation_deg: float = 180.0,  # if curriculum active change this gradually
        translation_jitter_m: float = 0.005,  # 5mm
        return_corr_indices: bool = False,
        return_normals: bool = True,
        curr_epoch: int = 0,
        matching_radius: Optional[float] = None,
        overfitting_index: Optional[int] = None,
        dataset_proto: Optional[model_config_pb2.Dataset] = None,
        calib_folder: Optional[str] = None,
        log_dir: Optional[str] = None,  # Directory for logging
        per_sample_cache_dir: Optional[str] = None,  # Directory for per-sample progressive caching
        use_pinned_memory: bool = False,
        debug: Optional[bool] = False,
    ):
        """
        Args:
            dataset_root: path containing train.pkl / val.pkl / test.pkl OR a roboeye converted folder
            subset: 'train'|'val'|'test'
            num_points: number of points the dataset sampler should produce
            voxel_size: downsample voxel size in meters (default 0.009)
            deterministic: if True, seed sampling by index (useful for debugging)
            return_normals: whether to return normals
        """
        super(RoboeyePairDataset, self).__init__()
        assert subset in ["train", "val", "test"]
        self.dataset_root = dataset_root
        self.subset = subset
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.normalize = normalize
        self.deterministic = deterministic
        self.use_augmentation = use_augmentation
        self.so3_augmentation = so3_augmentation
        self.so3_curriculum_epochs = so3_curriculum_epochs
        self.max_so3_rotation_deg = max_so3_rotation_deg
        self.translation_jitter_m = translation_jitter_m
        self.return_corr_indices = return_corr_indices
        self.return_normals = return_normals
        self.matching_radius = matching_radius
        self.overfitting_index = overfitting_index
        self.curr_epoch = curr_epoch
        self.calib_folder = calib_folder
        self.use_pinned_memory = use_pinned_memory
        self.debug = debug
        self.logger = setup_logger(name=__name__, log_dir=log_dir)
        self.per_sample_cache_dir = (
            os.path.expanduser(per_sample_cache_dir) if per_sample_cache_dir is not None else None
        )
        if self.per_sample_cache_dir is not None:
            os.makedirs(self.per_sample_cache_dir, exist_ok=True)
            self.logger.info(
                f"Per-sample progressive caching enabled at: {self.per_sample_cache_dir}"
            )

        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.logger.debug("dataset_proto = %s", dataset_proto)

        if dataset_proto is not None:
            self.logger.debug("Using dataset_proto path")
            self.data_list = []
            failures = []
            subset_mapping = {
                "train": dataset_proto.train,
                "val": dataset_proto.val,
                "test": dataset_proto.test,
            }
            if self.subset not in subset_mapping or not subset_mapping[self.subset]:
                raise ValueError(f"No {self.subset} data found in dataset_proto")
            dataset_proto_ = subset_mapping[self.subset][0]

            # Store dataset_id instead of dataset object (for fork-safe lazy loading)
            self.dataset_id = dataset_proto_.dataset_id
            self.split_start = dataset_proto_.split.start
            self.split_stop = dataset_proto_.split.stop
            self._dataset = None  # Lazy-loaded per worker
            self._worker_id = None  # Track which worker loaded the dataset

            indices = range(self.split_start, self.split_stop)
            workers = max(8, max(1, (cpu_count() or 1) - (cpu_count() // 2)))  # tune
            per_task_timeout = 60.0  # seconds
            max_retries = 1

            entries, failures = self._build_data_list_parallel(
                dataset_id=self.dataset_id,
                indices=indices,
                subset=self.subset,
                workers=workers,
                per_task_timeout=per_task_timeout,
                max_retries=max_retries,
            )

            self.data_list.extend(entries)

            # clean up
            if entries is not None:
                del entries
            gc.collect()

            try:
                failure_log_dir = log_dir or dataset_root or "."
                failures_path = osp.join(failure_log_dir, f"{self.subset}_dataset_failures.log")
                if failures:
                    with open(failures_path, "w") as fh:
                        import json

                        json.dump(failures, fh, indent=2)
                    self.logger.warning(
                        "Wrote %d dataset failures to %s", len(failures), failures_path
                    )
            except Exception:
                self.logger.exception("Failed to write dataset failures log")
            finally:
                if failures is not None:
                    del failures
                gc.collect()

        if self.overfitting_index is not None and self.deterministic:
            self.data_list = [self.data_list[self.overfitting_index]]

    def __len__(self):
        """
        Return number of point cloud pairs in the dataset

        """
        return len(self.data_list)

    def __getitem__(self, index: int):
        if self.overfitting_index is not None:
            index = self.overfitting_index

        # data entry for this index (used throughout)
        data_entry = self.data_list[index]

        cached = None
        if self.per_sample_cache_dir is not None:
            cached = self._load_from_per_sample_cache(index)

        if cached is not None:
            try:
                # load from preprocessed cache
                ref_points = cached["ref_points"]
                src_points = cached["src_points"]
                ref_normals = cached.get("ref_normals", None)
                src_normals = cached.get("src_normals", None)
                rotation = cached["rotation"]
                translation = cached["translation"]
                preproc_R_ref = cached["preproc_R_ref"]
                preproc_t_ref = cached["preproc_t_ref"]
                preproc_R_src = cached["preproc_R_src"]
                preproc_t_src = cached["preproc_t_src"]
                label = cached.get("label", -1)

                # Apply post-processing steps
                pair_dict = preprocess_pair(
                    curr_epoch=self.curr_epoch,
                    ref_points=ref_points,
                    src_points=src_points,
                    ref_normals=ref_normals,
                    src_normals=src_normals,
                    voxel_size=self.voxel_size,
                    normalize=self.normalize,
                    estimate_missing_normals=self.return_normals,
                    return_corr_indices=self.return_corr_indices and self.subset != "test",
                    matching_radius=self.matching_radius,
                    rotation=rotation,
                    translation=translation,
                    preproc_R_ref=preproc_R_ref,
                    preproc_t_ref=preproc_t_ref,
                    preproc_R_src=preproc_R_src,
                    preproc_t_src=preproc_t_src,
                    use_augmentation=self.use_augmentation,
                    so3_augmentation=self.so3_augmentation,
                    so3_curriculum_epochs=self.so3_curriculum_epochs,
                    max_so3_rotation_deg=self.max_so3_rotation_deg,
                    translation_jitter_m=self.translation_jitter_m,
                    subset=self.subset,
                    is_preprocessed=True,
                    device=torch.device("cpu"),
                    debug=self.debug,
                )

                data_dict = {
                    "ref_points": pair_dict["ref_points"],
                    "src_points": pair_dict["src_points"],
                    "ref_feats": pair_dict["ref_feats"],
                    "src_feats": pair_dict["src_feats"],
                    "transform": pair_dict["transform"],
                    "raw_gt_rotation": cached["raw_gt_rotation"] if self.debug else None,
                    "raw_gt_translation": cached["raw_gt_translation"] if self.debug else None,
                    "preproc_R_ref": pair_dict["preproc_R_ref"] if self.debug else None,
                    "preproc_t_ref": pair_dict["preproc_t_ref"] if self.debug else None,
                    "preproc_R_src": pair_dict["preproc_R_src"] if self.debug else None,
                    "preproc_t_src": pair_dict["preproc_t_src"] if self.debug else None,
                    "canonical_flipped": pair_dict.get("canonical_flipped", False),
                    "preproc_scale": pair_dict["preproc_scale"] if self.debug else None,
                    "label": torch.tensor(label, dtype=torch.long),
                    "index": torch.tensor(index, dtype=torch.long),
                }
                if self.return_normals:
                    data_dict["ref_normals"] = pair_dict.get("ref_normals", None)
                    data_dict["src_normals"] = pair_dict.get("src_normals", None)

                if self.return_corr_indices and self.matching_radius is not None:
                    data_dict["corr_indices"] = pair_dict["corr_indices"]

                # cleanup
                del cached, ref_points, src_points, ref_normals, src_normals, rotation, translation
                del preproc_R_ref, preproc_t_ref, preproc_R_src, preproc_t_src, label
                gc.collect()

                return data_dict
            except Exception as e:
                self.logger.error(
                    f"Failed to load from cache for index {index} "
                    f"(sample_idx={data_entry['sample_idx']}, label_idx={data_entry['label_idx']}): {e}"
                )

        # deterministic seeding
        if self.deterministic:
            np.random.seed(index)
            random.seed(index)
            try:
                torch.manual_seed(index)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(index)
            except Exception:
                pass

        try:
            ref_points, ref_normals = self._load_points_from_scans(data_entry)
            src_points, src_normals = self._load_points_from_template(data_entry)

        except Exception as e:
            self.logger.error(
                f"Failed to load source points for index {index} "
                f"(sample_idx={data_entry['sample_idx']}, label_idx={data_entry['label_idx']}): {e}"
            )
            # skip sample
            return None

        # initial template downsampling
        src_points, src_normals = safe_unpack_func_result(
            random_sample_points(src_points, self.num_points, normals=src_normals)
        )

        # get transformation
        rotation = (
            data_entry["rotation"] if "rotation" in data_entry else np.eye(3, dtype=np.float32)
        )
        translation = (
            data_entry["translation"]
            if "translation" in data_entry
            else np.zeros(3, dtype=np.float32)
        )

        # clean up
        del data_entry
        gc.collect()

        # preprocess pair
        pair_dict = preprocess_pair(
            curr_epoch=self.curr_epoch,
            ref_points=ref_points,
            src_points=src_points,
            ref_normals=ref_normals,
            src_normals=src_normals,
            voxel_size=self.voxel_size,
            normalize=self.normalize,
            estimate_missing_normals=self.return_normals,
            return_corr_indices=self.return_corr_indices and self.subset != "test",
            matching_radius=self.matching_radius,
            rotation=rotation,
            translation=translation,
            preproc_R_ref=None,
            preproc_t_ref=None,
            preproc_R_src=None,
            preproc_t_src=None,
            use_augmentation=self.use_augmentation,
            so3_augmentation=self.so3_augmentation,
            so3_curriculum_epochs=self.so3_curriculum_epochs,
            max_so3_rotation_deg=self.max_so3_rotation_deg,
            translation_jitter_m=self.translation_jitter_m,
            subset=self.subset,
            is_preprocessed=False,
            device=torch.device("cpu"),
            debug=self.debug,
        )

        # clean up
        del ref_points, src_points, ref_normals, src_normals, rotation, translation
        gc.collect()

        # save preprocessed cache
        if self.per_sample_cache_dir is not None:
            try:
                preprocessed_cache_data = {
                    "stage": "preprocessed",
                    "ref_points": pair_dict["pre_ref_points"],
                    "src_points": pair_dict["pre_src_points"],
                    "ref_normals": pair_dict.get("pre_ref_normals", None),
                    "src_normals": pair_dict.get("pre_src_normals", None),
                    "rotation": pair_dict["pre_rotation"],
                    "translation": pair_dict["pre_translation"],
                    "transform": pair_dict["pre_transform"],
                    "raw_gt_rotation": pair_dict["raw_gt_rotation"],
                    "raw_gt_translation": pair_dict["raw_gt_translation"],
                    "preproc_R_ref": pair_dict["preproc_R_ref"],
                    "preproc_t_ref": pair_dict["preproc_t_ref"],
                    "preproc_R_src": pair_dict["preproc_R_src"],
                    "preproc_t_src": pair_dict["preproc_t_src"],
                    "preproc_scale": pair_dict["preproc_scale"],
                    "label": index,
                }
                self._save_to_per_sample_cache(index, preprocessed_cache_data)
                del preprocessed_cache_data
                gc.collect()
            except Exception as e:
                self.logger.warning(f"Failed to save preprocessed cache for index {index}: {e}")

        data_dict = {
            "raw_gt_rotation": pair_dict["raw_gt_rotation"] if self.debug else None,
            "raw_gt_translation": pair_dict["raw_gt_translation"] if self.debug else None,
            "ref_points": pair_dict["ref_points"],
            "src_points": pair_dict["src_points"],
            "ref_feats": pair_dict["ref_feats"],
            "src_feats": pair_dict["src_feats"],
            "transform": pair_dict["transform"],
            "preproc_R_ref": pair_dict["preproc_R_ref"] if self.debug else None,
            "preproc_t_ref": pair_dict["preproc_t_ref"] if self.debug else None,
            "preproc_R_src": pair_dict["preproc_R_src"] if self.debug else None,
            "preproc_t_src": pair_dict["preproc_t_src"] if self.debug else None,
            "canonical_flipped": pair_dict.get("canonical_flipped", False),
            "preproc_scale": pair_dict["preproc_scale"] if self.debug else None,
            "label": torch.tensor(index, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.return_normals:
            data_dict["ref_normals"] = pair_dict.get("ref_normals", None)
            data_dict["src_normals"] = pair_dict.get("src_normals", None)

        if self.return_corr_indices and self.matching_radius is not None:
            data_dict["corr_indices"] = pair_dict["corr_indices"]

        # cleanup
        del pair_dict
        gc.collect()

        # pinned memory
        if self.use_pinned_memory and torch.cuda.is_available():
            for key in data_dict:
                if torch.is_tensor(data_dict[key]):
                    data_dict[key] = data_dict[key].pin_memory()

        return data_dict

    # getters
    def _get_dataset(self):
        """
        Lazy-load Nova dataset per worker to avoid fork() issues.

        Each DataLoader worker process gets its own dataset instance, avoiding
        segmentation faults from shared C++ state after fork.
        """
        current_worker = os.getpid()

        # Check if we need to load dataset for this worker
        if self._dataset is None or self._worker_id != current_worker:
            if hasattr(self, "dataset_id"):
                self.logger.debug(
                    f"Loading Nova dataset {self.dataset_id} in worker {current_worker}"
                )
                self._dataset = dataset_factory.get_dataset(
                    self.dataset_id, get=True, overwrite=True
                )
                self._worker_id = current_worker
            else:
                raise RuntimeError("dataset_id not set - cannot lazy-load dataset")

        return self._dataset

    def _get_cache_key(self, index: int) -> str:
        """Generate a unique cache key for a sample based on preprocessing parameters"""
        data_entry = self.data_list[index]
        if "scene_name" in data_entry:
            base_key = (
                f"{data_entry['scene_name']}_s{data_entry['sample_idx']}_l{data_entry['label_idx']}"
            )
        else:
            base_key = f"sample_{index}"

        key = f"{base_key}_n{self.num_points}_v{self.voxel_size}_norm{int(self.normalize)}"
        return key

    def _get_per_sample_cache_path(self, index: int) -> str:
        """Get the cache file path for a given sample index"""
        cache_key = self._get_cache_key(index)
        return osp.join(self.per_sample_cache_dir, f"{cache_key}.pkl")

    # setters
    def update_epoch(self, epoch: int):
        """Update current epoch for curriculum learning"""
        self.curr_epoch = epoch
        self.logger.debug(f"Dataset epoch updated to {epoch} for SO3 curriculum learning")

    # loaders
    def _load_points_from_template(self, entry: Dict):
        """
        Load points + normals from decimated mesh viewpoints from roboeye dataset (template/cad)
        """
        template = entry["template"]
        try:
            if "scene_name" in entry:
                # load ideal template
                points = template.master.points.astype(np.float32)
                norms = (
                    None
                    if getattr(template.master, "normals", None) is None
                    else template.master.normals.astype(np.float32)
                )
            return points, norms
        except Exception as e:
            self.logger.warning(
                f"Failed to load template points for sample {entry['sample_idx']}: {e}"
            )
            raise

    def _load_points_from_scans(self, entry: Dict):
        """
        Load points + normals from a roboeye dataset sample and label index.
        Returns empty arrays if loading fails.
        """
        sample_idx = entry["sample_idx"]
        label_idx = entry["label_idx"]
        sample, dataset = None, None

        try:  # Lazy-load dataset in each worker (fork-safe)
            dataset = self._get_dataset()
            sample = dataset[sample_idx]
            sample.ensure_depth()
            sample.ensure_point_cloud()
        except Exception as e:
            self.logger.warning(f"Failed to load sample {sample_idx} from Roboeye dataset: {e}")
            # Clean up before returning empty arrays
            if dataset is not None:
                del dataset
            gc.collect()
            raise

        # Handle test case without labels
        if self.subset == "test" and (not hasattr(sample, "labels") or sample.labels == -1):
            try:
                points = sample.point_cloud.points
                normals = (
                    None
                    if getattr(sample.point_cloud, "normals", None) is None
                    else sample.point_cloud.normals
                )
                result = (
                    points.astype(np.float32),
                    normals.astype(np.float32) if normals is not None else None,
                )
                return result
            except Exception as e:
                self.logger.warning(f"Failed to get points from sample {sample_idx}: {e}")
                raise
            finally:
                # Clean up
                if sample is not None:
                    del sample
                if dataset is not None:
                    del dataset
                gc.collect()

        try:
            # instance mask to point cloud to get reference points
            if sample.is_prepared is False:
                sample.prepare_sample(debug=False, headless=True, data_prep_cfg=None)
            mask = sample.labels.masks.data[label_idx] > 0
            sample.point_cloud.apply_mask_to_maps(mask)
            points = sample.point_cloud.points
            normals = (
                None
                if getattr(sample.point_cloud, "normals", None) is None
                else sample.point_cloud.normals
            )
            result = (
                points.astype(np.float32),
                normals.astype(np.float32) if normals is not None else None,
            )
            return result
        except Exception as e:
            self.logger.warning(
                f"Failed to load label {label_idx} from sample {sample_idx} in Roboeye dataset: {e}"
            )
            raise
        finally:
            # Always clean up, regardless of success or failure
            if sample is not None:
                del sample
            if dataset is not None:
                del dataset
            gc.collect()

    # cacher
    def _load_from_per_sample_cache(self, index: int) -> Optional[Dict]:
        """
        Load from progressive cache - returns the most complete cached state available

        Returns dict with keys indicating what's cached:
        - 'stage': 'raw', 'preprocessed', or None
        - 'raw_*': raw data (if stage >= 'raw')
        - 'preprocessed_*': preprocessed data (if stage >= 'preprocessed')
        """
        if self.per_sample_cache_dir is None:
            return None

        cache_path = self._get_per_sample_cache_path(index)
        if not osp.exists(cache_path):
            return None

        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            return cached_data
        except Exception as e:
            self.logger.warning(f"Failed to load cache for index {index}: {e}")
            return None

    def _save_to_per_sample_cache(self, index: int, data_dict: Dict):
        """
        Save to progressive cache - updates existing cache or creates new one

        Automatically cleans up redundant data (e.g., removes raw data once preprocessed is saved)
        Thread-safe for parallel DataLoader workers using file locking
        """
        if self.per_sample_cache_dir is None:
            return

        cache_path = self._get_per_sample_cache_path(index)
        lock_path = cache_path + ".lock"

        try:
            # Acquire exclusive lock for this cache file
            os.makedirs(osp.dirname(lock_path), exist_ok=True)
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

                try:
                    # Load existing cache if it exists
                    existing_cache = {}
                    if osp.exists(cache_path):
                        try:
                            with open(cache_path, "rb") as f:
                                existing_cache = pickle.load(f)
                        except Exception:
                            pass

                    # Merge new data with existing cache
                    cache_data = existing_cache.copy()
                    for key, value in data_dict.items():
                        if torch.is_tensor(value):
                            cache_data[key] = value.cpu()
                        else:
                            cache_data[key] = value

                    # Cleanup: if preprocessed stage is complete, remove raw-only data
                    if cache_data.get("stage") == "preprocessed":
                        # Remove redundant raw data (we have preprocessed voxel-downsampled data)
                        keys_to_remove = [
                            k
                            for k in cache_data.keys()
                            if k.startswith("raw_ref_") or k.startswith("raw_src_")
                        ]
                        for k in keys_to_remove:
                            cache_data.pop(k, None)

                    # Write atomically using a temp file
                    temp_path = cache_path + ".tmp"
                    with open(temp_path, "wb") as f:
                        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(temp_path, cache_path)

                finally:
                    pass

        except Exception as e:
            self.logger.warning(f"Failed to save cache for index {index}: {e}")
            temp_path = cache_path + ".tmp"
            if osp.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    # data loader helpers
    def _build_data_list_parallel(
        self,
        dataset_id,
        indices,
        subset,
        workers: Optional[int] = None,
        per_task_timeout: Optional[float] = None,
        max_retries: int = 0,
    ):
        if workers is None:
            workers = max(1, (cpu_count() or 1) - (cpu_count() // 2))

        args_iter = [(dataset_id, int(idx), subset) for idx in indices]
        data_list_entries = []
        failures = []

        if workers <= 1:  # sequential fallback
            for args in args_iter:
                idx, entries, fail = _process_single_sample(args)
                if fail is not None:
                    failures.append(fail)
                else:
                    data_list_entries.extend(entries)
                if len(data_list_entries) % 100 == 0:
                    gc.collect()
            return data_list_entries, failures

        ctx = get_context("spawn")
        try:
            with ctx.Pool(processes=workers) as pool:
                async_results = [pool.apply_async(_process_single_sample, (a,)) for a in args_iter]
                for i, async_res in enumerate(async_results):
                    idx = args_iter[i][1]
                    tries = 0
                    while True:
                        tries += 1
                        try:
                            if per_task_timeout is not None:
                                res_idx, entries, fail = async_res.get(timeout=per_task_timeout)
                            else:
                                res_idx, entries, fail = async_res.get()
                            break
                        except ctx.TimeoutError:
                            if tries <= max_retries:
                                async_res = pool.apply_async(
                                    _process_single_sample, (args_iter[i],)
                                )
                                continue
                            else:
                                fail = {
                                    "index": idx,
                                    "name": None,
                                    "dataset_id": dataset_id,
                                    "error": f"Timeout after {per_task_timeout}s",
                                }
                                entries = []
                                break
                        except Exception as e:
                            if tries <= max_retries:
                                async_res = pool.apply_async(
                                    _process_single_sample, (args_iter[i],)
                                )
                                continue
                            else:
                                fail = {
                                    "index": idx,
                                    "name": None,
                                    "dataset_id": dataset_id,
                                    "error": str(e),
                                }
                                entries = []
                                break

                    if fail is not None:
                        failures.append(fail)
                    else:
                        data_list_entries.extend(entries)
                    # cleanup
                    del async_res
                    gc.collect()

                    if len(data_list_entries) % 50 == 0:
                        gc.collect()

        except Exception as e:
            self.logger.exception("Parallel validation failed: %s", e)
            data_list_entries = []
            failures = []
            for args in args_iter:
                idx, entries, fail = _process_single_sample(args)
                if fail is not None:
                    failures.append(fail)
                else:
                    data_list_entries.extend(entries)
                if len(data_list_entries) % 100 == 0:
                    gc.collect()

        return data_list_entries, failures

    def cleanup(self):
        """Cleanup any resources held by the dataset"""
        self.logger.info("Cleaning up RoboeyePairDataset resources")
        self._dataset = None
        self._worker_id = None
        if hasattr(self, "data_list"):
            self.data_list.clear()
        gc.collect()
