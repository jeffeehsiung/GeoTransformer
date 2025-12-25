import os
from typing import Any, Tuple

import torch
from easydict import EasyDict as edict
from geotransformer.utils.data import (
    build_dataloader_stack_mode,
    calibrate_neighbors_stack_mode,
    registration_collate_fn_stack_mode,
    validate_dataset,
)

from geoTransformer.GeoTransformer.experiments.roboeye.dataset import RoboeyePairDataset


def train_data_loader(cfg: edict, distributed: bool, cache_bool: bool = True) -> Tuple[Any, Any]:
    # Setup cache directory
    per_sample_train_cache = None
    if cache_bool:
        cache_base = os.path.expanduser(getattr(cfg.data, "cache_folder_base", "~/cache/roboeye"))
        if cfg.data.cache_folder_base is None:
            cache_base = os.path.join(getattr(cfg, "output_dir", "."), "cache")
        os.makedirs(cache_base, exist_ok=True)
        if getattr(cfg.data, "per_sample_cache", False):
            per_sample_train_cache = os.path.join(cache_base, "train_per_sample")
            os.makedirs(per_sample_train_cache, exist_ok=True)

    train_dataset = RoboeyePairDataset(
        dataset_root=cfg.data.dataset_root,
        subset="train",
        num_points=cfg.data.num_points,
        voxel_size=cfg.data.voxel_size,
        normalize=cfg.data.normalize,
        deterministic=cfg.data.deterministic,
        use_augmentation=cfg.train.use_augmentation,
        so3_augmentation=cfg.train.so3_augmentation,
        so3_curriculum_epochs=cfg.train.so3_curriculum_epochs,  # if >0, curriculum must be handled externally
        max_so3_rotation_deg=cfg.train.max_so3_rotation_deg,  # if curriculum active change this gradually
        translation_jitter_m=cfg.train.translation_jitter_m,
        return_corr_indices=cfg.train.return_corr_indices,
        return_normals=cfg.train.return_normals,
        matching_radius=cfg.train.matching_radius,
        overfitting_index=cfg.train.overfitting_index,
        curr_epoch=cfg.train.curr_epoch,
        dataset_proto=cfg.data.dataset_proto,
        calib_folder=cfg.data.calib_folder,
        log_dir=getattr(cfg, "log_dir", None),
        per_sample_cache_dir=per_sample_train_cache,
    )
    # train_dataset, failed_train = validate_dataset(train_dataset)
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
        device=torch.device("cpu"),
    )
    return train_loader, neighbor_limits


def val_data_loader(cfg: edict, distributed: bool, cache_bool: bool = True) -> Tuple[Any, Any]:
    # Setup cache directory
    per_sample_val_cache = None
    if cache_bool:
        cache_base = os.path.expanduser(getattr(cfg.data, "cache_folder_base", "~/cache/roboeye"))
        if cfg.data.cache_folder_base is None:
            cache_base = os.path.join(getattr(cfg, "output_dir", "."), "cache")
        os.makedirs(cache_base, exist_ok=True)
        if getattr(cfg.data, "per_sample_cache", False):
            per_sample_val_cache = os.path.join(cache_base, "val_per_sample")
            os.makedirs(per_sample_val_cache, exist_ok=True)

    valid_dataset = RoboeyePairDataset(
        dataset_root=cfg.data.dataset_root,
        subset="val",
        num_points=cfg.data.num_points,
        voxel_size=cfg.data.voxel_size,
        normalize=cfg.data.normalize,
        deterministic=cfg.data.deterministic,
        use_augmentation=cfg.test.use_augmentation,
        so3_augmentation=cfg.test.so3_augmentation,
        so3_curriculum_epochs=cfg.test.so3_curriculum_epochs,
        max_so3_rotation_deg=cfg.test.max_so3_rotation_deg,
        translation_jitter_m=cfg.test.translation_jitter_m,
        return_corr_indices=cfg.test.return_corr_indices,
        return_normals=cfg.test.return_normals,
        matching_radius=cfg.test.matching_radius,
        overfitting_index=cfg.test.overfitting_index,
        curr_epoch=cfg.test.curr_epoch,
        dataset_proto=cfg.data.dataset_proto,
        calib_folder=cfg.data.calib_folder,
        log_dir=getattr(cfg, "log_dir", None),
        per_sample_cache_dir=per_sample_val_cache,
    )
    # valid_dataset, failed_valid = validate_dataset(valid_dataset)
    neighbor_limits = calibrate_neighbors_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        device=torch.device("cpu"),
    )

    return valid_loader, neighbor_limits


def test_data_loader(cfg: edict, distributed: bool, cache_bool: bool = True) -> Tuple[Any, Any]:
    per_sample_test_cache = None
    if cache_bool:
        cache_base = os.path.expanduser(getattr(cfg.data, "cache_folder_base", "~/cache/roboeye"))
        if cfg.data.cache_folder_base is None:
            cache_base = os.path.join(getattr(cfg, "output_dir", "."), "cache")
        os.makedirs(cache_base, exist_ok=True)
        if getattr(cfg.data, "per_sample_cache", False):
            per_sample_test_cache = os.path.join(cache_base, "test_per_sample")
            os.makedirs(per_sample_test_cache, exist_ok=True)

    test_dataset = RoboeyePairDataset(
        dataset_root=cfg.data.dataset_root,
        subset="test",
        num_points=cfg.data.num_points,
        voxel_size=cfg.data.voxel_size,
        normalize=cfg.data.normalize,
        deterministic=cfg.data.deterministic,
        use_augmentation=cfg.test.use_augmentation,
        so3_augmentation=cfg.test.so3_augmentation,
        so3_curriculum_epochs=cfg.test.so3_curriculum_epochs,
        max_so3_rotation_deg=cfg.test.max_so3_rotation_deg,
        translation_jitter_m=cfg.test.translation_jitter_m,
        return_corr_indices=cfg.test.return_corr_indices,
        return_normals=cfg.test.return_normals,
        matching_radius=cfg.test.matching_radius,
        overfitting_index=cfg.test.overfitting_index,
        curr_epoch=cfg.test.curr_epoch,
        dataset_proto=cfg.data.dataset_proto,
        calib_folder=cfg.data.calib_folder,
        log_dir=getattr(cfg, "log_dir", None),
        per_sample_cache_dir=per_sample_test_cache,
    )
    test_dataset, failed_test = validate_dataset(test_dataset)
    neighbor_limits = calibrate_neighbors_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        device=torch.device("cpu"),
    )

    return test_loader, neighbor_limits
