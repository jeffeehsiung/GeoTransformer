import argparse
import logging
import os
import os.path as osp
from pathlib import Path
from typing import Optional

from easydict import EasyDict as edict
from geotransformer.utils.common import ensure_dir
from nova.datasets import dataset_factory
from nova.proto.model import model_config_pb2

from geoTransformer.GeoTransformer.experiments.roboeye.logger import setup_logger
from iris.utils import model_config_utils

# Global logger - will be properly configured when make_cfg is called with log_dir
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)

# Directory paths - will be set up only when make_cfg is called with output directory configuration
_C.output_dir = None
_C.snapshot_dir = None
_C.log_dir = None
_C.event_dir = None
_C.feature_dir = None
_C.registration_dir = None

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 4
_C.backbone.init_voxel_size = 0.025  # in meters (2.5cm)
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.05  # in meters (5cm)
_C.model.num_points_in_patch = (
    64  # Reduced from 64 to handle sparse point clouds (num_points=512, voxel_size=0.015)
)
_C.model.num_sinkhorn_iterations = 100

# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 256
_C.coarse_matching.dual_normalization = True

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 1024
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ["self", "cross", "self", "cross", "self", "cross"]
_C.geotransformer.sigma_d = 2 * _C.backbone.init_sigma
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = "max"

# model - Fine Matching
_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius = 0.1
_C.fine_matching.mutual = True
_C.fine_matching.confidence_threshold = 0.05
_C.fine_matching.use_dustbin = False
_C.fine_matching.use_global_score = True
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.05


# loss - Symmetric loss
_C.symmetric_loss = edict()
_C.symmetric_loss.use_coarse_features = True
_C.symmetric_loss.use_fine_features = False  # Enable if you have fine features
_C.symmetric_loss.temperature = 0.1  # Lower = sharper, higher = softer similarities

# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0
_C.loss.use_symmetric_loss = True
_C.loss.weight_symmetric_loss = 0.1

# data
_C.data = edict()
_C.data.dataset_id = None  # Will be set dynamically or use default
_C.data.dataset_root = None  # Will be set dynamically
_C.data.dataset_type = None  # Will be detected automatically ('roboeye_custom' or 'bop')
_C.data.calib_folder = None  # optional calibration folder path
_C.data.num_points = 2048
_C.data.voxel_size = 0.002  # Set voxel size to reduce memory (2mm voxels)
_C.data.normalize = True
_C.data.deterministic = True
_C.data.rotation_magnitude = 45.0
_C.data.translation_magnitude = 0.5
_C.data.keep_ratio = 0.7
_C.data.crop_method = "plane"
_C.data.asymmetric = True
_C.data.twice_sample = True
_C.data.twice_transform = False
_C.data.dataset_proto = None  # Will be created dynamically when needed


# Cache configuration for progressive per-sample caching
_C.data.cache_folder_base = None  # Will be set in make_cfg to output_dir/cache
_C.data.per_sample_cache = True  # Enable progressive per-sample caching (survives crashes)
_C.data.cache_progressive = (
    True  # Single file with auto-cleanup (removes raw data when preprocessed complete)
)


# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 4  # Reduce workers to free memory (was 4)
_C.train.point_limit = 30000  # Reduce point limit to save memory (was 30000)
_C.train.gradient_accumulation_steps = 4  # Accumulate gradients to simulate batch_size=4
_C.train.use_augmentation = True
_C.train.so3_augmentation = True
_C.train.so3_curriculum_epochs = 50  # if >0, curriculum must be handled externally
_C.train.max_so3_rotation_deg = 180.0  # if curriculum active change this
_C.train.translation_jitter_m = 0.005  # 5mm
_C.train.augmentation_noise = 0.01
_C.train.augmentation_min_scale = 0.8
_C.train.augmentation_max_scale = 1.2
_C.train.augmentation_shift = 2.0
_C.train.augmentation_rotation = 1.0
_C.train.noise_magnitude = 0.05
_C.train.return_corr_indices = True
_C.train.return_normals = True
_C.train.matching_radius = _C.model.ground_truth_matching_radius
_C.train.overfitting_index = None
_C.train.curr_epoch = 0
_C.train.class_indices = "all"


# test data
_C.test = edict()
_C.test.batch_size = _C.train.batch_size
_C.test.num_workers = _C.train.num_workers
_C.test.point_limit = _C.train.point_limit
_C.test.use_augmentation = False  # no augmentation for testing/validation
_C.test.so3_augmentation = False
_C.test.so3_curriculum_epochs = _C.train.so3_curriculum_epochs
_C.test.max_so3_rotation_deg = _C.train.max_so3_rotation_deg
_C.test.translation_jitter_m = 0.0
_C.test.augmentation_noise = 0.0
_C.test.augmentation_min_scale = _C.train.augmentation_min_scale
_C.test.augmentation_max_scale = _C.train.augmentation_max_scale
_C.test.augmentation_shift = 0.0
_C.test.augmentation_rotation = 0.0
_C.test.noise_magnitude = 0.05
_C.test.return_corr_indices = True
_C.test.return_normals = False
_C.test.matching_radius = _C.model.ground_truth_matching_radius
_C.test.overfitting_index = None
_C.test.curr_epoch = 0
_C.test.class_indices = "all"

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.1
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rmse_threshold = 0.2
_C.eval.rre_threshold = 15.0
_C.eval.rte_threshold = 0.3

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.05
_C.ransac.num_points = 3
_C.ransac.num_iterations = 1000

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.warmup_steps = 10000
_C.optim.eta_init = 0.1
_C.optim.eta_min = 0.1
_C.optim.max_epoch = 40
_C.optim.max_iteration = 400000
_C.optim.snapshot_steps = 10000
_C.optim.grad_acc_steps = 1


def _setup_logger_once(logger: logging.Logger = None, log_dir=None):
    """Setup logger once globally and suppress verbose Nova library logs"""
    logger = setup_logger(__name__, log_dir=log_dir)
    return logger


def make_cfg(
    output_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    snapshot_dir: Optional[str] = None,
    phase: Optional[int] = None,
) -> edict:
    """
    Create configuration with optional dataset and directory setup

    Args:
        output_dir: Output directory for logs and snapshots
        log_dir: Log directory
        snapshot_dir: Snapshot directory
        phase: Training phase for auto-generated output directory
    """
    cfg = edict(_C.copy())  # Ensure it's an EasyDict

    # Setup directories if any are provided
    if output_dir or phase is not None:
        if output_dir:
            cfg.output_dir = osp.join(
                cfg.root_dir, "output", f"transfer_learning_phase_{phase}", f"{output_dir}"
            )
        else:
            # Auto-generate output directory based on phase
            cfg.output_dir = osp.join(cfg.root_dir, "output", f"transfer_learning_phase_{phase}")

        # Setup subdirectories
        if not snapshot_dir:
            snapshot_dir = osp.join(cfg.output_dir, "snapshots")
        if not log_dir:
            log_dir = osp.join(cfg.output_dir, "logs")

        cfg.snapshot_dir = snapshot_dir
        cfg.log_dir = log_dir
    else:
        cfg.output_dir = osp.join(cfg.root_dir, "output", cfg.exp_name)
        cfg.snapshot_dir = osp.join(cfg.output_dir, "snapshots")
        cfg.log_dir = osp.join(cfg.output_dir, "logs")

    cfg.event_dir = osp.join(cfg.output_dir, "wandb_events")
    cfg.feature_dir = osp.join(cfg.output_dir, "features")
    cfg.registration_dir = osp.join(cfg.output_dir, "registration")

    # Setup cache directory for dataset materialization
    if cfg.data.cache_folder_base is None:
        cfg.data.cache_folder_base = osp.join(cfg.output_dir, "cache")

    # Ensure directories exist
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.snapshot_dir)
    ensure_dir(cfg.log_dir)
    ensure_dir(cfg.event_dir)
    ensure_dir(cfg.feature_dir)
    ensure_dir(cfg.registration_dir)

    # Create cache directory if materialization is enabled
    # Setup logger once with centralized log_dir
    _setup_logger_once(logger=logger, log_dir=cfg.log_dir)

    return cfg


def setup_dataset_proto(cfg: edict, model_config: model_config_pb2.ModelConfig) -> edict:

    cfg.data.dataset_proto = model_config.datasets
    try:
        # validate accessing train and val
        train_dataset = cfg.data.dataset_proto.train
        val_dataset = cfg.data.dataset_proto.val
        test_dataset = cfg.data.dataset_proto.test

        # check nonoe
        if train_dataset is None:
            raise RuntimeError(f"Model Config: {model_config} train dataset is none")
        if val_dataset is None:
            raise RuntimeError(f"Model Config: {model_config} validation dataset is none")
        if test_dataset is None:
            raise RuntimeError(f"Model Config: {model_config} test dataset is none")
    except Exception as e:
        raise RuntimeError(f"Model Config: accessing dataset failed: {e}")

    return cfg


def setup_dataset_config(
    cfg: edict, dataset_id: Optional[str] = None, dataset_root: Optional[str] = None
) -> edict:
    """
    Dynamically configure dataset settings based on provided parameters.

    Args:
        cfg: Configuration object
        dataset_id: RoboEye dataset ID (for Nova factory)
        dataset_root: Path to BOP dataset root
    """
    # Default values - can be overridden
    default_dataset_id = "roboeye/yolo_pretrain/2025-08-08-14-41-05"
    default_dataset_root = Path.home() / "repos/roboeye/datasets/"
    # Set dataset parameters
    if dataset_id is not None:
        cfg.data.dataset_id = dataset_id
        cfg.data.dataset_type = "roboeye_custom"

        # Try to create dataset proto first
        try:
            dataset = dataset_factory.get_dataset(cfg.data.dataset_id, get=True, overwrite=False)

            # Create a model config and populate it with dataset info
            model_cfg = model_config_pb2.ModelConfig()
            model_config_utils.set_filtered_datasets(
                cfg=model_cfg,
                train_dataset=dataset,
                train_split=cfg.data.keep_ratio,
                val_split=1 - cfg.data.keep_ratio,
            )

            # The dataset info is now in model_cfg, not returned from set_datasets
            cfg.data.dataset_proto = model_cfg.datasets

            if cfg.data.dataset_proto is not None:
                cfg.data.dataset_root = str(default_dataset_root)  # Placeholder
            else:
                raise ValueError("set_datasets returned None")

        except Exception as e:
            dataset_path = Path(default_dataset_root) / dataset_id
            cfg.data.dataset_root = str(dataset_path)
            raise RuntimeError(
                f"Failed to create dataset proto for dataset_id: {dataset_id} with error: {e}. "
                f"Falling back to dataset_root: {dataset_root}"
            )

    elif dataset_root is not None:
        # Expand user path
        if isinstance(dataset_root, str) and dataset_root.startswith("~"):
            dataset_root = Path(dataset_root).expanduser()
        cfg.data.dataset_root = str(Path(dataset_root))
        cfg.data.dataset_type = "bop"
        cfg.data.dataset_proto = None

    else:
        # Use defaults
        cfg.data.dataset_id = default_dataset_id
        cfg.data.dataset_type = "roboeye_custom"
        cfg.data.dataset_root = str(default_dataset_root)  # Ensure dataset_root is always set

        # Create dataset proto for default dataset
        dataset = dataset_factory.get_dataset(cfg.data.dataset_id, get=True, overwrite=False)
        model_cfg = model_config_pb2.ModelConfig()
        model_config_utils.set_filtered_datasets(
            cfg=model_cfg,
            train_dataset=dataset,
            train_split=cfg.data.keep_ratio,
            val_split=1 - cfg.data.keep_ratio,
        )
        cfg.data.dataset_proto = model_cfg.datasets

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--link_output", dest="link_output", action="store_true", help="link output dir"
    )
    args = parser.parse_args()
    return args


def main() -> None:
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, "output")


if __name__ == "__main__":
    main()
