"""
Demo script to test transfer learning setup without actual training.
This validates the configuration and model setup.
"""

import os
import sys
from pathlib import Path

import click
import torch
from easydict import EasyDict as edict

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from geoTransformer.GeoTransformer.experiments.roboeye.model import create_model
from geoTransformer.GeoTransformer.experiments.roboeye.utils import (
    create_transfer_learning_cfg,
    freeze_model_components,
    load_pretrained_weights,
)


def test_transfer_learning_setup(config: edict) -> bool:
    """Test the transfer learning setup"""
    print("=== Transfer Learning Setup Test ===\n")

    # Test configuration creation
    print("1. Testing configuration creation...")
    try:
        print("   ✓ Configuration created successfully")
        print(f"   Scale factor: {config.transfer_learning['scale_factor']:.3f}")
        print(f"   Learning rates: {config.transfer_learning['learning_rates']}")
        print(f"   Data Voxel size: {config.data.voxel_size}")
        print(f"   Dataset type: {config.data.dataset_type}")
        print(f"   Backbone Voxel size: {config.backbone.init_voxel_size}")
        print(f"   Backbone Base radius: {config.backbone.base_radius}")
        print(f"   Backbone Init radius: {config.backbone.init_radius}")
        print(f"   Backbone Init sigma: {config.backbone.init_sigma}")
        print(f"   GeoTransformer sigma_d: {config.geotransformer.sigma_d}")
        print(f"   GeoTransformer sigma_a: {config.geotransformer.sigma_a}")
        print(f"   GeoTransformer angle_k: {config.geotransformer.angle_k}")
        print(f"   Fine matching acceptance_radius: {config.fine_matching.acceptance_radius}")
        print(f"   GT matching radius: {config.model.ground_truth_matching_radius}")
    except Exception as e:
        print(f"   ✗ Configuration creation failed: {e}")
        return False

    # Test model creation
    print("2. Testing model creation...")
    try:
        model = create_model(config)
        print(" Model created successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        return False

    # Test progressive freezing
    print("\n3. Testing progressive freezing...")
    for phase in range(4):
        try:
            # Reset model (unfreeze all)
            for param in model.parameters():
                param.requires_grad = True

            # Apply freezing for this phase
            freeze_model_components(model, phase)

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

            print(f"   Phase {phase}: {trainable_params:,} trainable, {frozen_params:,} frozen")
        except Exception as e:
            print(f"   ✗ Phase {phase} freezing failed: {e}")
            return False

    # Test pretrained weights loading (if provided)
    if config.transfer_learning.pretrained_weights and os.path.exists(
        config.transfer_learning.pretrained_weights
    ):
        print("4. Testing pretrained weights loading...")
        try:
            num_loaded, num_failed = load_pretrained_weights(
                model, config.transfer_learning.pretrained_weights
            )
            print(f"   ✓ Loaded {num_loaded} parameters, {num_failed} failed")
        except Exception as e:
            print(f"   ✗ Pretrained weights loading failed: {e}")
            return False
    else:
        print("4. Skipping pretrained weights test (no valid path provided)")

    # Test GPU availability
    print("5. Testing GPU availability...")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name()}")
        try:
            model = model.cuda()
            print("   ✓ Model moved to GPU successfully")
        except Exception as e:
            print(f"   ✗ Failed to move model to GPU: {e}")
    else:
        print("   ⚠ CUDA not available, will use CPU")
    print("\n=== Transfer Learning Setup Test Complete ===")
    print("✓ All tests passed! Ready for transfer learning.")
    return True


@click.command()
@click.option(
    "--dataset_id",
    type=str,
    default="HST-YZ/CAD6608043632/2025-10-14-rvbust_RE_20_synthetic",
    help='RoboEye dataset ID (e.g., "roboeye/yolo_pretrain/2025-08-08-14-41-05")',
)
@click.option(
    "--dataset_root",
    type=str,
    default=None,
    help='BOP dataset root path (e.g., "~/repos/roboeye/datasets/ITODD_converted")',
)
@click.option(
    "--cache_bool",
    is_flag=True,
    default=False,
    help="Save train and validation dataset preprocessed or not",
)
@click.option("--train_split", type=float, default=0.7, help="Data fraction for training [0-1]")
@click.option(
    "--pretrained_weights", type=str, default=None, help="Path to pretrained weights (.pth.tar)"
)
@click.option(
    "--pretrained_voxel_size",
    type=float,
    default=0.025,
    help="Voxel size of pretrained model (default: 0.025m)",
)
@click.option(
    "--target_voxel_size", type=float, default=0.009, help="Target voxel size (default: 0.009m)"
)
@click.option(
    "--voxel_strategy",
    type=click.Choice(["conservative", "balanced", "robust"]),
    default="conservative",
    help="Voxel size recommendation strategy (default: conservative)",
)
@click.option("--phase", type=int, default=0, help="Training phase (0-4, default: 0)")
@click.option(
    "--enable_progressive_unfreezing",
    is_flag=True,
    default=False,
    help="Enable automatic progressive unfreezing (enhanced 5-phase strategy)",
)
@click.option(
    "--freeze_epochs",
    type=str,
    default="8 24 40 56",
    help="Custom freeze epoch schedule (e.g., --freeze_epochs '8 20 32 42')",
)
@click.option(
    "--learning_rates",
    type=float,
    multiple=True,
    default=[5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
    help="Custom learning rate schedule for 5 phases (e.g., --learning_rates 5e-6 1e-5 2e-5 5e-5 1e-4)",
)
@click.option(
    "--optim_max_epoch", type=int, default=None, help="Number of epochs (overrides config default)"
)
@click.option(
    "--optim_lr", type=float, default=None, help="Learning rate (overrides phase default)"
)
@click.option(
    "--gradient_accumulation_steps",
    type=int,
    default=4,
    help="Number of steps to accumulate gradients before performing an optimizer update",
)
@click.option(
    "--so3_curriculum_epochs",
    type=int,
    default=50,
    help="Number of epochs to reach full SO3 rotation (default: 50)",
)
@click.option(
    "--max_so3_rotation_deg",
    type=float,
    default=180.0,
    help="Maximum SO3 rotation angle in degrees (default: 180.0)",
)
@click.option(
    "--enable_so3_curriculum",
    is_flag=True,
    default=False,
    help="Enable SO3 curriculum learning (default: True)",
)
@click.option(
    "--resume", is_flag=True, default=False, help="Resume from specific checkpoint path",
)
@click.option(
    "--continue_optimizer",
    is_flag=True,
    default=True,
    help="Continue optimizer state when resuming (same phase)",
)
@click.option(
    "--output_dir", type=str, default=None, help="Output directory (default: auto-generated)"
)
def main(
    dataset_id: str,
    dataset_root: str,
    cache_bool: bool,
    train_split: float,
    pretrained_weights: str,
    pretrained_voxel_size: float,
    target_voxel_size: float,
    voxel_strategy: str,
    phase: int,
    enable_progressive_unfreezing: bool,
    freeze_epochs: str,
    learning_rates: tuple[float, ...],
    optim_max_epoch: int,
    optim_lr: float,
    gradient_accumulation_steps: int,
    so3_curriculum_epochs: int,
    max_so3_rotation_deg: float,
    enable_so3_curriculum: bool,
    resume: bool,
    continue_optimizer: bool,
    output_dir: str,
) -> None:
    """Main training function"""
    # Validate arguments
    if dataset_id and dataset_root:
        raise ValueError("Error: Cannot specify both --dataset_id and --dataset_root")

    # Pack arguments into a params object (simulate argparse.Namespace)
    class Params:
        pass

    params = Params()
    params.dataset_id = dataset_id
    params.dataset_root = dataset_root
    params.cache_bool = cache_bool
    params.train_split = train_split
    params.pretrained_weights = pretrained_weights
    params.pretrained_voxel_size = pretrained_voxel_size
    params.target_voxel_size = target_voxel_size
    params.voxel_strategy = voxel_strategy
    params.phase = phase
    params.enable_progressive_unfreezing = enable_progressive_unfreezing
    if isinstance(freeze_epochs, str):
        freeze_epochs_list = [int(e) for e in freeze_epochs.replace(",", " ").split() if e.strip()]
    else:
        freeze_epochs_list = list(freeze_epochs)
    params.freeze_epochs = freeze_epochs_list
    if isinstance(learning_rates, str):
        learning_rates = [
            float(lr) for lr in learning_rates.replace(",", " ").split() if lr.strip()
        ]
    else:
        learning_rates = list(learning_rates)
    params.learning_rates = learning_rates
    params.optim_max_epoch = optim_max_epoch
    params.optim_lr = optim_lr
    params.gradient_accumulation_steps = gradient_accumulation_steps
    params.curriculum_learning = Params()
    params.curriculum_learning.enable_so3_curriculum = enable_so3_curriculum
    params.curriculum_learning.so3_curriculum_epochs = so3_curriculum_epochs
    params.curriculum_learning.max_so3_rotation_deg = max_so3_rotation_deg
    params.resume = resume
    params.continue_optimizer = continue_optimizer
    params.output_dir = output_dir

    training_cfg = create_transfer_learning_cfg(
        model_config_proto=None,
        params=params,
        dataset_id=dataset_id,
        dataset_root=dataset_root,
        output_dir=output_dir,
        log_dir=None,
        snapshot_dir=None,
        phase=phase,
    )
    success = test_transfer_learning_setup(config=training_cfg,)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
