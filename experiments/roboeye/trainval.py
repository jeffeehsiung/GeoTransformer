#!/usr/bin/env python3
import os
import os.path as osp
from typing import Dict, Tuple

import click
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from easydict import EasyDict as edict

mp.set_start_method("spawn", force=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.benchmark = False  # Disable for deterministic but saves memory

import faulthandler
import math
import signal

from geotransformer.engine import EpochBasedTrainer
from geotransformer.utils.torch import build_warmup_cosine_lr_scheduler

from geoTransformer.GeoTransformer.experiments.roboeye.dataloader import train_valid_data_loader
from geoTransformer.GeoTransformer.experiments.roboeye.loss import Evaluator, OverallLoss
from geoTransformer.GeoTransformer.experiments.roboeye.model import create_model
from geoTransformer.GeoTransformer.experiments.roboeye.utils import create_transfer_learning_cfg

faulthandler.register(signal.SIGUSR1)


class GeoTransformerTrainer(EpochBasedTrainer):
    def __init__(self, training_cfg: edict) -> None:
        self.training_cfg = training_cfg

        # Initialize best validation tracking for smart checkpoint management
        self.best_val_loss = float("inf")
        self.best_val_epoch = 0

        # Disable default snapshot saving - we'll handle it ourselves with smart saving
        super().__init__(
            training_cfg,
            max_epoch=training_cfg.transfer_learning.optim_max_epoch,
            save_all_snapshots=False,  # Disable default per-epoch saving
            grad_acc_steps=training_cfg.transfer_learning.gradient_accumulation_steps,  # Enable gradient accumulation
        )

        # Create dataloaders (same as RoboEye Trainer)
        self.setup_dataloaders()

        self.logger.info(
            f"Target voxel size: {self.training_cfg.transfer_learning.target_voxel_size*1000:.1f}mm"
        )
        self.logger.info(f"Output directory: {self.training_cfg.output_dir}")

        # Create model (same as RoboEye Trainer)
        self.setup_model()

        # Setup optimizer and scheduler with transfer learning settings (MUST be after model setup)
        self.setup_optimizer()

        # Setup loss and evaluator (same as RoboEye Trainer)
        self.setup_loss_and_evaluator()

    def setup_dataloaders(self) -> None:
        """Setup dataloaders (same as RoboEye Trainer)"""
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(
            self.training_cfg,
            self.distributed,
            cache_bool=self.training_cfg.transfer_learning.cache_bool,
        )
        self.register_loader(train_loader, val_loader)

    def setup_model(self) -> None:
        """Setup model (same as RoboEye Trainer)"""

        model = create_model(self.training_cfg).cuda()
        self.model = self.register_model(model)

        device = next(self.model.parameters()).device
        if device.type != "cuda":
            self.logger.warning(f"WARNING: Model is on {device}, not GPU!")

    def setup_loss_and_evaluator(self):
        """Setup loss function and evaluator (same as RoboEye Trainer)"""
        self.loss_func = OverallLoss(self.training_cfg).cuda()
        self.evaluator = Evaluator(self.training_cfg).cuda()

    def setup_optimizer(self) -> None:
        """Setup optimizer and scheduler specifically for transfer learning"""

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_cfg.optim.lr,
            weight_decay=self.training_cfg.optim.weight_decay,
        )
        self.register_optimizer(optimizer)

        total_steps = self.training_cfg.optim.max_iteration
        warmup_steps = self.training_cfg.optim.warmup_steps
        scheduler = build_warmup_cosine_lr_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            eta_init=self.training_cfg.optim.eta_init,
            eta_min=self.training_cfg.optim.eta_min,
            grad_acc_steps=self.training_cfg.transfer_learning.gradient_accumulation_steps,
        )
        self.register_optimizer(optimizer)
        self.register_scheduler(scheduler)

        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Train step with gradient accumulation support"""
        super().train_step(epoch, iteration, data_dict)

        if epoch == 1 and iteration == 0:
            self.logger.info("=== GPU DEVICE CHECK (First Training Step) ===")
            for key, value in data_dict.items():
                if hasattr(value, "device"):
                    self.logger.info(f"Data {key} device: {value.device}")
                else:
                    self.logger.info(f"Data {key} type: {type(value)} (no device attribute)")
            model_device = next(self.model.parameters()).device
            self.logger.info(f"Model device: {model_device}")

            # Check specifically for transform
            if "transform" in data_dict:
                transform = data_dict["transform"]
                self.logger.info(f"Transform type: {type(transform)}")
                if isinstance(transform, list):
                    self.logger.error(f"ERROR: Transform is a list with {len(transform)} elements")
                    if len(transform) > 0:
                        self.logger.info(f"First transform element type: {type(transform[0])}")
                        if hasattr(transform[0], "shape"):
                            self.logger.info(f"First transform element shape: {transform[0].shape}")

            # Log gradient accumulation setup
            grad_accum_steps = self.training_cfg.train.gradient_accumulation_steps

        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)

        # Gradient accumulation: scale loss by accumulation steps
        grad_accum_steps = self.training_cfg.train.gradient_accumulation_steps
        if grad_accum_steps > 1:
            # Scale loss to average over accumulation steps
            for key in loss_dict:
                if "loss" in key.lower() and isinstance(loss_dict[key], torch.Tensor):
                    loss_dict[key] = loss_dict[key] / grad_accum_steps

        # Clear cache after each step to prevent fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_dict, loss_dict

    def val_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Validation step (same as RoboEye Trainer)"""
        super().val_step(epoch, iteration, data_dict)
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)

        return output_dict, loss_dict

    def after_val_step(
        self,
        epoch: int,
        iteration: int,
        data_dict: Dict[str, torch.Tensor],
        output_dict: Dict[str, torch.Tensor],
        result_dict: Dict[str, torch.Tensor],
    ) -> None:
        """Override to capture validation summary for best checkpoint tracking"""
        super().after_val_step(epoch, iteration, data_dict, output_dict, result_dict)
        # Only main process saves checkpoints in DDP
        if hasattr(self, "local_rank") and getattr(self, "local_rank", 0) == 0:
            checkpoint_strategy = self.training_cfg.transfer_learning.checkpoint_strategy
            save_best_only = checkpoint_strategy.save_best_only
            cleanup_old = checkpoint_strategy.cleanup_old_checkpoints
            max_checkpoints = checkpoint_strategy.max_checkpoints

            if save_best_only and self.is_best_validation_epoch(epoch, result_dict):
                self.save_best_checkpoint(epoch)
                if cleanup_old:
                    self.cleanup_old_checkpoints(max_checkpoints, epoch)

    def before_train_epoch(self, epoch: int) -> None:
        """Update dataset epoch for SO3 curriculum learning before training epoch"""
        super().before_train_epoch(epoch)

        # Update curr_epoch in training dataset for SO3 curriculum learning
        if hasattr(self.train_loader, "dataset") and hasattr(
            self.train_loader.dataset, "update_epoch"
        ):
            self.train_loader.dataset.update_epoch(epoch)

    def before_val_epoch(self, epoch: int) -> None:
        """Update dataset epoch for SO3 curriculum learning before validation epoch"""
        super().before_val_epoch(epoch)

        # Update curr_epoch in validation dataset for SO3 curriculum learning
        if hasattr(self.val_loader, "dataset") and hasattr(self.val_loader.dataset, "update_epoch"):
            self.val_loader.dataset.update_epoch(epoch)

    def after_train_epoch(self, epoch: int) -> None:
        """Enhanced checkpoint management with smart saving to prevent memory bloat"""
        # Call parent method for standard logging
        super().after_train_epoch(epoch)

        # Only main process saves checkpoints
        if hasattr(self, "local_rank") and getattr(self, "local_rank", 0) == 0:
            checkpoint_strategy = self.training_cfg.transfer_learning.checkpoint_strategy
            cleanup_old = checkpoint_strategy.cleanup_old_checkpoints
            max_checkpoints = checkpoint_strategy.max_checkpoints
            # Cleanup old checkpoints if enabled
            if cleanup_old:
                self.cleanup_old_checkpoints(max_checkpoints, epoch)

    def is_best_validation_epoch(self, epoch: int, result_dict: Dict[str, torch.Tensor]) -> bool:
        """Check if this epoch has the best validation performance so far"""
        if "RMSE" not in result_dict:
            self.logger.warning(f"No RMSE in result dict for epoch {epoch}")
            return False

        current_val_loss = result_dict["RMSE"]

        if math.isnan(current_val_loss) or math.isinf(current_val_loss):
            self.logger.warning(
                f"Epoch {epoch}: Invalid validation loss (NaN or Inf), skipping best checkpoint save"
            )
            return False

        # Check if this is the best validation loss so far
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_val_epoch = epoch
            self.logger.info(f"New best validation loss: {current_val_loss:.6f} at epoch {epoch}")
            return True

        return False

    def cleanup_old_checkpoints(self, max_checkpoints: int, epoch: int) -> None:
        """Remove old checkpoints to free up disk space"""
        try:
            import glob

            epoch_pattern = osp.join(self.training_cfg.snapshot_dir, f"epoch-{epoch}*.pth.tar")
            epoch_files = glob.glob(epoch_pattern)
            if len(epoch_files) >= (max_checkpoints // 2):
                # Sort by modification time (oldest first)
                epoch_files.sort(key=lambda x: os.path.getmtime(x))

                files_to_remove = epoch_files[: -(max_checkpoints // 2)]
                for file_path in files_to_remove:
                    try:
                        os.remove(file_path)
                        self.logger.info(
                            f"Cleaned up epoch-specific file: {os.path.basename(file_path)}"
                        )
                    except OSError as e:
                        self.logger.warning(f"Failed to remove {file_path}: {e}")

        except Exception as e:
            self.logger.warning(f"Checkpoint cleanup failed: {e}")

    def save_best_checkpoint(self, epoch: int) -> None:
        """Save the best validation checkpoint with a dedicated 'best' filename"""
        best_checkpoint_name = "best_epoch.pth.tar"
        best_checkpoint_path = osp.join(self.training_cfg.snapshot_dir, best_checkpoint_name)

        checkpoint = {
            "epoch": epoch,
            "is_best": True,
            "val_loss": self.best_val_loss,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict()
            if hasattr(self, "scheduler") and self.scheduler is not None
            else None,
            "cfg": self.training_cfg,
            "target_voxel_size": self.training_cfg.transfer_learning.target_voxel_size,
            "pretrained_voxel_size": self.training_cfg.transfer_learning.pretrained_voxel_size,
        }

        torch.save(checkpoint, best_checkpoint_path)
        self.logger.info(
            f"✅ Best checkpoint saved: {best_checkpoint_name} (epoch {epoch}, val_loss: {self.best_val_loss:.6f})"
        )
        # Log summary of saved weights
        param_names = [name for name, _ in self.model.named_parameters() if _.requires_grad]
        self.logger.info(f"Best checkpoint trainable parameter names: {param_names}")


@click.command()
@click.option(
    "--dataset_id",
    type=str,
    default=None,
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
@click.option("--val_split", type=float, default=0.3, help="Data fraction for validation [0-1]")
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
@click.option(
    "--optim_max_epoch", type=int, default=None, help="Number of epochs (overrides config default)"
)
@click.option("--optim_lr", type=float, default=None, help="Learning rate (overrides default)")
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
@click.option("--resume", is_flag=True, default=False, help="Resume from specific checkpoint path")
@click.option(
    "--output_dir", type=str, default=None, help="Output directory (default: auto-generated)"
)
def main(
    dataset_id,
    dataset_root,
    cache_bool,
    train_split,
    val_split,
    pretrained_voxel_size,
    target_voxel_size,
    voxel_strategy,
    optim_max_epoch,
    optim_lr,
    gradient_accumulation_steps,
    so3_curriculum_epochs,
    max_so3_rotation_deg,
    enable_so3_curriculum,
    resume,
    output_dir,
):
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
    params.val_split = val_split
    params.pretrained_voxel_size = pretrained_voxel_size
    params.target_voxel_size = target_voxel_size
    params.voxel_strategy = voxel_strategy
    params.optim_max_epoch = optim_max_epoch
    params.optim_lr = optim_lr
    params.gradient_accumulation_steps = gradient_accumulation_steps
    params.so3_curriculum_epochs = so3_curriculum_epochs
    params.max_so3_rotation_deg = max_so3_rotation_deg
    params.enable_so3_curriculum = enable_so3_curriculum
    params.resume = resume
    params.output_dir = output_dir

    training_cfg = create_transfer_learning_cfg(
        model_config_proto=None,
        params=params,
        dataset_id=dataset_id,
        dataset_root=dataset_root,
        output_dir=output_dir,
        log_dir=None,
        snapshot_dir=None,
        phase=None,
    )
    # Create and run trainer
    trainer = GeoTransformerTrainer(training_cfg)
    trainer.run()


if __name__ == "__main__":
    main()
