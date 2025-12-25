#!/usr/bin/env python3
import os
import os.path as osp
from typing import Any, Dict, Tuple

import click
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from easydict import EasyDict as edict

mp.set_start_method("spawn", force=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.benchmark = False  # Disable for deterministic but saves memory

import faulthandler
import signal

from geotransformer.engine import EpochBasedTrainer
from geotransformer.utils.torch import build_warmup_cosine_lr_scheduler

from geoTransformer.GeoTransformer.experiments.roboeye.dataloader import (
    train_data_loader,
    val_data_loader,
)
from geoTransformer.GeoTransformer.experiments.roboeye.loss import Evaluator, OverallLoss
from geoTransformer.GeoTransformer.experiments.roboeye.model import create_model
from geoTransformer.GeoTransformer.experiments.roboeye.utils import (
    create_transfer_learning_cfg,
    freeze_model_components,
    load_pretrained_weights,
)

faulthandler.register(signal.SIGUSR1)


class GeoTransformerTLTrainer(EpochBasedTrainer):
    def __init__(self, training_cfg: edict):
        self.training_cfg = training_cfg

        # Initialize best validation tracking for smart checkpoint management
        self.best_val_loss = float("inf")
        self.best_val_epoch = 0

        # Initialize progressive unfreezing state (for future use)
        self._current_phase = training_cfg.transfer_learning.phase

        # Disable default snapshot saving - we'll handle it ourselves with smart saving
        super().__init__(
            training_cfg,
            max_epoch=training_cfg.transfer_learning.optim_max_epoch,
            save_all_snapshots=False,  # Disable default per-epoch saving
            grad_acc_steps=training_cfg.transfer_learning.gradient_accumulation_steps,  # Enable gradient accumulation
        )

        # Create dataloaders (same as RoboEye Trainer)
        self.setup_dataloaders()

        # Log transfer learning setup info
        self.logger.info("=== Transfer Learning Training ===")
        self.logger.info(f"Phase: {self._current_phase}")
        self.logger.info(
            f"Pretrained weights: {self.training_cfg.transfer_learning.pretrained_weights}"
        )
        self.logger.info(
            f"Target voxel size: {self.training_cfg.transfer_learning.target_voxel_size*1000:.1f}mm"
        )
        self.logger.info(f"Output directory: {self.training_cfg.output_dir}")

        # Create model (same as RoboEye Trainer)
        self.setup_model()

        # Load pretrained weights after model is created
        self.load_pretrained_weights()

        # Setup progressive freezing
        freeze_model_components(self.model, self._current_phase)

        # Setup optimizer and scheduler with transfer learning settings (MUST be after model setup)
        self.setup_transfer_learning_optimizer()

        # Setup loss and evaluator (same as RoboEye Trainer)
        self.setup_loss_and_evaluator()

        # Log trainable parameters
        self.log_trainable_parameters()

        # Always try to resume (will auto-detect previous phase if needed)
        self.resume_from_checkpoint()

    def setup_dataloaders(self) -> None:
        """Setup dataloaders (same as RoboEye Trainer)"""
        train_loader, neighbor_limits = train_data_loader(
            self.training_cfg,
            self.distributed,
            cache_bool=self.training_cfg.transfer_learning.cache_bool,
        )
        val_loader, _ = val_data_loader(
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

    def setup_transfer_learning_optimizer(self) -> None:
        """Setup optimizer and scheduler specifically for transfer learning"""

        lr = self.training_cfg.transfer_learning["learning_rates"][self._current_phase]

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer = optim.Adam(
            trainable_params, lr=lr, weight_decay=self.training_cfg.optim.weight_decay
        )
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

        self.logger.info("Transfer Learning Optimizer setup:")
        self.logger.info(f"  Phase {self._current_phase} LR: {lr}")
        self.logger.info(f"  Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        self.logger.info(
            f"  Eta init: {self.training_cfg.optim.eta_init}, Eta min: {self.training_cfg.optim.eta_min}"
        )

    def train_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validation step (same as RoboEye Trainer)"""
        super().val_step(epoch, iteration, data_dict)
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_dict, loss_dict

    def after_val_step(
        self,
        epoch: int,
        iteration: int,
        data_dict: Dict[str, Any],
        output_dict: Dict[str, Any],
        result_dict: Dict[str, Any],
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

    def load_pretrained_weights(self) -> None:
        """Load pretrained weights"""
        num_loaded, num_failed = load_pretrained_weights(
            self.model, self.training_cfg.transfer_learning.pretrained_weights
        )
        self.logger.info(f"Loaded {num_loaded} parameters, {num_failed} failed")

    def log_trainable_parameters(self) -> None:
        """Log information about trainable parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(
            f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)"
        )

        # Log which components are trainable
        trainable_components = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                component = name.split(".")[0]
                if component not in trainable_components:
                    trainable_components.append(component)

        self.logger.info(f"Trainable components: {trainable_components}")

        # Log phase-specific freezing strategy
        phase_descriptions = {
            0: "Phase 0: Scale Validation + Architectural Compatibility (optimal_transport + geometric embedding)",
            1: "Phase A: Coarse Feature Adaptation + Early Attention (optimal_transport + first 2 self-attention layers)",
            2: "Phase B: Cross-Modal Learning + Mid-Level Features (optimal_transport + transformer layers 0-3)",
            3: "Phase C1: Backbone Integration + Full Transformer (optimal_transport + full transformer + decoder)",
            4: "Phase C2: Full Model Polish + Domain Adaptation (all parameters)",
        }
        self.logger.info(
            f"Phase {self._current_phase} strategy: {phase_descriptions.get(self._current_phase, 'Unknown')}"
        )

    def update_progressive_unfreezing(self, epoch: int) -> None:
        """
        Update model freezing based on freeze_epochs schedule
        This implements automatic progressive unfreezing - should only be called when enabled
        """
        if (
            not hasattr(self.training_cfg, "transfer_learning")
            or "freeze_epochs" not in self.training_cfg.transfer_learning
        ):
            return

        freeze_schedule = self.training_cfg.transfer_learning["freeze_epochs"]

        # Determine current phase based on epoch
        current_auto_phase = 0
        for i, freeze_epoch in enumerate(freeze_schedule):
            if epoch >= freeze_epoch:
                current_auto_phase = i + 1

        # Check if phase has changed (no manual phase conflicts since this only runs when progressive is enabled)
        if not hasattr(self, "_current_phase") or self._current_phase != current_auto_phase:
            old_phase = getattr(self, "_current_phase", -1)
            self._current_phase = current_auto_phase

            self.logger.info(
                f"=== Progressive Unfreezing: Epoch {epoch} → Phase {current_auto_phase} ==="
            )

            if old_phase >= 0:
                self.logger.info(f"Phase transition: {old_phase} → {current_auto_phase}")

            # Apply new freezing strategy
            freeze_model_components(self.model, current_auto_phase)

            # Update optimizer for new trainable parameters
            self.update_optimizer_for_phase(current_auto_phase)

            # Log the change
            self.log_phase_transition(old_phase, current_auto_phase, epoch)

    def update_optimizer_for_phase(self, phase: int) -> None:
        """Update optimizer when phase changes to include newly unfrozen parameters"""
        # Get new learning rate for this phase
        if (
            hasattr(self.training_cfg, "transfer_learning")
            and "learning_rates" in self.training_cfg.transfer_learning
        ):
            lr_schedule = self.training_cfg.transfer_learning["learning_rates"]
            if phase < len(lr_schedule):
                new_lr = lr_schedule[phase]
            else:
                new_lr = lr_schedule[-1]  # Use last LR if phase exceeds schedule
        else:
            new_lr = self.training_cfg.optim.lr

        # Get newly trainable parameters
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        # Create new optimizer with new parameters and LR
        new_optimizer = optim.Adam(
            trainable_params, lr=new_lr, weight_decay=self.training_cfg.optim.weight_decay
        )

        # Update optimizer (scheduler remains unchanged - it works at optimizer level)
        self.register_optimizer(new_optimizer)
        self.optimizer = new_optimizer

        self.logger.info(f"Optimizer updated for phase {phase}: LR={new_lr}")
        self.logger.info("Scheduler continues unchanged (cosine annealing schedule maintained)")

    def log_phase_transition(self, old_phase: int, new_phase: int, epoch: int) -> None:
        """Log details of enhanced 5-phase transition"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Phase {new_phase} at epoch {epoch}:")
        self.logger.info(
            f"  Trainable parameters: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.1f}%)"
        )

        # Log which components are now trainable
        trainable_components = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                component = name.split(".")[0]
                if component not in trainable_components:
                    trainable_components.append(component)

        self.logger.info(f"  Trainable components: {trainable_components}")

        # Enhanced phase descriptions
        phase_descriptions = {
            0: "Phase 0: Scale Validation + Architectural Compatibility (optimal_transport + geometric embedding)",
            1: "Phase A: Coarse Feature Adaptation + Early Attention (optimal_transport + first 2 self-attention layers)",
            2: "Phase B: Cross-Modal Learning + Mid-Level Features (optimal_transport + transformer layers 0-3)",
            3: "Phase C1: Backbone Integration + Full Transformer (optimal_transport + full transformer + decoder)",
            4: "Phase C2: Full Model Polish + Domain Adaptation (all parameters)",
        }
        self.logger.info(f"  Strategy: {phase_descriptions.get(new_phase, 'Unknown phase')}")

    def before_train_epoch(self, epoch: int) -> None:
        """Update dataset epoch for SO3 curriculum learning before training epoch"""
        super().before_train_epoch(epoch)

        # Only check progressive unfreezing if explicitly enabled via --enable_progressive_unfreezing
        progressive_enabled = self.training_cfg.transfer_learning.enable_progressive_unfreezing

        if progressive_enabled:
            self.update_progressive_unfreezing(epoch)
        else:
            self.logger.warning("=== Manual Phase Control (Default) ===")
            self.logger.warning(
                "Progressive unfreezing is DISABLED - no automatic phase transitions will occur"
            )

        # Update curr_epoch in training dataset for SO3 curriculum learning
        if hasattr(self.train_loader, "dataset") and hasattr(
            self.train_loader.dataset, "update_epoch"
        ):
            self.train_loader.dataset.update_epoch(epoch)

    def before_val_epoch(self, epoch: int):
        """Update dataset epoch for SO3 curriculum learning before validation epoch"""
        super().before_val_epoch(epoch)

        # Update curr_epoch in validation dataset for SO3 curriculum learning
        if hasattr(self.val_loader, "dataset") and hasattr(self.val_loader.dataset, "update_epoch"):
            self.val_loader.dataset.update_epoch(epoch)

    def resume_from_checkpoint(self) -> None:
        """Resume training from checkpoint"""
        resume_path = None

        # Check for explicit resume path
        if self.training_cfg.transfer_learning.resume:
            # Try best checkpoint for current phase
            best_checkpoint = osp.join(
                self.training_cfg.snapshot_dir, f"best_phase_{self._current_phase}.pth.tar"
            )
            if os.path.exists(best_checkpoint):
                resume_path = best_checkpoint
                self.logger.info(f"Auto-resuming from best checkpoint: {best_checkpoint}")
            else:
                # Try latest checkpoint for current phase
                latest_checkpoint = osp.join(
                    self.training_cfg.snapshot_dir, f"latest_phase_{self._current_phase}.pth.tar"
                )
                if os.path.exists(latest_checkpoint):
                    resume_path = latest_checkpoint
                    self.logger.info(f"Auto-resuming from latest checkpoint: {latest_checkpoint}")

        # If still no checkpoint and starting a new phase, try to find previous phase completion
        if not resume_path and self._current_phase > 0:
            # Look for completion checkpoint from previous phase
            previous_phase = self._current_phase - 1
            voxel_mm = int(self.training_cfg.transfer_learning.target_voxel_size * 1000)
            scale_factor = int(
                (
                    self.training_cfg.transfer_learning.target_voxel_size
                    / self.training_cfg.transfer_learning.pretrained_voxel_size
                )
                * 1000
            )

            # Try latest phase checkpoint first
            latest_prev_phase = osp.join(
                self.training_cfg.snapshot_dir, f"latest_phase_{previous_phase}.pth.tar"
            )
            if os.path.exists(latest_prev_phase):
                resume_path = latest_prev_phase
                self.logger.info(
                    f"Auto-resuming from previous phase {previous_phase}: {latest_prev_phase}"
                )
            else:
                # Try specific completion checkpoint
                phase_completion_name = (
                    f"phase_{previous_phase}_complete_voxel{voxel_mm}mm_scale{scale_factor}.pth.tar"
                )
                phase_completion_path = osp.join(
                    self.training_cfg.snapshot_dir, phase_completion_name
                )
                if os.path.exists(phase_completion_path):
                    resume_path = phase_completion_path
                    self.logger.info(
                        f"Auto-resuming from previous phase {previous_phase}: {phase_completion_name}"
                    )

        if not resume_path:
            if self._current_phase > 0:
                self.logger.warning(
                    f"No checkpoint found from previous phase {self._current_phase - 1}. Starting phase {self._current_phase} from pretrained weights."
                )
            return

        self.logger.info(f"Resuming from checkpoint: {resume_path}")

        checkpoint = torch.load(resume_path, map_location="cpu")

        # Load model state
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
            self.logger.info("Model state loaded from checkpoint")
            # Log summary of loaded weights
            param_names = [name for name, _ in self.model.named_parameters() if _.requires_grad]
            self.logger.info(f"Trainable parameter names after checkpoint load: {param_names}")

        # For phase transitions, don't load optimizer/scheduler (start fresh for new phase)
        # Only load optimizer/scheduler if continuing same phase
        load_optimizer = False
        if "phase" in checkpoint and checkpoint["phase"] == self._current_phase:
            load_optimizer = (
                hasattr(self.training_cfg.transfer_learning, "continue_optimizer")
                and self.training_cfg.transfer_learning.continue_optimizer
            )
            if load_optimizer:
                self.logger.info("Same phase continuation - will load optimizer/scheduler state")
            else:
                self.logger.info(
                    "Same phase but continue_optimizer=False - starting with fresh optimizer/scheduler"
                )
        else:
            self.logger.info(
                f"Phase transition ({checkpoint.get('phase', 'unknown')} -> {self._current_phase}) - starting with fresh optimizer/scheduler"
            )

        # Load optimizer state if continuing same phase
        if load_optimizer and "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.logger.info("Optimizer state loaded from checkpoint")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load optimizer state: {e} - continuing with fresh optimizer"
                )

        # Load scheduler state if continuing same phase
        if load_optimizer and "scheduler" in checkpoint:
            try:
                if hasattr(self, "scheduler") and self.scheduler is not None:
                    self.scheduler.load_state_dict(checkpoint["scheduler"])
                    self.logger.info("Scheduler state loaded from checkpoint")
                else:
                    self.logger.warning(
                        "No scheduler registered to load state into; skipping scheduler restore"
                    )
            except Exception as e:
                self.logger.warning(
                    f"Failed to load scheduler state: {e} - continuing with fresh scheduler"
                )

        # Load epoch info only if continuing same phase
        if load_optimizer and "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
            self.logger.info(f"Resuming from epoch {self.start_epoch}")
        else:
            self.logger.info("Starting new phase from epoch 1")

    def after_train_epoch(self, epoch: int) -> None:
        """Enhanced checkpoint management with smart saving to prevent memory bloat"""
        # Call parent method for standard logging
        super().after_train_epoch(epoch)

        # Log phase-specific information
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.logger.info(
            f"Phase {self._current_phase} - Epoch {epoch} completed - Current LR: {current_lr:.6f}"
        )

        # Only main process saves checkpoints
        if hasattr(self, "local_rank") and getattr(self, "local_rank", 0) == 0:
            checkpoint_strategy = self.training_cfg.transfer_learning.checkpoint_strategy
            save_frequency = checkpoint_strategy.save_frequency
            save_phase_transitions = checkpoint_strategy.save_phase_transitions
            cleanup_old = checkpoint_strategy.cleanup_old_checkpoints
            max_checkpoints = checkpoint_strategy.max_checkpoints

            # Determine if we should save checkpoint this epoch
            should_save = False
            save_reason = ""

            # Always save at phase completion
            if epoch == self.max_epoch:
                should_save = True
                save_reason = "phase_completion"

            # Save at regular intervals (but not every epoch)
            elif epoch % save_frequency == 0:
                should_save = True
                save_reason = "regular_interval"

            # Save at phase transitions (for progressive unfreezing)
            elif save_phase_transitions and self.is_phase_transition_epoch(epoch):
                should_save = True
                save_reason = "phase_transition"

            if should_save:
                self.logger.info(f"Saving checkpoint (reason: {save_reason})")

                if save_reason == "phase_completion":
                    self.save_phase_completion_checkpoint(epoch)
                else:
                    self.save_transfer_learning_checkpoint(epoch, save_reason)

                # Cleanup old checkpoints if enabled
                if cleanup_old:
                    self.cleanup_old_checkpoints(max_checkpoints, epoch)
            else:
                self.logger.debug(
                    f"Skipping checkpoint save for epoch {epoch} (smart saving enabled)"
                )

    def is_best_validation_epoch(self, epoch: int, result_dict: Dict[str, Any]) -> bool:
        """Check if this epoch has the best validation performance so far"""
        if "RMSE" not in result_dict:
            self.logger.warning(f"No RMSE in result dict for epoch {epoch}")
            return False

        current_val_loss = result_dict["RMSE"]

        # Check for NaN - skip if validation failed
        import math

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

    def is_phase_transition_epoch(self, epoch: int):
        """Check if this epoch marks a phase transition"""
        if not hasattr(self.training_cfg, "transfer_learning"):
            return False

        freeze_epochs = self.training_cfg.transfer_learning.freeze_epochs
        return epoch in freeze_epochs

    def cleanup_old_checkpoints(self, max_checkpoints: int, epoch: int) -> None:
        """Remove old checkpoints to free up disk space"""
        try:
            import glob

            # Get all checkpoint files for this phase
            checkpoint_pattern = osp.join(
                self.training_cfg.snapshot_dir,
                f"checkpoint_epoch_*_phase_{self._current_phase}_*.pth.tar",
            )
            checkpoint_files = glob.glob(checkpoint_pattern)

            if len(checkpoint_files) > max_checkpoints:
                # Sort by modification time (oldest first)
                checkpoint_files.sort(key=lambda x: os.path.getmtime(x))

                # Remove oldest files, keeping only max_checkpoints
                files_to_remove = checkpoint_files[:-max_checkpoints]

                for file_path in files_to_remove:
                    try:
                        os.remove(file_path)
                        self.logger.info(
                            f"Cleaned up old checkpoint: {os.path.basename(file_path)}"
                        )
                    except OSError as e:
                        self.logger.warning(f"Failed to remove {file_path}: {e}")

                self.logger.info(
                    f"Checkpoint cleanup: removed {len(files_to_remove)} old files, keeping {len(checkpoint_files) - len(files_to_remove)}"
                )
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

    def save_phase_completion_checkpoint(self, epoch: int) -> None:
        """Save a special checkpoint marking completion of this phase"""
        voxel_mm = int(self.training_cfg.transfer_learning.target_voxel_size * 1000)
        scale_factor = int(
            (
                self.training_cfg.transfer_learning.target_voxel_size
                / self.training_cfg.transfer_learning.pretrained_voxel_size
            )
            * 1000
        )

        phase_completion_name = (
            f"phase_{self._current_phase}_complete_voxel{voxel_mm}mm_scale{scale_factor}.pth.tar"
        )
        phase_completion_path = osp.join(self.training_cfg.snapshot_dir, phase_completion_name)

        checkpoint = {
            "epoch": epoch,
            "phase": self._current_phase,
            "phase_complete": True,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict()
            if hasattr(self, "scheduler") and self.scheduler is not None
            else None,
            "cfg": self.training_cfg,
            "target_voxel_size": self.training_cfg.transfer_learning.target_voxel_size,
            "pretrained_voxel_size": self.training_cfg.transfer_learning.pretrained_voxel_size,
        }

        torch.save(checkpoint, phase_completion_path)
        self.logger.info(
            f"Phase {self._current_phase} completion checkpoint saved: {phase_completion_name}"
        )

        # Also update a latest phase checkpoint for easy discovery
        latest_phase_path = osp.join(
            self.training_cfg.snapshot_dir, f"latest_phase_{self._current_phase}.pth.tar"
        )
        torch.save(checkpoint, latest_phase_path)

    def save_best_checkpoint(self, epoch: int) -> None:
        """Save the best validation checkpoint with a dedicated 'best' filename"""
        best_checkpoint_name = f"best_phase_{self._current_phase}.pth.tar"
        best_checkpoint_path = osp.join(self.training_cfg.snapshot_dir, best_checkpoint_name)

        checkpoint = {
            "epoch": epoch,
            "phase": self._current_phase,
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

    def save_transfer_learning_checkpoint(self, epoch: int, save_reason: str = "regular") -> None:
        """Save checkpoint with transfer learning specific naming and enhanced metadata"""
        # Generate descriptive checkpoint name with save reason
        voxel_mm = int(self.training_cfg.backbone.init_voxel_size * 1000)
        sigma_d_mm = int(self.training_cfg.geotransformer.sigma_d * 1000)
        sigma_a = int(self.training_cfg.geotransformer.sigma_a)
        angle_k = int(self.training_cfg.geotransformer.angle_k)
        scale_factor = int(
            (
                self.training_cfg.backbone.init_voxel_size
                / self.training_cfg.transfer_learning.pretrained_voxel_size
            )
            * 1000
        )

        # Include save reason in filename for better organization
        checkpoint_name = f"checkpoint_epoch_{epoch}_phase_{self._current_phase}_{save_reason}_voxel{voxel_mm}mm_sigma-d{sigma_d_mm}mm_sigma-a{sigma_a}_angle-k{angle_k}_scale{scale_factor}.pth.tar"
        checkpoint_path = osp.join(self.training_cfg.snapshot_dir, checkpoint_name)

        # Enhanced checkpoint data with strategic framework metadata
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict()
            if hasattr(self, "scheduler") and self.scheduler is not None
            else None,
            "cfg": self.training_cfg,
            "phase": self._current_phase,
            "target_voxel_size": self.training_cfg.transfer_learning.target_voxel_size,
            "pretrained_voxel_size": self.training_cfg.transfer_learning.pretrained_voxel_size,
            "save_reason": save_reason,
            "strategic_phase_name": {
                0: "Phase 0: Scale Validation + Architectural Compatibility",
                1: "Phase A: Coarse Feature Adaptation + Early Attention",
                2: "Phase B: Cross-Modal Learning + Mid-Level Features",
                3: "Phase C1: Backbone Integration + Full Transformer",
                4: "Phase C2: Full Model Polish + Domain Adaptation",
            }.get(self._current_phase, f"Unknown Phase {self._current_phase}"),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "framework_version": "enhanced_5_phase_v1.0",
        }

        torch.save(checkpoint, checkpoint_path)

        # Only save as latest checkpoint if this is a significant save (not just regular interval)
        # This avoids redundant latest snapshots and lets users resume from best/important checkpoints
        if save_reason in ["best_validation", "phase_completion", "phase_transition"]:
            latest_path = osp.join(
                self.training_cfg.snapshot_dir,
                f"latest_phase_{self._current_phase}_epoch_{epoch}.pth.tar",
            )
            torch.save(checkpoint, latest_path)
            self.logger.info(
                f"Updated latest checkpoint: latest_phase_{self._current_phase}_epoch_{epoch}.pth.tar"
            )

        self.logger.info(f"Transfer learning checkpoint saved: {checkpoint_name}")

        # Log checkpoint statistics
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        self.logger.info(f"Checkpoint size: {file_size_mb:.1f} MB (reason: {save_reason})")


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
    default=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    help="Custom learning rate schedule for 5 phases (e.g., --learning_rates 5e-6 1e-5 2e-5 5e-5 1e-4)",
)
@click.option(
    "--optim_max_epoch", type=int, default=80, help="Number of epochs (overrides config default)"
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
    default=False,
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
    # Create and run trainer
    trainer = GeoTransformerTLTrainer(training_cfg)
    trainer.run()


if __name__ == "__main__":
    main()
