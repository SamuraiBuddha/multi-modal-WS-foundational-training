"""
Callback classes for training intervention and monitoring.

Provides hooks into the training loop for:
- Early stopping
- Model checkpointing
- Learning rate scheduling
- Topology rewiring
- Metric logging
"""

import torch
from typing import Dict, Optional, Callable, Any
from pathlib import Path
import json


class Callback:
    """Base class for training callbacks."""

    def on_train_begin(self, trainer):
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer, history: Dict):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer, epoch: int):
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict
    ) -> bool:
        """
        Called at the end of each epoch.

        Returns:
            True to stop training early
        """
        pass

    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Stop training when a metric stops improving.
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
        restore_best: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            monitor: Metric to monitor
            patience: Epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for metric
            restore_best: Restore best model weights at end
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict
    ) -> bool:
        # Get monitored value
        if 'val' in self.monitor:
            metrics = val_metrics
            key = self.monitor.replace('val_', '')
        else:
            metrics = train_metrics
            key = self.monitor.replace('train_', '')

        current = metrics.get(key, metrics.get(self.monitor))
        if current is None:
            return False

        # Check for improvement
        if self.mode == 'min':
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta

        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best:
                self.best_weights = {
                    k: v.cpu().clone()
                    for k, v in trainer.model.state_dict().items()
                }
        else:
            self.wait += 1

        if self.wait >= self.patience:
            if self.restore_best and self.best_weights:
                trainer.model.load_state_dict(self.best_weights)
            return True  # Stop training

        return False


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_frequency: int = 1,
    ):
        """
        Initialize checkpoint callback.

        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model
            mode: 'min' or 'max'
            save_best_only: Only save when metric improves
            save_frequency: Save every N epochs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency

        self.best_value = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict
    ) -> bool:
        # Get monitored value
        metrics = val_metrics if 'val' in self.monitor else train_metrics
        key = self.monitor.replace('val_', '').replace('train_', '')
        current = metrics.get(key, metrics.get(self.monitor))

        should_save = False

        if current is not None:
            if self.mode == 'min':
                improved = current < self.best_value
            else:
                improved = current > self.best_value

            if improved:
                self.best_value = current
                should_save = True

        if not self.save_best_only and (epoch + 1) % self.save_frequency == 0:
            should_save = True

        if should_save:
            path = self.save_dir / f"checkpoint_epoch{epoch + 1}.pt"
            trainer.save(str(path))

            # Also save as best
            if current is not None:
                best_path = self.save_dir / "best_model.pt"
                trainer.save(str(best_path))

        return False


class LearningRateScheduler(Callback):
    """
    Adjust learning rate during training.
    """

    def __init__(
        self,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        schedule_type: str = 'cosine',
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
    ):
        """
        Initialize LR scheduler.

        Args:
            scheduler: Custom scheduler (or None to use built-in)
            schedule_type: 'cosine', 'step', 'exponential'
            warmup_epochs: Linear warmup epochs
            min_lr: Minimum learning rate
        """
        self.scheduler = scheduler
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.initial_lr = None

    def on_train_begin(self, trainer):
        self.initial_lr = trainer.config.learning_rate

        if self.scheduler is None:
            if self.schedule_type == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    trainer.optimizer,
                    T_max=trainer.config.epochs - self.warmup_epochs,
                    eta_min=self.min_lr,
                )
            elif self.schedule_type == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    trainer.optimizer,
                    step_size=30,
                    gamma=0.1,
                )
            elif self.schedule_type == 'exponential':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    trainer.optimizer,
                    gamma=0.95,
                )

    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict
    ) -> bool:
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = self.initial_lr * warmup_factor
        else:
            if self.scheduler:
                self.scheduler.step()

        return False


class TopologyRewiring(Callback):
    """
    Trigger topology rewiring during training.

    Works with sparse models that support rewiring (SET, DEEP R).
    """

    def __init__(
        self,
        rewire_frequency: int = 100,
        method: str = 'set',
        prune_rate: float = 0.3,
    ):
        """
        Initialize rewiring callback.

        Args:
            rewire_frequency: Steps between rewiring
            method: 'set' or 'deep_r'
            prune_rate: Fraction of connections to rewire
        """
        self.rewire_frequency = rewire_frequency
        self.method = method
        self.prune_rate = prune_rate

    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        if trainer.global_step % self.rewire_frequency != 0:
            return

        # Find all sparse layers and rewire
        for module in trainer.model.modules():
            if hasattr(module, 'rewire'):
                module.rewire(
                    method=self.method,
                    prune_rate=self.prune_rate,
                )


class MetricLogger(Callback):
    """
    Log metrics during training.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        print_frequency: int = 1,
    ):
        """
        Initialize metric logger.

        Args:
            log_dir: Directory to save logs (optional)
            print_frequency: Print metrics every N epochs
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.print_frequency = print_frequency
        self.log_data = []

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict
    ) -> bool:
        # Collect metrics
        entry = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
            'lr': trainer.optimizer.param_groups[0]['lr'],
        }

        # Add sparsity stats if available
        if hasattr(trainer.model, 'get_sparsity_stats'):
            entry['sparsity'] = trainer.model.get_sparsity_stats()

        self.log_data.append(entry)

        # Print
        if (epoch + 1) % self.print_frequency == 0:
            train_loss = train_metrics.get('loss', 0)
            train_acc = train_metrics.get('accuracy', 0)
            val_loss = val_metrics.get('loss', 0)
            val_acc = val_metrics.get('accuracy', 0)

            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
            )

        # Save to file
        if self.log_dir:
            log_path = self.log_dir / "training_log.json"
            with open(log_path, 'w') as f:
                json.dump(self.log_data, f, indent=2)

        return False
