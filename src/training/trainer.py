"""
Main Trainer class for neural network training.

Provides a flexible training loop with callbacks, logging, and evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import time
from pathlib import Path
import json


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 64
    device: str = "auto"
    mixed_precision: bool = False
    gradient_clip: Optional[float] = 1.0
    log_frequency: int = 10
    eval_frequency: int = 1
    save_dir: Optional[str] = None
    seed: int = 42


class Trainer:
    """
    Flexible trainer for neural networks.

    Supports:
    - Customizable training loops
    - Callback system for extensibility
    - Mixed precision training
    - Gradient clipping
    - Automatic device selection
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        callbacks: Optional[List] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            config: Training configuration
            optimizer: Optional optimizer (created if not provided)
            criterion: Loss function (defaults to CrossEntropyLoss)
            callbacks: List of callback objects
        """
        self.config = config
        self.callbacks = callbacks or []

        # Setup device
        if config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            )
        else:
            self.device = torch.device(config.device)

        # Move model to device
        self.model = model.to(self.device)

        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Setup criterion
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (overrides config)

        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.epochs

        # Callbacks: on_train_begin
        self._call_callbacks('on_train_begin')

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Callbacks: on_epoch_begin
            self._call_callbacks('on_epoch_begin', epoch=epoch)

            # Training
            train_metrics = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics.get('accuracy', 0))

            # Validation
            if val_loader is not None and (epoch + 1) % self.config.eval_frequency == 0:
                val_metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics.get('accuracy', 0))
            else:
                val_metrics = {}

            # Callbacks: on_epoch_end
            stop_training = self._call_callbacks(
                'on_epoch_end',
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

            if stop_training:
                print(f"[INFO] Early stopping at epoch {epoch + 1}")
                break

        # Callbacks: on_train_end
        self._call_callbacks('on_train_end', history=self.history)

        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs = batch[:-1]
                    targets = batch[-1]
            elif isinstance(batch, dict):
                targets = batch.pop('target', batch.pop('label', None))
                inputs = batch
            else:
                raise ValueError(f"Unknown batch format: {type(batch)}")

            # Move to device
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast('cuda'):
                    outputs = self._forward(inputs)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()

                if self.config.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self._forward(inputs)
                loss = self.criterion(outputs, targets)

                # Add auxiliary loss if present (e.g., MoE)
                if hasattr(self.model, 'aux_loss'):
                    loss = loss + self.model.aux_loss

                loss.backward()

                if self.config.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )

                self.optimizer.step()

            # Callbacks: on_batch_end
            self._call_callbacks(
                'on_batch_end',
                batch_idx=batch_idx,
                loss=loss.item(),
            )

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            self.global_step += 1

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    def _forward(self, inputs):
        """Forward pass handling different input formats."""
        if isinstance(inputs, dict):
            return self.model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            return self.model(*inputs)
        else:
            return self.model(inputs)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on data.

        Args:
            data_loader: Data loader for evaluation

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[-1]
            elif isinstance(batch, dict):
                targets = batch.pop('target', batch.pop('label', None))
                inputs = batch
            else:
                raise ValueError(f"Unknown batch format: {type(batch)}")

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            targets = targets.to(self.device)

            outputs = self._forward(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return {
            'loss': total_loss / len(data_loader),
            'accuracy': 100.0 * correct / total,
        }

    def _call_callbacks(self, event: str, **kwargs) -> bool:
        """Call all callbacks for an event."""
        stop = False
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                result = method(self, **kwargs)
                if result is True:
                    stop = True
        return stop

    def save(self, path: str):
        """Save model and training state."""
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'history': self.history,
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load model and training state."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.current_epoch = state.get('epoch', 0)
        self.global_step = state.get('global_step', 0)
        self.history = state.get('history', self.history)
