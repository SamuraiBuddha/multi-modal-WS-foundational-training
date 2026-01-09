"""
Sparse training utilities.

Implements training algorithms that maintain and evolve sparse connectivity:
- SET (Sparse Evolutionary Training)
- DEEP R (Deep Rewiring)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from .trainer import Trainer, TrainingConfig
from ..topology.rewiring import set_rewire, deep_r_rewire


@dataclass
class SETSchedule:
    """Schedule for SET algorithm."""
    prune_rate: float = 0.3
    regrow_rate: float = 0.3
    update_frequency: int = 100
    start_epoch: int = 0
    end_epoch: Optional[int] = None


@dataclass
class DEEPRSchedule:
    """Schedule for DEEP R algorithm."""
    temperature: float = 1.0
    prune_rate: float = 0.3
    update_frequency: int = 100
    temperature_decay: float = 0.99


class SparseTrainer(Trainer):
    """
    Trainer with sparse connectivity management.

    Extends base Trainer with:
    - SET/DEEP R rewiring during training
    - Sparsity tracking
    - Gradient accumulation for sparse layers
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        sparse_schedule: Optional[SETSchedule] = None,
        deep_r_schedule: Optional[DEEPRSchedule] = None,
        **kwargs
    ):
        """
        Initialize sparse trainer.

        Args:
            model: Model with sparse layers
            config: Training configuration
            sparse_schedule: SET algorithm schedule
            deep_r_schedule: DEEP R algorithm schedule
            **kwargs: Additional arguments for base Trainer
        """
        super().__init__(model, config, **kwargs)

        self.sparse_schedule = sparse_schedule or SETSchedule()
        self.deep_r_schedule = deep_r_schedule

        # Track sparsity history
        self.sparsity_history = []

        # Collect sparse layers
        self.sparse_layers = []
        for module in model.modules():
            if hasattr(module, 'mask') and hasattr(module, 'rewire'):
                self.sparse_layers.append(module)

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Training epoch with sparse updates."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            # Handle batch format
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[-1]
            elif isinstance(batch, dict):
                targets = batch.pop('target', batch.pop('label', None))
                inputs = batch
            else:
                raise ValueError(f"Unknown batch format")

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self._forward(inputs)
            loss = self.criterion(outputs, targets)

            if hasattr(self.model, 'aux_loss'):
                loss = loss + self.model.aux_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            # Store gradients for DEEP R before optimizer step
            if self.deep_r_schedule:
                self._store_gradients()

            self.optimizer.step()

            # Sparse rewiring
            self._maybe_rewire()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            self.global_step += 1

            # Callbacks
            self._call_callbacks('on_batch_end', batch_idx=batch_idx, loss=loss.item())

        # Record sparsity
        self._record_sparsity()

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total,
        }

    def _store_gradients(self):
        """Store gradients for DEEP R rewiring."""
        for layer in self.sparse_layers:
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                layer._stored_grad = layer.weight.grad.clone()

    def _maybe_rewire(self):
        """Perform rewiring if scheduled."""
        schedule = self.sparse_schedule

        # Check if we should rewire
        if self.global_step % schedule.update_frequency != 0:
            return

        if self.current_epoch < schedule.start_epoch:
            return

        if schedule.end_epoch and self.current_epoch >= schedule.end_epoch:
            return

        # Perform rewiring
        for layer in self.sparse_layers:
            if self.deep_r_schedule and hasattr(layer, '_stored_grad'):
                # Use DEEP R
                layer.mask.data = deep_r_rewire(
                    layer.weight.data,
                    layer.mask,
                    layer._stored_grad,
                    temperature=self.deep_r_schedule.temperature,
                    prune_rate=self.deep_r_schedule.prune_rate,
                )
                # Decay temperature
                self.deep_r_schedule.temperature *= self.deep_r_schedule.temperature_decay
            else:
                # Use SET
                layer.mask.data = set_rewire(
                    layer.weight.data,
                    layer.mask,
                    prune_rate=schedule.prune_rate,
                    regrow_rate=schedule.regrow_rate,
                )

    def _record_sparsity(self):
        """Record sparsity levels for all sparse layers."""
        sparsities = {}
        for i, layer in enumerate(self.sparse_layers):
            if hasattr(layer, 'get_sparsity'):
                sparsities[f'layer_{i}'] = layer.get_sparsity()
            elif hasattr(layer, 'mask'):
                total = layer.mask.numel()
                nonzero = layer.mask.count_nonzero().item()
                sparsities[f'layer_{i}'] = 1.0 - (nonzero / total)

        if sparsities:
            sparsities['epoch'] = self.current_epoch
            sparsities['step'] = self.global_step
            self.sparsity_history.append(sparsities)

    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get current sparsity statistics."""
        stats = {}
        total_params = 0
        active_params = 0

        for i, layer in enumerate(self.sparse_layers):
            if hasattr(layer, 'mask'):
                mask = layer.mask
                total = mask.numel()
                active = mask.count_nonzero().item()
                total_params += total
                active_params += active
                stats[f'layer_{i}_sparsity'] = 1.0 - (active / total)

        if total_params > 0:
            stats['overall_sparsity'] = 1.0 - (active_params / total_params)
            stats['compression_ratio'] = total_params / max(1, active_params)

        return stats


def create_sparse_model_from_dense(
    dense_model: nn.Module,
    target_sparsity: float = 0.9,
    topology: str = 'random',
) -> nn.Module:
    """
    Convert a dense model to sparse by replacing Linear layers.

    Args:
        dense_model: Original dense model
        target_sparsity: Target sparsity level
        topology: 'random' or 'ws' for Watts-Strogatz

    Returns:
        Model with sparse layers
    """
    from ..models.layers import SparseLinear

    density = 1.0 - target_sparsity

    def replace_linear(module, prefix=''):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace with sparse version
                sparse_layer = SparseLinear(
                    child.in_features,
                    child.out_features,
                    density=density,
                    bias=child.bias is not None,
                    mask_type='ws' if topology == 'ws' else 'random',
                )
                # Copy weights
                with torch.no_grad():
                    sparse_layer.weight.data = child.weight.data.clone()
                    if child.bias is not None:
                        sparse_layer.bias.data = child.bias.data.clone()

                setattr(module, name, sparse_layer)
            else:
                replace_linear(child, prefix=f"{prefix}{name}.")

    replace_linear(dense_model)
    return dense_model
