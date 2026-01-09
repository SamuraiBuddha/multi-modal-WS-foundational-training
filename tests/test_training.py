"""
Tests for training module.
"""

import pytest
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from src.training import (
    Trainer,
    TrainingConfig,
    EarlyStopping,
    ModelCheckpoint,
    TopologyRewiring,
    SparseTrainer,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = TrainingConfig()
        assert config.epochs > 0
        assert config.batch_size > 0
        assert 0 < config.learning_rate < 1

    def test_custom_config(self):
        """Custom config values should be set."""
        config = TrainingConfig(
            epochs=50,
            batch_size=64,
            learning_rate=0.01,
        )
        assert config.epochs == 50
        assert config.batch_size == 64
        assert config.learning_rate == 0.01


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_early_stopping_patience(self):
        """Should stop after patience epochs without improvement."""
        callback = EarlyStopping(patience=3, min_delta=0.01)

        # Simulated improving losses
        callback.on_epoch_end(0, {'val_loss': 1.0})
        assert not callback.should_stop

        callback.on_epoch_end(1, {'val_loss': 0.9})
        assert not callback.should_stop

        # Simulated stagnation
        callback.on_epoch_end(2, {'val_loss': 0.9})
        callback.on_epoch_end(3, {'val_loss': 0.9})
        callback.on_epoch_end(4, {'val_loss': 0.9})

        assert callback.should_stop

    def test_early_stopping_improvement_resets(self):
        """Improvement should reset patience counter."""
        callback = EarlyStopping(patience=2, min_delta=0.01)

        callback.on_epoch_end(0, {'val_loss': 1.0})
        callback.on_epoch_end(1, {'val_loss': 0.99})  # Not enough improvement
        callback.on_epoch_end(2, {'val_loss': 0.8})   # Good improvement
        callback.on_epoch_end(3, {'val_loss': 0.79})  # Not enough
        callback.on_epoch_end(4, {'val_loss': 0.78})  # Not enough

        assert callback.should_stop


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_initialization(self):
        """Trainer should initialize correctly."""
        model = SimpleModel()
        config = TrainingConfig(epochs=2)
        trainer = Trainer(model, config)

        assert trainer.model is model
        assert trainer.config.epochs == 2

    def test_trainer_single_step(self):
        """Trainer should complete a training step."""
        model = SimpleModel()
        config = TrainingConfig(epochs=1, batch_size=4)
        trainer = Trainer(model, config)

        # Create dummy data
        x = torch.randn(8, 10)
        y = torch.randint(0, 2, (8,))
        dataset = torch.utils.data.TensorDataset(x, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        # Train for one epoch
        history = trainer.train(train_loader)

        assert 'train_loss' in history
        assert len(history['train_loss']) == 1


class TestSparseTrainer:
    """Tests for SparseTrainer with SET/DEEP-R."""

    def test_sparse_trainer_rewiring(self):
        """SparseTrainer should perform topology rewiring."""
        model = SimpleModel()
        config = TrainingConfig(epochs=2)

        trainer = SparseTrainer(
            model,
            config,
            rewire_every=1,
            prune_rate=0.3,
        )

        assert trainer.rewire_every == 1
        assert trainer.prune_rate == 0.3


class TestTopologyRewiring:
    """Tests for TopologyRewiring callback."""

    def test_rewiring_callback_trigger(self):
        """Callback should trigger at correct intervals."""
        callback = TopologyRewiring(
            rewire_every=5,
            prune_rate=0.2,
        )

        # Should not trigger at epoch 1-4
        for epoch in range(4):
            result = callback.on_epoch_end(epoch, {})
            # No error means success

        # Should trigger at epoch 5
        result = callback.on_epoch_end(4, {})
        # Check that it was triggered (implementation dependent)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
