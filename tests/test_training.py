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
    SETSchedule,
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


class MockTrainer:
    """Mock trainer for callback testing."""
    def __init__(self):
        self.model = SimpleModel()


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_early_stopping_patience(self):
        """Should stop after patience epochs without improvement."""
        callback = EarlyStopping(patience=3, min_delta=0.01)
        trainer = MockTrainer()

        # Simulated improving losses
        stop = callback.on_epoch_end(trainer, 0, {'loss': 1.0}, {'loss': 1.0})
        assert not stop

        stop = callback.on_epoch_end(trainer, 1, {'loss': 0.9}, {'loss': 0.9})
        assert not stop

        # Simulated stagnation
        callback.on_epoch_end(trainer, 2, {'loss': 0.9}, {'loss': 0.9})
        callback.on_epoch_end(trainer, 3, {'loss': 0.9}, {'loss': 0.9})
        stop = callback.on_epoch_end(trainer, 4, {'loss': 0.9}, {'loss': 0.9})

        assert stop

    def test_early_stopping_improvement_resets(self):
        """Improvement should reset patience counter."""
        callback = EarlyStopping(patience=2, min_delta=0.01)
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, 0, {'loss': 1.0}, {'loss': 1.0})
        callback.on_epoch_end(trainer, 1, {'loss': 0.99}, {'loss': 0.99})  # Not enough improvement
        callback.on_epoch_end(trainer, 2, {'loss': 0.8}, {'loss': 0.8})    # Good improvement - resets
        callback.on_epoch_end(trainer, 3, {'loss': 0.8}, {'loss': 0.8})    # No improvement (same)
        stop = callback.on_epoch_end(trainer, 4, {'loss': 0.8}, {'loss': 0.8})  # No improvement (same)

        assert stop


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
        history = trainer.fit(train_loader)

        assert 'train_loss' in history
        assert len(history['train_loss']) == 1


class TestSparseTrainer:
    """Tests for SparseTrainer with SET/DEEP-R."""

    def test_sparse_trainer_rewiring(self):
        """SparseTrainer should perform topology rewiring."""
        model = SimpleModel()
        config = TrainingConfig(epochs=2)

        schedule = SETSchedule(update_frequency=1, prune_rate=0.3)
        trainer = SparseTrainer(
            model,
            config,
            sparse_schedule=schedule,
        )

        assert trainer.sparse_schedule.update_frequency == 1
        assert trainer.sparse_schedule.prune_rate == 0.3


class TestTopologyRewiring:
    """Tests for TopologyRewiring callback."""

    def test_rewiring_callback_initialization(self):
        """Callback should initialize with correct parameters."""
        callback = TopologyRewiring(
            rewire_frequency=5,
            prune_rate=0.2,
        )

        assert callback.rewire_frequency == 5
        assert callback.prune_rate == 0.2
        assert callback.method == 'set'  # default method


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
