"""
Training module for neural network training utilities.

Provides:
- Trainer class for training loops
- Callbacks for monitoring and intervention
- Optimizers and schedulers
- Sparse training helpers
"""

from .trainer import Trainer, TrainingConfig
from .callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    TopologyRewiring,
    MetricLogger,
)
from .sparse_training import (
    SparseTrainer,
    SETSchedule,
    DEEPRSchedule,
)

__all__ = [
    "Trainer",
    "TrainingConfig",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "TopologyRewiring",
    "MetricLogger",
    "SparseTrainer",
    "SETSchedule",
    "DEEPRSchedule",
]
