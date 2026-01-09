"""
Data loading and preprocessing module.

Provides:
- Dataset loaders for common benchmarks
- Data transforms and augmentation
- Multi-modal data loading utilities
"""

from .datasets import (
    get_mnist,
    get_fashion_mnist,
    get_cifar10,
    SyntheticMultiModal,
)
from .transforms import (
    ToTensor,
    Normalize,
    RandomNoise,
    default_image_transform,
)
from .loaders import (
    create_dataloaders,
    MultiModalDataLoader,
)

__all__ = [
    "get_mnist",
    "get_fashion_mnist",
    "get_cifar10",
    "SyntheticMultiModal",
    "ToTensor",
    "Normalize",
    "RandomNoise",
    "default_image_transform",
    "create_dataloaders",
    "MultiModalDataLoader",
]
