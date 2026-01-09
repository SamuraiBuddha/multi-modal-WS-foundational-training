"""
Data transforms and augmentation.

Provides transforms for different data modalities.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union, Callable


class ToTensor:
    """Convert numpy array to PyTorch tensor."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).to(self.dtype)


class Normalize:
    """Normalize tensor to zero mean and unit variance."""

    def __init__(
        self,
        mean: Union[float, Tuple[float, ...]] = 0.0,
        std: Union[float, Tuple[float, ...]] = 1.0,
    ):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.mean, (list, tuple)):
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
        else:
            mean = self.mean
            std = self.std

        return (x - mean) / std


class RandomNoise:
    """Add random Gaussian noise to tensor."""

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std
        return x + noise


class RandomCrop:
    """Randomly crop image tensor."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        return x[:, top:top+new_h, left:left+new_w]


class RandomHorizontalFlip:
    """Randomly flip image horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() < self.p:
            return torch.flip(x, dims=[-1])
        return x


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def default_image_transform(
    image_size: Tuple[int, int] = (28, 28),
    normalize: bool = True,
    augment: bool = False,
) -> Callable:
    """
    Create default image transform pipeline.

    Args:
        image_size: Target image size
        normalize: Apply normalization
        augment: Include augmentation

    Returns:
        Transform function
    """
    transforms = [ToTensor()]

    if augment:
        transforms.extend([
            RandomHorizontalFlip(0.5),
            RandomNoise(0.05),
        ])

    if normalize:
        transforms.append(Normalize(0.5, 0.5))

    return Compose(transforms)


def default_text_transform(
    max_length: int = 128,
    pad_token: int = 0,
) -> Callable:
    """
    Create default text transform (padding/truncation).

    Args:
        max_length: Maximum sequence length
        pad_token: Token to use for padding

    Returns:
        Transform function
    """
    def transform(tokens: Union[list, np.ndarray]) -> torch.Tensor:
        tokens = list(tokens)

        # Truncate
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        # Pad
        if len(tokens) < max_length:
            tokens = tokens + [pad_token] * (max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)

    return transform


def default_audio_transform(
    target_length: int = 64,
    normalize: bool = True,
) -> Callable:
    """
    Create default audio transform.

    Args:
        target_length: Target time dimension
        normalize: Apply normalization

    Returns:
        Transform function
    """
    def transform(spectrogram: np.ndarray) -> torch.Tensor:
        # Convert to tensor
        x = torch.from_numpy(spectrogram).float()

        # Pad or truncate time dimension
        if x.shape[-1] < target_length:
            pad_size = target_length - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad_size))
        elif x.shape[-1] > target_length:
            x = x[..., :target_length]

        # Normalize
        if normalize:
            x = (x - x.mean()) / (x.std() + 1e-8)

        return x

    return transform
