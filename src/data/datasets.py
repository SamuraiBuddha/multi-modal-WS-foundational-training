"""
Dataset implementations and loaders.

Provides easy access to common datasets for the curriculum.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Callable, Dict, List
from pathlib import Path


def get_mnist(
    data_dir: str = './data',
    train: bool = True,
    transform: Optional[Callable] = None,
    download: bool = True,
) -> Dataset:
    """
    Load MNIST dataset.

    Args:
        data_dir: Directory to store/load data
        train: Load training or test split
        transform: Optional transform to apply
        download: Download if not present

    Returns:
        MNIST dataset
    """
    try:
        from torchvision.datasets import MNIST
        from torchvision import transforms

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        return MNIST(
            root=data_dir,
            train=train,
            transform=transform,
            download=download,
        )
    except ImportError:
        print("[WARN] torchvision not available, using synthetic data")
        return SyntheticImageDataset(
            n_samples=60000 if train else 10000,
            image_size=(1, 28, 28),
            n_classes=10,
        )


def get_fashion_mnist(
    data_dir: str = './data',
    train: bool = True,
    transform: Optional[Callable] = None,
    download: bool = True,
) -> Dataset:
    """
    Load Fashion-MNIST dataset.

    Args:
        data_dir: Directory to store/load data
        train: Load training or test split
        transform: Optional transform to apply
        download: Download if not present

    Returns:
        Fashion-MNIST dataset
    """
    try:
        from torchvision.datasets import FashionMNIST
        from torchvision import transforms

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])

        return FashionMNIST(
            root=data_dir,
            train=train,
            transform=transform,
            download=download,
        )
    except ImportError:
        print("[WARN] torchvision not available, using synthetic data")
        return SyntheticImageDataset(
            n_samples=60000 if train else 10000,
            image_size=(1, 28, 28),
            n_classes=10,
        )


def get_cifar10(
    data_dir: str = './data',
    train: bool = True,
    transform: Optional[Callable] = None,
    download: bool = True,
) -> Dataset:
    """
    Load CIFAR-10 dataset.

    Args:
        data_dir: Directory to store/load data
        train: Load training or test split
        transform: Optional transform to apply
        download: Download if not present

    Returns:
        CIFAR-10 dataset
    """
    try:
        from torchvision.datasets import CIFAR10
        from torchvision import transforms

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)
                )
            ])

        return CIFAR10(
            root=data_dir,
            train=train,
            transform=transform,
            download=download,
        )
    except ImportError:
        print("[WARN] torchvision not available, using synthetic data")
        return SyntheticImageDataset(
            n_samples=50000 if train else 10000,
            image_size=(3, 32, 32),
            n_classes=10,
        )


class SyntheticImageDataset(Dataset):
    """
    Synthetic image dataset for testing without downloads.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        image_size: Tuple[int, int, int] = (1, 28, 28),
        n_classes: int = 10,
        seed: int = 42,
    ):
        super().__init__()
        np.random.seed(seed)

        self.n_samples = n_samples
        self.image_size = image_size
        self.n_classes = n_classes

        # Generate synthetic data
        self.images = np.random.randn(n_samples, *image_size).astype(np.float32)
        self.labels = np.random.randint(0, n_classes, size=n_samples)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = torch.from_numpy(self.images[idx])
        label = int(self.labels[idx])
        return image, label


class SyntheticMultiModal(Dataset):
    """
    Synthetic multi-modal dataset for testing and demonstration.

    Generates correlated visual, text, and audio representations.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        visual_dim: Tuple[int, int, int] = (1, 28, 28),
        text_seq_len: int = 32,
        vocab_size: int = 1000,
        audio_dim: Tuple[int, int] = (128, 64),  # mel_bins, time
        n_classes: int = 10,
        correlation: float = 0.5,
        seed: int = 42,
    ):
        """
        Initialize synthetic multi-modal dataset.

        Args:
            n_samples: Number of samples
            visual_dim: Visual input dimensions (C, H, W)
            text_seq_len: Text sequence length
            vocab_size: Vocabulary size for text
            audio_dim: Audio dimensions (mel_bins, time)
            n_classes: Number of classes
            correlation: Correlation between modalities (0-1)
            seed: Random seed
        """
        super().__init__()

        np.random.seed(seed)

        self.n_samples = n_samples
        self.n_classes = n_classes
        self.correlation = correlation

        # Generate class assignments
        self.labels = np.random.randint(0, n_classes, size=n_samples)

        # Generate class prototypes
        visual_flat = int(np.prod(visual_dim))
        self.visual_prototypes = np.random.randn(n_classes, visual_flat)
        self.text_prototypes = np.random.randint(0, vocab_size, (n_classes, text_seq_len))
        self.audio_prototypes = np.random.randn(n_classes, *audio_dim)

        # Generate samples with correlation to class
        self.visual_data = np.zeros((n_samples, *visual_dim), dtype=np.float32)
        self.text_data = np.zeros((n_samples, text_seq_len), dtype=np.int64)
        self.audio_data = np.zeros((n_samples, *audio_dim), dtype=np.float32)

        for i in range(n_samples):
            label = self.labels[i]

            # Visual: prototype + noise
            visual = (
                correlation * self.visual_prototypes[label] +
                (1 - correlation) * np.random.randn(visual_flat)
            )
            self.visual_data[i] = visual.reshape(visual_dim).astype(np.float32)

            # Text: prototype with random substitutions
            text = self.text_prototypes[label].copy()
            n_substitute = int((1 - correlation) * text_seq_len)
            sub_indices = np.random.choice(text_seq_len, n_substitute, replace=False)
            text[sub_indices] = np.random.randint(0, vocab_size, n_substitute)
            self.text_data[i] = text

            # Audio: prototype + noise
            audio = (
                correlation * self.audio_prototypes[label] +
                (1 - correlation) * np.random.randn(*audio_dim)
            )
            self.audio_data[i] = audio.astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'visual': torch.from_numpy(self.visual_data[idx]),
            'text': torch.from_numpy(self.text_data[idx]),
            'audio': torch.from_numpy(self.audio_data[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


class GraphDataset(Dataset):
    """
    Dataset of graphs for network topology exercises.
    """

    def __init__(
        self,
        n_graphs: int = 100,
        n_nodes_range: Tuple[int, int] = (10, 50),
        graph_types: List[str] = ['ws', 'er', 'ba'],
        seed: int = 42,
    ):
        """
        Initialize graph dataset.

        Args:
            n_graphs: Number of graphs to generate
            n_nodes_range: Range of node counts
            graph_types: Types of graphs to include
            seed: Random seed
        """
        super().__init__()

        np.random.seed(seed)

        from ..topology.graphs import (
            watts_strogatz_graph,
            erdos_renyi_graph,
            barabasi_albert_graph,
        )

        self.graphs = []
        self.labels = []
        self.graph_type_names = graph_types

        type_to_label = {t: i for i, t in enumerate(graph_types)}

        for _ in range(n_graphs):
            n_nodes = np.random.randint(*n_nodes_range)
            graph_type = np.random.choice(graph_types)

            if graph_type == 'ws':
                k = min(4, n_nodes - 1)
                beta = np.random.uniform(0.1, 0.5)
                adj = watts_strogatz_graph(n_nodes, k, beta)
            elif graph_type == 'er':
                p = np.random.uniform(0.1, 0.3)
                adj = erdos_renyi_graph(n_nodes, p)
            else:  # ba
                m = np.random.randint(1, 4)
                adj = barabasi_albert_graph(n_nodes, m)

            self.graphs.append(adj)
            self.labels.append(type_to_label[graph_type])

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        adj = torch.from_numpy(self.graphs[idx])
        label = self.labels[idx]
        return adj, label
