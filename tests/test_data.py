"""
Tests for data module.
"""

import pytest
import torch
import sys
sys.path.insert(0, '..')

from src.data import (
    SyntheticMultiModal,
    GraphDataset,
    create_dataloaders,
    ToTensor,
    Normalize,
    RandomNoise,
)


class TestSyntheticMultiModal:
    """Tests for synthetic multi-modal dataset."""

    def test_dataset_length(self):
        """Dataset should have correct number of samples."""
        dataset = SyntheticMultiModal(n_samples=100)
        assert len(dataset) == 100

    def test_dataset_sample_keys(self):
        """Each sample should have all modality keys."""
        dataset = SyntheticMultiModal(n_samples=10)
        sample = dataset[0]

        assert 'visual' in sample
        assert 'text' in sample
        assert 'audio' in sample
        assert 'label' in sample

    def test_dataset_shapes(self):
        """Sample shapes should match configuration."""
        dataset = SyntheticMultiModal(
            n_samples=10,
            visual_dim=(1, 28, 28),
            text_seq_len=32,
            audio_dim=(128, 64),
        )
        sample = dataset[0]

        assert sample['visual'].shape == (1, 28, 28)
        assert sample['text'].shape == (32,)
        assert sample['audio'].shape == (128, 64)

    def test_dataset_reproducibility(self):
        """Same seed should produce same data."""
        dataset1 = SyntheticMultiModal(n_samples=10, seed=42)
        dataset2 = SyntheticMultiModal(n_samples=10, seed=42)

        sample1 = dataset1[0]
        sample2 = dataset2[0]

        assert torch.allclose(sample1['visual'], sample2['visual'])


class TestGraphDataset:
    """Tests for graph dataset."""

    def test_graph_dataset_length(self):
        """Graph dataset should have correct length."""
        dataset = GraphDataset(n_graphs=50)
        assert len(dataset) == 50

    def test_graph_dataset_sample(self):
        """Each sample should have adjacency matrix and label."""
        dataset = GraphDataset(n_graphs=10, n_nodes_range=(20, 21))
        adj, label = dataset[0]

        # Adjacency matrix should be 2D (variable size based on n_nodes_range)
        assert adj.dim() == 2
        assert isinstance(label, int)


class TestTransforms:
    """Tests for data transforms."""

    def test_to_tensor(self):
        """ToTensor should convert numpy to tensor."""
        import numpy as np
        transform = ToTensor()
        data = np.array([1.0, 2.0, 3.0])
        result = transform(data)
        assert isinstance(result, torch.Tensor)

    def test_normalize(self):
        """Normalize should center and scale data."""
        transform = Normalize(mean=0.5, std=0.5)
        x = torch.ones(3, 3) * 0.5
        result = transform(x)
        assert torch.allclose(result, torch.zeros(3, 3))

    def test_random_noise(self):
        """RandomNoise should add noise to tensor."""
        transform = RandomNoise(std=0.1)
        x = torch.zeros(10, 10)
        result = transform(x)
        assert not torch.allclose(result, x)


class TestDataLoaders:
    """Tests for data loader creation."""

    def test_create_dataloaders(self):
        """create_dataloaders should return dict of loaders."""
        # Create dummy train and val datasets
        x_train = torch.randn(80, 10)
        y_train = torch.randint(0, 2, (80,))
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

        x_val = torch.randn(20, 10)
        y_val = torch.randint(0, 2, (20,))
        val_dataset = torch.utils.data.TensorDataset(x_val, y_val)

        loaders = create_dataloaders(
            train_dataset,
            val_dataset=val_dataset,
            batch_size=16,
        )

        assert 'train' in loaders
        assert 'val' in loaders

    def test_dataloader_batch_size(self):
        """Data loaders should use correct batch size."""
        x = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = torch.utils.data.TensorDataset(x, y)

        loaders = create_dataloaders(
            dataset,
            batch_size=32,
        )

        batch = next(iter(loaders['train']))
        assert batch[0].shape[0] == 32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
