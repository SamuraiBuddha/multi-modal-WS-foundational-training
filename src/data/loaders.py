"""
Data loader utilities.

Provides specialized data loaders for different scenarios.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Optional, Dict, List, Iterator, Tuple
import numpy as np


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test data loaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer

    Returns:
        Dictionary of data loaders
    """
    loaders = {}

    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    if val_dataset is not None:
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if test_dataset is not None:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loaders


class MultiModalDataLoader:
    """
    Data loader for multi-modal datasets.

    Handles alignment of different modalities and missing data.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        modality_dropout: float = 0.0,
    ):
        """
        Initialize multi-modal data loader.

        Args:
            dataset: Multi-modal dataset
            batch_size: Batch size
            shuffle: Shuffle data
            num_workers: Number of workers
            pin_memory: Pin memory
            drop_last: Drop incomplete batches
            modality_dropout: Probability of dropping each modality
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.modality_dropout = modality_dropout

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for multi-modal data."""
        collated = {}

        # Get all keys from first sample
        keys = batch[0].keys()

        for key in keys:
            values = [sample[key] for sample in batch]

            # Stack tensors
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values

        # Apply modality dropout during training
        if self.modality_dropout > 0:
            for key in list(collated.keys()):
                if key not in ['label', 'target']:
                    if np.random.random() < self.modality_dropout:
                        # Zero out this modality
                        collated[key] = torch.zeros_like(collated[key])

        return collated

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)


class BalancedBatchSampler(Sampler):
    """
    Sampler that ensures balanced classes in each batch.
    """

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        drop_last: bool = False,
    ):
        """
        Initialize balanced sampler.

        Args:
            labels: List of labels for all samples
            batch_size: Batch size
            drop_last: Drop incomplete batches
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group indices by class
        self.class_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        self.n_classes = len(self.class_indices)
        self.samples_per_class = batch_size // self.n_classes

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within each class
        shuffled = {
            c: np.random.permutation(indices).tolist()
            for c, indices in self.class_indices.items()
        }

        # Track position in each class
        positions = {c: 0 for c in self.class_indices}

        batches = []
        while True:
            batch = []

            for c in self.class_indices:
                # Get samples from this class
                pos = positions[c]
                indices = shuffled[c]

                for _ in range(self.samples_per_class):
                    if pos >= len(indices):
                        # Reshuffle and reset
                        shuffled[c] = np.random.permutation(
                            self.class_indices[c]
                        ).tolist()
                        indices = shuffled[c]
                        pos = 0

                    batch.append(indices[pos])
                    pos += 1

                positions[c] = pos

            if len(batch) == self.batch_size:
                batches.append(batch)

            # Stop after one epoch worth of data
            if len(batches) * self.batch_size >= len(self.labels):
                break

        # Shuffle batches
        np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        n_batches = len(self.labels) // self.batch_size
        if not self.drop_last and len(self.labels) % self.batch_size != 0:
            n_batches += 1
        return n_batches


class InfiniteDataLoader:
    """
    Data loader that cycles indefinitely.

    Useful for training without explicit epochs.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
        self.iterator = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return batch
