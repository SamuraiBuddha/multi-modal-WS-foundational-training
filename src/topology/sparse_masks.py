"""
Sparse mask generation and manipulation for neural network layers.

Provides functions to create and manage sparse connectivity patterns
for neural networks based on various graph topologies.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union
import scipy.sparse as sp


def ws_sparse_mask(
    in_features: int,
    out_features: int,
    k: int,
    beta: float,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Create a Watts-Strogatz sparse mask for a neural network layer.

    Maps the WS small-world topology to a weight matrix mask.
    For non-square matrices, adapts the topology appropriately.

    Args:
        in_features: Number of input features (columns)
        out_features: Number of output features (rows)
        k: Initial neighbors per node (controls density)
        beta: Rewiring probability
        seed: Random seed
        dtype: PyTorch dtype for the mask
        device: Device to place mask on

    Returns:
        Binary mask tensor of shape (out_features, in_features)

    Example:
        >>> mask = ws_sparse_mask(784, 256, k=4, beta=0.3)
        >>> mask.shape
        torch.Size([256, 784])
        >>> mask.sum() / mask.numel()  # Sparsity level
        tensor(...)
    """
    if seed is not None:
        np.random.seed(seed)

    # For non-square matrices, we create a bipartite-like WS structure
    # Map input nodes to output nodes using WS rewiring logic

    mask = np.zeros((out_features, in_features), dtype=np.float32)

    # Determine base connectivity per output node
    base_connections = max(1, min(k, in_features))

    for out_idx in range(out_features):
        # Base pattern: connect to nearby input indices (ring-like)
        # Map output index to input space
        base_in_idx = int(out_idx * in_features / out_features)

        # Create initial local connections
        connections = set()
        for offset in range(-(base_connections // 2), base_connections // 2 + 1):
            if offset == 0 and base_connections % 2 == 0:
                continue
            in_idx = (base_in_idx + offset) % in_features
            connections.add(in_idx)

        # Ensure we have exactly base_connections
        while len(connections) < base_connections:
            in_idx = np.random.randint(0, in_features)
            connections.add(in_idx)

        connections = list(connections)[:base_connections]

        # Rewire with probability beta
        for i, in_idx in enumerate(connections):
            if np.random.random() < beta:
                # Rewire to random input
                new_idx = np.random.randint(0, in_features)
                while new_idx in connections:
                    new_idx = np.random.randint(0, in_features)
                connections[i] = new_idx

        # Set mask
        for in_idx in connections:
            mask[out_idx, in_idx] = 1.0

    mask_tensor = torch.tensor(mask, dtype=dtype)
    if device is not None:
        mask_tensor = mask_tensor.to(device)

    return mask_tensor


def random_sparse_mask(
    in_features: int,
    out_features: int,
    density: float,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Create a random sparse mask with specified density.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        density: Fraction of connections to keep (0 to 1)
        seed: Random seed
        dtype: PyTorch dtype
        device: Device to place mask on

    Returns:
        Binary mask tensor

    Example:
        >>> mask = random_sparse_mask(100, 50, density=0.1)
        >>> actual_density = mask.sum() / mask.numel()
        >>> abs(actual_density - 0.1) < 0.02  # Should be close to target
        True
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Generate random mask
    rand = torch.rand(out_features, in_features)
    mask = (rand < density).to(dtype)

    if device is not None:
        mask = mask.to(device)

    return mask


def structured_sparse_mask(
    in_features: int,
    out_features: int,
    block_size: int,
    density: float,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Create a block-structured sparse mask.

    Useful for hardware-efficient sparse computation.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        block_size: Size of dense blocks
        density: Fraction of blocks to keep
        seed: Random seed
        dtype: PyTorch dtype
        device: Device to place mask on

    Returns:
        Block-sparse mask tensor
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate block grid dimensions
    n_blocks_out = (out_features + block_size - 1) // block_size
    n_blocks_in = (in_features + block_size - 1) // block_size

    # Create block selection mask
    block_mask = np.random.random((n_blocks_out, n_blocks_in)) < density

    # Expand to full mask
    mask = np.zeros((out_features, in_features), dtype=np.float32)
    for i in range(n_blocks_out):
        for j in range(n_blocks_in):
            if block_mask[i, j]:
                row_start = i * block_size
                row_end = min((i + 1) * block_size, out_features)
                col_start = j * block_size
                col_end = min((j + 1) * block_size, in_features)
                mask[row_start:row_end, col_start:col_end] = 1.0

    mask_tensor = torch.tensor(mask, dtype=dtype)
    if device is not None:
        mask_tensor = mask_tensor.to(device)

    return mask_tensor


def mask_to_adjacency(mask: torch.Tensor) -> np.ndarray:
    """
    Convert a weight mask to an adjacency matrix representation.

    For visualization and graph analysis purposes.

    Args:
        mask: Binary mask tensor

    Returns:
        Adjacency matrix as numpy array
    """
    return mask.detach().cpu().numpy()


def adjacency_to_mask(
    adj: np.ndarray,
    dtype: torch.dtype = torch.float32,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Convert an adjacency matrix to a weight mask.

    Args:
        adj: Adjacency matrix as numpy array
        dtype: PyTorch dtype
        device: Device to place mask on

    Returns:
        Binary mask tensor
    """
    mask = torch.tensor(adj, dtype=dtype)
    if device is not None:
        mask = mask.to(device)
    return mask


def get_sparsity(mask: torch.Tensor) -> float:
    """
    Calculate the sparsity (fraction of zeros) in a mask.

    Args:
        mask: Binary mask tensor

    Returns:
        Sparsity level (0 to 1, where 1 = all zeros)
    """
    total = mask.numel()
    nonzero = mask.count_nonzero().item()
    return 1.0 - (nonzero / total)


def get_density(mask: torch.Tensor) -> float:
    """
    Calculate the density (fraction of ones) in a mask.

    Args:
        mask: Binary mask tensor

    Returns:
        Density level (0 to 1, where 1 = all ones)
    """
    return 1.0 - get_sparsity(mask)


def mask_to_sparse_tensor(
    mask: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Convert a dense mask to a sparse tensor.

    Args:
        mask: Dense binary mask
        weights: Optional weights to apply (mask * weights)

    Returns:
        Sparse COO tensor
    """
    if weights is not None:
        values = mask * weights
    else:
        values = mask

    return values.to_sparse()


def sparse_tensor_to_mask(
    sparse: torch.Tensor,
    size: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    Convert a sparse tensor to a dense binary mask.

    Args:
        sparse: Sparse tensor
        size: Output size (if different from sparse tensor size)

    Returns:
        Dense binary mask
    """
    dense = sparse.to_dense()
    mask = (dense != 0).float()

    if size is not None and mask.shape != size:
        # Resize if needed
        new_mask = torch.zeros(size)
        min_rows = min(mask.shape[0], size[0])
        min_cols = min(mask.shape[1], size[1])
        new_mask[:min_rows, :min_cols] = mask[:min_rows, :min_cols]
        mask = new_mask

    return mask


def prune_mask_magnitude(
    weights: torch.Tensor,
    mask: torch.Tensor,
    prune_ratio: float
) -> torch.Tensor:
    """
    Prune connections with smallest magnitude weights.

    Used in SET algorithm for removing weak connections.

    Args:
        weights: Weight tensor
        mask: Current binary mask
        prune_ratio: Fraction of connections to prune

    Returns:
        Updated mask with pruned connections set to 0
    """
    # Get masked weights
    masked_weights = weights * mask

    # Get magnitudes of active connections
    active_mask = mask > 0
    active_weights = masked_weights[active_mask].abs()

    if active_weights.numel() == 0:
        return mask

    # Find threshold for pruning
    n_prune = int(prune_ratio * active_weights.numel())
    if n_prune == 0:
        return mask

    threshold = torch.kthvalue(active_weights, n_prune).values

    # Create new mask, keeping connections above threshold
    new_mask = mask.clone()
    prune_locations = (masked_weights.abs() <= threshold) & active_mask
    new_mask[prune_locations] = 0

    return new_mask


def grow_mask_random(
    mask: torch.Tensor,
    grow_ratio: float,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Grow new random connections.

    Used in SET algorithm for adding new connections.

    Args:
        mask: Current binary mask
        grow_ratio: Fraction of current connections to add

    Returns:
        Updated mask with new connections
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Count current connections
    n_current = mask.sum().item()
    n_grow = int(grow_ratio * n_current)

    if n_grow == 0:
        return mask

    # Find locations where we can add connections
    inactive_mask = mask == 0
    inactive_indices = torch.nonzero(inactive_mask)

    if len(inactive_indices) == 0:
        return mask

    # Randomly select locations to activate
    n_grow = min(n_grow, len(inactive_indices))
    perm = torch.randperm(len(inactive_indices))[:n_grow]
    grow_indices = inactive_indices[perm]

    # Create new mask
    new_mask = mask.clone()
    for idx in grow_indices:
        new_mask[idx[0], idx[1]] = 1

    return new_mask
