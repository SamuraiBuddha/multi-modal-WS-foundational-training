"""
Graph generation functions for various network topologies.

Implements:
- Watts-Strogatz small-world networks
- Erdos-Renyi random graphs
- Barabasi-Albert scale-free networks
- Ring lattice (base for Watts-Strogatz)
"""

import numpy as np
import networkx as nx
from typing import Optional, Union
import torch


def ring_lattice(n: int, k: int) -> np.ndarray:
    """
    Create a ring lattice graph where each node connects to k nearest neighbors.

    This is the starting point for Watts-Strogatz small-world networks.
    Each node i is connected to nodes (i +/- 1), (i +/- 2), ..., (i +/- k//2).

    Args:
        n: Number of nodes
        k: Each node connects to k nearest neighbors (must be even)

    Returns:
        Adjacency matrix as numpy array of shape (n, n)

    Example:
        >>> adj = ring_lattice(10, 4)
        >>> adj.shape
        (10, 10)
        >>> adj.sum(axis=1)  # Each node has exactly k connections
        array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4.])
    """
    if k % 2 != 0:
        raise ValueError(f"k must be even, got {k}")
    if k >= n:
        raise ValueError(f"k must be less than n, got k={k}, n={n}")

    adj = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(1, k // 2 + 1):
            # Connect to j-th neighbor on each side (with wraparound)
            right_neighbor = (i + j) % n
            left_neighbor = (i - j) % n
            adj[i, right_neighbor] = 1
            adj[i, left_neighbor] = 1

    return adj


def watts_strogatz_graph(
    n: int,
    k: int,
    beta: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a Watts-Strogatz small-world network.

    The algorithm:
    1. Start with a ring lattice of n nodes, each connected to k neighbors
    2. For each edge, with probability beta, rewire it to a random node

    Small-world properties emerge for intermediate beta values:
    - beta=0: Regular lattice (high clustering, high path length)
    - beta=1: Random graph (low clustering, low path length)
    - beta~0.1-0.3: Small-world (high clustering, low path length)

    Args:
        n: Number of nodes
        k: Each node starts with k neighbors (must be even)
        beta: Rewiring probability (0 to 1)
        seed: Random seed for reproducibility

    Returns:
        Adjacency matrix as numpy array of shape (n, n)

    Example:
        >>> adj = watts_strogatz_graph(100, 4, 0.3, seed=42)
        >>> adj.shape
        (100, 100)
    """
    if seed is not None:
        np.random.seed(seed)

    # Start with ring lattice
    adj = ring_lattice(n, k)

    # Rewire edges with probability beta
    for i in range(n):
        for j in range(1, k // 2 + 1):
            # Consider edge from i to (i+j) mod n
            target = (i + j) % n

            if np.random.random() < beta:
                # Rewire this edge
                adj[i, target] = 0
                adj[target, i] = 0

                # Choose new target (not self, not existing neighbor)
                candidates = [
                    node for node in range(n)
                    if node != i and adj[i, node] == 0
                ]

                if candidates:
                    new_target = np.random.choice(candidates)
                    adj[i, new_target] = 1
                    adj[new_target, i] = 1
                else:
                    # No valid candidates, keep original
                    adj[i, target] = 1
                    adj[target, i] = 1

    return adj


def erdos_renyi_graph(
    n: int,
    p: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate an Erdos-Renyi random graph.

    Each possible edge exists independently with probability p.

    Args:
        n: Number of nodes
        p: Probability of each edge existing (0 to 1)
        seed: Random seed for reproducibility

    Returns:
        Adjacency matrix as numpy array of shape (n, n)

    Example:
        >>> adj = erdos_renyi_graph(100, 0.1, seed=42)
        >>> # Expected number of edges: n*(n-1)/2 * p
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random symmetric matrix
    rand_matrix = np.random.random((n, n))
    adj = (rand_matrix < p).astype(np.float32)

    # Make symmetric (undirected graph)
    adj = np.triu(adj, k=1)
    adj = adj + adj.T

    # No self-loops
    np.fill_diagonal(adj, 0)

    return adj


def barabasi_albert_graph(
    n: int,
    m: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a Barabasi-Albert scale-free network.

    The preferential attachment algorithm:
    1. Start with m+1 fully connected nodes
    2. Add nodes one at a time, each connecting to m existing nodes
    3. Probability of connecting to node i is proportional to its degree

    This creates networks with power-law degree distribution ("rich get richer").

    Args:
        n: Total number of nodes
        m: Number of edges each new node creates (must be <= initial nodes)
        seed: Random seed for reproducibility

    Returns:
        Adjacency matrix as numpy array of shape (n, n)

    Example:
        >>> adj = barabasi_albert_graph(100, 2, seed=42)
        >>> # Degree distribution follows power law
    """
    if seed is not None:
        np.random.seed(seed)

    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if m >= n:
        raise ValueError(f"m must be < n, got m={m}, n={n}")

    adj = np.zeros((n, n), dtype=np.float32)

    # Start with m+1 fully connected nodes
    initial_nodes = m + 1
    for i in range(initial_nodes):
        for j in range(i + 1, initial_nodes):
            adj[i, j] = 1
            adj[j, i] = 1

    # Track degrees for preferential attachment
    degrees = adj.sum(axis=1)

    # Add remaining nodes
    for new_node in range(initial_nodes, n):
        # Select m existing nodes with probability proportional to degree
        existing_nodes = np.arange(new_node)
        probs = degrees[:new_node] / degrees[:new_node].sum()

        # Sample without replacement
        targets = np.random.choice(
            existing_nodes, size=m, replace=False, p=probs
        )

        # Create edges
        for target in targets:
            adj[new_node, target] = 1
            adj[target, new_node] = 1
            degrees[new_node] += 1
            degrees[target] += 1

    return adj


def complete_graph(n: int) -> np.ndarray:
    """
    Generate a complete graph where every node connects to every other node.

    Args:
        n: Number of nodes

    Returns:
        Adjacency matrix as numpy array of shape (n, n)
    """
    adj = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(adj, 0)
    return adj


def adjacency_to_networkx(adj: np.ndarray) -> nx.Graph:
    """
    Convert adjacency matrix to NetworkX graph.

    Args:
        adj: Adjacency matrix as numpy array

    Returns:
        NetworkX Graph object
    """
    return nx.from_numpy_array(adj)


def networkx_to_adjacency(G: nx.Graph) -> np.ndarray:
    """
    Convert NetworkX graph to adjacency matrix.

    Args:
        G: NetworkX Graph object

    Returns:
        Adjacency matrix as numpy array
    """
    return nx.to_numpy_array(G, dtype=np.float32)


def adjacency_to_torch(
    adj: np.ndarray,
    sparse: bool = False,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Convert adjacency matrix to PyTorch tensor.

    Args:
        adj: Adjacency matrix as numpy array
        sparse: If True, return sparse tensor
        device: Device to place tensor on

    Returns:
        PyTorch tensor (dense or sparse)
    """
    if sparse:
        # Convert to COO format
        indices = np.nonzero(adj)
        values = adj[indices]
        indices = np.stack(indices)

        tensor = torch.sparse_coo_tensor(
            torch.from_numpy(indices),
            torch.from_numpy(values),
            adj.shape
        )
    else:
        tensor = torch.from_numpy(adj)

    if device is not None:
        tensor = tensor.to(device)

    return tensor
