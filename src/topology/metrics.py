"""
Network topology metrics for analyzing graph properties.

Implements key metrics for understanding small-world networks:
- Clustering coefficient (local and global)
- Average/characteristic path length
- Small-world coefficient (sigma)
- Degree distribution
"""

import numpy as np
import networkx as nx
from typing import Union, Optional, Tuple
import warnings


def clustering_coefficient(
    adj: np.ndarray,
    local: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate clustering coefficient of a graph.

    The clustering coefficient measures the degree to which nodes cluster together.
    For a node, it's the fraction of possible triangles that exist.

    Global clustering = (3 * number of triangles) / number of connected triplets

    Args:
        adj: Adjacency matrix as numpy array
        local: If True, return per-node clustering coefficients

    Returns:
        Global clustering coefficient (float) or array of local coefficients

    Example:
        >>> adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # Triangle
        >>> clustering_coefficient(adj)
        1.0
    """
    G = nx.from_numpy_array(adj)

    if local:
        local_cc = nx.clustering(G)
        return np.array([local_cc[i] for i in range(len(adj))])
    else:
        return nx.average_clustering(G)


def average_path_length(
    adj: np.ndarray,
    disconnected_value: Optional[float] = None
) -> float:
    """
    Calculate the average shortest path length of a graph.

    For small-world networks, this should be O(log n) even with high clustering.

    Args:
        adj: Adjacency matrix as numpy array
        disconnected_value: Value to return if graph is disconnected.
                           If None, raises an error.

    Returns:
        Average shortest path length

    Example:
        >>> adj = ring_lattice(10, 4)
        >>> apl = average_path_length(adj)
        >>> apl < 3  # Ring lattice has relatively short paths
        True
    """
    G = nx.from_numpy_array(adj)

    if not nx.is_connected(G):
        if disconnected_value is not None:
            return disconnected_value
        else:
            # Calculate for largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc)
            warnings.warn(
                "Graph is disconnected. Computing path length for "
                f"largest component with {len(largest_cc)} nodes."
            )

    return nx.average_shortest_path_length(G)


def characteristic_path_length(adj: np.ndarray) -> float:
    """
    Alias for average_path_length (terminology from Watts-Strogatz paper).

    Args:
        adj: Adjacency matrix

    Returns:
        Characteristic path length L
    """
    return average_path_length(adj)


def small_world_coefficient(
    adj: np.ndarray,
    n_random: int = 10,
    seed: Optional[int] = None
) -> Tuple[float, dict]:
    """
    Calculate the small-world coefficient (sigma) of a graph.

    sigma = (C / C_random) / (L / L_random)

    Where:
    - C is the clustering coefficient of the graph
    - L is the average path length of the graph
    - C_random, L_random are values for equivalent random graphs

    sigma > 1 indicates small-world properties.
    Typical small-world networks have sigma >> 1.

    Args:
        adj: Adjacency matrix as numpy array
        n_random: Number of random graphs to average over
        seed: Random seed for reproducibility

    Returns:
        Tuple of (sigma coefficient, dict with component values)

    Example:
        >>> adj = watts_strogatz_graph(100, 4, 0.1, seed=42)
        >>> sigma, details = small_world_coefficient(adj)
        >>> sigma > 1  # Should exhibit small-world properties
        True
    """
    if seed is not None:
        np.random.seed(seed)

    G = nx.from_numpy_array(adj)
    n = len(adj)
    m = int(adj.sum() / 2)  # Number of edges

    # Calculate properties of original graph
    C = clustering_coefficient(adj)
    L = average_path_length(adj, disconnected_value=float('inf'))

    # Generate random graphs with same n and m
    C_random_list = []
    L_random_list = []

    for _ in range(n_random):
        G_random = nx.gnm_random_graph(n, m)
        adj_random = nx.to_numpy_array(G_random)

        C_random_list.append(clustering_coefficient(adj_random))
        L_random_list.append(
            average_path_length(adj_random, disconnected_value=float('inf'))
        )

    C_random = np.mean(C_random_list)
    L_random = np.mean(L_random_list)

    # Calculate sigma
    if C_random == 0 or L_random == 0:
        sigma = float('inf')
    else:
        gamma = C / C_random  # Clustering ratio
        lambda_val = L / L_random  # Path length ratio
        sigma = gamma / lambda_val if lambda_val > 0 else float('inf')

    details = {
        'C': C,
        'L': L,
        'C_random': C_random,
        'L_random': L_random,
        'gamma': C / C_random if C_random > 0 else float('inf'),
        'lambda': L / L_random if L_random > 0 else float('inf'),
    }

    return sigma, details


def degree_distribution(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the degree distribution of a graph.

    Args:
        adj: Adjacency matrix as numpy array

    Returns:
        Tuple of (degrees, counts) where:
        - degrees: unique degree values
        - counts: number of nodes with each degree

    Example:
        >>> adj = ring_lattice(10, 4)
        >>> degrees, counts = degree_distribution(adj)
        >>> degrees
        array([4])
        >>> counts
        array([10])  # All nodes have degree 4
    """
    node_degrees = adj.sum(axis=1).astype(int)
    degrees, counts = np.unique(node_degrees, return_counts=True)
    return degrees, counts


def degree_statistics(adj: np.ndarray) -> dict:
    """
    Calculate various degree statistics.

    Args:
        adj: Adjacency matrix

    Returns:
        Dictionary with mean, std, min, max degree
    """
    degrees = adj.sum(axis=1)
    return {
        'mean': float(np.mean(degrees)),
        'std': float(np.std(degrees)),
        'min': int(np.min(degrees)),
        'max': int(np.max(degrees)),
        'median': float(np.median(degrees)),
    }


def graph_density(adj: np.ndarray) -> float:
    """
    Calculate the density of a graph.

    Density = 2 * E / (N * (N-1)) for undirected graphs

    Args:
        adj: Adjacency matrix

    Returns:
        Graph density (0 to 1)
    """
    n = len(adj)
    if n <= 1:
        return 0.0
    edges = adj.sum() / 2  # Undirected, so divide by 2
    max_edges = n * (n - 1) / 2
    return edges / max_edges


def connected_components(adj: np.ndarray) -> Tuple[int, list]:
    """
    Find connected components in a graph.

    Args:
        adj: Adjacency matrix

    Returns:
        Tuple of (number of components, list of component node sets)
    """
    G = nx.from_numpy_array(adj)
    components = list(nx.connected_components(G))
    return len(components), [list(c) for c in components]


def diameter(adj: np.ndarray) -> int:
    """
    Calculate the diameter (longest shortest path) of a graph.

    Args:
        adj: Adjacency matrix

    Returns:
        Graph diameter, or -1 if disconnected
    """
    G = nx.from_numpy_array(adj)
    if not nx.is_connected(G):
        return -1
    return nx.diameter(G)


def efficiency(adj: np.ndarray) -> float:
    """
    Calculate the global efficiency of a graph.

    Efficiency is the average of 1/d(i,j) for all pairs of nodes.
    More robust to disconnected graphs than average path length.

    Args:
        adj: Adjacency matrix

    Returns:
        Global efficiency (0 to 1)
    """
    G = nx.from_numpy_array(adj)
    return nx.global_efficiency(G)
