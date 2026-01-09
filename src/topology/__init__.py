"""
Topology module for graph generation, metrics, and sparse connectivity.

This module provides:
- Graph generation (Watts-Strogatz, Erdos-Renyi, Barabasi-Albert)
- Network metrics (clustering coefficient, path length, small-world coefficient)
- Sparse mask generation and manipulation
- Dynamic rewiring algorithms (SET, DEEP R)
"""

from .graphs import (
    watts_strogatz_graph,
    erdos_renyi_graph,
    barabasi_albert_graph,
    ring_lattice,
    complete_graph,
)
from .metrics import (
    clustering_coefficient,
    average_path_length,
    small_world_coefficient,
    degree_distribution,
    characteristic_path_length,
)
from .sparse_masks import (
    ws_sparse_mask,
    random_sparse_mask,
    mask_to_adjacency,
    adjacency_to_mask,
    get_sparsity,
)
from .rewiring import (
    rewire_edges,
    set_rewire,
    deep_r_rewire,
    adaptive_rewire,
)

__all__ = [
    # Graph generation
    "watts_strogatz_graph",
    "erdos_renyi_graph",
    "barabasi_albert_graph",
    "ring_lattice",
    "complete_graph",
    # Metrics
    "clustering_coefficient",
    "average_path_length",
    "small_world_coefficient",
    "degree_distribution",
    "characteristic_path_length",
    # Sparse masks
    "ws_sparse_mask",
    "random_sparse_mask",
    "mask_to_adjacency",
    "adjacency_to_mask",
    "get_sparsity",
    # Rewiring
    "rewire_edges",
    "set_rewire",
    "deep_r_rewire",
    "adaptive_rewire",
]
