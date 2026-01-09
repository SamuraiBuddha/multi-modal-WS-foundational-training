"""
Visualization module for graphs, training, and topology.

Provides tools for:
- Graph and network visualization
- Training progress monitoring
- Topology analysis visualization
- Interactive widgets for notebooks
"""

from .graph_viz import (
    plot_graph,
    plot_adjacency_matrix,
    plot_degree_distribution,
    plot_ws_rewiring,
)
from .training_viz import (
    plot_loss_curves,
    plot_metrics,
    plot_sparsity_evolution,
    create_training_dashboard,
)
from .topology_viz import (
    plot_topology_comparison,
    plot_small_world_metrics,
    visualize_layer_connectivity,
)

__all__ = [
    # Graph visualization
    "plot_graph",
    "plot_adjacency_matrix",
    "plot_degree_distribution",
    "plot_ws_rewiring",
    # Training visualization
    "plot_loss_curves",
    "plot_metrics",
    "plot_sparsity_evolution",
    "create_training_dashboard",
    # Topology visualization
    "plot_topology_comparison",
    "plot_small_world_metrics",
    "visualize_layer_connectivity",
]
