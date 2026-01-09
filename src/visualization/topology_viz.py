"""
Topology-specific visualization utilities.

Visualizations for analyzing and comparing network topologies.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
import networkx as nx


def plot_topology_comparison(
    topologies: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Compare different network topologies side by side.

    Args:
        topologies: Dictionary mapping names to adjacency matrices
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_topo = len(topologies)
    fig, axes = plt.subplots(1, n_topo, figsize=figsize)

    if n_topo == 1:
        axes = [axes]

    for ax, (name, adj) in zip(axes, topologies.items()):
        G = nx.from_numpy_array(adj)
        pos = nx.circular_layout(G)

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#cccccc', width=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, node_color='#1f77b4')

        # Calculate metrics
        from ..topology.metrics import clustering_coefficient, average_path_length

        cc = clustering_coefficient(adj)
        apl = average_path_length(adj, disconnected_value=float('inf'))

        ax.set_title(f'{name}\nC={cc:.3f}, L={apl:.2f}')
        ax.axis('off')

    plt.tight_layout()
    return fig


def plot_small_world_metrics(
    betas: List[float],
    n: int = 100,
    k: int = 4,
    n_trials: int = 5,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot clustering and path length as function of beta.

    Reproduces the classic Watts-Strogatz figure.

    Args:
        betas: List of beta values
        n: Number of nodes
        k: Initial neighbors
        n_trials: Trials to average over
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from ..topology.graphs import watts_strogatz_graph
    from ..topology.metrics import clustering_coefficient, average_path_length

    # Get reference values at beta=0
    adj_regular = watts_strogatz_graph(n, k, 0.0, seed=42)
    C0 = clustering_coefficient(adj_regular)
    L0 = average_path_length(adj_regular)

    # Compute metrics for each beta
    C_values = []
    L_values = []

    for beta in betas:
        C_trials = []
        L_trials = []

        for trial in range(n_trials):
            adj = watts_strogatz_graph(n, k, beta, seed=42 + trial)
            C_trials.append(clustering_coefficient(adj))
            L_trials.append(average_path_length(adj, disconnected_value=float('inf')))

        C_values.append(np.mean(C_trials))
        L_values.append(np.mean(L_trials))

    # Normalize
    C_norm = np.array(C_values) / C0
    L_norm = np.array(L_values) / L0

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.semilogx(betas, C_norm, 'o-', label='C(beta)/C(0)', markersize=6)
    ax.semilogx(betas, L_norm, 's-', label='L(beta)/L(0)', markersize=6)

    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.1, color='green', linestyle=':', alpha=0.5, label='Small-world regime')

    ax.set_xlabel('Rewiring probability (beta)')
    ax.set_ylabel('Normalized metric')
    ax.set_title('Small-World Effect: High clustering with short paths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    return fig


def visualize_layer_connectivity(
    mask: np.ndarray,
    layer_name: str = 'Layer',
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Visualize connectivity pattern of a neural network layer.

    Args:
        mask: Binary mask tensor (as numpy array)
        layer_name: Name for the layer
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Adjacency matrix view
    ax1 = axes[0]
    im = ax1.imshow(mask, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax1, shrink=0.8)
    ax1.set_title(f'{layer_name} Connectivity Matrix')
    ax1.set_xlabel('Input Features')
    ax1.set_ylabel('Output Features')

    # Degree distribution
    ax2 = axes[1]
    out_degree = mask.sum(axis=1)
    in_degree = mask.sum(axis=0)

    ax2.hist(out_degree, bins=20, alpha=0.5, label='Out-degree')
    ax2.hist(in_degree, bins=20, alpha=0.5, label='In-degree')
    ax2.axvline(out_degree.mean(), color='blue', linestyle='--')
    ax2.axvline(in_degree.mean(), color='orange', linestyle='--')

    ax2.set_xlabel('Degree')
    ax2.set_ylabel('Count')
    ax2.set_title('Degree Distribution')
    ax2.legend()

    # Add stats
    sparsity = 1.0 - (mask.sum() / mask.size)
    fig.suptitle(f'{layer_name}: {mask.shape[0]}x{mask.shape[1]}, Sparsity: {sparsity:.1%}')

    plt.tight_layout()
    return fig


def plot_topology_evolution(
    masks: List[np.ndarray],
    steps: List[int],
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Visualize how layer topology changes during training.

    Args:
        masks: List of masks at different training steps
        steps: Corresponding step numbers
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_snapshots = min(5, len(masks))
    indices = np.linspace(0, len(masks) - 1, n_snapshots).astype(int)

    fig, axes = plt.subplots(1, n_snapshots, figsize=figsize)

    for i, idx in enumerate(indices):
        ax = axes[i]
        mask = masks[idx]

        ax.imshow(mask, cmap='Blues', aspect='auto')
        sparsity = 1.0 - (mask.sum() / mask.size)
        ax.set_title(f'Step {steps[idx]}\nSparsity: {sparsity:.1%}')
        ax.axis('off')

    plt.suptitle('Topology Evolution During Training')
    plt.tight_layout()
    return fig


def plot_inter_module_connectivity(
    module_adj: np.ndarray,
    module_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Visualize connectivity between modules in a multi-modal network.

    Args:
        module_adj: Adjacency matrix between modules
        module_names: Names of modules
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_modules = module_adj.shape[0]

    if module_names is None:
        module_names = [f'Module {i}' for i in range(n_modules)]

    fig, ax = plt.subplots(figsize=figsize)

    G = nx.from_numpy_array(module_adj)
    pos = nx.circular_layout(G)

    # Draw with custom styling
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#666666',
        width=2,
        alpha=0.6,
        arrows=True,
        arrowsize=20,
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=2000,
        node_color=['#ff7f0e', '#1f77b4', '#2ca02c'][:n_modules],
        alpha=0.8,
    )

    # Add labels
    labels = {i: name for i, name in enumerate(module_names)}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight='bold')

    ax.set_title('Inter-Module Connectivity')
    ax.axis('off')

    return fig


def create_topology_report(
    adj: np.ndarray,
    name: str = 'Network',
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Create a comprehensive topology analysis report.

    Args:
        adj: Adjacency matrix
        name: Network name
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from ..topology.metrics import (
        clustering_coefficient,
        average_path_length,
        degree_distribution,
        small_world_coefficient,
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Graph visualization
    ax1 = axes[0, 0]
    G = nx.from_numpy_array(adj)
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='#cccccc', width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=30, node_color='#1f77b4')
    ax1.set_title(f'{name} - Graph Visualization')
    ax1.axis('off')

    # Adjacency matrix
    ax2 = axes[0, 1]
    im = ax2.imshow(adj, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax2, shrink=0.8)
    ax2.set_title('Adjacency Matrix')

    # Degree distribution
    ax3 = axes[1, 0]
    degrees, counts = degree_distribution(adj)
    ax3.bar(degrees, counts, alpha=0.7)
    ax3.set_xlabel('Degree')
    ax3.set_ylabel('Count')
    ax3.set_title('Degree Distribution')

    # Metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    cc = clustering_coefficient(adj)
    apl = average_path_length(adj, disconnected_value=float('inf'))
    try:
        sigma, details = small_world_coefficient(adj, n_random=3)
    except Exception:
        sigma = 'N/A'
        details = {}

    metrics_text = f"""
    Network Metrics for {name}
    {'=' * 30}

    Nodes: {adj.shape[0]}
    Edges: {int(adj.sum() / 2)}
    Density: {adj.sum() / (adj.shape[0] * (adj.shape[0] - 1)):.4f}

    Clustering Coefficient: {cc:.4f}
    Avg Path Length: {apl:.4f}
    Small-World Coefficient: {sigma if isinstance(sigma, str) else f'{sigma:.4f}'}

    Degree Statistics:
    - Mean: {degrees.mean():.2f}
    - Min: {degrees.min()}
    - Max: {degrees.max()}
    """

    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.suptitle(f'Topology Report: {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig
