"""
Graph and network visualization utilities.

Uses matplotlib and networkx for static visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List
import networkx as nx


def plot_graph(
    adj: np.ndarray,
    ax: Optional[plt.Axes] = None,
    layout: str = 'spring',
    node_size: int = 50,
    node_color: str = '#1f77b4',
    edge_color: str = '#cccccc',
    edge_width: float = 0.5,
    title: Optional[str] = None,
    show_labels: bool = False,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Visualize a graph from its adjacency matrix.

    Args:
        adj: Adjacency matrix
        ax: Matplotlib axes (creates new figure if None)
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        node_size: Size of nodes
        node_color: Color of nodes
        edge_color: Color of edges
        edge_width: Width of edges
        title: Plot title
        show_labels: Show node labels
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Create NetworkX graph
    G = nx.from_numpy_array(adj)

    # Compute layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Draw
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_color,
        width=edge_width,
        alpha=0.6,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_size,
        node_color=node_color,
    )

    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    if title:
        ax.set_title(title)

    ax.axis('off')
    ax.set_aspect('equal')

    return fig


def plot_adjacency_matrix(
    adj: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'Blues',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Visualize adjacency matrix as a heatmap.

    Args:
        adj: Adjacency matrix
        ax: Matplotlib axes
        cmap: Colormap
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(adj, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)

    if title:
        ax.set_title(title)

    ax.set_xlabel('Node')
    ax.set_ylabel('Node')

    return fig


def plot_degree_distribution(
    adj: np.ndarray,
    ax: Optional[plt.Axes] = None,
    bins: int = 20,
    title: str = 'Degree Distribution',
    figsize: Tuple[int, int] = (8, 5),
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot the degree distribution of a graph.

    Args:
        adj: Adjacency matrix
        ax: Matplotlib axes
        bins: Number of histogram bins
        title: Plot title
        figsize: Figure size
        log_scale: Use log scale for y-axis

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    degrees = adj.sum(axis=1)

    ax.hist(degrees, bins=bins, edgecolor='white', alpha=0.7)
    ax.axvline(degrees.mean(), color='red', linestyle='--', label=f'Mean: {degrees.mean():.1f}')

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()

    return fig


def plot_ws_rewiring(
    n: int = 20,
    k: int = 4,
    betas: List[float] = [0, 0.01, 0.1, 0.5, 1.0],
    figsize: Tuple[int, int] = (15, 3),
    seed: int = 42,
) -> plt.Figure:
    """
    Visualize Watts-Strogatz rewiring at different beta values.

    Shows how network structure changes from regular to random.

    Args:
        n: Number of nodes
        k: Initial neighbors
        betas: List of beta values to show
        figsize: Figure size
        seed: Random seed

    Returns:
        Matplotlib figure
    """
    from ..topology.graphs import watts_strogatz_graph

    fig, axes = plt.subplots(1, len(betas), figsize=figsize)

    for ax, beta in zip(axes, betas):
        adj = watts_strogatz_graph(n, k, beta, seed=seed)
        G = nx.from_numpy_array(adj)

        pos = nx.circular_layout(G)

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#cccccc', width=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, node_color='#1f77b4')

        ax.set_title(f'beta = {beta}')
        ax.axis('off')

    plt.tight_layout()
    return fig


def plot_graph_comparison(
    graphs: List[np.ndarray],
    titles: List[str],
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Compare multiple graphs side by side.

    Args:
        graphs: List of adjacency matrices
        titles: Titles for each graph
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_graphs = len(graphs)
    fig, axes = plt.subplots(1, n_graphs, figsize=figsize)

    if n_graphs == 1:
        axes = [axes]

    for ax, adj, title in zip(axes, graphs, titles):
        plot_graph(adj, ax=ax, title=title, layout='circular')

    plt.tight_layout()
    return fig


def animate_rewiring(
    n: int = 30,
    k: int = 4,
    n_frames: int = 50,
    save_path: Optional[str] = None,
) -> None:
    """
    Create an animation of WS rewiring (requires matplotlib animation).

    Args:
        n: Number of nodes
        k: Initial neighbors
        n_frames: Number of animation frames
        save_path: Path to save animation (optional)
    """
    from ..topology.graphs import watts_strogatz_graph
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(8, 8))

    betas = np.linspace(0, 1, n_frames)

    def update(frame):
        ax.clear()
        beta = betas[frame]
        adj = watts_strogatz_graph(n, k, beta, seed=42)
        G = nx.from_numpy_array(adj)
        pos = nx.circular_layout(G)

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#cccccc', width=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, node_color='#1f77b4')

        ax.set_title(f'Watts-Strogatz (beta = {beta:.2f})')
        ax.axis('off')
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=100, blit=True
    )

    if save_path:
        anim.save(save_path, writer='pillow')

    plt.close()
    return anim
