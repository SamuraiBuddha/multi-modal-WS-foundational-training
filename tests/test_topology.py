"""
Tests for topology module.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')

from src.topology import (
    watts_strogatz_graph,
    erdos_renyi_graph,
    barabasi_albert_graph,
    ring_lattice,
    clustering_coefficient,
    average_path_length,
    small_world_coefficient,
    degree_distribution,
    ws_sparse_mask,
    random_sparse_mask,
    get_sparsity,
    set_rewire,
)


class TestGraphGeneration:
    """Tests for graph generation functions."""

    def test_ring_lattice_shape(self):
        """Ring lattice should have correct shape."""
        adj = ring_lattice(10, 4)
        assert adj.shape == (10, 10)

    def test_ring_lattice_symmetric(self):
        """Ring lattice should be symmetric."""
        adj = ring_lattice(10, 4)
        assert np.allclose(adj, adj.T)

    def test_ring_lattice_degree(self):
        """Each node in ring lattice should have degree k."""
        k = 4
        adj = ring_lattice(10, k)
        degrees = adj.sum(axis=1)
        assert np.all(degrees == k)

    def test_watts_strogatz_shape(self):
        """WS graph should have correct shape."""
        adj = watts_strogatz_graph(20, 4, 0.3)
        assert adj.shape == (20, 20)

    def test_watts_strogatz_symmetric(self):
        """WS graph should be symmetric (undirected)."""
        adj = watts_strogatz_graph(20, 4, 0.3)
        assert np.allclose(adj, adj.T)

    def test_watts_strogatz_beta_zero(self):
        """WS with beta=0 should equal ring lattice."""
        adj_ws = watts_strogatz_graph(10, 4, 0.0, seed=42)
        adj_ring = ring_lattice(10, 4)
        assert np.allclose(adj_ws, adj_ring)

    def test_erdos_renyi_density(self):
        """ER graph density should approximate p."""
        p = 0.2
        adj = erdos_renyi_graph(100, p, seed=42)
        # Count edges (divide by 2 for undirected)
        actual_density = adj.sum() / (100 * 99)
        assert abs(actual_density - p) < 0.1  # Within 10%

    def test_barabasi_albert_edges(self):
        """BA graph should have approximately n*m edges."""
        n, m = 100, 3
        adj = barabasi_albert_graph(n, m, seed=42)
        expected_edges = (m + 1) * m / 2 + (n - m - 1) * m
        actual_edges = adj.sum() / 2
        assert abs(actual_edges - expected_edges) < 10


class TestMetrics:
    """Tests for network metrics."""

    def test_clustering_triangle(self):
        """Triangle should have clustering = 1."""
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.float32)
        cc = clustering_coefficient(adj)
        assert abs(cc - 1.0) < 0.01

    def test_clustering_line(self):
        """Line graph should have clustering = 0."""
        adj = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=np.float32)
        cc = clustering_coefficient(adj)
        assert cc == 0.0

    def test_average_path_length_triangle(self):
        """Triangle should have avg path length = 1."""
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.float32)
        apl = average_path_length(adj)
        assert abs(apl - 1.0) < 0.01

    def test_degree_distribution(self):
        """Degree distribution should sum to n."""
        adj = watts_strogatz_graph(50, 4, 0.3, seed=42)
        degrees, counts = degree_distribution(adj)
        assert counts.sum() == 50

    def test_small_world_coefficient_positive(self):
        """Small-world coefficient should be positive for WS networks."""
        adj = watts_strogatz_graph(50, 4, 0.1, seed=42)
        sigma, _ = small_world_coefficient(adj, n_random=3)
        assert sigma > 0


class TestSparseMasks:
    """Tests for sparse mask generation."""

    def test_ws_sparse_mask_shape(self):
        """WS sparse mask should have correct shape."""
        mask = ws_sparse_mask(100, 50, k=4, beta=0.3)
        assert mask.shape == (50, 100)

    def test_random_sparse_mask_density(self):
        """Random sparse mask should approximate target density."""
        density = 0.1
        mask = random_sparse_mask(1000, 500, density=density, seed=42)
        actual_density = 1.0 - get_sparsity(mask)
        assert abs(actual_density - density) < 0.02

    def test_get_sparsity_range(self):
        """Sparsity should be between 0 and 1."""
        mask = random_sparse_mask(100, 50, density=0.3)
        sparsity = get_sparsity(mask)
        assert 0 <= sparsity <= 1


class TestRewiring:
    """Tests for rewiring algorithms."""

    def test_set_rewire_preserves_density(self):
        """SET rewiring should approximately preserve density."""
        import torch

        weights = torch.randn(50, 100)
        mask = random_sparse_mask(100, 50, density=0.2)

        initial_density = 1.0 - get_sparsity(mask)
        new_mask = set_rewire(weights, mask, prune_rate=0.3)
        final_density = 1.0 - get_sparsity(new_mask)

        # Should be within 5% of original
        assert abs(final_density - initial_density) < 0.05


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
