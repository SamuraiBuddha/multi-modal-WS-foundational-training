"""
Pytest configuration and fixtures for Multi-Modal WS Training tests.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def device():
    """Get available device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def seed():
    """Set random seeds for reproducibility."""
    seed_val = 42
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    return seed_val


@pytest.fixture
def small_ws_graph():
    """Create a small WS graph for testing."""
    from src.topology import watts_strogatz_graph
    return watts_strogatz_graph(n=20, k=4, beta=0.3, seed=42)


@pytest.fixture
def dummy_batch():
    """Create dummy multi-modal batch."""
    return {
        'visual': torch.randn(4, 1, 28, 28),
        'text': torch.randint(0, 100, (4, 32)),
        'audio': torch.randn(4, 128, 64),
        'label': torch.randint(0, 10, (4,)),
    }


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    import torch.nn as nn

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    return SimpleNet()


@pytest.fixture
def sparse_mask():
    """Create sparse mask for testing."""
    from src.topology import random_sparse_mask
    return random_sparse_mask(100, 50, density=0.3, seed=42)
