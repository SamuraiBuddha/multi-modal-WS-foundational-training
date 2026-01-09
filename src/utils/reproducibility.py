"""
Reproducibility utilities.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_reproducible(seed: int = 42, deterministic: bool = True):
    """
    Configure PyTorch for reproducible results.

    Note: This may impact performance.

    Args:
        seed: Random seed
        deterministic: Use deterministic algorithms
    """
    set_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass  # Some operations don't have deterministic implementations


def get_rng_state() -> dict:
    """
    Get current random number generator states.

    Returns:
        Dictionary of RNG states
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state: dict):
    """
    Restore random number generator states.

    Args:
        state: Dictionary of RNG states from get_rng_state()
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])
