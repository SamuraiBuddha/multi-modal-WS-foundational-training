"""
Dynamic rewiring algorithms for sparse neural networks.

Implements:
- SET (Sparse Evolutionary Training) - prune by magnitude, grow random
- DEEP R (Deep Rewiring) - gradient-based connection importance
- Adaptive rewiring with learnable parameters
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from .sparse_masks import prune_mask_magnitude, grow_mask_random, get_sparsity


def rewire_edges(
    mask: torch.Tensor,
    rewire_fraction: float,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Randomly rewire a fraction of edges in a mask.

    Simulates Watts-Strogatz rewiring on the weight mask.

    Args:
        mask: Binary mask tensor
        rewire_fraction: Fraction of edges to rewire
        seed: Random seed

    Returns:
        Rewired mask tensor
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    new_mask = mask.clone()

    # Find active connections
    active_indices = torch.nonzero(mask)
    n_active = len(active_indices)
    n_rewire = int(rewire_fraction * n_active)

    if n_rewire == 0:
        return new_mask

    # Randomly select edges to rewire
    rewire_perm = torch.randperm(n_active)[:n_rewire]
    rewire_indices = active_indices[rewire_perm]

    # Find inactive locations for new edges
    inactive_mask = mask == 0
    inactive_indices = torch.nonzero(inactive_mask)

    if len(inactive_indices) < n_rewire:
        n_rewire = len(inactive_indices)
        rewire_indices = rewire_indices[:n_rewire]

    # Select random new locations
    new_perm = torch.randperm(len(inactive_indices))[:n_rewire]
    new_indices = inactive_indices[new_perm]

    # Perform rewiring
    for old_idx, new_idx in zip(rewire_indices, new_indices):
        new_mask[old_idx[0], old_idx[1]] = 0
        new_mask[new_idx[0], new_idx[1]] = 1

    return new_mask


def set_rewire(
    weights: torch.Tensor,
    mask: torch.Tensor,
    prune_rate: float = 0.3,
    regrow_rate: Optional[float] = None
) -> torch.Tensor:
    """
    SET (Sparse Evolutionary Training) rewiring algorithm.

    Algorithm:
    1. Prune connections with smallest magnitude weights
    2. Regrow random new connections

    Reference: Mocanu et al., "Scalable training of artificial neural
    networks with adaptive sparse connectivity inspired by network science"

    Args:
        weights: Weight tensor
        mask: Current binary mask
        prune_rate: Fraction of connections to prune
        regrow_rate: Fraction to regrow (defaults to prune_rate)

    Returns:
        Updated mask after SET rewiring

    Example:
        >>> weights = torch.randn(256, 784)
        >>> mask = random_sparse_mask(784, 256, density=0.1)
        >>> new_mask = set_rewire(weights, mask, prune_rate=0.2)
        >>> # Sparsity is preserved
        >>> abs(get_sparsity(new_mask) - get_sparsity(mask)) < 0.01
        True
    """
    if regrow_rate is None:
        regrow_rate = prune_rate

    # Step 1: Prune smallest magnitude connections
    pruned_mask = prune_mask_magnitude(weights, mask, prune_rate)

    # Step 2: Grow new random connections
    # Calculate how many to grow to maintain sparsity
    n_pruned = (mask.sum() - pruned_mask.sum()).item()
    actual_grow_rate = n_pruned / max(1, pruned_mask.sum().item())

    new_mask = grow_mask_random(pruned_mask, actual_grow_rate)

    return new_mask


def deep_r_rewire(
    weights: torch.Tensor,
    mask: torch.Tensor,
    gradients: torch.Tensor,
    temperature: float = 1.0,
    prune_rate: float = 0.3
) -> torch.Tensor:
    """
    DEEP R (Deep Rewiring) algorithm.

    Uses gradient information to determine connection importance.
    Connections with small |weight * gradient| are more likely to be pruned.

    Reference: Bellec et al., "Deep Rewiring: Training very sparse deep networks"

    Args:
        weights: Weight tensor
        mask: Current binary mask
        gradients: Gradient tensor (same shape as weights)
        temperature: Temperature for soft thresholding
        prune_rate: Fraction of connections to consider for pruning

    Returns:
        Updated mask after DEEP R rewiring
    """
    # Calculate connection importance as |w * grad|
    # This approximates the change in loss if connection is removed
    importance = (weights * gradients).abs() * mask

    # Get active connections
    active_mask = mask > 0
    active_importance = importance[active_mask]

    if active_importance.numel() == 0:
        return mask

    # Compute pruning probabilities (inverse importance)
    # Normalize importance scores
    max_imp = active_importance.max()
    if max_imp > 0:
        norm_importance = active_importance / max_imp
    else:
        norm_importance = active_importance

    # Pruning probability: low importance -> high prune prob
    prune_probs = torch.exp(-norm_importance / temperature)
    prune_probs = prune_probs / prune_probs.sum()

    # Sample connections to prune
    n_prune = int(prune_rate * len(active_importance))
    if n_prune == 0:
        return mask

    # Use importance-weighted sampling
    prune_indices = torch.multinomial(prune_probs, n_prune, replacement=False)

    # Get actual indices in mask
    active_indices = torch.nonzero(active_mask)
    prune_locations = active_indices[prune_indices]

    # Create new mask
    new_mask = mask.clone()
    for loc in prune_locations:
        new_mask[loc[0], loc[1]] = 0

    # Grow new connections randomly
    new_mask = grow_mask_random(new_mask, n_prune / max(1, new_mask.sum().item()))

    return new_mask


def adaptive_rewire(
    weights: torch.Tensor,
    mask: torch.Tensor,
    gradients: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    prune_rate: float = 0.3,
    method: str = "hybrid"
) -> torch.Tensor:
    """
    Adaptive rewiring combining SET and DEEP R approaches.

    Args:
        weights: Weight tensor
        mask: Current binary mask
        gradients: Optional gradient tensor
        beta: Balance between magnitude (0) and gradient (1) importance
        prune_rate: Fraction of connections to prune
        method: "set", "deep_r", or "hybrid"

    Returns:
        Updated mask
    """
    if method == "set" or gradients is None:
        return set_rewire(weights, mask, prune_rate)
    elif method == "deep_r":
        return deep_r_rewire(weights, mask, gradients, prune_rate=prune_rate)
    else:
        # Hybrid: combine magnitude and gradient importance
        magnitude_importance = weights.abs() * mask

        if gradients is not None:
            gradient_importance = (weights * gradients).abs() * mask
            # Normalize both
            mag_norm = magnitude_importance / (magnitude_importance.max() + 1e-8)
            grad_norm = gradient_importance / (gradient_importance.max() + 1e-8)
            # Combine
            importance = (1 - beta) * mag_norm + beta * grad_norm
        else:
            importance = magnitude_importance

        # Prune low importance connections
        active_mask = mask > 0
        active_importance = importance[active_mask]

        if active_importance.numel() == 0:
            return mask

        n_prune = int(prune_rate * len(active_importance))
        if n_prune == 0:
            return mask

        threshold = torch.kthvalue(active_importance, n_prune).values

        new_mask = mask.clone()
        prune_locations = (importance <= threshold) & active_mask
        new_mask[prune_locations] = 0

        # Grow to maintain density
        n_pruned = prune_locations.sum().item()
        new_mask = grow_mask_random(new_mask, n_pruned / max(1, new_mask.sum().item()))

        return new_mask


class DynamicSparseLayer(nn.Module):
    """
    A linear layer with dynamic sparse connectivity.

    Supports SET and DEEP R rewiring during training.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        density: float = 0.1,
        rewire_method: str = "set",
        rewire_frequency: int = 100,
        prune_rate: float = 0.3,
        bias: bool = True
    ):
        """
        Initialize dynamic sparse layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            density: Initial connection density
            rewire_method: "set", "deep_r", or "adaptive"
            rewire_frequency: Steps between rewiring
            prune_rate: Fraction to prune during rewiring
            bias: Include bias term
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.density = density
        self.rewire_method = rewire_method
        self.rewire_frequency = rewire_frequency
        self.prune_rate = prune_rate

        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize sparse mask
        mask = torch.rand(out_features, in_features) < density
        self.register_buffer('mask', mask.float())

        # Step counter for rewiring
        self.register_buffer('step_count', torch.tensor(0))

        # Store gradients for DEEP R
        self.weight_grad = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse weights."""
        # Apply mask to weights
        sparse_weight = self.weight * self.mask

        output = torch.nn.functional.linear(x, sparse_weight, self.bias)

        return output

    def rewire(self, gradients: Optional[torch.Tensor] = None):
        """Perform rewiring based on configured method."""
        with torch.no_grad():
            if self.rewire_method == "set":
                self.mask.data = set_rewire(
                    self.weight.data, self.mask,
                    prune_rate=self.prune_rate
                )
            elif self.rewire_method == "deep_r" and gradients is not None:
                self.mask.data = deep_r_rewire(
                    self.weight.data, self.mask, gradients,
                    prune_rate=self.prune_rate
                )
            else:
                self.mask.data = adaptive_rewire(
                    self.weight.data, self.mask, gradients,
                    prune_rate=self.prune_rate
                )

    def maybe_rewire(self, gradients: Optional[torch.Tensor] = None):
        """Rewire if step count matches frequency."""
        self.step_count += 1
        if self.step_count % self.rewire_frequency == 0:
            self.rewire(gradients)

    def get_sparsity(self) -> float:
        """Return current sparsity level."""
        return get_sparsity(self.mask)

    def get_effective_params(self) -> int:
        """Return number of active parameters."""
        return int(self.mask.sum().item())


class LearnableBetaRewiring(nn.Module):
    """
    Rewiring with learnable beta parameter (Watts-Strogatz style).

    Beta controls the balance between local and random connectivity.
    """

    def __init__(
        self,
        initial_beta: float = 0.3,
        min_beta: float = 0.0,
        max_beta: float = 1.0
    ):
        super().__init__()

        # Use unconstrained parameter with sigmoid for [min, max] range
        self.beta_logit = nn.Parameter(torch.tensor(0.0))
        self.min_beta = min_beta
        self.max_beta = max_beta

        # Initialize to desired beta
        with torch.no_grad():
            target = (initial_beta - min_beta) / (max_beta - min_beta)
            self.beta_logit.data = torch.logit(torch.tensor(target))

    @property
    def beta(self) -> torch.Tensor:
        """Get current beta value."""
        return self.min_beta + (self.max_beta - self.min_beta) * torch.sigmoid(self.beta_logit)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply beta-controlled rewiring to mask."""
        # This is a soft approximation of rewiring
        # In practice, you'd call rewire() during training steps
        return mask

    def rewire(self, mask: torch.Tensor) -> torch.Tensor:
        """Rewire mask using current beta value."""
        return rewire_edges(mask, self.beta.item())
