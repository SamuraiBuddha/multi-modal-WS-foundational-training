"""
Neural network layers with sparse connectivity support.

Implements sparse linear layers using various topologies
including Watts-Strogatz small-world patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import numpy as np

from ..topology.sparse_masks import (
    ws_sparse_mask,
    random_sparse_mask,
    get_sparsity,
)
from ..topology.rewiring import set_rewire, deep_r_rewire


class SparseLinear(nn.Module):
    """
    Linear layer with sparse connectivity.

    Maintains a fixed sparsity pattern that can be optionally rewired
    during training using SET or DEEP R algorithms.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        density: float = 0.1,
        bias: bool = True,
        mask_type: str = "random",
        ws_k: int = 4,
        ws_beta: float = 0.3,
    ):
        """
        Initialize sparse linear layer.

        Args:
            in_features: Size of input
            out_features: Size of output
            density: Fraction of connections to keep
            bias: Include bias term
            mask_type: "random" or "ws" (Watts-Strogatz)
            ws_k: Neighbors for WS topology
            ws_beta: Rewiring probability for WS
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.density = density

        # Initialize weight parameter
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            fan_in = in_features
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        # Create sparse mask
        if mask_type == "ws":
            mask = ws_sparse_mask(
                in_features, out_features, k=ws_k, beta=ws_beta
            )
        else:
            mask = random_sparse_mask(in_features, out_features, density)

        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with masked weights."""
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)

    def get_sparsity(self) -> float:
        """Return current sparsity level."""
        return get_sparsity(self.mask)

    def get_num_params(self) -> int:
        """Return number of active parameters."""
        n_weights = int(self.mask.sum().item())
        n_bias = self.out_features if self.bias is not None else 0
        return n_weights + n_bias

    def rewire(
        self,
        method: str = "set",
        prune_rate: float = 0.3,
        gradients: Optional[torch.Tensor] = None
    ):
        """
        Rewire connections using specified method.

        Args:
            method: "set" or "deep_r"
            prune_rate: Fraction of connections to rewire
            gradients: Gradient tensor for DEEP R
        """
        with torch.no_grad():
            if method == "set":
                self.mask.data = set_rewire(
                    self.weight.data, self.mask, prune_rate=prune_rate
                )
            elif method == "deep_r" and gradients is not None:
                self.mask.data = deep_r_rewire(
                    self.weight.data, self.mask, gradients, prune_rate=prune_rate
                )


class WSLinear(SparseLinear):
    """
    Linear layer with Watts-Strogatz connectivity pattern.

    A convenience wrapper around SparseLinear with WS topology.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int = 4,
        beta: float = 0.3,
        bias: bool = True,
    ):
        """
        Initialize WS linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            k: Number of initial neighbors
            beta: Rewiring probability
            bias: Include bias term
        """
        # Calculate approximate density from k
        density = k / in_features

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            density=density,
            bias=bias,
            mask_type="ws",
            ws_k=k,
            ws_beta=beta,
        )

        self.k = k
        self.beta = beta


class DenseBlock(nn.Module):
    """
    A dense (fully connected) block with optional nonlinearity and normalization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()

        layers = [nn.Linear(in_features, out_features)]

        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))

        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif activation != "none":
            raise ValueError(f"Unknown activation: {activation}")

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SparseBlock(nn.Module):
    """
    A sparse block with configurable topology, activation, and normalization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        density: float = 0.1,
        topology: str = "random",
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        ws_k: int = 4,
        ws_beta: float = 0.3,
    ):
        super().__init__()

        self.sparse_linear = SparseLinear(
            in_features, out_features,
            density=density,
            mask_type="ws" if topology == "ws" else "random",
            ws_k=ws_k,
            ws_beta=ws_beta,
        )

        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sparse_linear(x)

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def get_sparsity(self) -> float:
        return self.sparse_linear.get_sparsity()

    def rewire(self, **kwargs):
        self.sparse_linear.rewire(**kwargs)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer with sparse gating.

    Uses a learned router to select which experts process each input.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_experts: int = 4,
        top_k: int = 2,
        expert_hidden: int = 256,
        sparse_experts: bool = False,
        expert_density: float = 0.3,
    ):
        """
        Initialize MoE layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            n_experts: Number of expert networks
            top_k: Number of experts to use per input
            expert_hidden: Hidden dimension in each expert
            sparse_experts: Use sparse connectivity in experts
            expert_density: Density if using sparse experts
        """
        super().__init__()

        self.n_experts = n_experts
        self.top_k = top_k

        # Router network
        self.router = nn.Linear(in_features, n_experts)

        # Expert networks
        self.experts = nn.ModuleList()
        for _ in range(n_experts):
            if sparse_experts:
                expert = nn.Sequential(
                    SparseLinear(in_features, expert_hidden, density=expert_density),
                    nn.ReLU(),
                    SparseLinear(expert_hidden, out_features, density=expert_density),
                )
            else:
                expert = nn.Sequential(
                    nn.Linear(in_features, expert_hidden),
                    nn.ReLU(),
                    nn.Linear(expert_hidden, out_features),
                )
            self.experts.append(expert)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE.

        Args:
            x: Input tensor of shape (batch, in_features)

        Returns:
            Tuple of (output, router_probs) for auxiliary loss
        """
        batch_size = x.shape[0]

        # Get router logits and probabilities
        router_logits = self.router(x)  # (batch, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        output = torch.zeros(batch_size, self.experts[0][-1].out_features, device=x.device)

        for i, expert in enumerate(self.experts):
            # Find inputs routed to this expert
            expert_mask = (top_k_indices == i).any(dim=-1)
            if not expert_mask.any():
                continue

            expert_inputs = x[expert_mask]
            expert_outputs = expert(expert_inputs)

            # Get weights for this expert
            expert_weights = torch.where(
                top_k_indices[expert_mask] == i,
                top_k_probs[expert_mask],
                torch.zeros_like(top_k_probs[expert_mask])
            ).sum(dim=-1, keepdim=True)

            output[expert_mask] += expert_outputs * expert_weights

        return output, router_probs
