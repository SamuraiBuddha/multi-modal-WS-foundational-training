"""
Inter-module connectors with Watts-Strogatz topology.

These connectors wire together different modules (e.g., modality encoders)
using small-world network patterns for efficient information flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import numpy as np

from ..topology.graphs import watts_strogatz_graph
from ..topology.sparse_masks import ws_sparse_mask, get_sparsity
from ..topology.rewiring import rewire_edges, set_rewire


class WSConnector(nn.Module):
    """
    Watts-Strogatz inter-module connector.

    Connects multiple input modules using WS small-world topology.
    Each module's output is transformed and sparsely connected to others.
    """

    def __init__(
        self,
        module_dims: List[int],
        hidden_dim: int = 128,
        k: int = 4,
        beta: float = 0.3,
        n_layers: int = 2,
    ):
        """
        Initialize WS connector.

        Args:
            module_dims: List of input dimensions from each module
            hidden_dim: Hidden dimension for inter-module communication
            k: Initial neighbors in WS topology
            beta: Rewiring probability
            n_layers: Number of message passing layers
        """
        super().__init__()

        self.n_modules = len(module_dims)
        self.hidden_dim = hidden_dim
        self.k = k
        self.beta = beta
        self.n_layers = n_layers

        # Project each module to common hidden dimension
        self.input_projs = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in module_dims
        ])

        # Create WS adjacency matrix for inter-module connectivity
        # This defines which modules can directly communicate
        if self.n_modules > 1:
            adj = watts_strogatz_graph(
                self.n_modules, min(k, self.n_modules - 1), beta
            )
            self.register_buffer('module_adj', torch.from_numpy(adj))
        else:
            self.register_buffer('module_adj', torch.ones(1, 1))

        # Message passing layers
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for _ in range(n_layers):
            # Message function: aggregate from connected modules
            self.message_layers.append(
                nn.Linear(hidden_dim, hidden_dim)
            )
            # Update function: combine with self
            self.update_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                )
            )

        # Layer norm for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, module_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Pass messages between modules using WS connectivity.

        Args:
            module_outputs: List of tensors from each module
                           Each has shape (batch, module_dim[i])

        Returns:
            List of updated module representations
        """
        batch_size = module_outputs[0].shape[0]

        # Project to common dimension
        hidden_states = [
            proj(out) for proj, out in zip(self.input_projs, module_outputs)
        ]  # List of (batch, hidden_dim)

        # Stack for easier manipulation: (batch, n_modules, hidden_dim)
        hidden = torch.stack(hidden_states, dim=1)

        # Message passing
        for layer_idx in range(self.n_layers):
            # Compute messages from each module
            messages = self.message_layers[layer_idx](hidden)  # (batch, n_modules, hidden_dim)

            # Aggregate messages according to WS adjacency
            # module_adj[i,j] = 1 if module j sends to module i
            aggregated = torch.matmul(
                self.module_adj.unsqueeze(0),  # (1, n_modules, n_modules)
                messages  # (batch, n_modules, hidden_dim)
            )  # (batch, n_modules, hidden_dim)

            # Normalize by degree
            degrees = self.module_adj.sum(dim=1, keepdim=True).clamp(min=1)
            aggregated = aggregated / degrees.unsqueeze(0)

            # Update hidden states
            combined = torch.cat([hidden, aggregated], dim=-1)
            hidden = self.update_layers[layer_idx](combined)
            hidden = self.layer_norms[layer_idx](hidden)

        # Split back to list
        return [hidden[:, i, :] for i in range(self.n_modules)]

    def get_connectivity_stats(self) -> Dict[str, float]:
        """Get statistics about inter-module connectivity."""
        adj = self.module_adj.cpu().numpy()
        return {
            'n_connections': float(adj.sum()),
            'density': float(adj.sum() / (self.n_modules * (self.n_modules - 1) + 1e-8)),
            'avg_degree': float(adj.sum(axis=1).mean()),
        }


class LearnableWSConnector(nn.Module):
    """
    WS connector with learnable beta parameter.

    The rewiring probability beta is learned during training,
    allowing the network to discover optimal connectivity.
    """

    def __init__(
        self,
        module_dims: List[int],
        hidden_dim: int = 128,
        k: int = 4,
        initial_beta: float = 0.3,
        n_layers: int = 2,
        rewire_frequency: int = 100,
    ):
        """
        Initialize learnable WS connector.

        Args:
            module_dims: Input dimensions from each module
            hidden_dim: Hidden dimension
            k: Initial neighbors
            initial_beta: Starting rewiring probability
            n_layers: Message passing layers
            rewire_frequency: Steps between topology updates
        """
        super().__init__()

        self.n_modules = len(module_dims)
        self.hidden_dim = hidden_dim
        self.k = k
        self.n_layers = n_layers
        self.rewire_frequency = rewire_frequency

        # Learnable beta parameter (sigmoid constrained to [0, 1])
        self.beta_logit = nn.Parameter(torch.tensor(0.0))
        with torch.no_grad():
            # Initialize to desired beta
            self.beta_logit.data = torch.logit(torch.tensor(initial_beta))

        # Module projections
        self.input_projs = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in module_dims
        ])

        # Initialize adjacency with initial beta
        adj = watts_strogatz_graph(
            self.n_modules, min(k, self.n_modules - 1), initial_beta
        )
        self.register_buffer('module_adj', torch.from_numpy(adj))

        # Soft attention for differentiable routing (gradient path for beta)
        self.route_attention = nn.Parameter(torch.ones(self.n_modules, self.n_modules))

        # Message passing
        self.message_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.update_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
            ) for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # Step counter
        self.register_buffer('step_count', torch.tensor(0))

    @property
    def beta(self) -> torch.Tensor:
        """Get current beta value."""
        return torch.sigmoid(self.beta_logit)

    def forward(self, module_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass with soft attention routing."""
        batch_size = module_outputs[0].shape[0]

        # Project inputs
        hidden_states = [
            proj(out) for proj, out in zip(self.input_projs, module_outputs)
        ]
        hidden = torch.stack(hidden_states, dim=1)

        # Compute soft attention weights
        # Combine hard adjacency with soft attention
        soft_adj = F.softmax(self.route_attention, dim=-1)
        effective_adj = self.module_adj * soft_adj

        # Message passing
        for layer_idx in range(self.n_layers):
            messages = self.message_layers[layer_idx](hidden)

            # Use effective adjacency for routing
            aggregated = torch.matmul(
                effective_adj.unsqueeze(0),
                messages
            )

            degrees = effective_adj.sum(dim=1, keepdim=True).clamp(min=1)
            aggregated = aggregated / degrees.unsqueeze(0)

            combined = torch.cat([hidden, aggregated], dim=-1)
            hidden = self.update_layers[layer_idx](combined)
            hidden = self.layer_norms[layer_idx](hidden)

        return [hidden[:, i, :] for i in range(self.n_modules)]

    def maybe_rewire(self):
        """Rewire topology based on current beta (called during training)."""
        self.step_count += 1
        if self.step_count % self.rewire_frequency == 0:
            with torch.no_grad():
                current_beta = self.beta.item()
                new_adj = watts_strogatz_graph(
                    self.n_modules, min(self.k, self.n_modules - 1), current_beta
                )
                self.module_adj.data = torch.from_numpy(new_adj).to(self.module_adj.device)


class AdaptiveConnector(nn.Module):
    """
    Adaptive connector that learns both topology and weights.

    Uses attention mechanisms to discover optimal inter-module routing.
    """

    def __init__(
        self,
        module_dims: List[int],
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize adaptive connector.

        Args:
            module_dims: Input dimensions from each module
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.n_modules = len(module_dims)
        self.hidden_dim = hidden_dim

        # Project each module to common dimension
        self.input_projs = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in module_dims
        ])

        # Multi-head attention for inter-module communication
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, module_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Route information between modules using attention.

        Args:
            module_outputs: List of module output tensors

        Returns:
            Updated module representations
        """
        # Project and stack
        hidden_states = [
            proj(out) for proj, out in zip(self.input_projs, module_outputs)
        ]
        hidden = torch.stack(hidden_states, dim=1)  # (batch, n_modules, hidden_dim)

        # Self-attention between modules
        attn_out, attn_weights = self.attention(hidden, hidden, hidden)
        hidden = self.norm1(hidden + attn_out)

        # Feed-forward
        ffn_out = self.ffn(hidden)
        hidden = self.norm2(hidden + ffn_out)

        return [hidden[:, i, :] for i in range(self.n_modules)]

    def get_attention_pattern(
        self,
        module_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Get attention weights for visualization."""
        hidden_states = [
            proj(out) for proj, out in zip(self.input_projs, module_outputs)
        ]
        hidden = torch.stack(hidden_states, dim=1)

        _, attn_weights = self.attention(hidden, hidden, hidden)
        return attn_weights
