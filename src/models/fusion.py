"""
Multi-modal fusion strategies.

Provides various methods to combine representations from different modalities:
- Concatenation fusion
- Attention-based fusion
- Gated fusion
- Cross-modal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion.

    Concatenates all modality embeddings and projects to output dimension.
    """

    def __init__(
        self,
        input_dims: List[int],
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize concatenation fusion.

        Args:
            input_dims: Dimensions of each modality input
            output_dim: Output dimension
            hidden_dim: Optional hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()

        total_input = sum(input_dims)

        if hidden_dim is not None:
            self.fusion = nn.Sequential(
                nn.Linear(total_input, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.fusion = nn.Linear(total_input, output_dim)

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse embeddings by concatenation.

        Args:
            embeddings: List of modality embeddings

        Returns:
            Fused representation
        """
        concatenated = torch.cat(embeddings, dim=-1)
        return self.fusion(concatenated)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion.

    Uses attention mechanism to weight contributions from each modality.
    """

    def __init__(
        self,
        input_dim: int,
        n_modalities: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize attention fusion.

        Args:
            input_dim: Dimension of each modality embedding (must be same)
            n_modalities: Number of modalities
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_modalities = n_modalities

        # Query for fusion (learnable)
        self.query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm
        self.norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        embeddings: List[torch.Tensor],
        return_weights: bool = False
    ) -> torch.Tensor:
        """
        Fuse embeddings using attention.

        Args:
            embeddings: List of modality embeddings (batch, input_dim)
            return_weights: Return attention weights for visualization

        Returns:
            Fused representation, optionally with attention weights
        """
        batch_size = embeddings[0].shape[0]

        # Stack embeddings: (batch, n_modalities, input_dim)
        stacked = torch.stack(embeddings, dim=1)

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)

        # Attention: query attends to all modality embeddings
        fused, attn_weights = self.attention(query, stacked, stacked)

        # Remove sequence dimension and normalize
        fused = self.norm(fused.squeeze(1))

        if return_weights:
            return fused, attn_weights.squeeze(1)
        return fused


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism.

    Uses learned gates to control contribution of each modality.
    """

    def __init__(
        self,
        input_dim: int,
        n_modalities: int,
        output_dim: Optional[int] = None,
        gate_type: str = "softmax",
    ):
        """
        Initialize gated fusion.

        Args:
            input_dim: Dimension of each modality embedding
            n_modalities: Number of modalities
            output_dim: Output dimension (defaults to input_dim)
            gate_type: "softmax", "sigmoid", or "sparsemax"
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_modalities = n_modalities
        self.output_dim = output_dim or input_dim
        self.gate_type = gate_type

        # Gate network: takes concatenated inputs, outputs gate values
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim * n_modalities, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, n_modalities),
        )

        # Output projection
        if self.output_dim != input_dim:
            self.output_proj = nn.Linear(input_dim, self.output_dim)
        else:
            self.output_proj = nn.Identity()

    def forward(
        self,
        embeddings: List[torch.Tensor],
        return_gates: bool = False
    ) -> torch.Tensor:
        """
        Fuse embeddings using learned gates.

        Args:
            embeddings: List of modality embeddings
            return_gates: Return gate values for visualization

        Returns:
            Fused representation, optionally with gate values
        """
        # Concatenate for gate computation
        concatenated = torch.cat(embeddings, dim=-1)

        # Compute gate values
        gate_logits = self.gate_network(concatenated)

        if self.gate_type == "softmax":
            gates = F.softmax(gate_logits, dim=-1)
        elif self.gate_type == "sigmoid":
            gates = torch.sigmoid(gate_logits)
        else:
            gates = F.softmax(gate_logits, dim=-1)

        # Stack embeddings and apply gates
        stacked = torch.stack(embeddings, dim=1)  # (batch, n_mod, dim)
        gates = gates.unsqueeze(-1)  # (batch, n_mod, 1)

        # Weighted sum
        fused = (stacked * gates).sum(dim=1)

        # Project output
        fused = self.output_proj(fused)

        if return_gates:
            return fused, gates.squeeze(-1)
        return fused


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism.

    Allows each modality to attend to others for information exchange.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        """
        Initialize cross-modal attention.

        Args:
            input_dim: Dimension of modality embeddings
            n_heads: Number of attention heads
            dropout: Dropout rate
            bidirectional: Apply attention in both directions
        """
        super().__init__()

        self.input_dim = input_dim
        self.bidirectional = bidirectional

        # Cross-attention layers
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward for each modality after attention
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(
        self,
        query_embedding: torch.Tensor,
        key_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention.

        Args:
            query_embedding: Query modality (batch, dim)
            key_embedding: Key/value modality (batch, dim)

        Returns:
            Updated embeddings for both modalities
        """
        # Add sequence dimension
        query = query_embedding.unsqueeze(1)
        key = key_embedding.unsqueeze(1)

        # Cross attention: query attends to key
        attn_out, _ = self.cross_attn(query, key, key)
        query_updated = self.norm1(query + attn_out)
        query_updated = self.norm2(query_updated + self.ffn(query_updated))

        if self.bidirectional:
            # Key attends to query
            attn_out, _ = self.cross_attn(key, query, query)
            key_updated = self.norm1(key + attn_out)
            key_updated = self.norm2(key_updated + self.ffn(key_updated))
            return query_updated.squeeze(1), key_updated.squeeze(1)

        return query_updated.squeeze(1), key_embedding


class MultiModalFusion(nn.Module):
    """
    Complete multi-modal fusion module combining multiple strategies.
    """

    def __init__(
        self,
        modality_dims: List[int],
        hidden_dim: int = 128,
        output_dim: int = 64,
        fusion_type: str = "attention",
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-modal fusion.

        Args:
            modality_dims: Input dimensions for each modality
            hidden_dim: Hidden dimension for fusion
            output_dim: Final output dimension
            fusion_type: "concat", "attention", "gated", or "cross"
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.n_modalities = len(modality_dims)
        self.fusion_type = fusion_type

        # Project all modalities to same dimension first
        self.modality_projs = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in modality_dims
        ])

        # Fusion mechanism
        if fusion_type == "concat":
            self.fusion = ConcatFusion(
                [hidden_dim] * self.n_modalities,
                output_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif fusion_type == "attention":
            self.fusion = AttentionFusion(
                hidden_dim, self.n_modalities, n_heads, dropout
            )
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        elif fusion_type == "gated":
            self.fusion = GatedFusion(
                hidden_dim, self.n_modalities, output_dim
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        embeddings: List[torch.Tensor],
        return_details: bool = False
    ) -> torch.Tensor:
        """
        Fuse multiple modality embeddings.

        Args:
            embeddings: List of modality embeddings
            return_details: Return fusion details (weights, gates, etc.)

        Returns:
            Fused representation
        """
        # Project to common dimension
        projected = [
            proj(emb) for proj, emb in zip(self.modality_projs, embeddings)
        ]

        # Apply fusion
        if self.fusion_type == "attention":
            if return_details:
                fused, weights = self.fusion(projected, return_weights=True)
                return self.output_proj(fused), weights
            fused = self.fusion(projected)
            return self.output_proj(fused)
        elif self.fusion_type == "gated" and return_details:
            return self.fusion(projected, return_gates=True)
        else:
            return self.fusion(projected)
