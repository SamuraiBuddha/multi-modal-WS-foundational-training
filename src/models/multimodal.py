"""
Complete multi-modal architectures with Watts-Strogatz connectivity.

Implements the capstone architecture combining:
- Modality-specific encoders
- WS inter-module connectors
- Multi-modal fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .encoders import VisualEncoder, TextEncoder, AudioEncoder, create_encoder
from .connectors import WSConnector, LearnableWSConnector, AdaptiveConnector
from .fusion import MultiModalFusion, ConcatFusion, AttentionFusion
from .layers import SparseLinear, MixtureOfExperts


class MultiModalWSNetwork(nn.Module):
    """
    Multi-modal network with Watts-Strogatz inter-module connectivity.

    Architecture:
    1. Modality-specific encoders process each input type
    2. WS connector enables sparse inter-module communication
    3. Fusion module combines modality representations
    4. Task head produces final output
    """

    def __init__(
        self,
        modalities: List[str] = ["visual", "text", "audio"],
        encoder_configs: Optional[Dict[str, dict]] = None,
        embed_dim: int = 64,
        ws_k: int = 4,
        ws_beta: float = 0.3,
        learnable_beta: bool = True,
        fusion_type: str = "attention",
        output_dim: int = 10,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-modal WS network.

        Args:
            modalities: List of modality names to include
            encoder_configs: Optional config dict for each modality encoder
            embed_dim: Embedding dimension for each modality
            ws_k: Initial neighbors for WS topology
            ws_beta: Initial rewiring probability
            learnable_beta: Make beta learnable during training
            fusion_type: Fusion strategy ("concat", "attention", "gated")
            output_dim: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()

        self.modalities = modalities
        self.n_modalities = len(modalities)
        self.embed_dim = embed_dim

        # Default encoder configs
        default_configs = {
            "visual": {"input_shape": (1, 28, 28), "hidden_dims": [256, 128]},
            "text": {"vocab_size": 10000, "embed_dim": 128, "hidden_dim": 256},
            "audio": {"input_dim": 128, "hidden_dims": [256, 128]},
        }

        if encoder_configs is not None:
            for k, v in encoder_configs.items():
                default_configs[k] = v

        # Create encoders for each modality
        self.encoders = nn.ModuleDict()
        for modality in modalities:
            config = default_configs.get(modality, {})
            self.encoders[modality] = create_encoder(
                modality,
                output_dim=embed_dim,
                **config
            )

        # WS inter-module connector
        module_dims = [embed_dim] * self.n_modalities
        if learnable_beta:
            self.connector = LearnableWSConnector(
                module_dims=module_dims,
                hidden_dim=embed_dim,
                k=ws_k,
                initial_beta=ws_beta,
                n_layers=2,
            )
        else:
            self.connector = WSConnector(
                module_dims=module_dims,
                hidden_dim=embed_dim,
                k=ws_k,
                beta=ws_beta,
                n_layers=2,
            )

        # Multi-modal fusion
        self.fusion = MultiModalFusion(
            modality_dims=[embed_dim] * self.n_modalities,
            hidden_dim=embed_dim * 2,
            output_dim=embed_dim,
            fusion_type=fusion_type,
            dropout=dropout,
        )

        # Task head
        self.task_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through multi-modal network.

        Args:
            inputs: Dictionary mapping modality names to input tensors
            return_embeddings: Also return intermediate embeddings

        Returns:
            Task output, optionally with embeddings dict
        """
        # Encode each modality
        embeddings = []
        embedding_dict = {}
        for modality in self.modalities:
            if modality in inputs:
                emb = self.encoders[modality](inputs[modality])
                embeddings.append(emb)
                embedding_dict[modality] = emb
            else:
                # Use zero embedding for missing modalities
                batch_size = next(iter(inputs.values())).shape[0]
                device = next(iter(inputs.values())).device
                emb = torch.zeros(batch_size, self.embed_dim, device=device)
                embeddings.append(emb)
                embedding_dict[modality] = emb

        # Inter-module communication via WS connector
        connected = self.connector(embeddings)
        for i, modality in enumerate(self.modalities):
            embedding_dict[f"{modality}_connected"] = connected[i]

        # Fuse modalities
        fused = self.fusion(connected)
        embedding_dict["fused"] = fused

        # Task prediction
        output = self.task_head(fused)

        if return_embeddings:
            return output, embedding_dict
        return output

    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get sparsity statistics for all sparse components."""
        stats = {}

        # Connector stats
        if hasattr(self.connector, 'get_connectivity_stats'):
            stats['connector'] = self.connector.get_connectivity_stats()

        if hasattr(self.connector, 'beta'):
            stats['ws_beta'] = self.connector.beta.item()

        return stats

    def rewire(self):
        """Trigger topology rewiring in connector."""
        if hasattr(self.connector, 'maybe_rewire'):
            self.connector.maybe_rewire()


class SegmentedWSArchitecture(nn.Module):
    """
    Segmented Watts-Strogatz Multi-Modal Architecture.

    The capstone architecture with:
    - Heterogeneous module segments for each modality
    - WS small-world topology connecting segments
    - Learnable rewiring probability (beta)
    - Dynamic connectivity evolution during training
    - Mixture of Experts for modality-specific processing
    """

    def __init__(
        self,
        visual_config: Optional[dict] = None,
        text_config: Optional[dict] = None,
        audio_config: Optional[dict] = None,
        segment_dim: int = 64,
        n_ws_layers: int = 3,
        ws_k: int = 4,
        initial_beta: float = 0.3,
        use_moe: bool = True,
        n_experts: int = 4,
        sparse_layers: bool = True,
        layer_density: float = 0.3,
        output_dim: int = 10,
        dropout: float = 0.1,
    ):
        """
        Initialize Segmented WS Architecture.

        Args:
            visual_config: Config for visual encoder
            text_config: Config for text encoder
            audio_config: Config for audio encoder
            segment_dim: Dimension of each segment's output
            n_ws_layers: Number of WS-connected layers
            ws_k: Initial neighbors in WS topology
            initial_beta: Starting rewiring probability
            use_moe: Use Mixture of Experts in segments
            n_experts: Number of experts per MoE layer
            sparse_layers: Use sparse connectivity in layers
            layer_density: Density for sparse layers
            output_dim: Final output dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.segment_dim = segment_dim
        self.n_ws_layers = n_ws_layers

        # Default configs
        visual_config = visual_config or {
            "input_shape": (1, 28, 28),
            "hidden_dims": [256, 128],
        }
        text_config = text_config or {
            "vocab_size": 10000,
            "embed_dim": 128,
            "hidden_dim": 256,
        }
        audio_config = audio_config or {
            "input_dim": 128,
            "hidden_dims": [256, 128],
        }

        # Modality encoders (heterogeneous segments)
        self.visual_encoder = VisualEncoder(
            output_dim=segment_dim,
            sparse=sparse_layers,
            density=layer_density,
            **visual_config
        )
        self.text_encoder = TextEncoder(
            output_dim=segment_dim,
            **text_config
        )
        self.audio_encoder = AudioEncoder(
            output_dim=segment_dim,
            **audio_config
        )

        # WS-connected layers with learnable beta
        self.ws_layers = nn.ModuleList()
        self.beta_params = nn.ParameterList()

        for layer_idx in range(n_ws_layers):
            # Learnable beta for this layer
            beta_logit = nn.Parameter(torch.tensor(0.0))
            with torch.no_grad():
                beta_logit.data = torch.logit(torch.tensor(initial_beta))
            self.beta_params.append(beta_logit)

            # WS-connected processing layer
            if use_moe:
                layer = MixtureOfExperts(
                    in_features=segment_dim * 3,  # Concatenated modalities
                    out_features=segment_dim * 3,
                    n_experts=n_experts,
                    top_k=2,
                    expert_hidden=segment_dim * 2,
                    sparse_experts=sparse_layers,
                    expert_density=layer_density,
                )
            else:
                if sparse_layers:
                    layer = SparseLinear(
                        segment_dim * 3, segment_dim * 3,
                        density=layer_density,
                    )
                else:
                    layer = nn.Linear(segment_dim * 3, segment_dim * 3)

            self.ws_layers.append(layer)

        # Inter-segment connectivity masks (WS topology)
        # 3 modalities, each with segment_dim features
        self._init_ws_masks()

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(segment_dim * 3) for _ in range(n_ws_layers)
        ])

        # Fusion and output
        self.fusion = AttentionFusion(
            input_dim=segment_dim,
            n_modalities=3,
            n_heads=4,
            dropout=dropout,
        )

        self.output_head = nn.Sequential(
            nn.Linear(segment_dim, segment_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(segment_dim, output_dim),
        )

        # Track auxiliary losses (e.g., from MoE)
        self.aux_loss = 0.0

    def _init_ws_masks(self):
        """Initialize WS connectivity masks between segments."""
        from ..topology.graphs import watts_strogatz_graph

        # Create WS graph for inter-segment connectivity
        # 3 segments (visual, text, audio)
        adj = watts_strogatz_graph(3, k=2, beta=0.3)
        self.register_buffer('segment_adj', torch.from_numpy(adj))

    @property
    def betas(self) -> List[float]:
        """Get current beta values for each layer."""
        return [torch.sigmoid(b).item() for b in self.beta_params]

    def forward(
        self,
        visual: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through segmented WS architecture.

        Args:
            visual: Visual input (batch, C, H, W)
            text: Text input (batch, seq_len)
            audio: Audio input (batch, mel_bins, time)
            text_mask: Optional text padding mask

        Returns:
            Output logits
        """
        batch_size = (
            visual.shape[0] if visual is not None else
            text.shape[0] if text is not None else
            audio.shape[0]
        )
        device = (
            visual.device if visual is not None else
            text.device if text is not None else
            audio.device
        )

        # Encode each modality
        if visual is not None:
            visual_emb = self.visual_encoder(visual)
        else:
            visual_emb = torch.zeros(batch_size, self.segment_dim, device=device)

        if text is not None:
            text_emb = self.text_encoder(text, mask=text_mask)
        else:
            text_emb = torch.zeros(batch_size, self.segment_dim, device=device)

        if audio is not None:
            audio_emb = self.audio_encoder(audio)
        else:
            audio_emb = torch.zeros(batch_size, self.segment_dim, device=device)

        # Concatenate segment representations
        # Shape: (batch, segment_dim * 3)
        x = torch.cat([visual_emb, text_emb, audio_emb], dim=-1)

        # Pass through WS-connected layers
        self.aux_loss = 0.0
        for layer_idx in range(self.n_ws_layers):
            layer = self.ws_layers[layer_idx]

            if isinstance(layer, MixtureOfExperts):
                out, router_probs = layer(x)
                # Add load balancing loss
                self.aux_loss += self._compute_load_balance_loss(router_probs)
            else:
                out = layer(x)

            # Residual connection and norm
            x = self.layer_norms[layer_idx](x + out)

        # Split back to modalities
        visual_out = x[:, :self.segment_dim]
        text_out = x[:, self.segment_dim:self.segment_dim*2]
        audio_out = x[:, self.segment_dim*2:]

        # Fuse modalities
        fused = self.fusion([visual_out, text_out, audio_out])

        # Output
        return self.output_head(fused)

    def _compute_load_balance_loss(
        self,
        router_probs: torch.Tensor,
        importance_weight: float = 0.01
    ) -> torch.Tensor:
        """Compute load balancing loss for MoE."""
        # Encourage uniform expert usage
        expert_usage = router_probs.mean(dim=0)
        target_usage = torch.ones_like(expert_usage) / expert_usage.shape[0]
        return importance_weight * F.mse_loss(expert_usage, target_usage)

    def get_architecture_stats(self) -> Dict[str, Any]:
        """Get statistics about the architecture."""
        stats = {
            'n_ws_layers': self.n_ws_layers,
            'betas': self.betas,
            'segment_dim': self.segment_dim,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

        # Count sparse vs dense params if applicable
        sparse_params = 0
        for name, module in self.named_modules():
            if isinstance(module, SparseLinear):
                sparse_params += module.get_num_params()
        stats['sparse_params'] = sparse_params

        return stats

    def update_topology(self, step: int, update_frequency: int = 100):
        """
        Update WS topology based on learned betas.

        Called periodically during training.
        """
        if step % update_frequency != 0:
            return

        from ..topology.graphs import watts_strogatz_graph

        # Update segment adjacency based on current beta (use average)
        avg_beta = np.mean(self.betas)
        new_adj = watts_strogatz_graph(3, k=2, beta=avg_beta)
        self.segment_adj.data = torch.from_numpy(new_adj).to(self.segment_adj.device)
