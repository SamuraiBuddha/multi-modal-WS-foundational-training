"""
Modality-specific encoder networks.

Provides encoders for different data types:
- Visual (images)
- Text (sequences)
- Audio (spectrograms)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np

from .layers import SparseLinear, SparseBlock, DenseBlock


class VisualEncoder(nn.Module):
    """
    Encoder for visual/image data.

    Can use either convolutional or fully-connected architecture.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 28, 28),
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 64,
        use_conv: bool = False,
        sparse: bool = False,
        density: float = 0.3,
        dropout: float = 0.1,
    ):
        """
        Initialize visual encoder.

        Args:
            input_shape: Shape of input images (C, H, W)
            hidden_dims: Hidden layer dimensions
            output_dim: Output embedding dimension
            use_conv: Use convolutional layers (vs fully connected)
            sparse: Use sparse connectivity
            density: Connection density if sparse
            dropout: Dropout rate
        """
        super().__init__()

        self.input_shape = input_shape
        self.output_dim = output_dim

        if use_conv:
            self.encoder = self._build_conv_encoder(
                input_shape, hidden_dims, output_dim, dropout
            )
        else:
            input_dim = np.prod(input_shape)
            self.encoder = self._build_fc_encoder(
                input_dim, hidden_dims, output_dim, sparse, density, dropout
            )

        self.use_conv = use_conv

    def _build_conv_encoder(
        self,
        input_shape: Tuple[int, ...],
        hidden_dims: List[int],
        output_dim: int,
        dropout: float,
    ) -> nn.Module:
        """Build convolutional encoder."""
        c, h, w = input_shape

        layers = []

        # Conv layers
        in_channels = c
        for i, out_channels in enumerate([32, 64]):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])
            in_channels = out_channels
            h = (h + 1) // 2
            w = (w + 1) // 2

        layers.append(nn.Flatten())

        # FC layers
        fc_input = in_channels * h * w
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(fc_input, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            fc_input = hidden_dim

        layers.append(nn.Linear(fc_input, output_dim))

        return nn.Sequential(*layers)

    def _build_fc_encoder(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        sparse: bool,
        density: float,
        dropout: float,
    ) -> nn.Module:
        """Build fully connected encoder."""
        layers = [nn.Flatten()]

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            if sparse:
                layers.append(SparseBlock(
                    prev_dim, hidden_dim,
                    density=density,
                    activation="relu",
                    dropout=dropout,
                ))
            else:
                layers.append(DenseBlock(
                    prev_dim, hidden_dim,
                    activation="relu",
                    dropout=dropout,
                ))
            prev_dim = hidden_dim

        # Output layer (typically dense for final projection)
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode visual input to embedding."""
        return self.encoder(x)


class TextEncoder(nn.Module):
    """
    Encoder for text/sequence data.

    Uses embedding + transformer or LSTM architecture.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 64,
        n_layers: int = 2,
        architecture: str = "transformer",
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        """
        Initialize text encoder.

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden state dimension
            output_dim: Output embedding dimension
            n_layers: Number of encoder layers
            architecture: "transformer" or "lstm"
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()

        self.output_dim = output_dim
        self.architecture = architecture

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if architecture == "transformer":
            # Positional encoding
            self.pos_encoding = nn.Parameter(
                torch.randn(1, max_seq_len, embed_dim) * 0.02
            )

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            # Output projection
            self.output_proj = nn.Linear(embed_dim, output_dim)

        else:  # LSTM
            self.encoder = nn.LSTM(
                embed_dim, hidden_dim // 2,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_layers > 1 else 0,
            )
            self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text input to embedding.

        Args:
            x: Token indices of shape (batch, seq_len)
            mask: Optional attention mask

        Returns:
            Text embedding of shape (batch, output_dim)
        """
        # Embed tokens
        embedded = self.embedding(x)

        if self.architecture == "transformer":
            # Add positional encoding
            seq_len = x.shape[1]
            embedded = embedded + self.pos_encoding[:, :seq_len, :]

            # Encode
            encoded = self.encoder(embedded, src_key_padding_mask=mask)

            # Pool (mean over sequence)
            if mask is not None:
                # Mask out padding for mean
                mask_expanded = (~mask).unsqueeze(-1).float()
                pooled = (encoded * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
            else:
                pooled = encoded.mean(dim=1)

        else:  # LSTM
            if mask is not None:
                # Pack padded sequence
                lengths = (~mask).sum(dim=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    embedded, lengths, batch_first=True, enforce_sorted=False
                )
                _, (hidden, _) = self.encoder(packed)
            else:
                _, (hidden, _) = self.encoder(embedded)

            # Concatenate forward and backward hidden states
            pooled = torch.cat([hidden[-2], hidden[-1]], dim=-1)

        # Project to output dimension
        return self.output_proj(pooled)


class AudioEncoder(nn.Module):
    """
    Encoder for audio data (mel spectrograms).

    Uses 1D or 2D convolutions depending on input format.
    """

    def __init__(
        self,
        input_dim: int = 128,  # mel bins
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 64,
        use_conv: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize audio encoder.

        Args:
            input_dim: Number of mel frequency bins
            hidden_dims: Hidden layer dimensions
            output_dim: Output embedding dimension
            use_conv: Use convolutional layers
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if use_conv:
            self.encoder = self._build_conv_encoder(
                input_dim, hidden_dims, output_dim, dropout
            )
        else:
            self.encoder = self._build_fc_encoder(
                input_dim, hidden_dims, output_dim, dropout
            )

        self.use_conv = use_conv

    def _build_conv_encoder(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float,
    ) -> nn.Module:
        """Build 1D convolutional encoder for spectrograms."""
        layers = []

        # 1D conv layers over time dimension
        in_channels = input_dim
        for i, out_channels in enumerate([64, 128, 256]):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        # Global average pooling
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())

        # FC layers
        prev_dim = in_channels
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _build_fc_encoder(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float,
    ) -> nn.Module:
        """Build FC encoder (expects flattened input)."""
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode audio input to embedding.

        Args:
            x: Mel spectrogram of shape (batch, mel_bins, time) for conv
               or (batch, features) for FC

        Returns:
            Audio embedding of shape (batch, output_dim)
        """
        return self.encoder(x)


def create_encoder(
    modality: str,
    output_dim: int = 64,
    sparse: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create modality-specific encoders.

    Args:
        modality: "visual", "text", or "audio"
        output_dim: Output embedding dimension
        sparse: Use sparse connectivity (where applicable)
        **kwargs: Additional arguments for specific encoder

    Returns:
        Encoder module
    """
    if modality == "visual":
        return VisualEncoder(output_dim=output_dim, sparse=sparse, **kwargs)
    elif modality == "text":
        return TextEncoder(output_dim=output_dim, **kwargs)
    elif modality == "audio":
        return AudioEncoder(output_dim=output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown modality: {modality}")
