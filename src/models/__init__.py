"""
Models module for neural network architectures.

Provides:
- Sparse layers with various connectivity patterns
- Modality-specific encoders (visual, text, audio)
- Watts-Strogatz inter-module connectors
- Multi-modal fusion modules
- Complete multi-modal WS architecture
"""

from .layers import (
    SparseLinear,
    WSLinear,
    DenseBlock,
    SparseBlock,
    MixtureOfExperts,
)
from .encoders import (
    VisualEncoder,
    TextEncoder,
    AudioEncoder,
    create_encoder,
)
from .connectors import (
    WSConnector,
    LearnableWSConnector,
    AdaptiveConnector,
)
from .fusion import (
    ConcatFusion,
    AttentionFusion,
    GatedFusion,
    CrossModalAttention,
)
from .multimodal import (
    MultiModalWSNetwork,
    SegmentedWSArchitecture,
)

__all__ = [
    # Layers
    "SparseLinear",
    "WSLinear",
    "DenseBlock",
    "SparseBlock",
    "MixtureOfExperts",
    # Encoders
    "VisualEncoder",
    "TextEncoder",
    "AudioEncoder",
    "create_encoder",
    # Connectors
    "WSConnector",
    "LearnableWSConnector",
    "AdaptiveConnector",
    # Fusion
    "ConcatFusion",
    "AttentionFusion",
    "GatedFusion",
    "CrossModalAttention",
    # Multi-modal
    "MultiModalWSNetwork",
    "SegmentedWSArchitecture",
]
