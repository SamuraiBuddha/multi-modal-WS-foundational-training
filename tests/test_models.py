"""
Tests for models module.
"""

import pytest
import torch
import sys
sys.path.insert(0, '..')

from src.models import (
    SparseLinear,
    WSLinear,
    SparseBlock,
    MixtureOfExperts,
    VisualEncoder,
    TextEncoder,
    AudioEncoder,
    WSConnector,
    ConcatFusion,
    AttentionFusion,
    MultiModalWSNetwork,
)


class TestSparseLinear:
    """Tests for SparseLinear layer."""

    def test_sparse_linear_forward_shape(self):
        """Output shape should be correct."""
        layer = SparseLinear(100, 50, density=0.3)
        x = torch.randn(8, 100)
        out = layer(x)
        assert out.shape == (8, 50)

    def test_sparse_linear_sparsity(self):
        """Mask should have correct sparsity."""
        layer = SparseLinear(100, 50, density=0.3)
        actual_density = layer.mask.sum() / layer.mask.numel()
        assert abs(actual_density - 0.3) < 0.1

    def test_sparse_linear_gradient_flow(self):
        """Gradients should flow through sparse connections."""
        layer = SparseLinear(100, 50, density=0.3)
        x = torch.randn(4, 100, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.weight.grad is not None


class TestWSLinear:
    """Tests for WSLinear layer."""

    def test_ws_linear_forward(self):
        """WSLinear should produce correct output shape."""
        layer = WSLinear(100, 50, k=4, beta=0.3)
        x = torch.randn(8, 100)
        out = layer(x)
        assert out.shape == (8, 50)

    def test_ws_linear_topology(self):
        """WS mask should have small-world structure."""
        layer = WSLinear(100, 50, k=4, beta=0.3)
        # Check mask exists and has correct shape
        assert layer.mask.shape == (50, 100)


class TestMixtureOfExperts:
    """Tests for MoE layer."""

    def test_moe_forward_shape(self):
        """MoE should produce correct output shape."""
        moe = MixtureOfExperts(
            in_features=64,
            out_features=32,
            n_experts=4,
            top_k=2,
        )
        x = torch.randn(8, 64)
        out, router_probs = moe(x)
        assert out.shape == (8, 32)

    def test_moe_routing(self):
        """Top-k experts should be selected."""
        moe = MixtureOfExperts(
            in_features=64,
            out_features=32,
            n_experts=4,
            top_k=2,
        )
        x = torch.randn(8, 64)
        out, router_probs = moe(x)
        # Routing weights should sum to 1
        assert torch.allclose(router_probs.sum(dim=-1), torch.ones(8), atol=1e-5)


class TestEncoders:
    """Tests for modality encoders."""

    def test_visual_encoder(self):
        """Visual encoder should handle image input."""
        encoder = VisualEncoder(
            input_shape=(1, 28, 28),
            hidden_dims=[64, 32],
        )
        x = torch.randn(4, 1, 28, 28)
        out = encoder(x)
        assert out.shape[0] == 4

    def test_text_encoder(self):
        """Text encoder should handle token sequences."""
        encoder = TextEncoder(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=32,
        )
        x = torch.randint(0, 1000, (4, 32))
        out = encoder(x)
        assert out.shape[0] == 4

    def test_audio_encoder(self):
        """Audio encoder should handle spectrogram input."""
        encoder = AudioEncoder(
            input_dim=128,
            hidden_dims=[64, 32],
        )
        x = torch.randn(4, 128, 64)
        out = encoder(x)
        assert out.shape[0] == 4


class TestWSConnector:
    """Tests for WS inter-module connector."""

    def test_ws_connector_shape(self):
        """Connector should transform dimensions correctly."""
        connector = WSConnector(
            module_dims=[64, 64, 64],  # 3 modules with 64 dim each
            hidden_dim=128,
            k=2,
            beta=0.3,
        )
        # Connector expects list of module outputs
        inputs = [torch.randn(8, 64) for _ in range(3)]
        outputs = connector(inputs)
        # Returns list of updated representations
        assert len(outputs) == 3
        assert outputs[0].shape == (8, 128)


class TestFusion:
    """Tests for fusion modules."""

    def test_concat_fusion(self):
        """Concat fusion should combine modalities."""
        fusion = ConcatFusion(
            input_dims=[64, 64, 64],
            output_dim=128,
        )
        inputs = [torch.randn(4, 64) for _ in range(3)]
        out = fusion(inputs)
        assert out.shape == (4, 128)

    def test_attention_fusion(self):
        """Attention fusion should use attention mechanism."""
        fusion = AttentionFusion(
            input_dim=64,
            n_modalities=3,
        )
        inputs = [torch.randn(4, 64) for _ in range(3)]
        out = fusion(inputs)
        # Output is same dimension as input (no projection)
        assert out.shape == (4, 64)


class TestMultiModalWSNetwork:
    """Tests for the complete multi-modal WS network."""

    def test_multimodal_forward(self):
        """Network should handle all modality inputs."""
        model = MultiModalWSNetwork(
            modalities=['visual', 'text', 'audio'],
            encoder_configs={
                'visual': {'input_shape': (1, 28, 28), 'hidden_dims': [64]},
                'text': {'vocab_size': 100, 'embed_dim': 32, 'hidden_dim': 64},
                'audio': {'input_dim': 128, 'hidden_dims': [64]},
            },
            embed_dim=64,
            output_dim=10,
        )

        inputs = {
            'visual': torch.randn(4, 1, 28, 28),
            'text': torch.randint(0, 100, (4, 16)),
            'audio': torch.randn(4, 128, 32),
        }

        out = model(inputs)
        assert out.shape == (4, 10)

    def test_multimodal_single_modality(self):
        """Network should work with single modality."""
        model = MultiModalWSNetwork(
            modalities=['visual'],
            encoder_configs={
                'visual': {'input_shape': (1, 28, 28), 'hidden_dims': [64]},
            },
            embed_dim=64,
            output_dim=10,
        )

        inputs = {'visual': torch.randn(4, 1, 28, 28)}
        out = model(inputs)
        assert out.shape == (4, 10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
