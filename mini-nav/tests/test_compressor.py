"""Tests for PoolNetCompressor module."""

import pytest
import torch
from feature_compressor.core.compressor import PoolNetCompressor


class TestPoolNetCompressor:
    """Test suite for PoolNetCompressor class."""

    def test_compressor_init(self):
        """Test PoolNetCompressor initializes with correct parameters."""
        # This test will fail until we implement the module

        compressor = PoolNetCompressor(
            input_dim=1024,
            compression_dim=256,
            top_k_ratio=0.5,
            hidden_ratio=2.0,
            dropout_rate=0.1,
            use_residual=True,
        )

        assert compressor.input_dim == 1024
        assert compressor.compression_dim == 256
        assert compressor.top_k_ratio == 0.5

    def test_compressor_forward_shape(self):
        """Test output shape is [batch, compression_dim]."""

        compressor = PoolNetCompressor(
            input_dim=1024,
            compression_dim=256,
            top_k_ratio=0.5,
        )

        # Simulate DINOv2 output: batch=2, seq_len=257 (CLS+256 patches), dim=1024
        x = torch.randn(2, 257, 1024)
        out = compressor(x)

        assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"

    def test_attention_scores_shape(self):
        """Test attention scores have shape [batch, seq_len]."""

        compressor = PoolNetCompressor(input_dim=1024, compression_dim=256)

        x = torch.randn(2, 257, 1024)
        scores = compressor._compute_attention_scores(x)

        assert scores.shape == (2, 257), f"Expected (2, 257), got {scores.shape}"

    def test_top_k_selection(self):
        """Test that only top_k_ratio tokens are selected."""

        compressor = PoolNetCompressor(
            input_dim=1024, compression_dim=256, top_k_ratio=0.5
        )

        x = torch.randn(2, 257, 1024)
        pooled = compressor._apply_pooling(x, compressor._compute_attention_scores(x))

        # With top_k_ratio=0.5, should select 50% of tokens (int rounds down)
        expected_k = 128  # int(257 * 0.5) = 128
        assert pooled.shape[1] == expected_k, (
            f"Expected seq_len={expected_k}, got {pooled.shape[1]}"
        )

    def test_residual_connection(self):
        """Test residual adds input contribution to output."""

        compressor = PoolNetCompressor(
            input_dim=1024,
            compression_dim=256,
            use_residual=True,
        )

        x = torch.randn(2, 257, 1024)
        out1 = compressor(x)

        # Residual should affect output
        assert out1 is not None
        assert out1.shape == (2, 256)

    def test_gpu_device(self):
        """Test model moves to GPU correctly if available."""

        device = "cuda" if torch.cuda.is_available() else "cpu"

        compressor = PoolNetCompressor(
            input_dim=1024,
            compression_dim=256,
            device=device,
        )

        x = torch.randn(2, 257, 1024).to(device)
        out = compressor(x)

        assert out.device.type == device
