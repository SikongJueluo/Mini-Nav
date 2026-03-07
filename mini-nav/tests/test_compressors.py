"""Tests for compressor modules (HashCompressor, Pipeline)."""

import pytest
import torch
from compressors import (
    BinarySign,
    HashCompressor,
    HashPipeline,
    SAMHashPipeline,
    VideoPositiveMask,
    bits_to_hash,
    create_pipeline_from_config,
    hamming_distance,
    hamming_similarity,
    hash_to_bits,
)
from configs import cfg_manager
from PIL import Image


class TestHashCompressor:
    """Test suite for HashCompressor."""

    def test_hash_compressor_init(self):
        """Verify HashCompressor initializes with correct dimensions."""
        compressor = HashCompressor(input_dim=1024, hash_bits=512)
        assert compressor.input_dim == 1024
        assert compressor.hash_bits == 512

    def test_hash_compressor_forward(self):
        """Verify forward pass produces correct output shapes."""
        compressor = HashCompressor(input_dim=1024, hash_bits=512)
        tokens = torch.randn(4, 197, 1024)  # [B, N, input_dim]

        logits, hash_codes, bits = compressor(tokens)

        assert logits.shape == (4, 512)
        assert hash_codes.shape == (4, 512)
        assert bits.shape == (4, 512)
        # Verify bits are binary (0 or 1)
        assert torch.all((bits == 0) | (bits == 1))

    def test_hash_compressor_encode(self):
        """Verify encode method returns binary bits."""
        compressor = HashCompressor(input_dim=1024, hash_bits=512)
        tokens = torch.randn(2, 197, 1024)

        bits = compressor.encode(tokens)

        assert bits.shape == (2, 512)
        assert bits.dtype == torch.int32
        assert torch.all((bits == 0) | (bits == 1))

    def test_hash_compressor_similarity(self):
        """Verify compute_similarity returns correct shape."""
        compressor = HashCompressor(input_dim=1024, hash_bits=512)

        # Create random bits
        bits1 = torch.randint(0, 2, (3, 512))
        bits2 = torch.randint(0, 2, (5, 512))

        sim = compressor.compute_similarity(bits1, bits2)

        assert sim.shape == (3, 5)


class TestBinarySign:
    """Test suite for BinarySign function."""

    def test_binary_sign_forward(self):
        """Verify BinarySign produces {-1, +1} outputs."""
        x = torch.randn(4, 512)
        result = BinarySign.apply(x)

        assert torch.all((result == 1) | (result == -1))

    def test_binary_sign_round_trip(self):
        """Verify bits -> hash -> bits preserves values."""
        bits = torch.randint(0, 2, (4, 512))
        hash_codes = bits_to_hash(bits)
        bits_recovered = hash_to_bits(hash_codes)

        assert torch.equal(bits, bits_recovered)


class TestHammingMetrics:
    """Test suite for Hamming distance and similarity."""

    def test_hamming_distance_same_codes(self):
        """Verify hamming distance is 0 for identical single codes."""
        bits1 = torch.randint(0, 2, (512,))
        bits2 = bits1.clone()

        dist = hamming_distance(bits1, bits2)

        assert dist.item() == 0

    def test_hamming_distance_self_comparison(self):
        """Verify hamming distance diagonal is 0 (each code compared to itself)."""
        bits = torch.randint(0, 2, (10, 512))

        dist = hamming_distance(bits, bits)

        # Diagonal should be 0 (distance to self)
        diagonal = torch.diag(dist)
        assert torch.all(diagonal == 0)

    def test_hamming_distance_different(self):
        """Verify hamming distance is correct for different codes."""
        bits1 = torch.zeros(1, 512, dtype=torch.int32)
        bits2 = torch.ones(1, 512, dtype=torch.int32)

        dist = hamming_distance(bits1, bits2)

        assert dist.item() == 512

    def test_hamming_similarity(self):
        """Verify hamming similarity is positive for similar codes."""
        hash1 = torch.ones(1, 512)
        hash2 = torch.ones(1, 512)

        sim = hamming_similarity(hash1, hash2)

        assert sim.item() == 512  # Max similarity


class TestHashLoss:
    """Test suite for HashLoss."""

    def test_hash_loss_init(self):
        """Verify HashLoss initializes with correct parameters."""
        from compressors import HashLoss

        loss_fn = HashLoss(
            contrastive_weight=1.0,
            distill_weight=0.5,
            quant_weight=0.01,
            temperature=0.2,
        )

        assert loss_fn.contrastive_weight == 1.0
        assert loss_fn.distill_weight == 0.5
        assert loss_fn.quant_weight == 0.01
        assert loss_fn.temperature == 0.2

    def test_hash_loss_forward(self):
        """Verify HashLoss computes loss correctly."""
        from compressors import HashLoss

        loss_fn = HashLoss()

        batch_size = 4
        hash_bits = 512
        logits = torch.randn(batch_size, hash_bits)
        hash_codes = torch.sign(logits)
        teacher_embed = torch.randn(batch_size, 1024)
        positive_mask = torch.eye(batch_size, dtype=torch.bool)

        total_loss, components = loss_fn(
            logits=logits,
            hash_codes=hash_codes,
            teacher_embed=teacher_embed,
            positive_mask=positive_mask,
        )

        assert "contrastive" in components
        assert "distill" in components
        assert "quantization" in components
        assert "total" in components


class TestVideoPositiveMask:
    """Test suite for VideoPositiveMask."""

    def test_from_frame_indices(self):
        """Verify positive mask generation from frame indices."""
        mask_gen = VideoPositiveMask(temporal_window=2)

        frame_indices = torch.tensor([0, 1, 3, 5])

        mask = mask_gen.from_frame_indices(frame_indices)

        assert mask.shape == (4, 4)
        # Frame 0 and 1 should be positive (distance 1 <= 2)
        assert mask[0, 1] == True
        # Frame 0 and 3 should be negative (distance 3 > 2)
        assert mask[0, 3] == False

    def test_from_video_ids(self):
        """Verify positive mask generation from video IDs and frame indices."""
        mask_gen = VideoPositiveMask(temporal_window=2)

        video_ids = torch.tensor([0, 0, 1, 1])
        frame_indices = torch.tensor([0, 1, 0, 1])

        mask = mask_gen.from_video_ids(video_ids, frame_indices)

        assert mask.shape == (4, 4)
        # Same video and temporally close
        assert mask[0, 1] == True  # video 0, frames 0,1
        # Different video
        assert mask[0, 2] == False  # video 0 vs 1


class TestHashPipeline:
    """Test suite for HashPipeline."""

    def test_pipeline_init(self):
        """Verify pipeline initializes all components."""
        pipeline = HashPipeline(
            dino_model="facebook/dinov2-large",
            hash_bits=512,
        )

        assert pipeline.dino_model == "facebook/dinov2-large"
        assert pipeline.dino_dim == 1024

    def test_pipeline_hash_bits(self):
        """Verify pipeline uses correct hash bits."""
        pipeline = HashPipeline(hash_bits=256)
        assert pipeline.hash_bits == 256

    def test_pipeline_alias(self):
        """Verify SAMHashPipeline is alias for HashPipeline."""
        assert SAMHashPipeline is HashPipeline


class TestConfigIntegration:
    """Test suite for config integration with pipeline."""

    def test_create_pipeline_from_config(self):
        """Verify pipeline can be created from config."""
        config = cfg_manager.load()

        pipeline = create_pipeline_from_config(config)

        assert isinstance(pipeline, HashPipeline)
        assert pipeline.hash_bits == config.model.compression_dim

    def test_config_settings(self):
        """Verify config contains required settings."""
        config = cfg_manager.load()

        assert hasattr(config.model, "dino_model")
        assert hasattr(config.model, "compression_dim")


@pytest.mark.slow
class TestPipelineIntegration:
    """Integration tests for full pipeline (slow, requires model downloads)."""

    def test_pipeline_end_to_end(self):
        """Test full pipeline with actual models (slow test)."""
        # Skip if no GPU
        if not torch.cuda.is_available():
            pytest.skip("Requires CUDA")

        # Create a simple test image
        image = Image.new("RGB", (640, 480), color=(128, 128, 128))

        # Initialize pipeline (will download models on first run)
        pipeline = HashPipeline(
            dino_model="facebook/dinov2-large",
            hash_bits=512,
        )

        # Run pipeline
        hash_bits = pipeline(image)

        # Verify output shape
        assert hash_bits.dim() == 2
        assert hash_bits.shape[1] == 512
        assert torch.all((hash_bits == 0) | (hash_bits == 1))

    def test_extract_features(self):
        """Test feature extraction."""
        if not torch.cuda.is_available():
            pytest.skip("Requires CUDA")

        image = Image.new("RGB", (640, 480), color=(128, 128, 128))

        pipeline = HashPipeline(
            dino_model="facebook/dinov2-large",
        )

        features = pipeline.extract_features(image)

        # Should return DINO features (1024 for large)
        assert features.dim() == 2
        assert features.shape[1] == 1024
