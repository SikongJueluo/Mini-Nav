"""Tests for compressor modules (SAM, DINO, HashCompressor, Pipeline)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from configs import cfg_manager
from compressors import (
    BinarySign,
    DinoCompressor,
    HashCompressor,
    SegmentCompressor,
    SAMHashPipeline,
    create_pipeline_from_config,
    bits_to_hash,
    hash_to_bits,
    hamming_distance,
    hamming_similarity,
)


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


class TestSegmentCompressor:
    """Test suite for SegmentCompressor."""

    @pytest.fixture
    def mock_image(self):
        """Create a mock PIL image."""
        img = Image.new("RGB", (224, 224), color="red")
        return img

    def test_segment_compressor_init(self):
        """Verify SegmentCompressor initializes with correct parameters."""
        segmentor = SegmentCompressor(
            model_name="facebook/sam2.1-hiera-large",
            min_mask_area=100,
            max_masks=10,
        )

        assert segmentor.model_name == "facebook/sam2.1-hiera-large"
        assert segmentor.min_mask_area == 100
        assert segmentor.max_masks == 10

    def test_filter_masks(self):
        """Verify mask filtering logic."""
        # Create segmentor to get default filter params
        segmentor = SegmentCompressor()

        # Create mock masks tensor with different areas
        # Masks shape: [N, H, W]
        masks = []
        for area in [50, 200, 150, 300, 10]:
            mask = torch.zeros(100, 100)
            mask[:1, :area] = 1  # Create mask with specific area
            masks.append(mask)

        masks_tensor = torch.stack(masks)  # [5, 100, 100]
        valid = segmentor._filter_masks(masks_tensor)

        # Should filter out 50 and 10 (below min_mask_area=100)
        # Then keep top 3 (max_masks=10)
        assert len(valid) == 3
        # Verify sorted by area (descending)
        areas = [v["area"] for v in valid]
        assert areas == sorted(areas, reverse=True)


class TestDinoCompressor:
    """Test suite for DinoCompressor."""

    def test_dino_compressor_init(self):
        """Verify DinoCompressor initializes correctly."""
        dino = DinoCompressor()

        assert dino.model_name == "facebook/dinov2-large"

    def test_dino_compressor_with_compressor(self):
        """Verify DinoCompressor with HashCompressor."""
        hash_compressor = HashCompressor(input_dim=1024, hash_bits=512)
        dino = DinoCompressor(compressor=hash_compressor)

        assert dino.compressor is hash_compressor


class TestSAMHashPipeline:
    """Test suite for SAMHashPipeline."""

    def test_pipeline_init(self):
        """Verify pipeline initializes all components."""
        pipeline = SAMHashPipeline(
            sam_model="facebook/sam2.1-hiera-large",
            dino_model="facebook/dinov2-large",
            hash_bits=512,
        )

        assert isinstance(pipeline.segmentor, SegmentCompressor)
        assert isinstance(pipeline.dino, DinoCompressor)
        assert isinstance(pipeline.hash_compressor, HashCompressor)

    def test_pipeline_hash_bits(self):
        """Verify pipeline uses correct hash bits."""
        pipeline = SAMHashPipeline(hash_bits=256)
        assert pipeline.hash_compressor.hash_bits == 256


class TestConfigIntegration:
    """Test suite for config integration with pipeline."""

    def test_create_pipeline_from_config(self):
        """Verify pipeline can be created from config."""
        config = cfg_manager.load()

        pipeline = create_pipeline_from_config(config)

        assert isinstance(pipeline, SAMHashPipeline)
        assert pipeline.hash_compressor.hash_bits == config.model.compression_dim

    def test_config_sam_settings(self):
        """Verify config contains SAM settings."""
        config = cfg_manager.load()

        assert hasattr(config.model, "sam_model")
        assert hasattr(config.model, "sam_min_mask_area")
        assert hasattr(config.model, "sam_max_masks")
        assert config.model.sam_model == "facebook/sam2.1-hiera-large"
        assert config.model.sam_min_mask_area == 100
        assert config.model.sam_max_masks == 10


class TestPipelineIntegration:
    """Integration tests for full pipeline (slow, requires model downloads)."""

    @pytest.mark.slow
    def test_pipeline_end_to_end(self):
        """Test full pipeline with actual models (slow test)."""
        # Skip if no GPU
        if not torch.cuda.is_available():
            pytest.skip("Requires CUDA")

        # Create a simple test image
        image = Image.new("RGB", (640, 480), color=(128, 128, 128))

        # Initialize pipeline (will download models on first run)
        pipeline = SAMHashPipeline(
            sam_model="facebook/sam2.1-hiera-large",
            dino_model="facebook/dinov2-large",
            hash_bits=512,
            sam_min_mask_area=100,
            sam_max_masks=5,
        )

        # Run pipeline
        hash_codes = pipeline(image)

        # Verify output shape
        assert hash_codes.dim() == 2
        assert hash_codes.shape[1] == 512
        assert torch.all((hash_codes == 0) | (hash_codes == 1))

    @pytest.mark.slow
    def test_extract_features_without_hash(self):
        """Test feature extraction without hash compression."""
        if not torch.cuda.is_available():
            pytest.skip("Requires CUDA")

        image = Image.new("RGB", (640, 480), color=(128, 128, 128))

        pipeline = SAMHashPipeline(
            sam_model="facebook/sam2.1-hiera-large",
            dino_model="facebook/dinov2-large",
        )

        features = pipeline.extract_features(image, use_hash=False)

        # Should return DINO features (1024 for large)
        assert features.dim() == 2
        assert features.shape[1] == 1024

    @pytest.mark.slow
    def test_extract_masks_only(self):
        """Test mask extraction only."""
        if not torch.cuda.is_available():
            pytest.skip("Requires CUDA")

        image = Image.new("RGB", (640, 480), color=(128, 128, 128))

        pipeline = SAMHashPipeline(
            sam_model="facebook/sam2.1-hiera-large",
        )

        masks = pipeline.extract_masks(image)

        # Should return a list of masks
        assert isinstance(masks, list)
