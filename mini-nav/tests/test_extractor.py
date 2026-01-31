"""Tests for DINOv2FeatureExtractor module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from feature_compressor.core.extractor import DINOv2FeatureExtractor
from PIL import Image


class TestDINOv2FeatureExtractor:
    """Test suite for DINOv2FeatureExtractor class."""

    def test_extractor_init(self):
        """Test DINOv2FeatureExtractor initializes correctly."""

        extractor = DINOv2FeatureExtractor()

        assert extractor.model is not None
        assert extractor.processor is not None
        assert extractor.compressor is not None

    def test_single_image_processing(self):
        """Test processing a single image."""

        extractor = DINOv2FeatureExtractor()

        # Create a simple test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name)
            result = extractor.process_image(f.name)

        assert "original_features" in result
        assert "compressed_features" in result
        assert "metadata" in result

        # Check shapes
        assert result["original_features"].shape[0] == 1  # batch=1
        assert result["compressed_features"].shape == (1, 256)
        assert "compression_ratio" in result["metadata"]

    def test_output_structure(self):
        """Test output structure contains expected keys."""

        extractor = DINOv2FeatureExtractor()

        # Create test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name)
            result = extractor.process_image(f.name)

        required_keys = [
            "original_features",
            "compressed_features",
            "pooled_features",
            "metadata",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        metadata_keys = [
            "compression_ratio",
            "processing_time",
            "feature_norm",
            "device",
        ]
        for key in metadata_keys:
            assert key in result["metadata"], f"Missing metadata key: {key}"

    def test_feature_saving(self):
        """Test saving features to disk."""

        extractor = DINOv2FeatureExtractor()

        # Create test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                img.save(f.name)
                result = extractor.process_image(f.name)

            # Save features
            json_path = tmpdir / "features.json"
            from feature_compressor.utils.feature_utils import (
                save_features_to_json,
            )

            save_features_to_json(
                result["compressed_features"], json_path, result["metadata"]
            )

            assert json_path.exists()

            # Verify file can be loaded
            with open(json_path) as f:
                data = json.load(f)
            assert "features" in data
            assert "metadata" in data

    def test_batch_processing(self):
        """Test batch processing of multiple images."""

        extractor = DINOv2FeatureExtractor()

        # Create multiple test images
        images = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for i in range(3):
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img_path = tmpdir / f"test_{i}.jpg"
                img.save(img_path)
                images.append(str(img_path))

            results = extractor.process_batch(str(tmpdir), batch_size=2)

        assert len(results) == 3
        for result in results:
            assert result["compressed_features"].shape == (1, 256)

    def test_gpu_handling(self):
        """Test GPU device handling."""

        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = DINOv2FeatureExtractor(device=device)

        assert extractor.device.type == device

        # Create test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name)
            result = extractor.process_image(f.name)

        assert result["metadata"]["device"] == device
