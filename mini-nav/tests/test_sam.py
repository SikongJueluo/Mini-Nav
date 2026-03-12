"""Tests for SAM segmentation utilities.

Note: These tests mock the SAM model loading since SAM requires
heavy model weights. The actual SAM integration should be tested
separately in integration tests.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from PIL import Image


class TestSAMSegmentation:
    """Test suite for SAM segmentation utilities."""

    def test_segment_image_empty_masks(self):
        """Test segment_image returns empty list when no masks generated."""
        from utils.sam import segment_image

        # Create mock mask generator that returns empty list
        mock_generator = Mock()
        mock_generator.generate.return_value = []

        result = segment_image(mock_generator, Image.new("RGB", (100, 100)))

        assert result == []

    def test_segment_image_filters_small_masks(self):
        """Test segment_image filters masks below min_area threshold."""
        from utils.sam import segment_image

        # Create mock masks with different areas
        small_mask = {
            "segment": np.zeros((10, 10), dtype=bool),
            "area": 50,  # Below 32*32 = 1024
            "bbox": [0, 0, 10, 10],
            "predicted_iou": 0.9,
            "stability_score": 0.8,
        }
        large_mask = {
            "segment": np.ones((100, 100), dtype=bool),
            "area": 10000,  # Above threshold
            "bbox": [0, 0, 100, 100],
            "predicted_iou": 0.95,
            "stability_score": 0.9,
        }

        mock_generator = Mock()
        mock_generator.generate.return_value = [small_mask, large_mask]

        result = segment_image(
            mock_generator,
            Image.new("RGB", (100, 100)),
            min_area=32 * 32,
            max_masks=5,
        )

        # Should only return the large mask
        assert len(result) == 1
        assert result[0]["area"] == 10000

    def test_segment_image_limits_max_masks(self):
        """Test segment_image limits to max_masks largest masks."""
        from utils.sam import segment_image

        # Create 10 masks with different areas
        masks = [
            {
                "segment": np.ones((i + 1, i + 1), dtype=bool),
                "area": (i + 1) * (i + 1),
                "bbox": [0, 0, i + 1, i + 1],
                "predicted_iou": 0.9,
                "stability_score": 0.8,
            }
            for i in range(10)
        ]

        mock_generator = Mock()
        mock_generator.generate.return_value = masks

        result = segment_image(
            mock_generator,
            Image.new("RGB", (100, 100)),
            min_area=1,
            max_masks=3,
        )

        # Should only return top 3 largest masks
        assert len(result) == 3
        # Check they are sorted by area (largest first)
        areas = [m["area"] for m in result]
        assert areas == sorted(areas, reverse=True)

    def test_segment_image_sorted_by_area(self):
        """Test segment_image returns masks sorted by area descending."""
        from utils.sam import segment_image

        # Create masks with known areas (unordered)
        mask1 = {"segment": np.ones((5, 5), dtype=bool), "area": 25, "bbox": [0, 0, 5, 5]}
        mask2 = {"segment": np.ones((10, 10), dtype=bool), "area": 100, "bbox": [0, 0, 10, 10]}
        mask3 = {"segment": np.ones((3, 3), dtype=bool), "area": 9, "bbox": [0, 0, 3, 3]}

        mock_generator = Mock()
        mock_generator.generate.return_value = [mask1, mask2, mask3]

        result = segment_image(
            mock_generator,
            Image.new("RGB", (100, 100)),
            min_area=1,
            max_masks=10,
        )

        # Should be sorted by area descending
        assert result[0]["area"] == 100
        assert result[1]["area"] == 25
        assert result[2]["area"] == 9


class TestExtractMaskedRegion:
    """Test suite for extracting masked regions from images."""

    def test_extract_masked_region_binary(self):
        """Test extracting masked region with binary mask."""
        from utils.sam import extract_masked_region

        # Create a simple image
        image = Image.new("RGB", (10, 10), color=(255, 0, 0))

        # Create a binary mask (half kept, half masked)
        mask = np.zeros((10, 10), dtype=bool)
        mask[:, :5] = True

        result = extract_masked_region(image, mask)

        # Check that left half is red, right half is black
        result_np = np.array(result)
        left_half = result_np[:, :5, :]
        right_half = result_np[:, 5:, :]

        assert np.all(left_half == [255, 0, 0])
        assert np.all(right_half == [0, 0, 0])

    def test_extract_masked_region_all_masked(self):
        """Test extracting when entire image is masked."""
        from utils.sam import extract_masked_region

        image = Image.new("RGB", (10, 10), color=(255, 0, 0))
        mask = np.ones((10, 10), dtype=bool)

        result = extract_masked_region(image, mask)
        result_np = np.array(result)

        # Entire image should be preserved
        assert np.all(result_np == [255, 0, 0])

    def test_extract_masked_region_all_zero_mask(self):
        """Test extracting when mask is all zeros."""
        from utils.sam import extract_masked_region

        image = Image.new("RGB", (10, 10), color=(255, 0, 0))
        mask = np.zeros((10, 10), dtype=bool)

        result = extract_masked_region(image, mask)
        result_np = np.array(result)

        # Entire image should be black
        assert np.all(result_np == [0, 0, 0])
