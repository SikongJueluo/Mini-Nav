"""Tests for FeatureVisualizer module."""

import os
import tempfile

import numpy as np
import pytest
import torch
from feature_compressor.core.visualizer import FeatureVisualizer


class TestFeatureVisualizer:
    """Test suite for FeatureVisualizer class."""

    def test_histogram_generation(self):
        """Test histogram generation from features."""

        viz = FeatureVisualizer()
        features = torch.randn(20, 256)

        fig = viz.plot_histogram(features, title="Test Histogram")

        assert fig is not None
        assert "Test Histogram" in fig.layout.title.text

    def test_pca_2d_generation(self):
        """Test PCA 2D scatter plot generation."""

        viz = FeatureVisualizer()
        features = torch.randn(20, 256)
        labels = ["cat"] * 10 + ["dog"] * 10

        fig = viz.plot_pca_2d(features, labels=labels)

        assert fig is not None
        assert "PCA 2D" in fig.layout.title.text

    def test_comparison_plot_generation(self):
        """Test comparison plot generation."""

        viz = FeatureVisualizer()
        features_list = [torch.randn(20, 256), torch.randn(20, 256)]
        names = ["Set A", "Set B"]

        fig = viz.plot_comparison(features_list, names)

        assert fig is not None
        assert "Comparison" in fig.layout.title.text

    def test_html_export(self):
        """Test HTML export format."""

        viz = FeatureVisualizer()
        features = torch.randn(10, 256)

        fig = viz.plot_histogram(features)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_plot")
            viz.save(fig, output_path, formats=["html"])

            assert os.path.exists(output_path + ".html")

    def test_png_export(self):
        """Test PNG export format."""

        viz = FeatureVisualizer()
        features = torch.randn(10, 256)

        fig = viz.plot_histogram(features)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_plot")

            # Skip PNG export if Chrome not available
            try:
                viz.save(fig, output_path, formats=["png"])
                assert os.path.exists(output_path + ".png")
            except RuntimeError as e:
                if "Chrome" in str(e):
                    pass
                else:
                    raise

    def test_json_export(self):
        """Test JSON export format."""

        viz = FeatureVisualizer()
        features = torch.randn(10, 256)

        fig = viz.plot_histogram(features)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_plot")
            viz.save(fig, output_path, formats=["json"])

            assert os.path.exists(output_path + ".json")
