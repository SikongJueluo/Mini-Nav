"""Feature visualization using Plotly."""

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from configs import FeatureCompressorConfig, cfg_manager, load_yaml
from plotly.graph_objs import Figure

from ..utils.plot_utils import (
    apply_theme,
    create_comparison_plot,
    create_histogram,
    create_pca_scatter_2d,
    save_figure,
)


class FeatureVisualizer:
    """Visualize DINOv2 features with interactive Plotly charts.

    Supports histograms, PCA projections, and feature comparisons
    with multiple export formats.

    Args:
        config_path: Path to YAML configuration file
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config: FeatureCompressorConfig = self._load_config(config_path)

    def _load_config(
        self, config_path: Optional[str] = None
    ) -> FeatureCompressorConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file, or None for default

        Returns:
            Configuration Pydantic model
        """
        if config_path is None:
            return cfg_manager.get_or_load_config("feature_compressor")
        else:
            return load_yaml(Path(config_path), FeatureCompressorConfig)

    def plot_histogram(
        self, features: torch.Tensor, title: Optional[str] = None
    ) -> Figure:
        """Plot histogram of feature values.

        Args:
            features: Feature tensor [batch, dim]
            title: Plot title

        Returns:
            Plotly Figure object
        """
        features_np = features.cpu().numpy()
        fig = create_histogram(
            features_np, title="Feature Histogram" if title is None else title
        )

        fig = apply_theme(fig, self.config.visualization.plot_theme)
        fig.update_layout(
            width=self.config.visualization.fig_width,
            height=self.config.visualization.fig_height,
        )

        return fig

    def plot_pca_2d(
        self, features: torch.Tensor, labels: Optional[List] = None
    ) -> Figure:
        """Plot 2D PCA projection of features.

        Args:
            features: Feature tensor [n_samples, dim]
            labels: Optional labels for coloring

        Returns:
            Plotly Figure object
        """
        features_np = features.cpu().numpy()

        fig = create_pca_scatter_2d(
            features_np,
            labels=[i for i in range(len(features_np))] if labels is None else labels,
        )
        fig = apply_theme(fig, self.config.visualization.plot_theme)
        fig.update_traces(
            marker=dict(
                size=self.config.visualization.point_size,
                colorscale=self.config.visualization.color_scale,
            )
        )
        fig.update_layout(
            width=self.config.visualization.fig_width,
            height=self.config.visualization.fig_height,
        )

        return fig

    def plot_comparison(
        self, features_list: List[torch.Tensor], names: List[str]
    ) -> object:
        """Plot comparison of multiple feature sets.

        Args:
            features_list: List of feature tensors
            names: Names for each feature set

        Returns:
            Plotly Figure object
        """
        features_np_list = [f.cpu().numpy() for f in features_list]

        fig = create_comparison_plot(features_np_list, names)

        fig = apply_theme(fig, self.config.visualization.plot_theme)
        fig.update_layout(
            width=self.config.visualization.fig_width * len(features_list),
            height=self.config.visualization.fig_height,
        )

        return fig

    def generate_report(
        self, results: List[dict], output_dir: Union[str, Path]
    ) -> List[str]:
        """Generate full feature analysis report.

        Args:
            results: List of extractor results
            output_dir: Directory to save visualizations

        Returns:
            List of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        # Extract all compressed features
        all_features = torch.cat([r["compressed_features"] for r in results], dim=0)

        # Create histogram
        hist_fig = self.plot_histogram(all_features, "Compressed Feature Distribution")
        hist_path = output_dir / "feature_histogram"
        self.save(hist_fig, str(hist_path), formats=["html"])
        generated_files.append(str(hist_path) + ".html")

        # Create PCA
        pca_fig = self.plot_pca_2d(all_features)
        pca_path = output_dir / "feature_pca_2d"
        self.save(pca_fig, str(pca_path), formats=["html", "png"])
        generated_files.append(str(pca_path) + ".html")
        generated_files.append(str(pca_path) + ".png")

        return generated_files

    def save(self, fig: Figure, path: str, formats: List[str]) -> None:
        """Save figure in multiple formats.

        Args:
            fig: Plotly Figure object
            path: Output file path (without extension)
            formats: List of formats to export
        """
        if formats is None:
            formats = ["html"]

        for fmt in formats:
            if fmt == "png":
                save_figure(fig, path, format="png")
            else:
                save_figure(fig, path, format=fmt)
