"""Visualization example for DINOv2 Feature Compressor."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from dino_feature_compressor import FeatureVisualizer


def main():
    # Generate synthetic features for demonstration
    print("Generating synthetic features...")
    n_samples = 100
    n_features = 256

    # Create two clusters
    cluster1 = np.random.randn(50, n_features) + 2
    cluster2 = np.random.randn(50, n_features) - 2
    features = np.vstack([cluster1, cluster2])

    labels = ["Cluster A"] * 50 + ["Cluster B"] * 50

    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Initialize visualizer
    print("Initializing FeatureVisualizer...")
    viz = FeatureVisualizer()

    output_dir = Path(__file__).parent.parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create histogram
    print("Creating histogram...")
    fig_hist = viz.plot_histogram(features_tensor, title="Feature Distribution")
    viz.save(fig_hist, str(output_dir / "feature_histogram"), formats=["html", "json"])
    print(f"Saved histogram to {output_dir / 'feature_histogram.html'}")

    # Create PCA 2D projection
    print("Creating PCA 2D projection...")
    fig_pca = viz.plot_pca_2d(features_tensor, labels=labels)
    viz.save(fig_pca, str(output_dir / "feature_pca_2d"), formats=["html", "json"])
    print(f"Saved PCA to {output_dir / 'feature_pca_2d.html'}")

    # Create comparison plot
    print("Creating comparison plot...")
    features_list = [torch.tensor(cluster1), torch.tensor(cluster2)]
    names = ["Cluster A", "Cluster B"]
    fig_comp = viz.plot_comparison(features_list, names)
    viz.save(fig_comp, str(output_dir / "feature_comparison"), formats=["html", "json"])
    print(f"Saved comparison to {output_dir / 'feature_comparison.html'}")

    print("\nDone! All visualizations saved to outputs/ directory.")


if __name__ == "__main__":
    main()
