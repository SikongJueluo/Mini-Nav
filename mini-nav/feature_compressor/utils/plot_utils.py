"""Plotting utility functions for feature visualization."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_histogram(data: np.ndarray, title: str = None, **kwargs) -> go.Figure:
    """Create a histogram plot.

    Args:
        data: 1D array of values
        title: Plot title
        **kwargs: Additional histogram arguments

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data.flatten(),
            name="Feature Values",
            **kwargs,
        )
    )

    if title:
        fig.update_layout(title=title)

    fig.update_layout(
        xaxis_title="Value",
        yaxis_title="Count",
        hovermode="x unified",
    )

    return fig


def create_pca_scatter_2d(
    features: np.ndarray, labels: List = None, **kwargs
) -> go.Figure:
    """Create a 2D PCA scatter plot.

    Args:
        features: 2D array [n_samples, n_features]
        labels: Optional list of labels for coloring
        **kwargs: Additional scatter arguments

    Returns:
        Plotly Figure object
    """
    from sklearn.decomposition import PCA

    # Apply PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(features)

    explained_var = pca.explained_variance_ratio_ * 100

    fig = go.Figure()

    if labels is None:
        fig.add_trace(
            go.Scatter(
                x=components[:, 0],
                y=components[:, 1],
                mode="markers",
                marker=dict(size=8, opacity=0.7),
                **kwargs,
            )
        )
    else:
        for label in set(labels):
            mask = np.array(labels) == label
            fig.add_trace(
                go.Scatter(
                    x=components[mask, 0],
                    y=components[mask, 1],
                    mode="markers",
                    name=str(label),
                    marker=dict(size=8, opacity=0.7),
                )
            )

    fig.update_layout(
        title=f"PCA 2D Projection (Total Variance: {explained_var.sum():.1f}%)",
        xaxis_title=f"PC 1 ({explained_var[0]:.1f}%)",
        yaxis_title=f"PC 2 ({explained_var[1]:.1f}%)",
        hovermode="closest",
    )

    return fig


def create_comparison_plot(
    features_list: List[np.ndarray], names: List[str], **kwargs
) -> go.Figure:
    """Create a comparison plot of multiple feature sets.

    Args:
        features_list: List of feature arrays
        names: List of names for each feature set
        **kwargs: Additional histogram arguments

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(rows=1, cols=len(features_list), subplot_titles=names)

    for i, features in enumerate(features_list, 1):
        fig.add_trace(
            go.Histogram(
                x=features.flatten(),
                name=names[i - 1],
                showlegend=False,
                **kwargs,
            ),
            row=1,
            col=i,
        )

    fig.update_layout(
        title="Feature Distribution Comparison",
        hovermode="x unified",
    )

    return fig


def save_figure(fig: go.Figure, path: str, format: str = "html") -> None:
    """Save figure to file.

    Args:
        fig: Plotly Figure object
        path: Output file path (without extension)
        format: Output format ('html', 'png', 'json')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "html":
        fig.write_html(str(path) + ".html", include_plotlyjs="cdn")
    elif format == "png":
        fig.write_image(str(path) + ".png", scale=2)
    elif format == "json":
        fig.write_json(str(path) + ".json")
    else:
        raise ValueError(f"Unsupported format: {format}")


def apply_theme(fig: go.Figure, theme: str = "plotly_white") -> go.Figure:
    """Apply a theme to the figure.

    Args:
        fig: Plotly Figure object
        theme: Theme name

    Returns:
        Updated Plotly Figure object
    """
    fig.update_layout(template=theme)
    return fig
