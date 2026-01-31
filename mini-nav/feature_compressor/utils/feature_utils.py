"""Feature processing utilities."""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """L2-normalize features.

    Args:
        features: Tensor of shape [batch, dim] or [batch, seq, dim]

    Returns:
        L2-normalized features
    """
    norm = torch.norm(features, p=2, dim=-1, keepdim=True)
    return features / (norm + 1e-8)


def compute_feature_stats(features: torch.Tensor) -> Dict[str, float]:
    """Compute basic statistics for features.

    Args:
        features: Tensor of shape [batch, dim] or [batch, seq, dim]

    Returns:
        Dictionary with mean, std, min, max
    """
    with torch.no_grad():
        return {
            "mean": float(features.mean().item()),
            "std": float(features.std().item()),
            "min": float(features.min().item()),
            "max": float(features.max().item()),
        }


def save_features_to_json(
    features: torch.Tensor, path: Path, metadata: Dict = None
) -> None:
    """Save features to JSON file.

    Args:
        features: Tensor to save
        path: Output file path
        metadata: Optional metadata dictionary
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    features_np = features.cpu().numpy()

    data = {
        "features": features_np.tolist(),
        "shape": list(features.shape),
    }

    if metadata:
        data["metadata"] = metadata

    with open(path, "w") as f:
        import json

        json.dump(data, f, indent=2)


def save_features_to_csv(features: torch.Tensor, path: Path) -> None:
    """Save features to CSV file.

    Args:
        features: Tensor to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    features_np = features.cpu().numpy()

    np.savetxt(path, features_np, delimiter=",", fmt="%.6f")
