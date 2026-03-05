"""Feature extraction utilities for image models."""

from typing import Any, List, Union, cast

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from transformers import BitImageProcessorFast
from tqdm.auto import tqdm


def _extract_features_from_output(output: Any) -> torch.Tensor:
    """Extract features from model output, handling both HuggingFace ModelOutput and raw tensors.

    Args:
        output: Model output (either ModelOutput with .last_hidden_state or raw tensor).

    Returns:
        Feature tensor of shape [B, D].
    """
    # Handle HuggingFace ModelOutput (has .last_hidden_state)
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]  # [B, D] - CLS token
    # Handle raw tensor output (like DinoCompressor)
    return cast(torch.Tensor, output)


def infer_vector_dim(
    processor: BitImageProcessorFast,
    model: nn.Module,
    sample_image: Any,
) -> int:
    """Infer model output vector dimension via a single forward pass.

    Args:
        processor: Image preprocessor.
        model: Feature extraction model.
        sample_image: A sample image for dimension inference.

    Returns:
        Vector dimension.
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        inputs = processor(images=sample_image, return_tensors="pt")
        inputs.to(device)
        output = model(inputs)

    features = _extract_features_from_output(output)
    return features.shape[-1]


@torch.no_grad()
def extract_single_image_feature(
    processor: BitImageProcessorFast,
    model: nn.Module,
    image: Union[Image.Image, Any],
) -> List[float]:
    """Extract feature from a single image.

    Args:
        processor: Image preprocessor.
        model: Feature extraction model.
        image: A single image (PIL Image or other supported format).

    Returns:
        The extracted CLS token feature vector as a list of floats.
    """
    device = next(model.parameters()).device
    model.eval()

    inputs = processor(images=image, return_tensors="pt")
    inputs.to(device, non_blocking=True)
    outputs = model(inputs)

    features = _extract_features_from_output(outputs)  # [1, D]
    return features.cpu().squeeze(0).tolist()


@torch.no_grad()
def extract_batch_features(
    processor: BitImageProcessorFast,
    model: nn.Module,
    images: Union[List[Any], Any],
    batch_size: int = 32,
    show_progress: bool = False,
) -> torch.Tensor:
    """Extract features from a batch of images.

    Args:
        processor: Image preprocessor.
        model: Feature extraction model.
        images: List of images, DataLoader, or other iterable.
        batch_size: Batch size for processing.
        show_progress: Whether to show progress bar.

    Returns:
        Tensor of shape [batch_size, feature_dim].
    """
    device = next(model.parameters()).device
    model.eval()

    # Handle DataLoader input
    if isinstance(images, DataLoader):
        all_features = []
        iterator = tqdm(images, desc="Extracting features") if show_progress else images
        for batch in iterator:
            imgs = batch["img"] if isinstance(batch, dict) else batch[0]
            inputs = processor(images=imgs, return_tensors="pt")
            inputs.to(device)
            outputs = model(inputs)
            features = _extract_features_from_output(outputs)  # [B, D]
            all_features.append(features.cpu())
        return torch.cat(all_features, dim=0)

    # Handle list of images
    all_features = []
    iterator = tqdm(range(0, len(images), batch_size), desc="Extracting features") if show_progress else range(0, len(images), batch_size)
    for i in iterator:
        batch_imgs = images[i : i + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt")
        inputs.to(device)
        outputs = model(inputs)
        features = _extract_features_from_output(outputs)  # [B, D]
        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)
