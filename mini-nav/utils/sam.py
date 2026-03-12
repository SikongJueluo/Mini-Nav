"""SAM (Segment Anything Model) utilities for object segmentation."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def load_sam_model(
    model_name: str = "facebook/sam2.1-hiera-large",
    device: str = "cuda",
    checkpoint_dir: Path | None = None,
) -> tuple[Any, Any]:
    """Load SAM 2.1 model and mask generator.

    Args:
        model_name: SAM model name (currently supports facebook/sam2.1-hiera-*).
        device: Device to load model on (cuda or cpu).
        checkpoint_dir: Optional directory for model checkpoint cache.

    Returns:
        Tuple of (sam_model, mask_generator).
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Build SAM2 model
    sam_model = build_sam2(model_name, device=device)

    # Create automatic mask generator
    mask_generator = SAM2AutomaticMaskGenerator(sam_model)

    return sam_model, mask_generator


def segment_image(
    mask_generator: Any,
    image: Image.Image,
    min_area: int = 32 * 32,
    max_masks: int = 5,
) -> list[dict[str, Any]]:
    """Segment image using SAM to extract object masks.

    Args:
        mask_generator: SAM2AutomaticMaskGenerator instance.
        image: PIL Image to segment.
        min_area: Minimum mask area threshold in pixels.
        max_masks: Maximum number of masks to return.

    Returns:
        List of mask dictionaries with keys:
            - segment: Binary mask (numpy array)
            - area: Mask area in pixels
            - bbox: Bounding box [x, y, width, height]
            - predicted_iou: Model's confidence in the mask
            - stability_score: Stability score for the mask
    """
    # Convert PIL Image to numpy array
    image_np = np.array(image.convert("RGB"))

    # Generate masks
    masks = mask_generator.generate(image_np)

    if not masks:
        return []

    # Filter by minimum area
    filtered_masks = [m for m in masks if m["area"] >= min_area]

    if not filtered_masks:
        return []

    # Sort by area (largest first) and limit to max_masks
    sorted_masks = sorted(filtered_masks, key=lambda x: x["area"], reverse=True)
    return sorted_masks[:max_masks]


def extract_masked_region(
    image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """Extract masked region from image.

    Args:
        image: Original PIL Image.
        mask: Binary mask as numpy array (True = keep).

    Returns:
        PIL Image with only the masked region visible.
    """
    image_np = np.array(image.convert("RGB"))

    # Apply mask
    masked_np = image_np * mask[:, :, np.newaxis]

    return Image.fromarray(masked_np.astype(np.uint8))
