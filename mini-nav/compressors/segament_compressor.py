"""Segment Anything 2 feature extractor with mask filtering and image cropping.

Extracts object masks from images using SAM2.1, filters by area and confidence,
then crops the original image to obtain individual object regions.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor


class SegmentCompressor(nn.Module):
    """SAM2.1 based segmenter with mask filtering.

    Extracts object masks from images, filters by area and confidence,
    and crops the original image to produce individual object patches.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam2.1-hiera-large",
        min_mask_area: int = 100,
        max_masks: int = 10,
        device: Optional[str] = None,
    ):
        """Initialize SAM2.1 segmenter.

        Args:
            model_name: HuggingFace model name for SAM2.1
            min_mask_area: Minimum mask pixel area threshold
            max_masks: Maximum number of masks to keep
            device: Device to load model on (auto-detect if None)
        """
        super().__init__()

        self.model_name = model_name
        self.min_mask_area = min_mask_area
        self.max_masks = max_masks

        # Auto detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load SAM model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForMaskGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

    def forward(self, image: Image.Image) -> list[Image.Image]:
        """Extract object masks and crop object regions.

        Args:
            image: Input PIL Image

        Returns:
            List of cropped object images (one per valid mask)
        """
        # Run SAM inference
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process masks
        masks = self.processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )[0]

        # Filter masks by area and confidence
        valid_masks = self._filter_masks(masks)

        if len(valid_masks) == 0:
            return []

        # Crop object regions from original image
        cropped_objects = self._crop_objects(image, valid_masks)

        return cropped_objects

    def _filter_masks(self, masks: torch.Tensor) -> list[dict]:
        """Filter masks by area and keep top-N.

        Args:
            masks: Predicted masks [N, H, W]

        Returns:
            List of mask dictionaries with 'mask' and 'area'
        """
        valid_masks = []

        for mask in masks:
            # Calculate mask area
            area = mask.sum().item()

            # Filter by minimum area
            if area < self.min_mask_area:
                continue

            valid_masks.append({"mask": mask, "area": area})

        # Sort by area (descending) and keep top-N
        valid_masks = sorted(valid_masks, key=lambda x: x["area"], reverse=True)
        valid_masks = valid_masks[: self.max_masks]

        return valid_masks

    def _crop_objects(
        self, image: Image.Image, masks: list[dict]
    ) -> list[Image.Image]:
        """Crop object regions from image using masks.

        Args:
            image: Original PIL Image
            masks: List of mask dictionaries

        Returns:
            List of cropped object images
        """
        # Convert PIL to numpy for processing
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        cropped_objects = []

        for mask_info in masks:
            mask = mask_info["mask"].cpu().numpy()

            # Find bounding box from mask
            rows = mask.any(axis=1)
            cols = mask.any(axis=0)

            if not rows.any() or not cols.any():
                continue

            y_min, y_max = rows.argmax(), h - rows[::-1].argmax() - 1
            x_min, x_max = cols.argmax(), w - cols[::-1].argmax() - 1

            # Add small padding
            pad = 5
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)

            # Crop
            cropped = image.crop((x_min, y_min, x_max, y_max))
            cropped_objects.append(cropped)

        return cropped_objects

    @torch.no_grad()
    def extract_masks(self, image: Image.Image) -> list[torch.Tensor]:
        """Extract only masks without cropping (for debugging).

        Args:
            image: Input PIL Image

        Returns:
            List of binary masks [H, W]
        """
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        masks = self.processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )[0]

        valid_masks = self._filter_masks(masks)
        return [m["mask"] for m in valid_masks]
