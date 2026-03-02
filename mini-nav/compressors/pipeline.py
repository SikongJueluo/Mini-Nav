"""Complete pipeline for SAM + DINO + HashCompressor.

This pipeline extracts object masks from images using SAM2.1,
crops the objects, extracts features using DINOv2,
and compresses them to binary hash codes using HashCompressor.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image

from .dino_compressor import DinoCompressor
from .hash_compressor import HashCompressor
from .segament_compressor import SegmentCompressor


def create_pipeline_from_config(config) -> "SAMHashPipeline":
    """Create SAMHashPipeline from a config object.

    Args:
        config: Configuration object with model settings

    Returns:
        Initialized SAMHashPipeline
    """
    return SAMHashPipeline(
        sam_model=config.model.sam_model,
        dino_model=config.model.name,
        hash_bits=config.model.compression_dim,
        sam_min_mask_area=config.model.sam_min_mask_area,
        sam_max_masks=config.model.sam_max_masks,
        compressor_path=config.model.compressor_path,
        device=config.model.device if config.model.device != "auto" else None,
    )


class SAMHashPipeline(nn.Module):
    """Complete pipeline: SAM segmentation + DINO features + Hash compression.

    Pipeline flow:
        Image -> SAM (extract masks) -> Crop objects -> DINO (features) -> Hash (binary codes)

    Usage:
        # Initialize with config
        pipeline = SAMHashPipeline(
            sam_model="facebook/sam2.1-hiera-large",
            dino_model="facebook/dinov2-large",
            hash_bits=512,
        )

        # Process image
        image = Image.open("path/to/image.jpg")
        hash_codes = pipeline(image)  # [N, 512] binary bits
    """

    def __init__(
        self,
        sam_model: str = "facebook/sam2.1-hiera-large",
        dino_model: str = "facebook/dinov2-large",
        hash_bits: int = 512,
        sam_min_mask_area: int = 100,
        sam_max_masks: int = 10,
        compressor_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize the complete pipeline.

        Args:
            sam_model: SAM model name from HuggingFace
            dino_model: DINOv2 model name from HuggingFace
            hash_bits: Number of bits in hash code
            sam_min_mask_area: Minimum mask area threshold
            sam_max_masks: Maximum number of masks to keep
            compressor_path: Optional path to trained HashCompressor weights
            device: Device to run models on
        """
        super().__init__()

        # Auto detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Initialize components
        self.segmentor = SegmentCompressor(
            model_name=sam_model,
            min_mask_area=sam_min_mask_area,
            max_masks=sam_max_masks,
            device=device,
        )

        # HashCompressor expects DINO features (1024 dim for dinov2-large)
        dino_dim = 1024 if "large" in dino_model else 768
        self.hash_compressor = HashCompressor(
            input_dim=dino_dim, hash_bits=hash_bits
        ).to(device)

        # Load pretrained compressor if provided
        if compressor_path is not None:
            self.hash_compressor.load_state_dict(
                torch.load(compressor_path, map_location=device)
            )
            print(f"[OK] Loaded HashCompressor from {compressor_path}")

        self.dino = DinoCompressor(
            model_name=dino_model,
            compressor=self.hash_compressor,
            device=device,
        )

    def forward(self, image: Image.Image) -> torch.Tensor:
        """Process a single image through the complete pipeline.

        Args:
            image: Input PIL Image

        Returns:
            Binary hash codes [N, hash_bits] where N is number of detected objects
        """
        # Step 1: SAM - extract and crop objects
        cropped_objects = self.segmentor(image)

        if len(cropped_objects) == 0:
            # No objects detected, return empty tensor
            return torch.empty(
                0, self.hash_compressor.hash_bits, dtype=torch.int32, device=self.device
            )

        # Step 2: DINO - extract features from cropped objects
        # Step 3: HashCompressor - compress features to binary codes
        hash_codes = self.dino.encode(cropped_objects)

        return hash_codes

    def extract_features(
        self, image: Image.Image, use_hash: bool = False
    ) -> torch.Tensor:
        """Extract features from image with optional hash compression.

        Args:
            image: Input PIL Image
            use_hash: If True, return binary hash codes; else return DINO features

        Returns:
            Features [N, dim] where dim is 1024 (DINO) or 512 (hash)
        """
        cropped_objects = self.segmentor(image)

        if len(cropped_objects) == 0:
            dim = self.hash_compressor.hash_bits if use_hash else 1024
            return torch.empty(0, dim, device=self.device)

        if use_hash:
            return self.dino.encode(cropped_objects)
        else:
            return self.dino.extract_features(cropped_objects)

    def extract_masks(self, image: Image.Image) -> list[torch.Tensor]:
        """Extract only masks without full processing (for debugging).

        Args:
            image: Input PIL Image

        Returns:
            List of binary masks [H, W]
        """
        return self.segmentor.extract_masks(image)
