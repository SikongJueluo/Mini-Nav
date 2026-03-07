"""Hash compression pipeline with DINO feature extraction.

This pipeline extracts features using DINOv2 and compresses them
to binary hash codes using HashCompressor.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def create_pipeline_from_config(config) -> "HashPipeline":
    """Create HashPipeline from a config object.

    Args:
        config: Configuration object with model settings

    Returns:
        Initialized HashPipeline
    """
    return HashPipeline(
        dino_model=config.model.dino_model,
        hash_bits=config.model.compression_dim,
        compressor_path=config.model.compressor_path,
        device=config.model.device if config.model.device != "auto" else None,
    )


class HashPipeline(nn.Module):
    """Pipeline: DINO features + Hash compression.

    Pipeline flow:
        PIL Image -> DINO (features) -> Hash (binary codes)

    Usage:
        # Initialize with config
        pipeline = HashPipeline(
            dino_model="facebook/dinov2-large",
            hash_bits=512,
        )

        # Process image
        image = Image.open("path/to/image.jpg")
        hash_bits = pipeline(image)  # [1, 512] binary bits
    """

    def __init__(
        self,
        dino_model: str = "facebook/dinov2-large",
        hash_bits: int = 512,
        compressor_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize the pipeline.

        Args:
            dino_model: DINOv2 model name from HuggingFace
            hash_bits: Number of bits in hash code
            compressor_path: Optional path to trained HashCompressor weights
            device: Device to run models on
        """
        super().__init__()

        # Auto detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.dino_model = dino_model

        # Initialize DINO processor and model
        self.processor = AutoImageProcessor.from_pretrained(dino_model)
        self.dino = AutoModel.from_pretrained(dino_model).to(self.device)
        self.dino.eval()

        # Determine DINO feature dimension
        self.dino_dim = 1024 if "large" in dino_model else 768

        # Initialize HashCompressor
        self.hash_compressor = nn.Module()  # Placeholder, will be replaced
        self._init_hash_compressor(hash_bits, compressor_path)

    def _init_hash_compressor(
        self, hash_bits: int, compressor_path: Optional[str] = None
    ):
        """Initialize the hash compressor module.

        This is called during __init__ but we need to replace it properly.
        """
        # Import here to avoid circular imports
        from .hash_compressor import HashCompressor

        compressor = HashCompressor(input_dim=self.dino_dim, hash_bits=hash_bits).to(
            self.device
        )

        # Load pretrained compressor if provided
        if compressor_path is not None:
            compressor.load_state_dict(
                torch.load(compressor_path, map_location=self.device)
            )
            print(f"[OK] Loaded HashCompressor from {compressor_path}")

        # Replace the placeholder
        self.hash_compressor = compressor

    @property
    def hash_bits(self):
        """Return the number of hash bits."""
        return self.hash_compressor.hash_bits

    def forward(self, image: Image.Image) -> torch.Tensor:
        """Process a single image through the pipeline.

        Args:
            image: Input PIL Image

        Returns:
            Binary hash codes [1, hash_bits] as int32
        """
        # Extract DINO features
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.dino(**inputs)
            tokens = outputs.last_hidden_state  # [1, N, dim]

        # Compress to hash codes
        _, _, bits = self.hash_compressor(tokens)

        return bits

    def encode(self, image: Image.Image) -> torch.Tensor:
        """Encode an image to binary hash bits.

        Alias for forward().

        Args:
            image: Input PIL Image

        Returns:
            Binary hash codes [1, hash_bits] as int32
        """
        return self.forward(image)

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """Extract DINO features from an image.

        Args:
            image: Input PIL Image

        Returns:
            DINO features [1, dino_dim], normalized
        """
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.dino(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)  # [1, dim]
            features = F.normalize(features, dim=-1)

        return features


# Backward compatibility alias
SAMHashPipeline = HashPipeline
