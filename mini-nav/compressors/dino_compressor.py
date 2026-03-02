from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DinoCompressor(nn.Module):
    """DINOv2 feature extractor with optional hash compression.

    When compressor is None: returns normalized DINO embeddings.
    When compressor is provided: returns binary hash bits for CAM storage.

    Supports both PIL Image input and pre-extracted tokens.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        compressor: Optional[nn.Module] = None,
        device: Optional[str] = None,
    ):
        """Initialize DINOv2 extractor.

        Args:
            model_name: HuggingFace model name
            compressor: Optional hash compressor for producing binary codes
            device: Device to load model on
        """
        super().__init__()

        # Auto detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.dino = AutoModel.from_pretrained(model_name).to(self.device)
        self.dino.eval()

        self.compressor = compressor

    def forward(self, inputs):
        teacher_tokens = self.dino(**inputs).last_hidden_state  # [B,N,1024]

        teacher_embed = teacher_tokens.mean(dim=1)
        teacher_embed = F.normalize(teacher_embed, dim=-1)  # [B,1024]

        if self.compressor is None:
            return teacher_embed

        # HashCompressor returns (logits, hash_codes, bits)
        _, _, bits = self.compressor(teacher_tokens)
        return bits  # [B, 512] binary bits for CAM

    def extract_features(self, images: list[Image.Image]) -> torch.Tensor:
        """Extract DINO features from a list of cropped object images.

        Args:
            images: List of PIL Images (cropped objects)

        Returns:
            DINO features [N, feature_dim], normalized
        """
        if len(images) == 0:
            return torch.empty(0, self.dino.config.hidden_size, device=self.device)

        # Process batch of images
        inputs = self.processor(images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.dino(**inputs)

        # Pool tokens to get global representation
        features = outputs.last_hidden_state.mean(dim=1)  # [N, 1024]
        features = F.normalize(features, dim=-1)

        return features

    def encode(self, images: list[Image.Image]) -> torch.Tensor:
        """Extract features from images and optionally compress to hash codes.

        Args:
            images: List of PIL Images

        Returns:
            If compressor is None: DINO features [N, 1024]
            If compressor is set: Binary hash bits [N, 512]
        """
        if self.compressor is None:
            return self.extract_features(images)

        # Extract features first
        features = self.extract_features(images)  # [N, 1024]

        # Add sequence dimension for compressor (expects [B, N, dim])
        features = features.unsqueeze(1)  # [N, 1, 1024]

        # Compress to hash codes
        _, _, bits = self.compressor(features)

        return bits
