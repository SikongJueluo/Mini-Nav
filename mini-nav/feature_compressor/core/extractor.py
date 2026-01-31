"""DINOv2 feature extraction and compression pipeline."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from configs import FeatureCompressorConfig, cfg_manager, load_yaml
from transformers import AutoImageProcessor, AutoModel

from ..utils.image_utils import load_image, preprocess_image
from .compressor import PoolNetCompressor


class DINOv2FeatureExtractor:
    """End-to-end DINOv2 feature extraction with compression.

    Loads DINOv2 model, extracts last_hidden_state features,
    and applies PoolNetCompressor for dimensionality reduction.

    Args:
        config_path: Path to YAML configuration file
        device: Device to use ('auto', 'cpu', or 'cuda')
    """

    def __init__(self, config_path: Optional[str] = None, device: str = "auto"):
        self.config: FeatureCompressorConfig = self._load_config(config_path)

        # Set device
        if device == "auto":
            device = self.config.model.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load DINOv2 model and processor
        model_name = self.config.model.name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Initialize compressor
        self.compressor = PoolNetCompressor(
            input_dim=self.model.config.hidden_size,
            compression_dim=self.config.model.compression_dim,
            top_k_ratio=self.config.model.top_k_ratio,
            hidden_ratio=self.config.model.hidden_ratio,
            dropout_rate=self.config.model.dropout_rate,
            use_residual=self.config.model.use_residual,
            device=str(self.device),
        )

    def _load_config(
        self, config_path: Optional[str] = None
    ) -> FeatureCompressorConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file, or None for default

        Returns:
            FeatureCompressorConfig instance
        """
        if config_path is None:
            return cfg_manager.get_or_load_config("feature_compressor")
        else:
            return load_yaml(Path(config_path), FeatureCompressorConfig)

    def _extract_dinov2_features(self, images: List) -> torch.Tensor:
        """Extract DINOv2 last_hidden_state features.

        Args:
            images: List of PIL Images

        Returns:
            last_hidden_state [batch, seq_len, hidden_dim]
        """
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state

        return features

    def _compress_features(self, features: torch.Tensor) -> torch.Tensor:
        """Compress features using PoolNetCompressor.

        Args:
            features: [batch, seq_len, hidden_dim]

        Returns:
            compressed [batch, compression_dim]
        """
        with torch.no_grad():
            compressed = self.compressor(features)

        return compressed

    def process_image(
        self, image_path: str, visualize: bool = False
    ) -> Dict[str, object]:
        """Process a single image and extract compressed features.

        Args:
            image_path: Path to image file
            visualize: Whether to generate visualizations

        Returns:
            Dictionary with original_features, compressed_features, metadata
        """
        start_time = time.time()

        # Load and preprocess image
        image = load_image(image_path)
        image = preprocess_image(image, size=224)

        # Extract DINOv2 features
        original_features = self._extract_dinov2_features([image])

        # Compute feature stats for compression ratio
        original_dim = original_features.shape[-1]
        compressed_dim = self.compressor.compression_dim
        compression_ratio = original_dim / compressed_dim

        # Compress features
        compressed_features = self._compress_features(original_features)

        # Get pooled features (before compression) for analysis
        pooled_features = self.compressor._apply_pooling(
            original_features,
            self.compressor._compute_attention_scores(original_features),
        )
        pooled_features = pooled_features.mean(dim=1)

        # Compute feature norm
        feature_norm = torch.norm(compressed_features, p=2, dim=-1).mean().item()

        processing_time = time.time() - start_time

        # Build result dictionary
        result = {
            "original_features": original_features.cpu(),
            "compressed_features": compressed_features.cpu(),
            "pooled_features": pooled_features.cpu(),
            "metadata": {
                "image_path": str(image_path),
                "compression_ratio": compression_ratio,
                "processing_time": processing_time,
                "feature_norm": feature_norm,
                "device": str(self.device),
                "model_name": self.config.model.name,
            },
        }

        return result

    def process_batch(
        self,
        image_dir: Union[str, Path],
        batch_size: int = 8,
        save_features: bool = True,
    ) -> List[Dict[str, object]]:
        """Process multiple images in batches.

        Args:
            image_dir: Directory containing images
            batch_size: Number of images per batch
            save_features: Whether to save features to disk

        Returns:
            List of result dictionaries, one per image
        """
        image_dir = Path(image_dir)
        image_files = sorted(image_dir.glob("*.*"))

        results = []

        # Process in batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i : i + batch_size]

            # Load and preprocess batch
            images = [preprocess_image(load_image(f), size=224) for f in batch_files]

            # Extract features for batch
            original_features = self._extract_dinov2_features(images)
            compressed_features = self._compress_features(original_features)

            # Create individual results
            for j, file_path in enumerate(batch_files):
                pooled_features = self.compressor._apply_pooling(
                    original_features[j : j + 1],
                    self.compressor._compute_attention_scores(
                        original_features[j : j + 1]
                    ),
                ).mean(dim=1)

                result = {
                    "original_features": original_features[j : j + 1].cpu(),
                    "compressed_features": compressed_features[j : j + 1].cpu(),
                    "pooled_features": pooled_features.cpu(),
                    "metadata": {
                        "image_path": str(file_path),
                        "compression_ratio": original_features.shape[-1]
                        / self.compressor.compression_dim,
                        "processing_time": 0.0,
                        "feature_norm": torch.norm(
                            compressed_features[j : j + 1], p=2, dim=-1
                        )
                        .mean()
                        .item(),
                        "device": str(self.device),
                        "model_name": self.config.model.name,
                    },
                }

                results.append(result)

                # Save features if requested
                if save_features:
                    output_dir = Path(self.config.output.directory)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    output_path = output_dir / f"{file_path.stem}_features.json"
                    from ..utils.feature_utils import save_features_to_json

                    save_features_to_json(
                        result["compressed_features"],
                        output_path,
                        result["metadata"],
                    )

        return results
