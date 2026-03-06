import io
from typing import Dict, List, Optional, cast

import torch
from database import db_manager
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from torch import nn
from rich.progress import track
from transformers import (
    AutoImageProcessor,
    AutoModel,
    BitImageProcessorFast,
    Dinov2Model,
)
from utils.feature_extractor import extract_batch_features

from datasets import load_dataset


def pil_image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert a PIL Image to bytes in the specified format.

    Args:
        image: PIL Image to convert.
        format: Image format (e.g., 'PNG', 'JPEG').

    Returns:
        bytes: The encoded image bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


class FeatureRetrieval:
    """Singleton feature retrieval manager for image feature extraction."""

    _instance: Optional["FeatureRetrieval"] = None

    _initialized: bool = False
    processor: BitImageProcessorFast
    model: nn.Module

    def __new__(cls, *args, **kwargs) -> "FeatureRetrieval":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        processor: Optional[BitImageProcessorFast] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the singleton with processor and model.

        Args:
            processor: Image processor for preprocessing images.
            model: Model for feature extraction.
        """
        # 如果已经初始化过，直接返回
        if self._initialized:
            return

        # 首次初始化时必须提供 processor 和 model
        if processor is None or model is None:
            raise ValueError(
                "Processor and model must be provided on first initialization."
            )

        self.processor = processor
        self.model = model
        self._initialized = True

    @torch.no_grad()
    def establish_database(
        self,
        images: List[PngImageFile],
        labels: List[int] | List[str],
        batch_size: int = 64,
        label_map: Optional[Dict[int, str] | List[str]] = None,
    ) -> None:
        """Extract features from images and store them in the database.

        Args:
            images: List of images to process.
            labels: List of labels corresponding to images.
            batch_size: Number of images to process in a batch.
            label_map: Optional mapping from label indices to string names.
        """
        # Extract features using the utility function
        cls_tokens = extract_batch_features(
            self.processor, self.model, images, batch_size=batch_size
        )

        for i in track(range(len(labels)), description="Storing to database"):
            batch_label = labels[i] if label_map is None else label_map[labels[i]]

            # Store to database
            db_manager.table.add(
                [
                    {
                        "id": i,
                        "label": batch_label,
                        "vector": cls_tokens[i].numpy(),
                        "binary": pil_image_to_bytes(images[i]),
                    }
                ]
            )


if __name__ == "__main__":
    train_dataset = load_dataset("uoft-cs/cifar10", split="train")
    label_map = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    processor = cast(
        BitImageProcessorFast,
        AutoImageProcessor.from_pretrained("facebook/dinov2-large", device_map="cuda"),
    )
    model = cast(
        Dinov2Model,
        AutoModel.from_pretrained("facebook/dinov2-large", device_map="cuda"),
    )

    feature_retrieval = FeatureRetrieval(processor, model)

    feature_retrieval.establish_database(
        train_dataset["img"],
        train_dataset["label"],
        label_map=label_map,
    )
