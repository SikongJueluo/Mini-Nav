import io
from typing import Any, Dict, List, Optional, Union, cast

import torch
from database import db_manager
from datasets import load_dataset
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from torch import nn
from tqdm.auto import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    BitImageProcessorFast,
    Dinov2Model,
)


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
        device = self.model.device
        self.model.eval()

        for i in tqdm(range(0, len(images), batch_size)):
            batch_imgs = images[i : i + batch_size]

            inputs = self.processor(batch_imgs, return_tensors="pt")

            # 迁移数据到GPU
            inputs.to(device)

            outputs = self.model(**inputs)

            # 后处理
            feats = outputs.last_hidden_state  # [B, N, D]
            cls_tokens = feats[:, 0]  # Get CLS token (first token) for all batch items
            cls_tokens = cast(torch.Tensor, cls_tokens)

            # 迁移输出到CPU
            cls_tokens = cls_tokens.cpu()
            batch_labels = (
                labels[i : i + batch_size]
                if label_map is None
                else list(
                    map(lambda x: label_map[cast(int, x)], labels[i : i + batch_size])
                )
            )
            actual_batch_size = len(batch_labels)

            # 存库
            db_manager.table.add(
                [
                    {
                        "id": i + j,
                        "label": batch_labels[j],
                        "vector": cls_tokens[j].numpy(),
                        "binary": pil_image_to_bytes(batch_imgs[j]),
                    }
                    for j in range(actual_batch_size)
                ]
            )

    @torch.no_grad()
    def extract_single_image_feature(
        self, image: Union[Image.Image, Any]
    ) -> List[float]:
        """Extract feature from a single image without storing to database.

        Args:
            image: A single image (PIL Image or other supported format).

        Returns:
            pl.Series: The extracted CLS token feature vector as a Polars Series.
        """
        device = self.model.device
        self.model.eval()

        # 预处理图片
        inputs = self.processor(images=image, return_tensors="pt")
        inputs.to(device, non_blocking=True)

        # 提取特征
        outputs = self.model(**inputs)

        # 获取 CLS token
        feats = outputs.last_hidden_state  # [1, N, D]
        cls_token = feats[:, 0]  # [1, D]
        cls_token = cast(torch.Tensor, cls_token)

        # 返回 CLS List
        return cls_token.cpu().squeeze(0).tolist()


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
