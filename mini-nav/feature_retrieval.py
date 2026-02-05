from typing import Any, Dict, List, Optional, cast

import torch
from database import db_manager
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel


class FeatureRetrieval:
    """Singleton feature retrieval manager for image feature extraction."""

    _instance: Optional["FeatureRetrieval"] = None

    _initialized: bool = False
    processor: Any
    model: Any

    def __new__(cls, *args, **kwargs) -> "FeatureRetrieval":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, processor: Optional[Any] = None, model: Optional[Any] = None
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
        images: List[Any],
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

            inputs = self.processor(images=batch_imgs, return_tensors="pt")

            # 迁移数据到GPU
            inputs.to(device, non_blocking=True)

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
                    }
                    for j in range(actual_batch_size)
                ]
            )


if __name__ == "__main__":
    train_dataset = load_dataset("uoft-cs/cifar10", split="train")
    train_dataset = cast(Dataset, train_dataset)
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

    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-large", device_map="cuda"
    )
    model = AutoModel.from_pretrained("facebook/dinov2-large", device_map="cuda")

    feature_retrieval = FeatureRetrieval(processor, model)

    feature_retrieval.establish_database(
        train_dataset["img"],
        train_dataset["label"],
        label_map=label_map,
    )
