from typing import Any, Dict, List, Optional, cast

import torch
from database import db_manager
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel


@torch.no_grad()
def establish_database(
    processor,
    model,
    images: List[Any],
    labels: List[int] | List[str],
    batch_size=64,
    label_map: Optional[Dict[int, str] | List[str]] = None,
):
    device = model.device
    model.eval()

    for i in tqdm(range(0, len(images), batch_size)):
        batch_imgs = images[i : i + batch_size]

        inputs = processor(images=batch_imgs, return_tensors="pt")

        # 迁移数据到GPU
        inputs.to(device, non_blocking=True)

        outputs = model(**inputs)

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
                {"id": i + j, "label": batch_labels[j], "vector": cls_tokens[j].numpy()}
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

    establish_database(
        processor,
        model,
        train_dataset["img"],
        train_dataset["label"],
        label_map=label_map,
    )
