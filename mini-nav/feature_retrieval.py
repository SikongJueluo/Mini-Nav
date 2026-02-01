from typing import cast
import torch
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset
from transformers import AutoImageProcessor, AutoModel


@torch.no_grad()
def establish_database(processor, model, images, batch_size=64):
    device = model.device
    model.eval()

    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i : i + batch_size]

        inputs = processor(images=batch, return_tensors="pt")

        # 迁移数据到GPU
        inputs.to(device, non_blocking=True)

        outputs = model(**inputs)
        feats = outputs.last_hidden_state  # [B, N, D]
        # 后处理 / 存库

        


if __name__ == "__main__":
    train_dataset = load_dataset("uoft-cs/cifar10", split="train")
    train_dataset = cast(Dataset, train_dataset)

    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-large", device_map="cuda"
    )
    model = AutoModel.from_pretrained("facebook/dinov2-large", device_map="cuda")

    establish_database(processor, model, train_dataset["img"])
