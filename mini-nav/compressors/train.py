import torch
import torch.nn.functional as F
from compressors import FloatCompressor
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel


def train(dinov2: nn.Module, epoch_size: int, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset("uoft-cs/cifar10", split="train").with_format("torch")
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
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
        "facebook/dinov2-large", device_map=device
    )
    dino = AutoModel.from_pretrained("facebook/dinov2-large", device_map=device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False

    compressor = FloatCompressor().to(device)

    optimizer = torch.optim.AdamW(compressor.parameters(), lr=1e-4)

    for epoch in range(epoch_size):
        train_bar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epoch_size}]")

        for batch in train_bar:
            imgs = batch["img"]

            # ---- teacher forward ----
            with torch.no_grad():
                inputs = processor(imgs, return_tensors="pt").to(device)

                teacher_tokens = dino(**inputs).last_hidden_state
                # [B,N,1024]

                teacher_embed = teacher_tokens.mean(dim=1)
                teacher_embed = F.normalize(teacher_embed, dim=-1)
                # [B,1024]

            # ---- student forward ----
            z512, recon = compressor(teacher_tokens)

            # ---- loss ----
            mse_loss = F.mse_loss(recon, teacher_embed)

            cos_loss = 1 - F.cosine_similarity(recon, teacher_embed, dim=-1).mean()

            loss = mse_loss + cos_loss

            # ---- backward ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(loss=loss.item())
