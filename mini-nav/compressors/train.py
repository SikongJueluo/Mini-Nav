import os

import torch
import torch.nn.functional as F
from compressors import FloatCompressor
from configs import cfg_manager
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel


def save_checkpoint(model: nn.Module, optimizer, epoch, step, path="checkpoint.pt"):
    config = cfg_manager.get()
    path = config.output.directory / path

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    torch.save(ckpt, path)
    print(f"✅ Saved checkpoint to {path}")


def load_checkpoint(model: nn.Module, optimizer, path="checkpoint.pt"):
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    start_epoch = ckpt["epoch"]
    start_step = ckpt["step"]

    print(f"✅ Loaded checkpoint from {path}")
    print(f"➡️ Resume from epoch={start_epoch}, step={start_step}")

    return start_epoch, start_step


def train(
    dinov2: nn.Module, epoch_size: int, batch_size: int, checkpoint_path="checkpoint.pt"
):
    # Auto dectect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Global variables
    save_every = 500
    start_epoch = 0
    global_step = 0

    # Load dataset
    ds = load_dataset("uoft-cs/cifar10", split="train").with_format("torch")
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load processor
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-large", device_map=device
    )

    # Load model
    dino = AutoModel.from_pretrained("facebook/dinov2-large", device_map=device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False

    # Load compressor model
    compressor = FloatCompressor().to(device)

    # Load optimizer
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=1e-4)

    # Auto load checkpoint
    output_dir = cfg_manager.get().output.directory
    if os.path.exists(output_dir / checkpoint_path):
        start_epoch, global_step = load_checkpoint(
            compressor, optimizer, output_dir / checkpoint_path
        )

    try:
        for epoch in range(start_epoch, epoch_size):
            train_bar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epoch_size}]")

            for batch in train_bar:
                global_step += 1

                # ---- training step ----
                imgs = batch["img"]

                # ---- teacher forward ----
                with torch.no_grad():
                    inputs = processor(imgs, return_tensors="pt").to(device)

                    teacher_tokens = dino(**inputs).last_hidden_state  # [B,N,1024]

                    teacher_embed = teacher_tokens.mean(dim=1)
                    teacher_embed = F.normalize(teacher_embed, dim=-1)  # [B,1024]

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

                # ---- periodic save ----
                if global_step % save_every == 0:
                    save_checkpoint(compressor, optimizer, epoch, global_step)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted, saving checkpoint...")

        save_checkpoint(compressor, optimizer, epoch, global_step)

        print("✅ Checkpoint saved. Exiting.")
        return

    torch.save(compressor.state_dict(), output_dir / "compressor.pt")
    print("✅ Final compressor saved")
