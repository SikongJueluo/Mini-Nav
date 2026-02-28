"""Training script for hash compressor."""

import os

import torch
import torch.nn.functional as F
from compressors import HashCompressor, HashLoss
from configs import cfg_manager
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

from datasets import load_dataset


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
    epoch_size: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    checkpoint_path: str = "hash_checkpoint.pt",
):
    """Train hash compressor with batch-level retrieval loss.

    Args:
        epoch_size: Number of epochs to train
        batch_size: Batch size for training
        lr: Learning rate
        checkpoint_path: Path to save/load checkpoints
    """
    # Auto detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Global variables
    save_every = 500
    start_epoch = 0
    global_step = 0

    # Load dataset
    ds_train = load_dataset("uoft-cs/cifar10", split="train").with_format("torch")
    dataloader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Load processor
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-large", device_map=device
    )

    # Load DINO model (frozen)
    dino = AutoModel.from_pretrained("facebook/dinov2-large", device_map=device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False

    # Load hash compressor
    compressor = HashCompressor(input_dim=1024, hash_bits=512).to(device)

    # Load loss function
    loss_fn = HashLoss(
        contrastive_weight=1.0,
        distill_weight=0.5,
        quant_weight=0.01,
        temperature=0.2,
    )

    # Load optimizer
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=lr)

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
                logits, hash_codes, bits = compressor(teacher_tokens)

                # ---- generate positive mask ----
                labels = batch["label"]
                # positive_mask[i,j] = True if labels[i] == labels[j]
                positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]

                # ---- loss ----
                total_loss, components = loss_fn(
                    logits=logits,
                    hash_codes=hash_codes,
                    teacher_embed=teacher_embed,
                    positive_mask=positive_mask,
                )

                # ---- backward ----
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # ---- logging ----
                train_bar.set_postfix(
                    loss=f"{components['total']:.4f}",
                    cont=f"{components['contrastive']:.2f}",
                    distill=f"{components['distill']:.3f}",
                    quant=f"{components['quantization']:.2f}",
                )

                # ---- periodic save ----
                if global_step % save_every == 0:
                    save_checkpoint(
                        compressor, optimizer, epoch, global_step, checkpoint_path
                    )

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted, saving checkpoint...")
        save_checkpoint(compressor, optimizer, epoch, global_step, checkpoint_path)
        print("✅ Checkpoint saved. Exiting.")
        return

    # Save final model
    torch.save(compressor.state_dict(), output_dir / "hash_compressor.pt")
    print("✅ Final hash compressor saved")
