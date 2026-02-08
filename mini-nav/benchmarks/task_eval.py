from typing import Literal, cast

import polars as pl
import torch
from datasets import Dataset, load_dataset
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BitImageProcessorFast
from utils import get_device


def establish_database(
    processor: BitImageProcessorFast,
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
) -> pl.DataFrame:
    df = pl.DataFrame()

    model.eval()
    dataloader = DataLoader(
        dataset.with_format("torch"), batch_size=batch_size, shuffle=True, num_workers=4
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Establish Database"):
            imgs = batch["img"]
            labels = batch["label"]

            inputs = processor(imgs, return_tensors="pt").to(get_device())

            outputs = cast(Tensor, model(inputs))

    return df


def task_eval(
    processor: BitImageProcessorFast,
    model: nn.Module,
    dataset: Literal["CIFAR-10", "CIFAR-100"],
    top_k: int = 10,
    batch_size: int = 32,
):
    match dataset:
        case "CIFAR-10":
            train_dataset = load_dataset("uoft-cs/cifar10", split="train")
            test_dataset = load_dataset("uoft-cs/cifar10", split="test")
        case "CIFAR-100":
            train_dataset = load_dataset("uoft-cs/cifar100", split="train")
            test_dataset = load_dataset("uoft-cs/cifar100", split="test")
        case _:
            raise ValueError(
                f"Unknown dataset: {dataset}. Only support: 'CIFAR-10', 'CIFAR-100'."
            )

    # Establish database
    df = establish_database(processor, model, train_dataset, batch_size)

    # Test
    dataloader = DataLoader(
        test_dataset.with_format("torch"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test Evaluation"):
            imgs = batch["img"]
            labels = batch["label"]

            inputs = processor(imgs, return_tensors="pt").to(get_device())

            outputs = cast(Tensor, model(inputs))
            for vec in outputs:
                pass
