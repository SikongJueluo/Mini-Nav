from typing import Literal, cast

import torch
from compressors import DinoCompressor, FloatCompressor
from transformers import AutoImageProcessor, BitImageProcessorFast
from utils import get_device, get_output_diretory

from .task_eval import task_eval


def evaluate(
    compressor_model: Literal["Dinov2", "Dinov2WithCompressor"],
    dataset: Literal["CIFAR-10", "CIFAR-100"],
    benchmark: Literal["Recall@1", "Recall@10"],
):
    match compressor_model:
        case "Dinov2":
            processor = cast(
                BitImageProcessorFast,
                AutoImageProcessor.from_pretrained(
                    "facebook/dinov2-large", device_map=get_device()
                ),
            )
            model = DinoCompressor().to(get_device())
        case "Dinov2WithCompressor":
            processor = cast(
                BitImageProcessorFast,
                AutoImageProcessor.from_pretrained(
                    "facebook/dinov2-large", device_map=get_device()
                ),
            )

            compressor = FloatCompressor().load_state_dict(
                torch.load(get_output_diretory() / "compressor.pt")
            )
            model = DinoCompressor(compressor).to(get_device())
        case _:
            raise ValueError(f"Unknown compressor: {compressor_model}")

    match benchmark:
        case "Recall@1":
            task_eval(processor, model, dataset, 1)
        case "Recall@10":
            task_eval(processor, model, dataset, 10)
        case _:
            raise ValueError(f"Unknown benchmark: {benchmark}")


__all__ = ["task_eval", "evaluate"]
