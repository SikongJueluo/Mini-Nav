from typing import Literal, cast

import torch
from compressors import DinoCompressor, FloatCompressor
from configs import cfg_manager
from transformers import AutoImageProcessor, BitImageProcessorFast
from utils import get_device

from .task_eval import task_eval


def evaluate(
    compressor_model: Literal["Dinov2", "Dinov2WithCompressor"],
    dataset: Literal["CIFAR-10", "CIFAR-100"],
    benchmark: Literal["Recall@1", "Recall@10"],
):
    """运行模型评估。

    Args:
        compressor_model: 压缩模型类型。
        dataset: 数据集名称。
        benchmark: 评估指标。
    """
    device = get_device()

    match compressor_model:
        case "Dinov2":
            processor = cast(
                BitImageProcessorFast,
                AutoImageProcessor.from_pretrained(
                    "facebook/dinov2-large", device_map=device
                ),
            )
            model = DinoCompressor().to(device)
        case "Dinov2WithCompressor":
            processor = cast(
                BitImageProcessorFast,
                AutoImageProcessor.from_pretrained(
                    "facebook/dinov2-large", device_map=device
                ),
            )
            output_dir = cfg_manager.get().output.directory
            compressor = FloatCompressor()
            compressor.load_state_dict(torch.load(output_dir / "compressor.pt"))
            model = DinoCompressor(compressor).to(device)
        case _:
            raise ValueError(f"Unknown compressor: {compressor_model}")

    # 根据 benchmark 确定 top_k
    match benchmark:
        case "Recall@1":
            task_eval(processor, model, dataset, compressor_model, top_k=1)
        case "Recall@10":
            task_eval(processor, model, dataset, compressor_model, top_k=10)
        case _:
            raise ValueError(f"Unknown benchmark: {benchmark}")


__all__ = ["task_eval", "evaluate"]
