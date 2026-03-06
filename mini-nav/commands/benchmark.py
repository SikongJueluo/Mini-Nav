from typing import cast

import typer
from commands import app


@app.command()
def benchmark(
    ctx: typer.Context,
    model_path: str = typer.Option(
        None, "--model", "-m", help="Path to compressor model weights"
    ),
):
    import torch
    from benchmarks import run_benchmark
    from compressors import DinoCompressor
    from configs import cfg_manager
    from transformers import AutoImageProcessor, BitImageProcessorFast
    from utils import get_device

    config = cfg_manager.get()
    benchmark_cfg = config.benchmark

    device = get_device()

    model_cfg = config.model
    processor = cast(
        BitImageProcessorFast,
        AutoImageProcessor.from_pretrained(model_cfg.dino_model, device_map=device),
    )

    model = DinoCompressor().to(device)
    if model_path:
        from compressors import HashCompressor

        compressor = HashCompressor(
            input_dim=model_cfg.compression_dim,
            hash_bits=model_cfg.compression_dim,
        )
        compressor.load_state_dict(torch.load(model_path))
        model.compressor = compressor

    run_benchmark(
        model=model,
        processor=processor,
        config=benchmark_cfg,
        model_name="dinov2",
    )
