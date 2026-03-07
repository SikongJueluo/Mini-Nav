from typing import Any, Optional, cast

import typer
from commands import app


@app.command()
def benchmark(
    ctx: typer.Context,
    model_path: Optional[str] = typer.Option(
        None, "--model", "-m", help="Path to compressor model weights"
    ),
):
    import torch
    import torch.nn.functional as F
    from benchmarks import run_benchmark
    from configs import cfg_manager
    from transformers import AutoImageProcessor, AutoModel, BitImageProcessorFast
    from utils import get_device

    config = cfg_manager.get()
    benchmark_cfg = config.benchmark

    device = get_device()

    model_cfg = config.model
    processor = cast(
        BitImageProcessorFast,
        AutoImageProcessor.from_pretrained(model_cfg.dino_model, device_map=device),
    )

    # Load DINO model for feature extraction
    dino = AutoModel.from_pretrained(model_cfg.dino_model, device_map=device)
    dino.eval()

    # Optional hash compressor
    compressor = None
    if model_path:
        from compressors import HashCompressor

        compressor = HashCompressor(
            input_dim=model_cfg.compression_dim,
            hash_bits=model_cfg.compression_dim,
        )
        compressor.load_state_dict(torch.load(model_path))
        compressor.to(device)
        compressor.eval()

    # Create wrapper with extract_features method
    class DinoFeatureExtractor:
        def __init__(self, dino, compressor=None):
            self.dino = dino
            self.compressor = compressor

        def extract_features(self, images: list) -> torch.Tensor:
            inputs = processor(images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.dino(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
                features = F.normalize(features, dim=-1)
            return features

        def encode(self, images: list) -> torch.Tensor:
            if self.compressor is None:
                return self.extract_features(images)
            tokens = self.dino(**processor(images, return_tensors="pt").to(device)).last_hidden_state
            _, _, bits = self.compressor(tokens)
            return bits

    model = DinoFeatureExtractor(dino, compressor)

    run_benchmark(
        model=model,
        processor=processor,
        config=benchmark_cfg,
        model_name="dinov2",
    )
