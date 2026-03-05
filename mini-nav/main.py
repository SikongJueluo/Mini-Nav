import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["train", "benchmark", "visualize", "generate"],
        help="Action to perform: train, benchmark, visualize, or generate",
    )
    args = parser.parse_args()

    if args.action == "train":
        from compressors import train

        train(
            epoch_size=10, batch_size=64, lr=1e-4, checkpoint_path="hash_checkpoint.pt"
        )
    elif args.action == "benchmark":
        from typing import cast

        import torch
        from benchmarks import run_benchmark
        from compressors import DinoCompressor
        from configs import cfg_manager
        from transformers import AutoImageProcessor, BitImageProcessorFast
        from utils import get_device

        config = cfg_manager.get()
        benchmark_cfg = config.benchmark

        if not benchmark_cfg.enabled:
            print("Benchmark is not enabled. Set benchmark.enabled=true in config.yaml")
            exit(1)

        device = get_device()

        # Load model and processor based on config
        model_cfg = config.model
        processor = cast(
            BitImageProcessorFast,
            AutoImageProcessor.from_pretrained(model_cfg.dino_model, device_map=device),
        )

        # Load compressor weights if specified in model config
        model = DinoCompressor().to(device)
        if model_cfg.compressor_path is not None:
            from compressors import HashCompressor

            compressor = HashCompressor(
                input_dim=model_cfg.compression_dim,
                output_dim=model_cfg.compression_dim,
            )
            compressor.load_state_dict(torch.load(model_cfg.compressor_path))
            # Wrap with compressor if path is specified
            model.compressor = compressor

        # Run benchmark
        run_benchmark(
            model=model,
            processor=processor,
            config=benchmark_cfg,
            model_name="dinov2",
        )
    elif args.action == "visualize":
        from visualizer import app

        app.run(debug=True)
    else:  # generate
        from configs import cfg_manager
        from data_loading.synthesizer import ImageSynthesizer

        config = cfg_manager.get()
        dataset_cfg = config.dataset

        synthesizer = ImageSynthesizer(
            dataset_root=dataset_cfg.dataset_root,
            output_dir=dataset_cfg.output_dir,
            num_objects_range=dataset_cfg.num_objects_range,
            num_scenes=dataset_cfg.num_scenes,
            object_scale_range=dataset_cfg.object_scale_range,
            rotation_range=dataset_cfg.rotation_range,
            overlap_threshold=dataset_cfg.overlap_threshold,
            seed=dataset_cfg.seed,
        )

        generated_files = synthesizer.generate()
        print(
            f"Generated {len(generated_files)} synthesized images in {dataset_cfg.output_dir}"
        )
