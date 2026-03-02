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
        from benchmarks import evaluate

        evaluate("Dinov2", "CIFAR-10", "Recall@10")
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
        print(f"Generated {len(generated_files)} synthesized images in {dataset_cfg.output_dir}")
