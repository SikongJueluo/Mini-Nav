import typer


def generate(ctx: typer.Context):
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
    typer.echo(
        f"Generated {len(generated_files)} synthesized images in {dataset_cfg.output_dir}"
    )
