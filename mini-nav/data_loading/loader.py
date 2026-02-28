"""Data loaders for synthetic and validation datasets using Hugging Face datasets."""

from pathlib import Path
from typing import Any

from datasets import Dataset, Image


def load_synth_dataset(
    synth_dir: Path,
    annotations_suffix: str = ".txt",
) -> Dataset:
    """Load synthesized dataset for object detection training.

    Args:
        synth_dir: Directory containing synthesized images and annotations
        annotations_suffix: Suffix for annotation files

    Returns:
        Hugging Face Dataset with image and objects columns
    """
    synth_dir = Path(synth_dir)
    image_files = sorted(synth_dir.glob("synth_*.jpg"))

    if not image_files:
        return Dataset.from_dict({"image": [], "objects": []}).cast_column("image", Image())

    image_paths: list[str] = []
    all_objects: list[dict[str, Any]] = []

    for img_path in image_files:
        image_paths.append(str(img_path))

        anno_path = img_path.with_suffix(annotations_suffix)
        if not anno_path.exists():
            all_objects.append({"bbox": [], "category": [], "area": [], "id": []})
            continue

        bboxes: list[list[float]] = []
        categories: list[str] = []
        areas: list[float] = []
        ids: list[int] = []

        with open(anno_path, "r") as f:
            for idx, line in enumerate(f):
                if not (line := line.strip()):
                    continue

                parts = line.split()
                if len(parts) != 5:
                    continue

                xmin, ymin, xmax, ymax = map(int, parts[1:])
                width, height = xmax - xmin, ymax - ymin

                bboxes.append([float(xmin), float(ymin), float(width), float(height)])
                categories.append(parts[0])
                areas.append(float(width * height))
                ids.append(idx)

        all_objects.append({"bbox": bboxes, "category": categories, "area": areas, "id": ids})

    dataset = Dataset.from_dict({"image": image_paths, "objects": all_objects})
    return dataset.cast_column("image", Image())


def load_val_dataset(
    scenes_dir: Path,
    split: str = "easy",
) -> Dataset:
    """Load validation dataset from scene images.

    Args:
        scenes_dir: Directory containing scene subdirectories
        split: Scene split to load ('easy' or 'hard')

    Returns:
        Hugging Face Dataset with image and image_id columns
    """
    scenes_dir = Path(scenes_dir)
    split_dir = scenes_dir / split

    if not split_dir.exists():
        raise ValueError(f"Scene split directory not found: {split_dir}")

    rgb_files = sorted(split_dir.glob("*/rgb_*.jpg"))

    if not rgb_files:
        return Dataset.from_dict({"image": [], "image_id": []}).cast_column("image", Image())

    dataset = Dataset.from_dict({
        "image": [str(p) for p in rgb_files],
        "image_id": [p.stem for p in rgb_files],
    })

    return dataset.cast_column("image", Image())
