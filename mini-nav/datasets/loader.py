"""Data loaders for synthetic and validation datasets."""

from collections.abc import Iterator
from pathlib import Path

from PIL import Image


class SynthDataset:
    """Dataset loader for synthesized training images."""

    def __init__(self, synth_dir: Path, annotations_suffix: str = ".txt"):
        """
        Initialize the synthetic dataset loader.

        Args:
            synth_dir: Directory containing synthesized images and annotations
            annotations_suffix: Suffix for annotation files
        """
        self.synth_dir = Path(synth_dir)
        self.annotations_suffix = annotations_suffix

        # Find all images
        self.image_files = sorted(self.synth_dir.glob("synth_*.jpg"))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[Image.Image, list[tuple[str, int, int, int, int]]]:
        """Get a single item.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (image, annotations) where annotations is a list of
            (category, xmin, ymin, xmax, ymax)
        """
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        # Load annotations
        anno_path = img_path.with_suffix(self.annotations_suffix)
        annotations: list[tuple[str, int, int, int, int]] = []

        if anno_path.exists():
            with open(anno_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            category = parts[0]
                            xmin, ymin, xmax, ymax = map(int, parts[1:])
                            annotations.append((category, xmin, ymin, xmax, ymax))

        return image, annotations

    def __iter__(self) -> Iterator[tuple[Image.Image, list[tuple[str, int, int, int, int]]]]:
        """Iterate over the dataset."""
        for i in range(len(self)):
            yield self[i]


class ValDataset:
    """Dataset loader for validation scene images."""

    def __init__(self, scenes_dir: Path, split: str = "easy"):
        """
        Initialize the validation dataset loader.

        Args:
            scenes_dir: Directory containing scene subdirectories
            split: Scene split to load ('easy' or 'hard')
        """
        self.scenes_dir = Path(scenes_dir)
        self.split = split

        self.split_dir = self.scenes_dir / split
        if not self.split_dir.exists():
            raise ValueError(f"Scene split directory not found: {self.split_dir}")

        # Find all RGB images
        self.image_files = sorted(self.split_dir.glob("*/rgb_*.jpg"))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[Image.Image, Path]:
        """Get a single item.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (image, scene_path)
        """
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        return image, img_path.parent

    def __iter__(self) -> Iterator[tuple[Image.Image, Path]]:
        """Iterate over the dataset."""
        for i in range(len(self)):
            yield self[i]
