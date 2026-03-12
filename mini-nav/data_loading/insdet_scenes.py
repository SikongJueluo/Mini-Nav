"""InsDet Scenes dataset for multi-object retrieval benchmark."""

from pathlib import Path
from typing import Any

from benchmarks.base import BaseDataset
from data_loading.loader import load_val_dataset


class InsDetScenesDataset(BaseDataset):
    """InsDet-FULL/Scenes dataset with easy/hard splits.

    This dataset provides scene images with object annotations from the
    Instance Detection (InsDet) dataset, supporting easy and hard splits.
    """

    def __init__(
        self,
        scenes_dir: Path | str,
        split: str = "easy",
    ):
        """Initialize InsDet Scenes dataset.

        Args:
            scenes_dir: Path to the InsDet-FULL/Scenes directory.
            split: Scene split to use ('easy' or 'hard').
        """
        self.scenes_dir = Path(scenes_dir)
        self.split = split
        self._dataset = load_val_dataset(self.scenes_dir, split)

    def get_train_split(self) -> Any:
        """Get training split (same as test for this dataset).

        Returns:
            HuggingFace Dataset for training.
        """
        return self._dataset

    def get_test_split(self) -> Any:
        """Get test/evaluation split.

        Returns:
            HuggingFace Dataset for testing.
        """
        return self._dataset

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item.

        Returns:
            Dictionary containing:
                - image: PIL Image
                - image_id: Scene identifier
                - objects: dict with bbox, category, area, id
        """
        return self._dataset[idx]
