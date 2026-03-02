"""HuggingFace dataset loader for benchmark evaluation."""

from typing import Any

from datasets import load_dataset

from ..base import BaseDataset


class HuggingFaceDataset(BaseDataset):
    """Dataset loader for HuggingFace datasets."""

    def __init__(
        self,
        hf_id: str,
        img_column: str = "img",
        label_column: str = "label",
    ):
        """Initialize HuggingFace dataset loader.

        Args:
            hf_id: HuggingFace dataset ID.
            img_column: Name of the image column.
            label_column: Name of the label column.
        """
        self.hf_id = hf_id
        self.img_column = img_column
        self.label_column = label_column
        self._train_dataset: Any = None
        self._test_dataset: Any = None

    def _load(self) -> tuple[Any, Any]:
        """Load dataset from HuggingFace.

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        if self._train_dataset is None:
            dataset = load_dataset(self.hf_id)
            # Handle datasets that use 'train' and 'test' splits
            if "train" in dataset:
                self._train_dataset = dataset["train"]
            if "test" in dataset:
                self._test_dataset = dataset["test"]
            # Handle datasets that use 'train' and 'validation' splits
            elif "validation" in dataset:
                self._test_dataset = dataset["validation"]
        return self._train_dataset, self._test_dataset

    def get_train_split(self) -> Any:
        """Get training split of the dataset.

        Returns:
            Training dataset.
        """
        train, _ = self._load()
        return train

    def get_test_split(self) -> Any:
        """Get test/evaluation split of the dataset.

        Returns:
            Test dataset.
        """
        _, test = self._load()
        return test
