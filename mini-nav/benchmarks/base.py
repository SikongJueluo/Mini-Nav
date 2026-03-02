"""Base classes for benchmark datasets and tasks."""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import lancedb
from torch.utils.data import DataLoader


class BaseDataset(ABC):
    """Abstract base class for benchmark datasets."""

    @abstractmethod
    def get_train_split(self) -> Any:
        """Get training split of the dataset.

        Returns:
            Dataset object for training.
        """
        pass

    @abstractmethod
    def get_test_split(self) -> Any:
        """Get test/evaluation split of the dataset.

        Returns:
            Dataset object for testing.
        """
        pass


class BaseBenchmarkTask(ABC):
    """Abstract base class for benchmark evaluation tasks."""

    def __init__(self, **kwargs: Any):
        """Initialize the benchmark task.

        Args:
            **kwargs: Task-specific configuration parameters.
        """
        self.config = kwargs

    @abstractmethod
    def build_database(
        self,
        model: Any,
        processor: Any,
        train_dataset: Any,
        table: lancedb.table.Table,
        batch_size: int,
    ) -> None:
        """Build the evaluation database from training data.

        Args:
            model: Feature extraction model.
            processor: Image preprocessor.
            train_dataset: Training dataset.
            table: LanceDB table to store features.
            batch_size: Batch size for DataLoader.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        processor: Any,
        test_dataset: Any,
        table: lancedb.table.Table,
        batch_size: int,
    ) -> dict[str, Any]:
        """Evaluate the model on the test dataset.

        Args:
            model: Feature extraction model.
            processor: Image preprocessor.
            test_dataset: Test dataset.
            table: LanceDB table to search against.
            batch_size: Batch size for DataLoader.

        Returns:
            Dictionary containing evaluation results.
        """
        pass


class DatasetFactory(Protocol):
    """Protocol for dataset factory."""

    def __call__(self, config: Any) -> BaseDataset:
        """Create a dataset from configuration.

        Args:
            config: Dataset configuration.

        Returns:
            Dataset instance.
        """
        ...
