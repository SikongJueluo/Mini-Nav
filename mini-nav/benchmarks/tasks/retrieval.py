"""Retrieval task for benchmark evaluation (Recall@K)."""

from typing import Any

import lancedb
import pyarrow as pa
from benchmarks.base import BaseBenchmarkTask
from benchmarks.tasks.registry import RegisterTask
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BitImageProcessorFast

from utils.feature_extractor import extract_batch_features, infer_vector_dim


def _build_eval_schema(vector_dim: int) -> pa.Schema:
    """Build PyArrow schema for evaluation database table.

    Args:
        vector_dim: Feature vector dimension.

    Returns:
        PyArrow schema with id, label, and vector fields.
    """
    return pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("label", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


def _establish_eval_database(
    processor: BitImageProcessorFast,
    model: nn.Module,
    table: lancedb.table.Table,
    dataloader: DataLoader,
) -> None:
    """Extract features from training images and store them in a database table.

    Args:
        processor: Image preprocessor.
        model: Feature extraction model.
        table: LanceDB table to store features.
        dataloader: DataLoader for the training dataset.
    """
    # Extract all features using the utility function
    all_features = extract_batch_features(processor, model, dataloader, show_progress=True)

    # Store features to database
    global_idx = 0
    for batch in tqdm(dataloader, desc="Storing eval database"):
        labels = batch["label"]
        labels_list = labels.tolist()
        batch_size = len(labels_list)

        table.add(
            [
                {
                    "id": global_idx + j,
                    "label": labels_list[j],
                    "vector": all_features[global_idx + j].numpy(),
                }
                for j in range(batch_size)
            ]
        )
        global_idx += batch_size


def _evaluate_recall(
    processor: BitImageProcessorFast,
    model: nn.Module,
    table: lancedb.table.Table,
    dataloader: DataLoader,
    top_k: int,
) -> tuple[int, int]:
    """Evaluate Recall@K by searching the database for each test image.

    Args:
        processor: Image preprocessor.
        model: Feature extraction model.
        table: LanceDB table to search against.
        dataloader: DataLoader for the test dataset.
        top_k: Number of top results to retrieve.

    Returns:
        A tuple of (correct_count, total_count).
    """
    # Extract all features using the utility function
    all_features = extract_batch_features(processor, model, dataloader, show_progress=True)

    correct = 0
    total = 0
    feature_idx = 0

    for batch in tqdm(dataloader, desc=f"Evaluating Recall@{top_k}"):
        labels = batch["label"]
        labels_list = labels.tolist()

        for j in range(len(labels_list)):
            feature = all_features[feature_idx + j].tolist()
            true_label = labels_list[j]

            results = (
                table.search(feature)
                .select(["label", "_distance"])
                .limit(top_k)
                .to_polars()
            )

            retrieved_labels = results["label"].to_list()
            if true_label in retrieved_labels:
                correct += 1
            total += 1

        feature_idx += len(labels_list)

    return correct, total


@RegisterTask("retrieval")
class RetrievalTask(BaseBenchmarkTask):
    """Retrieval evaluation task (Recall@K)."""

    def __init__(self, top_k: int = 10):
        """Initialize retrieval task.

        Args:
            top_k: Number of top results to retrieve for recall calculation.
        """
        super().__init__(top_k=top_k)
        self.top_k = top_k

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
        # Get a sample image to infer vector dimension
        sample = train_dataset[0]
        sample_image = sample["img"]

        vector_dim = infer_vector_dim(processor, model, sample_image)
        expected_schema = _build_eval_schema(vector_dim)

        # Check schema compatibility
        if table.schema != expected_schema:
            raise ValueError(
                f"Table schema mismatch. Expected: {expected_schema}, "
                f"Got: {table.schema}"
            )

        # Build database
        train_loader = DataLoader(
            train_dataset.with_format("torch"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        _establish_eval_database(processor, model, table, train_loader)

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
            Dictionary containing evaluation results with keys:
                - accuracy: Recall@K accuracy (0.0 ~ 1.0)
                - correct: Number of correct predictions
                - total: Total number of test samples
                - top_k: The K value used
        """
        test_loader = DataLoader(
            test_dataset.with_format("torch"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        correct, total = _evaluate_recall(
            processor, model, table, test_loader, self.top_k
        )

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "top_k": self.top_k,
        }
