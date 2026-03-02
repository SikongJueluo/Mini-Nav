"""Retrieval task for benchmark evaluation (Recall@K)."""

from typing import Any, cast

import lancedb
import pyarrow as pa
import torch
from benchmarks.base import BaseBenchmarkTask
from benchmarks.tasks.registry import RegisterTask
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BitImageProcessorFast


def _infer_vector_dim(
    processor: BitImageProcessorFast,
    model: nn.Module,
    sample_image: Any,
) -> int:
    """Infer model output vector dimension via a single forward pass.

    Args:
        processor: Image preprocessor.
        model: Feature extraction model.
        sample_image: A sample image for dimension inference.

    Returns:
        Vector dimension.
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        inputs = processor(images=sample_image, return_tensors="pt")
        inputs.to(device)
        output = model(inputs)

    return output.shape[-1]


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


@torch.no_grad()
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
    device = next(model.parameters()).device
    model.eval()

    global_idx = 0
    for batch in tqdm(dataloader, desc="Building eval database"):
        imgs = batch["img"]
        labels = batch["label"]

        inputs = processor(imgs, return_tensors="pt")
        inputs.to(device)
        outputs = model(inputs)

        features = cast(torch.Tensor, outputs).cpu()
        labels_list = labels.tolist()

        batch_size = len(labels_list)
        table.add(
            [
                {
                    "id": global_idx + j,
                    "label": labels_list[j],
                    "vector": features[j].numpy(),
                }
                for j in range(batch_size)
            ]
        )
        global_idx += batch_size


@torch.no_grad()
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
    device = next(model.parameters()).device
    model.eval()

    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc=f"Evaluating Recall@{top_k}"):
        imgs = batch["img"]
        labels = batch["label"]

        inputs = processor(imgs, return_tensors="pt")
        inputs.to(device)
        outputs = model(inputs)

        features = cast(torch.Tensor, outputs).cpu()
        labels_list = labels.tolist()

        for j in range(len(labels_list)):
            feature = features[j].tolist()
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

        vector_dim = _infer_vector_dim(processor, model, sample_image)
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
