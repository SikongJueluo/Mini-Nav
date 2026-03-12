"""Multi-object retrieval benchmark task.

This benchmark evaluates retrieval accuracy using multiple objects from a cropped
scene region. It uses SAM for object segmentation, DINO+Hash pipeline for feature
extraction, and LanceDB for vector storage with scene-level score aggregation.
"""

import random
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
from benchmarks.base import BaseBenchmarkTask
from benchmarks.tasks.registry import RegisterTask
from configs.models import BenchmarkTaskConfig
from rich.progress import track
from torch import nn
from torch.utils.data import DataLoader
from transformers import BitImageProcessorFast
from utils.feature_extractor import extract_single_image_feature
from utils.sam import load_sam_model, segment_image
from utils.common import get_device


def _build_object_schema(vector_dim: int) -> pa.Schema:
    """Build PyArrow schema for object-level vectors.

    Args:
        vector_dim: Feature vector dimension.

    Returns:
        PyArrow schema with id, image_id, object_id, category, and vector fields.
    """
    return pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("image_id", pa.string()),
            pa.field("object_id", pa.string()),
            pa.field("category", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


def _compute_scene_score(
    query_object_ids: list[str],
    retrieved_results: dict[str, list[tuple[float, str]]],
    gamma: float,
) -> dict[str, float]:
    """Compute scene-level scores using co-occurrence penalty.

    Args:
        query_object_ids: List of query object IDs.
        retrieved_results: Dict mapping image_id to list of (distance, object_id) results.
        gamma: Co-occurrence penalty exponent.

    Returns:
        Dict mapping image_id to computed scene score.
    """
    scene_scores: dict[str, float] = {}

    for image_id, results in retrieved_results.items():
        # Build a set of retrieved object IDs for this scene
        retrieved_ids = {obj_id for _, obj_id in results}

        # Count how many query objects are found in this scene
        matched_count = sum(1 for q_id in query_object_ids if q_id in retrieved_ids)

        if matched_count == 0:
            scene_scores[image_id] = 0.0
            continue

        # Sum of best similarities (using distance as similarity: smaller = better)
        # We use 1/(1+distance) to convert distance to similarity
        similarities = []
        for dist, obj_id in results:
            if obj_id in query_object_ids:
                sim = 1.0 / (1.0 + dist)
                similarities.append(sim)

        sum_similarity = sum(similarities) if similarities else 0.0

        # Hit rate: ratio of matched objects
        hit_rate = matched_count / len(query_object_ids)

        # Final score: sum_similarity * (hit_rate)^gamma
        score = sum_similarity * (hit_rate ** gamma)
        scene_scores[image_id] = score

    return scene_scores


@RegisterTask("multi-object-retrieval")
class MultiObjectRetrievalTask(BaseBenchmarkTask):
    """Multi-object retrieval benchmark task."""

    def __init__(self, **kwargs: Any):
        """Initialize multi-object retrieval task.

        Args:
            **kwargs: Configuration parameters from BenchmarkTaskConfig.
        """
        # Use config from kwargs or load default config
        if kwargs:
            config_dict = kwargs
        else:
            config = BenchmarkTaskConfig(type="multi-object-retrieval")
            config_dict = config.model_dump()

        super().__init__(**config_dict)
        self.config = BenchmarkTaskConfig(**config_dict)

        # SAM settings from ModelConfig (passed via kwargs or use defaults)
        self.sam_model = kwargs.get("sam_model", "facebook/sam2.1-hiera-large")
        self.min_mask_area = kwargs.get("sam_min_mask_area", 32 * 32)
        self.max_masks_per_image = kwargs.get("sam_max_masks", 5)

        # Lazy-loaded resources
        self._sam_model = None
        self._mask_generator = None

    @property
    def sam_model(self) -> Any:
        """Lazy-load SAM model."""
        if self._sam_model is None:
            self._sam_model, self._mask_generator = load_sam_model(
                model_name=self.sam_model,
                device=str(get_device()),
            )
        return self._sam_model

    @property
    def mask_generator(self) -> Any:
        """Lazy-load mask generator."""
        if self._mask_generator is None:
            self._sam_model, self._mask_generator = load_sam_model(
                model_name=self.sam_model,
                device=str(get_device()),
            )
        return self._mask_generator

    def build_database(
        self,
        model: nn.Module,
        processor: BitImageProcessorFast,
        train_dataset: Any,
        table: lancedb.table.Table,
        batch_size: int,
    ) -> None:
        """Build the evaluation database with object-level vectors.

        Args:
            model: Feature extraction model.
            processor: Image preprocessor.
            train_dataset: Training dataset.
            table: LanceDB table to store features.
            batch_size: Batch size for DataLoader.
        """
        # Infer vector dimension from a sample
        sample = train_dataset[0]
        sample_image = sample["image"]

        # Get vector dimension by running a forward pass
        vector_dim = self._infer_vector_dim(processor, model, sample_image)
        expected_schema = _build_object_schema(vector_dim)

        # Check schema compatibility
        if table.schema != expected_schema:
            raise ValueError(
                f"Table schema mismatch. Expected: {expected_schema}, "
                f"Got: {table.schema}"
            )

        # Build database: segment each image, extract features per object
        record_id = 0
        records = []

        for idx in track(range(len(train_dataset)), description="Building object database"):
            item = train_dataset[idx]
            image = item["image"]
            image_id = item.get("image_id", f"image_{idx}")

            # Segment image using SAM
            masks = segment_image(
                self.mask_generator,
                image,
                min_area=self.min_mask_area,
                max_masks=self.max_masks_per_image,
            )

            if not masks:
                continue

            # Extract features for each mask
            for mask_idx, mask_info in enumerate(masks):
                # Extract masked region
                masked_image = self._apply_mask(image, mask_info["segment"])

                # Extract feature vector
                vector = extract_single_image_feature(processor, model, masked_image)

                # Create object ID
                object_id = f"{image_id}_obj_{mask_idx}"
                category = mask_info.get("category", "unknown")

                records.append({
                    "id": record_id,
                    "image_id": image_id,
                    "object_id": object_id,
                    "category": category,
                    "vector": vector,
                })
                record_id += 1

        # Add all records to table
        if records:
            table.add(records)

    def evaluate(
        self,
        model: nn.Module,
        processor: BitImageProcessorFast,
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
        top_k = self.config.top_k_per_object

        correct = 0
        total = 0

        for idx in track(range(len(test_dataset)), description=f"Evaluating Recall@{top_k}"):
            item = test_dataset[idx]
            image = item["image"]
            target_image_id = item.get("image_id", f"image_{idx}")

            # Segment query image
            masks = segment_image(
                self.mask_generator,
                image,
                min_area=self.min_mask_area,
                max_masks=self.max_masks_per_image,
            )

            if not masks:
                continue

            # Randomly sample query objects
            num_query = min(self.config.num_query_objects, len(masks))
            query_masks = random.sample(masks, num_query)

            # Extract features and search for each query object
            retrieved_results: dict[str, list[tuple[float, str]]] = {}

            for mask_info in query_masks:
                # Extract masked region
                masked_image = self._apply_mask(image, mask_info["segment"])

                # Extract feature vector
                vector = extract_single_image_feature(processor, model, masked_image)

                # Search in LanceDB
                results = (
                    table.search(vector)
                    .select(["image_id", "object_id", "_distance"])
                    .limit(top_k)
                    .to_polars()
                )

                # Aggregate results by scene
                for row in results.iter_rows():
                    image_id = row["image_id"]
                    object_id = row["object_id"]
                    distance = row["_distance"]

                    if image_id not in retrieved_results:
                        retrieved_results[image_id] = []
                    retrieved_results[image_id].append((distance, object_id))

            # Compute scene scores
            query_object_ids = [m.get("object_id", f"query_obj_{i}") for i, m in enumerate(query_masks)]
            scene_scores = _compute_scene_score(
                query_object_ids,
                retrieved_results,
                self.config.gamma,
            )

            # Rank scenes by score
            ranked_scenes = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)

            # Check if target is in top-K
            top_k_scenes = [scene_id for scene_id, _ in ranked_scenes[:top_k]]
            if target_image_id in top_k_scenes:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "top_k": top_k,
        }

    def _infer_vector_dim(
        self,
        processor: BitImageProcessorFast,
        model: nn.Module,
        sample_image: Any,
    ) -> int:
        """Infer vector dimension from model output."""
        vector = extract_single_image_feature(processor, model, sample_image)
        return len(vector)

    def _apply_mask(self, image: Any, mask: np.ndarray) -> Any:
        """Apply mask to image and return masked image.

        Args:
            image: PIL Image.
            mask: Binary mask as numpy array.

        Returns:
            Masked PIL Image.
        """
        import numpy as np
        from PIL import Image

        image_np = np.array(image.convert("RGB"))
        # Ensure mask is the right shape
        if mask.shape != image_np.shape[:2]:
            from skimage.transform import resize
            mask_resized = resize(mask, image_np.shape[:2], order=0, anti_aliasing=False)
        else:
            mask_resized = mask

        # Apply mask
        masked_np = image_np * mask_resized[:, :, np.newaxis]
        return Image.fromarray(masked_np.astype(np.uint8))
