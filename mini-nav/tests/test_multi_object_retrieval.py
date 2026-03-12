"""Integration tests for multi-object retrieval benchmark pipeline.

These tests verify the end-to-end functionality of the multi-object retrieval
benchmark, including schema building, database population, and evaluation.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image


class TestMultiObjectRetrievalIntegration:
    """Integration tests for multi-object retrieval benchmark."""

    @pytest.fixture
    def mock_model_processor(self):
        """Create mock model and processor."""
        mock_model = Mock()
        mock_processor = Mock()

        # Mock the feature extraction to return a fixed-size vector
        def mock_extract(processor, model, image):
            return [0.1] * 256  # 256-dim vector
        mock_processor.images = mock_extract

        return mock_model, mock_processor

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset with images and annotations."""
        # Create mock items
        items = []
        for i in range(3):
            item = {
                "image": Image.new("RGB", (224, 224), color=(i * 50, 100, 150)),
                "image_id": f"scene_{i}",
                "objects": {
                    "bbox": [[10, 10, 50, 50], [60, 60, 40, 40]],
                    "category": ["object_a", "object_b"],
                    "area": [2500, 1600],
                    "id": [0, 1],
                },
            }
            items.append(item)

        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=len(items))
        mock_dataset.__getitem__ = lambda self, idx: items[idx]
        mock_dataset.with_format = lambda fmt: mock_dataset

        return mock_dataset

    def test_build_object_schema(self):
        """Test that object schema is built correctly."""
        from benchmarks.tasks.multi_object_retrieval import _build_object_schema
        import pyarrow as pa

        vector_dim = 256
        schema = _build_object_schema(vector_dim)

        assert isinstance(schema, pa.Schema)
        assert "id" in schema.names
        assert "image_id" in schema.names
        assert "object_id" in schema.names
        assert "category" in schema.names
        assert "vector" in schema.names

        # Check vector field has correct dimension
        vector_field = schema.field("vector")
        assert isinstance(vector_field.type, pa.List)
        assert vector_field.type.value_type == pa.float32()

    @patch("benchmarks.tasks.multi_object_retrieval.load_sam_model")
    @patch("benchmarks.tasks.multi_object_retrieval.segment_image")
    def test_build_database_with_mocked_sam(
        self,
        mock_segment,
        mock_load_sam,
        mock_model_processor,
        mock_dataset,
    ):
        """Test database building with mocked SAM segmentation."""
        from benchmarks.tasks.multi_object_retrieval import (
            MultiObjectRetrievalTask,
            _build_object_schema,
        )

        mock_model, mock_processor = mock_model_processor

        # Mock SAM
        mock_load_sam.return_value = (Mock(), Mock())
        mock_segment.return_value = [
            {
                "segment": np.ones((224, 224), dtype=bool),
                "area": 50000,
                "bbox": [0, 0, 224, 224],
            }
        ]

        # Create task with config
        task = MultiObjectRetrievalTask(
            sam_model="facebook/sam2.1-hiera-large",
            min_mask_area=1024,
            max_masks_per_image=5,
            gamma=1.0,
            top_k_per_object=50,
            num_query_objects=3,
        )

        # Create mock table
        mock_table = Mock()
        mock_table.schema = _build_object_schema(256)

        # Build database (this should not raise)
        task.build_database(mock_model, mock_processor, mock_dataset, mock_table, batch_size=1)

        # Verify table.add was called
        assert mock_table.add.called

    @patch("benchmarks.tasks.multi_object_retrieval.load_sam_model")
    @patch("benchmarks.tasks.multi_object_retrieval.segment_image")
    def test_evaluate_with_mocked_sam(
        self,
        mock_segment,
        mock_load_sam,
        mock_model_processor,
        mock_dataset,
    ):
        """Test evaluation with mocked SAM segmentation."""
        from benchmarks.tasks.multi_object_retrieval import (
            MultiObjectRetrievalTask,
            _build_object_schema,
        )

        mock_model, mock_processor = mock_model_processor

        # Mock SAM
        mock_load_sam.return_value = (Mock(), Mock())
        mock_segment.return_value = [
            {
                "segment": np.ones((224, 224), dtype=bool),
                "area": 50000,
                "bbox": [0, 0, 224, 224],
                "object_id": "query_obj_0",
            }
        ]

        # Create mock table with search results
        mock_table = Mock()
        mock_table.schema = _build_object_schema(256)

        # Mock search to return matching result
        mock_result = Mock()
        mock_result.to_polars.return_value = {
            "image_id": ["scene_0"],
            "object_id": ["scene_0_obj_0"],
            "_distance": [0.1],
        }

        mock_table.search.return_value.select.return_value.limit.return_value = mock_result

        # Create task
        task = MultiObjectRetrievalTask(
            sam_model="facebook/sam2.1-hiera-large",
            min_mask_area=1024,
            max_masks_per_image=5,
            gamma=1.0,
            top_k_per_object=50,
            num_query_objects=1,
        )

        # Evaluate
        results = task.evaluate(mock_model, mock_processor, mock_dataset, mock_table, batch_size=1)

        # Verify results structure
        assert "accuracy" in results
        assert "correct" in results
        assert "total" in results
        assert "top_k" in results
        assert results["top_k"] == 50

    def test_task_initialization_with_config(self):
        """Test task initialization with custom config."""
        from benchmarks.tasks.multi_object_retrieval import MultiObjectRetrievalTask

        task = MultiObjectRetrievalTask(
            sam_model="facebook/sam2.1-hiera-small",
            min_mask_area=500,
            max_masks_per_image=3,
            gamma=0.5,
            top_k_per_object=100,
            num_query_objects=5,
        )

        assert task.sam_model == "facebook/sam2.1-hiera-small"
        assert task.min_mask_area == 500
        assert task.max_masks_per_image == 3
        assert task.config.gamma == 0.5
        assert task.config.top_k_per_object == 100
        assert task.config.num_query_objects == 5

    def test_task_initialization_defaults(self):
        """Test task initialization with default config."""
        from benchmarks.tasks.multi_object_retrieval import MultiObjectRetrievalTask

        task = MultiObjectRetrievalTask()

        # Check defaults from BenchmarkTaskConfig
        assert task.config.gamma == 1.0
        assert task.config.top_k_per_object == 50
        assert task.config.num_query_objects == 3
        # SAM settings from ModelConfig defaults
        assert task.sam_model == "facebook/sam2.1-hiera-large"
        assert task.min_mask_area == 1024
        assert task.max_masks_per_image == 5


class TestInsDetScenesDataset:
    """Tests for InsDetScenesDataset class."""

    def test_dataset_class_exists(self):
        """Test that InsDetScenesDataset can be imported."""
        from data_loading.insdet_scenes import InsDetScenesDataset

        assert InsDetScenesDataset is not None

    @patch("data_loading.insdet_scenes.load_val_dataset")
    def test_dataset_loads_correct_split(self, mock_load):
        """Test dataset loads correct split."""
        from data_loading.insdet_scenes import InsDetScenesDataset

        mock_load.return_value = Mock()

        dataset = InsDetScenesDataset("/path/to/scenes", split="easy")

        mock_load.assert_called_once_with("/path/to/scenes", "easy")
        assert dataset.split == "easy"
