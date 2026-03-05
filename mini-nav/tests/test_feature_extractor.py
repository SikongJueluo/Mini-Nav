"""Tests for feature extraction utilities."""

import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from utils.feature_extractor import (
    extract_batch_features,
    extract_single_image_feature,
    infer_vector_dim,
)

TEST_MODEL_NAME = "facebook/dinov2-base"


@pytest.fixture
def model_and_processor():
    processor = AutoImageProcessor.from_pretrained(TEST_MODEL_NAME)
    model = AutoModel.from_pretrained(TEST_MODEL_NAME)
    model.eval()
    yield processor, model
    del model
    del processor


def test_infer_vector_dim(model_and_processor):
    """Verify infer_vector_dim returns correct dimension."""
    processor, model = model_and_processor
    sample_image = Image.new("RGB", (224, 224), color="blue")
    dim = infer_vector_dim(processor, model, sample_image)
    assert dim == 768


def test_extract_single_image_feature(model_and_processor):
    """Verify single image feature extraction."""
    processor, model = model_and_processor
    sample_image = Image.new("RGB", (224, 224), color="red")
    features = extract_single_image_feature(processor, model, sample_image)
    assert isinstance(features, list)
    assert len(features) == 768


def test_extract_batch_features(model_and_processor):
    """Verify batch feature extraction."""
    processor, model = model_and_processor
    images = [Image.new("RGB", (224, 224), color="red") for _ in range(3)]
    features = extract_batch_features(processor, model, images)
    assert isinstance(features, torch.Tensor)
    assert features.shape == (3, 768)
