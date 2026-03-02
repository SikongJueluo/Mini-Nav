"""Dataset loaders for benchmark evaluation."""

from .huggingface import HuggingFaceDataset
from .local import LocalDataset

__all__ = ["HuggingFaceDataset", "LocalDataset"]