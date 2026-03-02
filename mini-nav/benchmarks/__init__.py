"""Benchmark evaluation module.

This module provides a modular benchmark system that supports:
- Multiple dataset sources (HuggingFace, local)
- Multiple evaluation tasks (retrieval, with extensibility for more)
- Configuration-driven execution
"""

from .base import BaseBenchmarkTask, BaseDataset
from .runner import run_benchmark

__all__ = ["BaseBenchmarkTask", "BaseDataset", "run_benchmark"]