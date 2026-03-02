"""Benchmark evaluation tasks."""

from .retrieval import RetrievalTask
from .registry import TASK_REGISTRY, get_task

__all__ = ["RetrievalTask", "TASK_REGISTRY", "get_task"]