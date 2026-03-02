"""Task registry for benchmark evaluation."""

from typing import Any, Type

from benchmarks.base import BaseBenchmarkTask

# Task registry: maps task type string to task class
TASK_REGISTRY: dict[str, Type[BaseBenchmarkTask]] = {}


class RegisterTask:
    """Decorator class to register a benchmark task.

    Usage:
        @register_task("retrieval")
        class RetrievalTask(BaseBenchmarkTask):
            ...
    """

    def __init__(self, task_type: str):
        """Initialize the decorator with task type.

        Args:
            task_type: Task type identifier.
        """
        self.task_type = task_type

    def __call__(self, cls: type[BaseBenchmarkTask]) -> type[BaseBenchmarkTask]:
        """Register the decorated class to the task registry.

        Args:
            cls: The class to be decorated.

        Returns:
            The unmodified class.
        """
        TASK_REGISTRY[self.task_type] = cls
        return cls


def get_task(task_type: str, **kwargs: Any) -> BaseBenchmarkTask:
    """Get a benchmark task instance by type.

    Args:
        task_type: Task type identifier.
        **kwargs: Additional arguments passed to task constructor.

    Returns:
        Task instance.

    Raises:
        ValueError: If task type is not registered.
    """
    if task_type not in TASK_REGISTRY:
        available = list(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task type: {task_type}. Available tasks: {available}"
        )
    return TASK_REGISTRY[task_type](**kwargs)
