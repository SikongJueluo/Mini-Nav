"""Generic YAML loader with Pydantic validation."""

from pathlib import Path
from typing import TypeVar, Type

import yaml
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class ConfigError(Exception):
    """Configuration loading and validation error."""

    pass


def load_yaml(path: Path, model_class: Type[T]) -> T:
    """Load and validate YAML configuration file.

    Args:
        path: Path to YAML file (str or Path object)
        model_class: Pydantic model class to validate against

    Returns:
        Validated model instance of type T

    Raises:
        ConfigError: On file not found, YAML parsing error, or validation failure
    """
    # Coerce str to Path if needed
    if isinstance(path, str):
        path = Path(path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Configuration file not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parsing error: {e}") from e

    try:
        return model_class.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"Configuration validation failed: {e}") from e


def save_yaml(path: Path, model: BaseModel) -> None:
    """Save Pydantic model to YAML file.

    Args:
        path: Path to YAML file (str or Path object)
        model: Pydantic model instance to save

    Raises:
        ConfigError: On file write error
    """
    # Coerce str to Path if needed
    if isinstance(path, str):
        path = Path(path)

    try:
        data = model.model_dump(exclude_unset=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        raise ConfigError(f"Failed to save configuration to {path}: {e}") from e
