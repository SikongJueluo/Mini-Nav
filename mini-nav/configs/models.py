"""Pydantic data models for feature compressor configuration."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PoolingType(str, Enum):
    """Enum for pooling types."""

    ATTENTION = "attention"


class ModelConfig(BaseModel):
    """Configuration for the vision model and compression."""

    model_config = ConfigDict(extra="ignore")

    name: str = "facebook/dinov2-large"
    compression_dim: int = Field(
        default=512, gt=0, description="Output feature dimension"
    )
    device: str = "auto"


class OutputConfig(BaseModel):
    """Configuration for output settings."""

    model_config = ConfigDict(extra="ignore")

    directory: Path = Path(__file__).parent.parent.parent / "outputs"

    @field_validator("directory", mode="after")
    def convert_to_absolute(cls, v: Path) -> Path:
        """
        Converts the path to an absolute path relative to the current working directory.
        This works even if the path doesn't exist on disk.
        """
        if v.is_absolute():
            return v
        return Path(__file__).parent.parent.parent / v


class Config(BaseModel):
    """Root configuration for the feature compressor."""

    model_config = ConfigDict(extra="ignore")

    model: ModelConfig
    output: OutputConfig
