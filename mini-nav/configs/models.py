"""Pydantic data models for feature compressor configuration."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for the vision model and compression."""

    model_config = ConfigDict(extra="ignore")

    name: str = "facebook/dinov2-large"
    compression_dim: int = Field(
        default=512, gt=0, description="Output feature dimension"
    )
    device: str = "auto"
    sam_model: str = Field(
        default="facebook/sam2.1-hiera-large",
        description="SAM model name from HuggingFace",
    )
    sam_min_mask_area: int = Field(
        default=100, gt=0, description="Minimum mask area threshold"
    )
    sam_max_masks: int = Field(
        default=10, gt=0, description="Maximum number of masks to keep"
    )
    compressor_path: Optional[str] = Field(
        default=None, description="Path to trained HashCompressor weights"
    )


class OutputConfig(BaseModel):
    """Configuration for output settings."""

    model_config = ConfigDict(extra="ignore")

    directory: Path = Path(__file__).parent.parent.parent / "outputs"

    @field_validator("directory", mode="after")
    def convert_to_absolute(cls, v: Path) -> Path:
        """Converts the path to an absolute path relative to the project root.

        This works even if the path doesn't exist on disk.
        """
        if v.is_absolute():
            return v
        return Path(__file__).parent.parent.parent / v


class DatasetConfig(BaseModel):
    """Configuration for synthetic dataset generation."""

    model_config = ConfigDict(extra="ignore")

    dataset_root: Path = (
        Path(__file__).parent.parent.parent / "datasets" / "InsDet-FULL"
    )
    output_dir: Path = (
        Path(__file__).parent.parent.parent / "datasets" / "InsDet-FULL" / "Synthesized"
    )
    num_objects_range: tuple[int, int] = (3, 8)
    num_scenes: int = 1000
    object_scale_range: tuple[float, float] = (0.1, 0.4)
    rotation_range: tuple[int, int] = (-30, 30)
    overlap_threshold: float = 0.3
    seed: int = 42

    @field_validator("dataset_root", "output_dir", mode="after")
    def convert_to_absolute(cls, v: Path) -> Path:
        """Converts the path to an absolute path relative to the project root.

        This works even if the path doesn't exist on disk.
        """
        if v.is_absolute():
            return v
        return Path(__file__).parent.parent.parent / v

    @field_validator("num_objects_range", mode="after")
    def validate_num_objects(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] < 1 or v[1] < v[0]:
            raise ValueError("num_objects_range must have min >= 1 and min <= max")
        return v

    @field_validator("object_scale_range", mode="after")
    def validate_scale(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] <= 0 or v[1] <= 0 or v[1] < v[0]:
            raise ValueError(
                "object_scale_range must have positive values and min <= max"
            )
        return v

    @field_validator("overlap_threshold", mode="after")
    def validate_overlap(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("overlap_threshold must be between 0 and 1")
        return v


class Config(BaseModel):
    """Root configuration for the feature compressor."""

    model_config = ConfigDict(extra="ignore")

    model: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
