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
        default=256, gt=0, description="Output feature dimension"
    )
    pooling_type: PoolingType = PoolingType.ATTENTION
    top_k_ratio: float = Field(
        default=0.5, ge=0, le=1, description="Ratio of tokens to keep"
    )
    hidden_ratio: float = Field(
        default=2.0, gt=0, description="MLP hidden dim as multiple of compression_dim"
    )
    dropout_rate: float = Field(
        default=0.1, ge=0, le=1, description="Dropout probability"
    )
    use_residual: bool = True
    device: str = "auto"


class VisualizationConfig(BaseModel):
    """Configuration for visualization settings."""

    model_config = ConfigDict(extra="ignore")

    plot_theme: str = "plotly_white"
    color_scale: str = "viridis"
    point_size: int = Field(default=8, gt=0)
    fig_width: int = Field(default=900, gt=0)
    fig_height: int = Field(default=600, gt=0)


class OutputConfig(BaseModel):
    """Configuration for output settings."""

    model_config = ConfigDict(extra="ignore")

    directory: Path = Path(__file__).parent.parent.parent / "outputs"
    html_self_contained: bool = True
    png_scale: int = Field(default=2, gt=0)

    @field_validator("directory", mode="after")
    def convert_to_absolute(cls, v: Path) -> Path:
        """
        Converts the path to an absolute path relative to the current working directory.
        This works even if the path doesn't exist on disk.
        """
        if v.is_absolute():
            return v
        return Path(__file__).parent.parent.parent / v


class FeatureCompressorConfig(BaseModel):
    """Root configuration for the feature compressor."""

    model_config = ConfigDict(extra="ignore")

    model: ModelConfig
    visualization: VisualizationConfig
    output: OutputConfig
