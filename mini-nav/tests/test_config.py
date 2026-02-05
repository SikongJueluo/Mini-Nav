"""Tests for configuration system using Pydantic models."""

import tempfile
from pathlib import Path

import pytest
import yaml
from configs import (
    ConfigError,
    ConfigManager,
    FeatureCompressorConfig,
    ModelConfig,
    OutputConfig,
    PoolingType,
    VisualizationConfig,
    cfg_manager,
    load_yaml,
    save_yaml,
)
from pydantic import ValidationError


class TestConfigModels:
    """Test suite for Pydantic configuration models."""

    def test_model_config_defaults(self):
        """Verify ModelConfig creates with correct defaults."""
        config = ModelConfig()
        assert config.name == "facebook/dinov2-large"
        assert config.compression_dim == 256
        assert config.pooling_type == PoolingType.ATTENTION
        assert config.top_k_ratio == 0.5
        assert config.hidden_ratio == 2.0
        assert config.dropout_rate == 0.1
        assert config.use_residual is True
        assert config.device == "auto"

    def test_model_config_validation(self):
        """Test validation constraints for ModelConfig."""
        # Test compression_dim > 0
        with pytest.raises(ValidationError, match="greater than 0"):
            ModelConfig(compression_dim=0)

        with pytest.raises(ValidationError, match="greater than 0"):
            ModelConfig(compression_dim=-1)

        # Test top_k_ratio in [0, 1]
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            ModelConfig(top_k_ratio=1.5)

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ModelConfig(top_k_ratio=-0.1)

        # Test dropout_rate in [0, 1]
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            ModelConfig(dropout_rate=1.5)

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ModelConfig(dropout_rate=-0.1)

        # Test hidden_ratio > 0
        with pytest.raises(ValidationError, match="greater than 0"):
            ModelConfig(hidden_ratio=0)

        with pytest.raises(ValidationError, match="greater than 0"):
            ModelConfig(hidden_ratio=-1)

    def test_visualization_config_defaults(self):
        """Verify VisualizationConfig creates with correct defaults."""
        config = VisualizationConfig()
        assert config.plot_theme == "plotly_white"
        assert config.color_scale == "viridis"
        assert config.point_size == 8
        assert config.fig_width == 900
        assert config.fig_height == 600

    def test_visualization_config_validation(self):
        """Test validation constraints for VisualizationConfig."""
        # Test fig_width > 0
        with pytest.raises(ValidationError, match="greater than 0"):
            VisualizationConfig(fig_width=0)

        with pytest.raises(ValidationError, match="greater than 0"):
            VisualizationConfig(fig_width=-1)

        # Test fig_height > 0
        with pytest.raises(ValidationError, match="greater than 0"):
            VisualizationConfig(fig_height=0)

        with pytest.raises(ValidationError, match="greater than 0"):
            VisualizationConfig(fig_height=-1)

        # Test point_size > 0
        with pytest.raises(ValidationError, match="greater than 0"):
            VisualizationConfig(point_size=0)

        with pytest.raises(ValidationError, match="greater than 0"):
            VisualizationConfig(point_size=-1)

    def test_output_config_defaults(self):
        """Verify OutputConfig creates with correct defaults."""
        config = OutputConfig()
        output_dir = Path(__file__).parent.parent.parent / "outputs"

        assert config.directory == output_dir
        assert config.html_self_contained is True
        assert config.png_scale == 2

    def test_output_config_validation(self):
        """Test validation constraints for OutputConfig."""
        # Test png_scale > 0
        with pytest.raises(ValidationError, match="greater than 0"):
            OutputConfig(png_scale=0)

        with pytest.raises(ValidationError, match="greater than 0"):
            OutputConfig(png_scale=-1)

    def test_pooling_type_enum(self):
        """Verify PoolingType enum values."""
        assert PoolingType.ATTENTION.value == "attention"
        assert PoolingType.ATTENTION == PoolingType("attention")

    def test_feature_compressor_config(self):
        """Verify FeatureCompressorConfig nests all models correctly."""
        model_cfg = ModelConfig(compression_dim=512)
        viz_cfg = VisualizationConfig(point_size=16)
        out_cfg = OutputConfig(directory="/tmp/outputs")

        config = FeatureCompressorConfig(
            model=model_cfg,
            visualization=viz_cfg,
            output=out_cfg,
        )

        assert config.model.compression_dim == 512
        assert config.visualization.point_size == 16
        assert config.output.directory == Path("/tmp/outputs")


class TestYamlLoader:
    """Test suite for YAML loading and saving."""

    def test_load_existing_yaml(self):
        """Load config.yaml and verify values."""
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        config = load_yaml(config_path, FeatureCompressorConfig)

        # Verify model config
        assert config.model.name == "facebook/dinov2-large"
        assert config.model.compression_dim == 256
        assert config.model.pooling_type == PoolingType.ATTENTION
        assert config.model.top_k_ratio == 0.5
        assert config.model.hidden_ratio == 2.0
        assert config.model.dropout_rate == 0.1
        assert config.model.use_residual is True

        # Verify visualization config
        assert config.visualization.plot_theme == "plotly_white"
        assert config.visualization.color_scale == "viridis"
        assert config.visualization.point_size == 8
        assert config.visualization.fig_width == 900
        assert config.visualization.fig_height == 600

        # Verify output config
        output_dir = Path(__file__).parent.parent.parent / "outputs"

        assert config.output.directory == output_dir
        assert config.output.html_self_contained is True
        assert config.output.png_scale == 2

    def test_load_yaml_validation(self):
        """Test that invalid data raises ConfigError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Write invalid config (missing required fields)
            yaml.dump({"invalid": "data"}, f)
            temp_path = f.name

        try:
            with pytest.raises(ConfigError, match="validation failed"):
                load_yaml(Path(temp_path), FeatureCompressorConfig)
        finally:
            Path(temp_path).unlink()

    def test_save_yaml_roundtrip(self):
        """Create config, save to temp, verify file exists with content."""
        original = cfg_manager.load()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_yaml(temp_path, original)

            # Verify file exists and has content
            assert Path(temp_path).exists()
            with open(temp_path, "r") as f:
                content = f.read()
                assert len(content) > 0
                assert "model" in content
                assert "visualization" in content
                assert "output" in content
        finally:
            Path(temp_path).unlink()

    def test_load_yaml_file_not_found(self):
        """Verify FileNotFoundError raises ConfigError."""
        with pytest.raises(ConfigError, match="not found"):
            load_yaml(Path("/nonexistent/path/config.yaml"), FeatureCompressorConfig)


class TestConfigManager:
    """Test suite for ConfigManager singleton with multi-config support."""

    def test_singleton_pattern(self):
        """Verify ConfigManager() returns same instance."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_load_config(self):
        """Test loading default config."""
        config = cfg_manager.load()

        assert config is not None
        assert config.model.compression_dim == 256
        assert config.visualization.point_size == 8

    def test_get_without_load(self):
        """Test that get() auto-loads config if not loaded."""
        # Reset the singleton's cached config
        cfg_manager._config = None

        # get() should auto-load
        config = cfg_manager.get()
        assert config is not None
        assert config.model.compression_dim == 256

    def test_save_config(self):
        """Test saving configuration to file."""
        config = FeatureCompressorConfig(
            model=ModelConfig(compression_dim=512),
            visualization=VisualizationConfig(),
            output=OutputConfig(),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            cfg_manager.save(config, path=temp_path)
            loaded_config = load_yaml(temp_path, FeatureCompressorConfig)

            assert loaded_config.model.compression_dim == 512
        finally:
            if temp_path.exists():
                temp_path.unlink()
