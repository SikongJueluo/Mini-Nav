"""Tests for configuration system using Pydantic models."""

import tempfile
from pathlib import Path

import pytest
import yaml
from configs import (
    Config,
    ConfigError,
    ConfigManager,
    ModelConfig,
    OutputConfig,
    PoolingType,
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
        assert config.compression_dim == 512
        assert config.device == "auto"

    def test_model_config_validation(self):
        """Test validation constraints for ModelConfig."""
        # Test compression_dim > 0
        with pytest.raises(ValidationError, match="greater than 0"):
            ModelConfig(compression_dim=0)

        with pytest.raises(ValidationError, match="greater than 0"):
            ModelConfig(compression_dim=-1)

    def test_output_config_defaults(self):
        """Verify OutputConfig creates with correct defaults."""
        config = OutputConfig()
        output_dir = Path(__file__).parent.parent.parent / "outputs"

        assert config.directory == output_dir

    def test_pooling_type_enum(self):
        """Verify PoolingType enum values."""
        assert PoolingType.ATTENTION.value == "attention"
        assert PoolingType.ATTENTION == PoolingType("attention")

    def test_feature_compressor_config(self):
        """Verify FeatureCompressorConfig nests all models correctly."""
        model_cfg = ModelConfig(compression_dim=512)
        out_cfg = OutputConfig(directory="/tmp/outputs")

        config = Config(
            model=model_cfg,
            output=out_cfg,
        )

        assert config.model.compression_dim == 512
        assert config.output.directory == Path("/tmp/outputs")


class TestYamlLoader:
    """Test suite for YAML loading and saving."""

    def test_load_existing_yaml(self):
        """Load config.yaml and verify values."""
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        config = load_yaml(config_path, Config)

        # Verify model config
        assert config.model.name == "facebook/dinov2-large"
        assert config.model.compression_dim == 256

        # Verify output config
        output_dir = Path(__file__).parent.parent.parent / "outputs"

        assert config.output.directory == output_dir

    def test_load_yaml_validation(self):
        """Test that invalid data raises ConfigError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Write invalid config (missing required fields)
            yaml.dump({"invalid": "data"}, f)
            temp_path = f.name

        try:
            with pytest.raises(ConfigError, match="validation failed"):
                load_yaml(Path(temp_path), Config)
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
            load_yaml(Path("/nonexistent/path/config.yaml"), Config)


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
        config = Config(
            model=ModelConfig(compression_dim=512),
            output=OutputConfig(),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            cfg_manager.save(config, path=temp_path)
            loaded_config = load_yaml(temp_path, Config)

            assert loaded_config.model.compression_dim == 512
        finally:
            if temp_path.exists():
                temp_path.unlink()
