"""Configuration manager for multiple configurations."""

from pathlib import Path
from typing import Dict, Optional

from .loader import load_yaml, save_yaml
from .models import FeatureCompressorConfig


class ConfigManager:
    """Singleton configuration manager supporting multiple configs."""

    _instance: Optional["ConfigManager"] = None
    _configs: Dict[str, FeatureCompressorConfig] = {}

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.config_dir = Path(__file__).parent

    def load_config(
        self, config_name: str = "feature_compressor"
    ) -> FeatureCompressorConfig:
        """Load configuration from YAML file.

        Args:
            config_name: Name of config file without extension

        Returns:
            Loaded and validated FeatureCompressorConfig instance
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        config = load_yaml(config_path, FeatureCompressorConfig)
        self._configs[config_name] = config
        return config

    def load_all_configs(self) -> Dict[str, FeatureCompressorConfig]:
        """Load all YAML configuration files from config directory.

        Returns:
            Dictionary mapping config names to FeatureCompressorConfig instances
        """
        config_files = list(self.config_dir.glob("*.yaml"))
        loaded_configs = {}

        for config_path in config_files:
            config_name = config_path.stem
            if config_name.startswith("_"):
                continue  # Skip private configs
            config = load_yaml(config_path, FeatureCompressorConfig)
            loaded_configs[config_name] = config

        self._configs.update(loaded_configs)
        return loaded_configs

    def get_config(
        self, config_name: str = "feature_compressor"
    ) -> FeatureCompressorConfig:
        """Get loaded configuration by name.

        Args:
            config_name: Name of configuration to retrieve

        Returns:
            FeatureCompressorConfig instance

        Raises:
            ValueError: If configuration not loaded
        """
        if config_name not in self._configs:
            raise ValueError(
                f"Configuration '{config_name}' not loaded. "
                f"Call load_config('{config_name}') or load_all_configs() first."
            )
        return self._configs[config_name]

    def get_or_load_config(
        self, config_name: str = "feature_compressor"
    ) -> FeatureCompressorConfig:
        """Get loaded configuration by name or load it if not loaded.

        Args:
            config_name: Name of configuration to retrieve

        Returns:
            FeatureCompressorConfig instance

        Raises:
            ValueError: If configuration not loaded
        """
        if config_name not in self._configs:
            return self.load_config(config_name)
        return self._configs[config_name]

    def list_configs(self) -> list[str]:
        """List names of all currently loaded configurations.

        Returns:
            List of configuration names
        """
        return list(self._configs.keys())

    def save_config(
        self,
        config_name: str = "feature_compressor",
        config: Optional[FeatureCompressorConfig] = None,
        path: Optional[Path] = None,
    ) -> None:
        """Save configuration to YAML file.

        Args:
            config_name: Name of config file without extension
            config: Configuration to save. If None, saves currently loaded config for that name.
            path: Optional custom path to save to. If None, saves to config_dir.

        Raises:
            ValueError: If no configuration loaded for the given name and config is None
        """
        if config is None:
            if config_name not in self._configs:
                raise ValueError(
                    f"No configuration loaded for '{config_name}'. "
                    f"Cannot save without providing config parameter."
                )
            config = self._configs[config_name]

        save_path = path if path else self.config_dir / f"{config_name}.yaml"
        save_yaml(save_path, config)
        self._configs[config_name] = config


# Global singleton instance
cfg_manager = ConfigManager()
