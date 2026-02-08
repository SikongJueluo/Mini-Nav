"""Configuration manager for unified config."""

from pathlib import Path
from typing import Optional

from .loader import load_yaml, save_yaml
from .models import Config


class ConfigManager:
    """Singleton configuration manager for unified config."""

    _instance: Optional["ConfigManager"] = None
    _config: Optional[Config] = None

    def __new__(cls) -> "ConfigManager":
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize config manager with config directory and path."""
        self.config_dir = Path(__file__).parent
        self.config_path = self.config_dir / "config.yaml"

    def load(self) -> Config:
        """Load configuration from config.yaml file.

        Returns:
            Loaded and validated FeatureCompressorConfig instance
        """
        config = load_yaml(self.config_path, Config)
        self._config = config
        return config

    def get(self) -> Config:
        """Get loaded configuration, auto-loading if not already loaded.

        Returns:
            FeatureCompressorConfig instance

        Note:
            Automatically loads config if not already loaded
        """
        # Auto-load if config not yet loaded
        if self._config is None:
            return self.load()
        return self._config

    def save(
        self,
        config: Optional[Config] = None,
        path: Optional[Path] = None,
    ) -> None:
        """Save configuration to YAML file.

        Args:
            config: Configuration to save. If None, saves currently loaded config.
            path: Optional custom path. If None, saves to default config.yaml.

        Raises:
            ValueError: If no configuration loaded and config is None
        """
        # Use provided config or fall back to loaded config
        if config is None:
            if self._config is None:
                raise ValueError(
                    "No configuration loaded. "
                    "Cannot save without providing config parameter."
                )
            config = self._config

        # Save to custom path or default config.yaml
        save_path = path if path else self.config_path
        save_yaml(save_path, config)
        self._config = config


# Global singleton instance
cfg_manager = ConfigManager()
