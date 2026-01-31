from enum import Enum
from pathlib import Path
from typing import Dict

import yaml


class Config(Enum):
    FEATURE_COMPRESSOR = "feature_compressor.yaml"


def get_config_dir() -> Path:
    return Path(__file__).parent


def get_default_config(config_type: Config) -> Dict[Unknown, Unknown]:
    config_path = get_config_dir() / config_type.value

    with open(config_path) as f:
        return yaml.safe_load(f)
