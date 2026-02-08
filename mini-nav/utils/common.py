from functools import lru_cache
from pathlib import Path

import torch
from configs import cfg_manager
from torch.types import Device


@lru_cache(maxsize=1)
def get_device() -> Device:
    config = cfg_manager.get()
    device = config.model.device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    return device


@lru_cache(maxsize=1)
def get_output_diretory() -> Path:
    config = cfg_manager.get()
    return config.output.directory
