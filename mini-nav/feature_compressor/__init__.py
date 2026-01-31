"""DINOv2 Feature Compressor - Extract and compress visual features."""

__version__ = "0.1.0"

from .core.compressor import PoolNetCompressor
from .core.extractor import DINOv2FeatureExtractor
from .core.visualizer import FeatureVisualizer

__all__ = [
    "PoolNetCompressor",
    "DINOv2FeatureExtractor",
    "FeatureVisualizer",
    "__version__",
]
