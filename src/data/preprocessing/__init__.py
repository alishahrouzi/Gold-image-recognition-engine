"""Deterministic image preprocessing (Sprint S1.8).

Converts loaded images to encoder-ready tensors. Does not load datasets,
modify files, or apply training augmentations.

Pipeline: RGB (in memory) → resize → float32 CHW tensor in [0, 1] → normalize.
"""

from .config import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_INTERPOLATION,
    DEFAULT_MEAN,
    DEFAULT_STD,
    ImagePreprocessingConfig,
)
from .pipeline import ImagePreprocessor, PreprocessedDataset

__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_INTERPOLATION",
    "DEFAULT_MEAN",
    "DEFAULT_STD",
    "ImagePreprocessingConfig",
    "ImagePreprocessor",
    "PreprocessedDataset",
]
