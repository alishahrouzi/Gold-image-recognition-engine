"""Image preprocessing and optional training-only augmentation.

Deterministic path (S1.8): RGB → resize → float32 CHW tensor → normalize.
Training path (S1.9): RGB → TrainingAugmentor → ImagePreprocessor.
"""

from .augmentation import (
    AugmentationConfig,
    BrightnessConfig,
    ColorConfig,
    ContrastConfig,
    DETERMINISTIC_ROLES,
    HorizontalFlipConfig,
    PIPELINE_ROLES,
    RandomCropConfig,
    RotationConfig,
    TRAINING_ROLE,
    TrainingAugmentor,
    augmentor_for_role,
)
from .config import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_INTERPOLATION,
    DEFAULT_MEAN,
    DEFAULT_STD,
    ImagePreprocessingConfig,
)
from .pipeline import ImagePreprocessor, PreprocessedDataset, build_preprocessed_dataset

__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_INTERPOLATION",
    "DEFAULT_MEAN",
    "DEFAULT_STD",
    "DETERMINISTIC_ROLES",
    "PIPELINE_ROLES",
    "TRAINING_ROLE",
    "AugmentationConfig",
    "BrightnessConfig",
    "ColorConfig",
    "ContrastConfig",
    "HorizontalFlipConfig",
    "ImagePreprocessingConfig",
    "ImagePreprocessor",
    "PreprocessedDataset",
    "RandomCropConfig",
    "RotationConfig",
    "TrainingAugmentor",
    "augmentor_for_role",
    "build_preprocessed_dataset",
]
