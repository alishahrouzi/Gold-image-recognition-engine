"""Read-only Dataset 1 visualization for data QA (S1.11).

Does not implement encoders, embeddings, retrieval, or pair generation.
"""

from .config import (
    DEFAULT_AUGMENTATION_SAMPLES,
    DEFAULT_NEGATIVE_PAIRS,
    DEFAULT_POSITIVE_PAIRS,
    DEFAULT_TEST_SAMPLES,
    DEFAULT_TRAIN_SAMPLES,
    DEFAULT_VALID_SAMPLES,
    DEFAULT_VISUALIZATION_SEED,
    VisualizationConfig,
)
from .generate import (
    VisualizationResult,
    generate_dataset_visualizations,
    load_visualization_inputs,
)
from .report import build_visualization_report, write_visualization_report
from .sampler import build_selection, sample_train_groups
from .types import GroupPanel, PairPanel, VisualizationSelection
from .validation import (
    validate_image_ready,
    validate_negative_pair,
    validate_positive_pair,
    validate_selection,
)

__all__ = [
    "DEFAULT_AUGMENTATION_SAMPLES",
    "DEFAULT_NEGATIVE_PAIRS",
    "DEFAULT_POSITIVE_PAIRS",
    "DEFAULT_TEST_SAMPLES",
    "DEFAULT_TRAIN_SAMPLES",
    "DEFAULT_VALID_SAMPLES",
    "DEFAULT_VISUALIZATION_SEED",
    "GroupPanel",
    "PairPanel",
    "VisualizationConfig",
    "VisualizationResult",
    "VisualizationSelection",
    "build_selection",
    "build_visualization_report",
    "generate_dataset_visualizations",
    "load_visualization_inputs",
    "sample_train_groups",
    "validate_image_ready",
    "validate_negative_pair",
    "validate_positive_pair",
    "validate_selection",
    "write_visualization_report",
]
