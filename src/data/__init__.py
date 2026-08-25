"""Dataset ingestion layer (Sprint S1.1 / S1.2).

Manifest → Samples → UnifiedDataset → DataLoader (with collate_samples).
Image-level readability / quality reports: inspect_samples.
"""

from .collate import CollatedBatch, collate_samples
from .constants import (
    ALLOWED_SPLITS,
    CATEGORY_TO_ID,
    ID_TO_CATEGORY,
    SOURCE_DATASET1,
    SPLIT_ORDER,
)
from .datasets.unified_dataset import UnifiedDataset
from .errors import DatasetIngestionError
from .image_quality import (
    ImageQualityResult,
    build_image_validation_report,
    inspect_sample,
    inspect_samples,
    write_image_validation_report,
)
from .interfaces import Dataset
from .loaders.image_loader import load_rgb_image
from .loaders.manifest import load_manifest
from .types import DatasetItem, Sample
from .validation import DatasetValidationReport, build_validation_report, validate_samples

__all__ = [
    "ALLOWED_SPLITS",
    "CATEGORY_TO_ID",
    "SPLIT_ORDER",
    "CollatedBatch",
    "Dataset",
    "DatasetIngestionError",
    "DatasetItem",
    "DatasetValidationReport",
    "ID_TO_CATEGORY",
    "ImageQualityResult",
    "SOURCE_DATASET1",
    "Sample",
    "UnifiedDataset",
    "build_image_validation_report",
    "build_validation_report",
    "collate_samples",
    "inspect_sample",
    "inspect_samples",
    "load_manifest",
    "load_rgb_image",
    "validate_samples",
    "write_image_validation_report",
]
