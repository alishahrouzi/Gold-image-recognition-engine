"""Dataset ingestion layer (Sprint S1.1 / S1.2 / S1.3 / S1.4).

Manifest → Samples → UnifiedDataset → DataLoader (with collate_samples).
Image-level readability / quality reports: inspect_samples.
Duplicate detection reports: detect_duplicates (read-only).
Cleaning audit reports: audit_dataset_cleaning (non-destructive).
"""

from .cleaning import (
    CleaningAuditConfig,
    audit_dataset_cleaning,
    write_cleaning_report,
)
from .collate import CollatedBatch, collate_samples
from .constants import (
    ALLOWED_SPLITS,
    CATEGORY_TO_ID,
    ID_TO_CATEGORY,
    SOURCE_DATASET1,
    SPLIT_ORDER,
)
from .duplicates import (
    DuplicateDetectionConfig,
    detect_duplicates,
    write_duplicate_report,
)
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

# UnifiedDataset subclasses torch.utils.data.Dataset. Import it lazily so
# S1.2/S1.3/S1.4 QA modules do not require PyTorch at import time.
from typing import Any as _Any

def __getattr__(name: str) -> _Any:
    if name == "UnifiedDataset":
        from .datasets.unified_dataset import UnifiedDataset

        return UnifiedDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ALLOWED_SPLITS",
    "CATEGORY_TO_ID",
    "SPLIT_ORDER",
    "CleaningAuditConfig",
    "CollatedBatch",
    "Dataset",
    "DatasetIngestionError",
    "DatasetItem",
    "DatasetValidationReport",
    "DuplicateDetectionConfig",
    "ID_TO_CATEGORY",
    "ImageQualityResult",
    "SOURCE_DATASET1",
    "Sample",
    "UnifiedDataset",
    "audit_dataset_cleaning",
    "build_image_validation_report",
    "build_validation_report",
    "collate_samples",
    "detect_duplicates",
    "inspect_sample",
    "inspect_samples",
    "load_manifest",
    "load_rgb_image",
    "validate_samples",
    "write_cleaning_report",
    "write_duplicate_report",
    "write_image_validation_report",
]
