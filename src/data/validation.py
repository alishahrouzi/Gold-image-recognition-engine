"""Integrity checks for ingested Samples.

Counts are derived from the manifest. Production totals are not hard-coded.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

from .constants import ALLOWED_SPLITS, CATEGORY_TO_ID, SOURCE_DATASET1, SPLIT_ORDER
from .errors import DatasetIngestionError
from .types import Sample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetValidationReport:
    """Summary produced after a successful contract check."""

    total_samples: int
    total_groups: int
    split_counts: Dict[str, int]
    category_counts: Dict[str, int]
    groups_per_split: Dict[str, int]


def validate_samples(
    samples: Sequence[Sample],
    *,
    validate_files: bool = True,
) -> DatasetValidationReport:
    """Validate group integrity, splits, sources, and optional file existence.

    Raises DatasetIngestionError on the first contract violation. Does not
    repair the manifest.
    """
    if not samples:
        raise DatasetIngestionError("Dataset contains no samples.")

    _validate_unique_image_ids(samples)
    _validate_group_ids(samples)
    _validate_cross_split_groups(samples)
    _validate_vocabulary(samples)

    if validate_files:
        _validate_image_files(samples)

    report = build_validation_report(samples)
    logger.info(
        "Dataset validation passed: samples=%s groups=%s splits=%s categories=%s",
        report.total_samples,
        report.total_groups,
        dict(report.split_counts),
        dict(report.category_counts),
    )
    return report


def build_validation_report(samples: Sequence[Sample]) -> DatasetValidationReport:
    """Compute counts from samples without additional contract checks."""
    split_counts = Counter(sample.split for sample in samples)
    category_counts = Counter(sample.category for sample in samples)
    groups_per_split: Dict[str, set[str]] = defaultdict(set)
    all_groups = set()
    for sample in samples:
        all_groups.add(sample.group_id)
        groups_per_split[sample.split].add(sample.group_id)

    return DatasetValidationReport(
        total_samples=len(samples),
        total_groups=len(all_groups),
        split_counts={split: split_counts.get(split, 0) for split in SPLIT_ORDER},
        category_counts={name: category_counts.get(name, 0) for name in CATEGORY_TO_ID},
        groups_per_split={
            split: len(groups_per_split.get(split, set())) for split in SPLIT_ORDER
        },
    )


def _validate_unique_image_ids(samples: Sequence[Sample]) -> None:
    seen: Dict[str, int] = {}
    for sample in samples:
        if sample.image_id in seen:
            raise DatasetIngestionError(
                f"Duplicate image_id {sample.image_id!r}."
            )
        seen[sample.image_id] = 1


def _validate_group_ids(samples: Sequence[Sample]) -> None:
    for sample in samples:
        if not sample.group_id:
            raise DatasetIngestionError(
                f"Missing group_id for image {sample.image_id!r}."
            )


def _validate_cross_split_groups(samples: Sequence[Sample]) -> None:
    splits_by_group: Dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        splits_by_group[sample.group_id].add(sample.split)

    for group_id, splits in splits_by_group.items():
        if len(splits) > 1:
            split_list = ", ".join(name for name in SPLIT_ORDER if name in splits)
            raise DatasetIngestionError(
                f"Group {group_id!r} appears in multiple splits: {split_list}."
            )


def _validate_vocabulary(samples: Sequence[Sample]) -> None:
    for sample in samples:
        if sample.category not in CATEGORY_TO_ID:
            raise DatasetIngestionError(
                f"Unknown category {sample.category!r} for image {sample.image_id!r}."
            )
        expected = CATEGORY_TO_ID[sample.category]
        if sample.category_id != expected:
            raise DatasetIngestionError(
                f"category_id {sample.category_id} does not match category "
                f"{sample.category!r} (expected {expected}) for image "
                f"{sample.image_id!r}."
            )
        if sample.split not in ALLOWED_SPLITS:
            allowed = "/".join(SPLIT_ORDER)
            raise DatasetIngestionError(
                f"Invalid split {sample.split!r}. Expected {allowed}."
            )
        if sample.source != SOURCE_DATASET1:
            raise DatasetIngestionError(
                f"Invalid source {sample.source!r} for image {sample.image_id!r}."
            )


def _validate_image_files(samples: Sequence[Sample]) -> None:
    for sample in samples:
        if not Path(sample.image_path).is_file():
            raise DatasetIngestionError(
                f"Image file does not exist: {sample.image_path}"
            )
