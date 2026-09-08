"""Dataset-1 integrity and leakage guards used during S2.8 evaluation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.constants import SOURCE_DATASET1, SPLIT_ORDER
from src.data.types import Sample
from src.data.validation import build_validation_report, validate_samples

from .errors import EvaluationError

DATASET1_EXPECTED = {
    "total_images": 4969,
    "total_groups": 2135,
    "train_images": 4328,
    "train_groups": 1494,
    "valid_images": 429,
    "valid_groups": 429,
    "test_images": 212,
    "test_groups": 212,
}


def manifest_sha256(manifest_path: str | Path) -> str:
    """Return the SHA-256 digest of the manifest file bytes."""
    path = Path(manifest_path)
    if not path.is_file():
        raise EvaluationError(f"Manifest does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_manifest_unchanged(manifest_path: str | Path, expected_sha256: str) -> None:
    """Fail if the manifest bytes changed during evaluation."""
    actual = manifest_sha256(manifest_path)
    if actual != expected_sha256:
        raise EvaluationError(
            "Manifest changed during evaluation; S2.8 is read-only with respect "
            f"to the dataset. expected={expected_sha256} actual={actual}"
        )


def verify_dataset1_contract(samples: Sequence[Sample], *, validate_files: bool = False) -> dict[str, Any]:
    """Validate Dataset 1 counts, split isolation, and source labels."""
    if validate_files:
        report = validate_samples(samples, validate_files=True)
    else:
        validate_samples(samples, validate_files=False)
        report = build_validation_report(samples)

    splits_by_group: dict[str, set[str]] = defaultdict(set)
    sources = {sample.source for sample in samples}
    for sample in samples:
        splits_by_group[sample.group_id].add(sample.split)
    cross_split_groups = sorted(
        group_id for group_id, splits in splits_by_group.items() if len(splits) > 1
    )

    expected = DATASET1_EXPECTED
    mismatches: dict[str, Any] = {}
    actual = {
        "total_images": report.total_samples,
        "total_groups": report.total_groups,
        "train_images": report.split_counts.get("train", 0),
        "train_groups": report.groups_per_split.get("train", 0),
        "valid_images": report.split_counts.get("valid", 0),
        "valid_groups": report.groups_per_split.get("valid", 0),
        "test_images": report.split_counts.get("test", 0),
        "test_groups": report.groups_per_split.get("test", 0),
    }
    for key, value in expected.items():
        if actual[key] != value:
            mismatches[key] = {"expected": value, "actual": actual[key]}

    if mismatches:
        raise EvaluationError(f"Dataset 1 contract mismatch: {mismatches}")
    if cross_split_groups:
        raise EvaluationError(
            "Cross-split product groups detected; evaluation is invalid: "
            f"{cross_split_groups[:10]}"
        )
    if sources != {SOURCE_DATASET1}:
        raise EvaluationError(f"Unexpected dataset sources: {sorted(sources)}")

    return {
        "dataset_source": SOURCE_DATASET1,
        "split_order": list(SPLIT_ORDER),
        "cross_split_groups": [],
        "counts": actual,
        "file_validation": validate_files,
    }


def assert_same_split_protocol(query_split: str, gallery_split: str) -> None:
    """S2.8 baseline evaluation is same-split leave-one-image-out.

    Dataset 1 has zero cross-split group overlap, so a query from valid/test
    against a train-only gallery cannot have a same-product positive.
    """
    if query_split != gallery_split:
        raise EvaluationError(
            "S2.8 baseline evaluation requires query_split == gallery_split. "
            f"Got query_split={query_split!r}, gallery_split={gallery_split!r}. "
            "Dataset 1 has no cross-split product groups, so a different-split "
            "gallery cannot contain a same-product positive."
        )


def assert_gallery_metadata_aligned(metadata: Sequence[Mapping[str, Any]], embeddings_rows: int) -> None:
    """Fail if gallery rows, IDs, or group labels are incomplete or duplicated."""
    if embeddings_rows != len(metadata):
        raise EvaluationError(
            "Gallery embeddings and metadata are misaligned: "
            f"embeddings={embeddings_rows} metadata={len(metadata)}"
        )
    image_ids: list[str] = []
    for index, item in enumerate(metadata):
        image_id = str(item.get("image_id") or "").strip()
        group_id = str(item.get("product_group") or item.get("group_id") or "").strip()
        if not image_id:
            raise EvaluationError(f"Gallery metadata row {index} is missing image_id.")
        if not group_id:
            raise EvaluationError(
                f"Gallery metadata row {index} (image_id={image_id!r}) is missing group_id."
            )
        image_ids.append(image_id)
    if len(image_ids) != len(set(image_ids)):
        raise EvaluationError("Gallery metadata contains duplicate image_id values.")
