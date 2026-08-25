"""Dataset cleaning audit for Dataset 1 (Sprint S1.4).

This module is a **non-destructive** audit / consistency layer. It verifies
that the cleaned Dataset 1 baseline matches the manifest and QA contracts.

It never deletes, moves, renames, recompresses, or overwrites images, and it
never regenerates or rewrites the authoritative manifest.

Destructive cleanup already happened manually during Sprint 0 / Sprint 1.
Historical removal counts that cannot be derived from repository evidence are
reported as ``not_available_from_repository``.

Reusable dependencies
---------------------
S1.1
    ``load_manifest``, ``validate_samples`` — contract, groups, splits.
S1.2
    ``inspect_samples`` — readability, corruption, WARNING vs INVALID.
S1.3
    ``detect_duplicates`` — exact / perceptual / near duplicate detection.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from PIL import Image, UnidentifiedImageError

from .constants import (
    ALLOWED_SPLITS,
    CATEGORY_TO_ID,
    MANIFEST_REQUIRED_FIELDS,
    SOURCE_DATASET1,
    SPLIT_ORDER,
)
from .duplicates import (
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    DuplicateDetectionConfig,
    detect_duplicates,
)
from .errors import DatasetIngestionError
from .image_quality import (
    REASON_DECODE_ERROR,
    STATUS_INVALID,
    STATUS_WARNING,
    inspect_samples,
)
from .loaders.manifest import load_manifest
from .types import Sample
from .validation import build_validation_report, validate_samples

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

SCHEMA_VERSION = "1.0"
MODE_AUDIT = "audit"
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_UNEXPECTED = "unexpected_dataset_state"

NOT_AVAILABLE = "not_available_from_repository"
ALREADY_COMPLIANT = "already_compliant"
EXTERNAL_METADATA_N_A = (
    "External metadata correction is not applicable to Dataset 1 MVP."
)

# Dataset 1 cleaned baseline (authoritative QA totals).
BASELINE_IMAGE_COUNT = 4969
BASELINE_SPLIT_COUNTS: Dict[str, int] = {
    "train": 4328,
    "valid": 429,
    "test": 212,
}
BASELINE_GROUP_COUNT = 2135
BASELINE_GROUP_SIZE_DISTRIBUTION: Dict[str, int] = {
    "1": 655,
    "2": 126,
    "3": 1354,
    "4+": 0,
}

ALLOWED_SOURCE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg")
ALLOWED_DECODED_FORMATS = frozenset({"JPEG"})


@dataclass(frozen=True)
class CleaningAuditConfig:
    """Controls the S1.4 cleaning audit (always non-destructive)."""

    mode: str = MODE_AUDIT
    run_image_inspection: bool = True
    run_duplicate_detection: bool = True
    near_duplicate_threshold: int = DEFAULT_NEAR_DUPLICATE_THRESHOLD
    image_extensions: Tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS

    def normalized(self) -> "CleaningAuditConfig":
        mode = self.mode.lower().strip()
        if mode != MODE_AUDIT:
            raise ValueError(
                f"Unsupported cleaning mode {self.mode!r}. "
                f"Only {MODE_AUDIT!r} is implemented; destructive cleaning is "
                "disabled by design."
            )
        if self.near_duplicate_threshold < 0:
            raise ValueError("near_duplicate_threshold must be >= 0.")
        extensions = tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in self.image_extensions
        )
        if not extensions:
            raise ValueError("image_extensions must not be empty.")
        return CleaningAuditConfig(
            mode=mode,
            run_image_inspection=self.run_image_inspection,
            run_duplicate_detection=self.run_duplicate_detection,
            near_duplicate_threshold=self.near_duplicate_threshold,
            image_extensions=extensions,
        )


def audit_dataset_cleaning(
    dataset_root: PathLike,
    manifest_path: PathLike,
    *,
    config: Optional[CleaningAuditConfig] = None,
    image_quality_results: Optional[Sequence[Any]] = None,
    duplicate_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a non-destructive Dataset 1 cleaning audit.

    Args:
        dataset_root: Path to Dataset 1 (train/valid/test).
        manifest_path: Authoritative CSV manifest.
        config: Audit configuration (audit-only).
        image_quality_results: Optional precomputed S1.2 results (tests).
        duplicate_report: Optional precomputed S1.3 report (tests).

    Returns:
        Machine-readable cleaning audit report (schema 1.0).
    """
    cfg = (config or CleaningAuditConfig()).normalized()
    root = Path(dataset_root).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest}")

    logger.info("S1.4 cleaning audit (mode=%s) root=%s", cfg.mode, root)

    samples = load_manifest(manifest, dataset_root=root, validate_files=False)
    contract = _audit_contract(samples)
    filesystem = _audit_filesystem(root, samples, cfg.image_extensions)
    manifest_validation = _audit_manifest_consistency(samples, filesystem)
    # Do not embed the full path inventory in the published report.
    filesystem_public = {
        key: value
        for key, value in filesystem.items()
        if key != "relative_paths"
    }
    group_validation = _audit_groups(samples)
    metadata_policy = _audit_metadata_policy(manifest)

    if image_quality_results is not None:
        quality_results = list(image_quality_results)
    elif cfg.run_image_inspection:
        quality_results = inspect_samples(samples)
    else:
        quality_results = []

    quality = _summarize_quality(quality_results) if quality_results else None

    if duplicate_report is not None:
        dup_report = dict(duplicate_report)
    elif cfg.run_duplicate_detection:
        dup_report = detect_duplicates(
            root,
            config=DuplicateDetectionConfig(
                image_extensions=cfg.image_extensions,
                threshold=cfg.near_duplicate_threshold,
                comparison_mode="intra",
            ),
            manifest_a=manifest,
            dataset_a_name=root.name,
        )
    else:
        dup_report = None

    duplicates = _summarize_duplicates(dup_report) if dup_report else None
    format_policy = _audit_format_policy(samples, quality_results)

    unexpected = _collect_unexpected_states(
        quality=quality,
        duplicates=duplicates,
        manifest_validation=manifest_validation,
        group_validation=group_validation,
        contract=contract,
    )

    requirements = _build_requirements(
        quality=quality,
        duplicates=duplicates,
        format_policy=format_policy,
        metadata_policy=metadata_policy,
        unexpected=unexpected,
    )

    status = STATUS_PASS if not unexpected and _requirements_pass(requirements) else STATUS_FAIL

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "mode": cfg.mode,
        "destructive_operations": False,
        "dataset": {
            "name": "dataset1",
            "root": str(root),
            "source": SOURCE_DATASET1,
            "manifest": str(manifest),
        },
        "baseline": {
            "image_count": BASELINE_IMAGE_COUNT,
            "train": BASELINE_SPLIT_COUNTS["train"],
            "valid": BASELINE_SPLIT_COUNTS["valid"],
            "test": BASELINE_SPLIT_COUNTS["test"],
            "groups": BASELINE_GROUP_COUNT,
            "group_size_distribution": dict(BASELINE_GROUP_SIZE_DISTRIBUTION),
        },
        "cleaning_summary": {
            "corrupted_removed": 0,
            "duplicates_removed": NOT_AVAILABLE,
            "invalid_removed": NOT_AVAILABLE,
            "format_converted": 0,
            "metadata_modified": 0,
            "note": (
                "Historical exact/perceptual duplicate and invalid removals "
                "were performed manually in Sprint 0/1. Exact removal counts "
                "are not recorded in repository artifacts."
            ),
        },
        "current_state": {
            "images": len(samples),
            "filesystem_images": filesystem["total_images"],
            "readable": None if quality is None else quality["readable"],
            "corrupted": None if quality is None else quality["corrupted"],
            "warnings": None if quality is None else quality["warnings"],
            "invalid": None if quality is None else quality["invalid"],
            "exact_duplicates": (
                None if duplicates is None else duplicates["exact_duplicate_groups"]
            ),
            "perceptual_duplicates": (
                None
                if duplicates is None
                else duplicates["perceptual_duplicate_groups"]
            ),
            "near_duplicate_pairs": (
                None if duplicates is None else duplicates["near_duplicate_pairs"]
            ),
            "groups": group_validation["total_groups"],
            "cross_split_groups": group_validation["cross_split_group_count"],
        },
        "contract_validation": contract,
        "filesystem_validation": filesystem_public,
        "format_policy": format_policy,
        "metadata_policy": metadata_policy,
        "manifest_validation": manifest_validation,
        "group_validation": group_validation,
        "image_quality": quality,
        "duplicates": duplicates,
        "requirements": requirements,
        "unexpected_dataset_state": unexpected,
        "status": status,
    }
    return report


def write_cleaning_report(report: Mapping[str, Any], output_path: PathLike) -> Path:
    """Write the cleaning audit JSON. Does not modify images or the manifest."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("Wrote dataset cleaning report: %s", path)
    return path


def count_filesystem_images(
    dataset_root: PathLike,
    *,
    extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> Dict[str, Any]:
    """Count image files under Dataset 1 without modifying anything."""
    root = Path(dataset_root)
    allowed = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
    }
    by_split: Dict[str, int] = {split: 0 for split in SPLIT_ORDER}
    by_category: Dict[str, int] = {name: 0 for name in CATEGORY_TO_ID}
    paths: List[str] = []
    unexpected_paths: List[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        relative = path.relative_to(root).as_posix()
        parts = Path(relative).parts
        if len(parts) < 3:
            unexpected_paths.append(relative)
            continue
        split, category = parts[0], parts[1]
        if split not in ALLOWED_SPLITS or category not in CATEGORY_TO_ID:
            unexpected_paths.append(relative)
            continue
        by_split[split] += 1
        by_category[category] += 1
        paths.append(relative)

    return {
        "total_images": len(paths),
        "split_counts": by_split,
        "category_counts": by_category,
        "unexpected_paths": unexpected_paths,
        "relative_paths": paths,
    }


def group_size_distribution(samples: Sequence[Sample]) -> Dict[str, int]:
    """Return group-size histogram keys ``1``, ``2``, ``3``, ``4+``."""
    sizes: Dict[str, int] = defaultdict(int)
    for sample in samples:
        sizes[sample.group_id] += 1
    distribution = {"1": 0, "2": 0, "3": 0, "4+": 0}
    for count in sizes.values():
        if count <= 3:
            distribution[str(count)] += 1
        else:
            distribution["4+"] += 1
    return distribution


def _audit_contract(samples: Sequence[Sample]) -> Dict[str, Any]:
    issues: List[str] = []
    try:
        validate_samples(samples, validate_files=False)
    except DatasetIngestionError as exc:
        issues.append(str(exc))
    report = build_validation_report(samples)
    return {
        "passed": not issues,
        "issues": issues,
        "total_samples": report.total_samples,
        "total_groups": report.total_groups,
        "split_counts": dict(report.split_counts),
        "category_counts": dict(report.category_counts),
        "groups_per_split": dict(report.groups_per_split),
        "source_consistent": all(s.source == SOURCE_DATASET1 for s in samples),
        "category_ids_consistent": all(
            s.category_id == CATEGORY_TO_ID[s.category] for s in samples
        ),
    }


def _audit_filesystem(
    root: Path,
    samples: Sequence[Sample],
    extensions: Sequence[str],
) -> Dict[str, Any]:
    counted = count_filesystem_images(root, extensions=extensions)
    return {
        "dataset_root": str(root),
        "total_images": counted["total_images"],
        "split_counts": counted["split_counts"],
        "category_counts": counted["category_counts"],
        "unexpected_path_count": len(counted["unexpected_paths"]),
        "unexpected_paths": counted["unexpected_paths"][:50],
        "relative_paths": counted["relative_paths"],
        "matches_baseline_total": counted["total_images"] == BASELINE_IMAGE_COUNT,
        "matches_baseline_splits": counted["split_counts"] == BASELINE_SPLIT_COUNTS,
    }


def _audit_manifest_consistency(
    samples: Sequence[Sample],
    filesystem: Mapping[str, Any],
) -> Dict[str, Any]:
    missing = [
        sample.image_id
        for sample in samples
        if not Path(sample.image_path).is_file()
    ]
    manifest_paths = {
        Path(sample.image_path).resolve().as_posix() for sample in samples
    }
    root = Path(filesystem["dataset_root"])
    fs_paths = {
        (root / relative).resolve().as_posix()
        for relative in filesystem.get("relative_paths", [])
    }
    # Prefer relative comparison when filesystem listing is available.
    relative_manifest = set()
    for sample in samples:
        path = Path(sample.image_path)
        try:
            relative_manifest.add(path.relative_to(root).as_posix())
        except ValueError:
            relative_manifest.add(path.as_posix())
    fs_relative = set(filesystem.get("relative_paths", []))
    orphan_files = sorted(fs_relative - relative_manifest) if fs_relative else []
    orphan_manifest = sorted(relative_manifest - fs_relative) if fs_relative else []

    split_counts = Counter(sample.split for sample in samples)
    split_match = {
        split: split_counts.get(split, 0) == filesystem["split_counts"].get(split, 0)
        for split in SPLIT_ORDER
    }
    count_match = len(samples) == filesystem["total_images"]
    baseline_match = (
        len(samples) == BASELINE_IMAGE_COUNT
        and all(
            split_counts.get(split, 0) == expected
            for split, expected in BASELINE_SPLIT_COUNTS.items()
        )
    )
    passed = (
        count_match
        and all(split_match.values())
        and not missing
        and not orphan_files
        and not orphan_manifest
        and filesystem["unexpected_path_count"] == 0
    )
    return {
        "passed": passed,
        "manifest_count": len(samples),
        "filesystem_count": filesystem["total_images"],
        "counts_equal": count_match,
        "baseline_match": baseline_match,
        "split_counts_manifest": {split: split_counts.get(split, 0) for split in SPLIT_ORDER},
        "split_counts_filesystem": dict(filesystem["split_counts"]),
        "split_counts_equal": split_match,
        "missing_manifest_files": missing[:50],
        "missing_manifest_file_count": len(missing),
        "orphan_filesystem_files": orphan_files[:50],
        "orphan_filesystem_file_count": len(orphan_files),
        "orphan_manifest_entries": orphan_manifest[:50],
        "orphan_manifest_entry_count": len(orphan_manifest),
        # Retain resolved path sets only for debugging size, not full dump.
        "resolved_manifest_paths": len(manifest_paths),
        "resolved_filesystem_paths": len(fs_paths),
    }


def _audit_groups(samples: Sequence[Sample]) -> Dict[str, Any]:
    splits_by_group: Dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        splits_by_group[sample.group_id].add(sample.split)
    leaking = sorted(
        group_id
        for group_id, splits in splits_by_group.items()
        if len(splits) > 1
    )
    distribution = group_size_distribution(samples)
    total_groups = len(splits_by_group)
    enforce_baseline = len(samples) == BASELINE_IMAGE_COUNT
    matches_baseline = (
        total_groups == BASELINE_GROUP_COUNT
        and distribution == BASELINE_GROUP_SIZE_DISTRIBUTION
    )
    passed = not leaking and (matches_baseline if enforce_baseline else True)
    return {
        "passed": passed,
        "total_groups": total_groups,
        "expected_groups": BASELINE_GROUP_COUNT if enforce_baseline else None,
        "group_size_distribution": distribution,
        "expected_group_size_distribution": (
            dict(BASELINE_GROUP_SIZE_DISTRIBUTION) if enforce_baseline else None
        ),
        "enforce_baseline": enforce_baseline,
        "matches_baseline_distribution": matches_baseline if enforce_baseline else None,
        "cross_split_group_count": len(leaking),
        "cross_split_groups": leaking[:50],
        "no_split_leakage": len(leaking) == 0,
    }


def _audit_metadata_policy(manifest_path: Path) -> Dict[str, Any]:
    import csv

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
    required = set(MANIFEST_REQUIRED_FIELDS)
    missing = sorted(required - columns)
    return {
        "status": "compliant" if not missing else "non_compliant",
        "required_internal_fields": sorted(required),
        "present_fields": sorted(columns),
        "missing_required_fields": missing,
        "external_metadata": EXTERNAL_METADATA_N_A,
        "dataset2_metadata": "out_of_scope",
        "metadata_modified": 0,
    }


def _summarize_quality(results: Sequence[Any]) -> Dict[str, Any]:
    readable = sum(1 for item in results if item.readable)
    invalid = [item for item in results if item.status == STATUS_INVALID]
    warnings = [item for item in results if item.status == STATUS_WARNING]
    corrupted = [
        item for item in invalid if getattr(item, "reason", None) == REASON_DECODE_ERROR
    ]
    rgb = sum(
        1
        for item in results
        if item.readable and getattr(item, "original_mode", None) == "RGB"
    )
    rgb_convertible = sum(
        1 for item in results if item.readable and item.rgb_convertible
    )
    return {
        "total": len(results),
        "readable": readable,
        "corrupted": len(corrupted),
        "warnings": len(warnings),
        "invalid": len(invalid),
        "rgb": rgb,
        "rgb_convertible": rgb_convertible,
        "warning_not_treated_as_invalid": True,
        "corrupted_image_ids": [item.image_id for item in corrupted[:50]],
        "invalid_image_ids": [item.image_id for item in invalid[:50]],
        "warning_image_ids": [item.image_id for item in warnings[:50]],
        "passed": len(corrupted) == 0 and len(invalid) == 0,
    }


def _summarize_duplicates(report: Mapping[str, Any]) -> Dict[str, Any]:
    summary = report.get("summary", {})
    exact = int(summary.get("exact_duplicate_groups", 0))
    perceptual = int(summary.get("perceptual_duplicate_groups", 0))
    near = int(summary.get("near_duplicate_pairs", 0))
    search = summary.get("near_duplicate_search", {})
    return {
        "exact_duplicate_groups": exact,
        "perceptual_duplicate_groups": perceptual,
        "near_duplicate_pairs": near,
        "unreadable_images": int(summary.get("unreadable_images", 0)),
        "near_duplicate_search_status": search.get("status"),
        "exact_duplicates_absent": exact == 0,
        "perceptual_candidates_remain": perceptual > 0,
        "near_candidates_remain": near > 0,
        "policy": (
            "Exact duplicates must be absent. Remaining perceptual/near hits "
            "are candidates (not auto-confirmed). Manual Sprint 0/1 review "
            "removed confirmed duplicates; candidate collisions may remain."
        ),
        "passed": exact == 0,
    }


def _audit_format_policy(
    samples: Sequence[Sample],
    quality_results: Sequence[Any],
) -> Dict[str, Any]:
    extension_counts: Counter[str] = Counter()
    non_jpeg_extensions: List[str] = []
    for sample in samples:
        suffix = Path(sample.image_path).suffix.lower()
        extension_counts[suffix] += 1
        if suffix not in ALLOWED_SOURCE_EXTENSIONS:
            non_jpeg_extensions.append(sample.image_id)

    decoded_formats: Counter[str] = Counter()
    non_jpeg_decoded: List[str] = []
    # Spot-check decoded format for a subset when quality results exist,
    # otherwise inspect extensions only and sample-decode first N files.
    decode_limit = min(len(samples), 64) if not quality_results else 0
    for sample in samples[:decode_limit]:
        path = Path(sample.image_path)
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                fmt = image.format or "UNKNOWN"
        except (UnidentifiedImageError, OSError):
            fmt = "UNREADABLE"
        decoded_formats[fmt] += 1
        if fmt not in ALLOWED_DECODED_FORMATS:
            non_jpeg_decoded.append(sample.image_id)

    # When full inspection ran, all readable images imply decode worked;
    # Dataset 1 policy is JPEG-on-disk, RGB convertible in memory.
    extension_ok = not non_jpeg_extensions and set(extension_counts) <= set(
        ALLOWED_SOURCE_EXTENSIONS
    )
    rgb_ok = True
    if quality_results:
        rgb_ok = all(
            (not item.readable) or item.rgb_convertible for item in quality_results
        )

    status = ALREADY_COMPLIANT if extension_ok and rgb_ok else "non_compliant"
    return {
        "status": status,
        "policy": {
            "source_files_unchanged": True,
            "allowed_extensions": list(ALLOWED_SOURCE_EXTENSIONS),
            "allowed_decoded_formats": sorted(ALLOWED_DECODED_FORMATS),
            "rgb_conversion": "in_memory_only",
            "reencode": False,
            "preprocessing_belongs_elsewhere": True,
        },
        "extension_counts": dict(sorted(extension_counts.items())),
        "non_jpeg_extension_count": len(non_jpeg_extensions),
        "non_jpeg_extension_image_ids": non_jpeg_extensions[:50],
        "decoded_format_sample_counts": dict(sorted(decoded_formats.items())),
        "non_jpeg_decoded_sample_ids": non_jpeg_decoded[:50],
        "rgb_policy_satisfied": rgb_ok,
        "format_converted": 0,
    }


def _collect_unexpected_states(
    *,
    quality: Optional[Mapping[str, Any]],
    duplicates: Optional[Mapping[str, Any]],
    manifest_validation: Mapping[str, Any],
    group_validation: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    unexpected: List[Dict[str, Any]] = []
    if quality is not None and quality["corrupted"] > 0:
        unexpected.append(
            {
                "code": STATUS_UNEXPECTED,
                "check": "corrupted_images",
                "message": (
                    f"Found {quality['corrupted']} corrupted image(s). "
                    "Destructive removal is disabled; investigate manually."
                ),
                "image_ids": quality.get("corrupted_image_ids", []),
            }
        )
    if quality is not None and quality["invalid"] > 0:
        unexpected.append(
            {
                "code": STATUS_UNEXPECTED,
                "check": "invalid_images",
                "message": (
                    f"Found {quality['invalid']} invalid image(s). "
                    "Destructive removal is disabled."
                ),
                "image_ids": quality.get("invalid_image_ids", []),
            }
        )
    if duplicates is not None and duplicates["exact_duplicate_groups"] > 0:
        unexpected.append(
            {
                "code": STATUS_UNEXPECTED,
                "check": "exact_duplicates",
                "message": (
                    f"Found {duplicates['exact_duplicate_groups']} exact "
                    "duplicate group(s). Baseline after Sprint 0/1 cleanup "
                    "expected 0. Files were not deleted."
                ),
            }
        )
    if not manifest_validation["passed"]:
        unexpected.append(
            {
                "code": STATUS_UNEXPECTED,
                "check": "manifest_consistency",
                "message": "Manifest and filesystem counts/paths are inconsistent.",
            }
        )
    if not group_validation["no_split_leakage"]:
        unexpected.append(
            {
                "code": STATUS_UNEXPECTED,
                "check": "group_split_leakage",
                "message": "One or more groups span train/valid/test.",
                "groups": group_validation.get("cross_split_groups", []),
            }
        )
    if (
        group_validation.get("enforce_baseline")
        and not group_validation.get("matches_baseline_distribution")
    ):
        unexpected.append(
            {
                "code": STATUS_UNEXPECTED,
                "check": "group_baseline",
                "message": (
                    "Group count or size distribution does not match the "
                    "Dataset 1 cleaned baseline."
                ),
            }
        )
    if not contract["passed"]:
        unexpected.append(
            {
                "code": STATUS_UNEXPECTED,
                "check": "contract_validation",
                "message": "Manifest contract validation failed.",
                "issues": contract.get("issues", []),
            }
        )
    return unexpected


def _build_requirements(
    *,
    quality: Optional[Mapping[str, Any]],
    duplicates: Optional[Mapping[str, Any]],
    format_policy: Mapping[str, Any],
    metadata_policy: Mapping[str, Any],
    unexpected: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    unexpected_codes = {item.get("check") for item in unexpected}

    corrupted_status = "PASS"
    corrupted_evidence = "image inspection not run"
    corrupted_action = "none"
    if quality is not None:
        if quality["corrupted"] == 0:
            corrupted_status = "PASS"
            corrupted_evidence = (
                f"readable={quality['readable']}, corrupted={quality['corrupted']}"
            )
            corrupted_action = "none (already clean)"
        else:
            corrupted_status = "FAIL"
            corrupted_evidence = (
                f"corrupted={quality['corrupted']} image_ids="
                f"{quality.get('corrupted_image_ids', [])}"
            )
            corrupted_action = "report unexpected_dataset_state; do not delete"

    if duplicates is None:
        dup_status = "SKIPPED"
        dup_evidence = "duplicate detection not run"
        dup_action = "none"
    elif duplicates["exact_duplicate_groups"] == 0:
        dup_status = "PASS"
        dup_evidence = (
            f"exact_groups=0; perceptual_candidates="
            f"{duplicates['perceptual_duplicate_groups']}; "
            f"near_pairs={duplicates['near_duplicate_pairs']}"
        )
        dup_action = (
            "none (exact duplicates absent; perceptual/near remain candidates)"
        )
    else:
        dup_status = "FAIL"
        dup_evidence = (
            f"exact_groups={duplicates['exact_duplicate_groups']}"
        )
        dup_action = "report only; do not delete"

    format_status = (
        "PASS" if format_policy["status"] == ALREADY_COMPLIANT else "FAIL"
    )
    meta_status = (
        "PASS" if metadata_policy["status"] == "compliant" else "FAIL"
    )

    if quality is None:
        invalid_status = "SKIPPED"
        invalid_evidence = "image inspection not run"
        invalid_action = "none"
    elif quality["invalid"] == 0:
        invalid_status = "PASS"
        invalid_evidence = (
            f"invalid=0; warnings={quality['warnings']} "
            "(warnings are not invalid)"
        )
        invalid_action = "none (warnings retained after manual review)"
    else:
        invalid_status = "FAIL"
        invalid_evidence = f"invalid={quality['invalid']}"
        invalid_action = "report unexpected_dataset_state; do not delete"

    return {
        "corrupted_images": {
            "requirement": "Remove corrupted images",
            "status": corrupted_status,
            "evidence": corrupted_evidence,
            "action": corrupted_action,
        },
        "duplicate_removal": {
            "requirement": "Remove duplicates",
            "status": dup_status,
            "evidence": dup_evidence,
            "action": dup_action,
        },
        "format_standardization": {
            "requirement": "Standardize image format",
            "status": format_status,
            "evidence": (
                f"status={format_policy['status']}; "
                f"extensions={format_policy['extension_counts']}"
            ),
            "action": (
                "none (already_compliant; no re-encode)"
                if format_status == "PASS"
                else "investigate non-compliant formats; do not auto-convert"
            ),
        },
        "metadata": {
            "requirement": "Correct metadata",
            "status": meta_status,
            "evidence": (
                f"internal_fields={metadata_policy['present_fields']}; "
                f"{EXTERNAL_METADATA_N_A}"
            ),
            "action": "none (internal manifest fields compliant)",
        },
        "invalid_data": {
            "requirement": "Remove invalid data",
            "status": invalid_status,
            "evidence": invalid_evidence,
            "action": invalid_action,
        },
        "notes": {
            "unexpected_checks": sorted(x for x in unexpected_codes if x),
        },
    }


def _requirements_pass(requirements: Mapping[str, Any]) -> bool:
    for key, value in requirements.items():
        if key == "notes":
            continue
        if not isinstance(value, Mapping):
            continue
        status = value.get("status")
        if status not in {"PASS", "SKIPPED"}:
            return False
    return True
