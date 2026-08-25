"""Unit tests for S1.4 non-destructive dataset cleaning audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import pytest
from PIL import Image, ImageDraw

from data.cleaning import (
    ALREADY_COMPLIANT,
    CleaningAuditConfig,
    audit_dataset_cleaning,
    write_cleaning_report,
)
from data.constants import CATEGORY_TO_ID, SOURCE_DATASET1
from data.errors import DatasetIngestionError
from tests.data.helpers import write_manifest


def _jewelry_jpeg(
    path: Path,
    *,
    size: Tuple[int, int] = (48, 48),
    background: Tuple[int, int, int] = (40, 80, 120),
    jewel: Tuple[int, int, int] = (200, 160, 50),
    mark: Tuple[int, int] = (2, 2),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.rectangle(
        (width // 4, height // 4, 3 * width // 4, 3 * height // 4),
        fill=jewel,
    )
    # Unique marker so fixtures do not collide on exact/perceptual hashes.
    draw.point(mark, fill=(255, 0, 0))
    draw.text((4, height - 12), path.stem[:6], fill=(0, 255, 0))
    image.save(path, format="JPEG", quality=95)
    return path


def _build_mini_dataset(root: Path) -> Path:
    """Create a tiny Dataset 1-shaped tree with matching manifest."""
    rows = []
    specs = [
        ("train", "Bracelet", "g_a", "DS1_IMG_000001", "IMG_0001_aaaa_jpg.rf.aaa.jpg", (10, 20, 30), (200, 100, 40), (1, 1)),
        ("train", "Ring", "g_b", "DS1_IMG_000002", "IMG_0002_bbbb_jpg.rf.bbb.jpg", (80, 40, 10), (40, 180, 90), (5, 7)),
        ("valid", "Necklace", "g_c", "DS1_IMG_000003", "IMG_0003_cccc_jpg.rf.ccc.jpg", (20, 90, 140), (220, 60, 30), (12, 3)),
        ("test", "Earrings", "g_d", "DS1_IMG_000004", "IMG_0004_dddd_jpg.rf.ddd.jpg", (150, 20, 80), (30, 40, 200), (20, 20)),
    ]
    for split, category, group_id, image_id, filename, bg, jewel, mark in specs:
        relative = f"{split}/{category}/{filename}"
        _jewelry_jpeg(root / relative, background=bg, jewel=jewel, mark=mark)
        rows.append(
            {
                "image_id": image_id,
                "image_path": relative.replace("\\", "/"),
                "group_id": group_id,
                "category": category,
                "category_id": str(CATEGORY_TO_ID[category]),
                "split": split,
                "source": SOURCE_DATASET1,
            }
        )
    manifest = write_manifest(root / "manifest.csv", rows)
    return manifest


def _snapshot(root: Path) -> Dict[str, Tuple[int, str]]:
    snapshot: Dict[str, Tuple[int, str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot[path.relative_to(root).as_posix()] = (path.stat().st_size, digest)
    return snapshot


def test_manifest_count_matches_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    manifest = _build_mini_dataset(dataset)
    report = audit_dataset_cleaning(
        dataset,
        manifest,
        config=CleaningAuditConfig(run_duplicate_detection=True),
    )
    assert report["manifest_validation"]["counts_equal"] is True
    assert report["manifest_validation"]["manifest_count"] == 4
    assert report["manifest_validation"]["filesystem_count"] == 4
    assert report["status"] == "PASS"


def test_missing_image_is_detected(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    manifest = _build_mini_dataset(dataset)
    target = next(dataset.rglob("*.jpg"))
    target.unlink()
    report = audit_dataset_cleaning(
        dataset,
        manifest,
        config=CleaningAuditConfig(run_duplicate_detection=False),
    )
    assert report["status"] == "FAIL"
    assert report["manifest_validation"]["missing_manifest_file_count"] >= 1
    assert report["image_quality"]["invalid"] >= 1


def test_corrupted_image_is_detected(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    manifest = _build_mini_dataset(dataset)
    target = next(dataset.rglob("*.jpg"))
    target.write_bytes(b"not-a-jpeg")
    report = audit_dataset_cleaning(
        dataset,
        manifest,
        config=CleaningAuditConfig(run_duplicate_detection=False),
    )
    assert report["status"] == "FAIL"
    assert report["image_quality"]["corrupted"] >= 1
    assert any(
        item["check"] == "corrupted_images"
        for item in report["unexpected_dataset_state"]
    )


def test_duplicate_detection_integration(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    manifest = _build_mini_dataset(dataset)
    first = next((dataset / "train").rglob("*.jpg"))
    twin = first.parent / "twin.jpg"
    twin.write_bytes(first.read_bytes())
    # Manifest still points at original files; filesystem now has an orphan twin.
    # Build a matching manifest that includes the twin so exact dups are audited.
    rows = []
    for path in sorted(dataset.rglob("*.jpg")):
        rel = path.relative_to(dataset).as_posix()
        parts = Path(rel).parts
        split, category = parts[0], parts[1]
        rows.append(
            {
                "image_id": f"id_{path.stem}",
                "image_path": rel,
                "group_id": f"g_{path.stem}",
                "category": category,
                "category_id": str(CATEGORY_TO_ID[category]),
                "split": split,
                "source": SOURCE_DATASET1,
            }
        )
    manifest = write_manifest(dataset / "manifest_with_dup.csv", rows)
    report = audit_dataset_cleaning(dataset, manifest)
    assert report["duplicates"]["exact_duplicate_groups"] >= 1
    assert report["status"] == "FAIL"
    assert report["requirements"]["duplicate_removal"]["status"] == "FAIL"


def test_invalid_category_is_detected(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    (dataset / "train" / "Bracelet").mkdir(parents=True)
    _jewelry_jpeg(dataset / "train" / "Bracelet" / "x.jpg")
    manifest = write_manifest(
        dataset / "m.csv",
        [
            {
                "image_id": "bad_cat",
                "image_path": "train/Bracelet/x.jpg",
                "group_id": "g1",
                "category": "NotACategory",
                "category_id": "0",
                "split": "train",
                "source": SOURCE_DATASET1,
            }
        ],
    )
    with pytest.raises(DatasetIngestionError, match="Unknown category"):
        audit_dataset_cleaning(
            dataset,
            manifest,
            config=CleaningAuditConfig(
                run_image_inspection=False,
                run_duplicate_detection=False,
            ),
        )


def test_invalid_split_is_detected(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    (dataset / "train" / "Bracelet").mkdir(parents=True)
    _jewelry_jpeg(dataset / "train" / "Bracelet" / "x.jpg")
    manifest = write_manifest(
        dataset / "m.csv",
        [
            {
                "image_id": "bad_split",
                "image_path": "train/Bracelet/x.jpg",
                "group_id": "g1",
                "category": "Bracelet",
                "category_id": "0",
                "split": "training",
                "source": SOURCE_DATASET1,
            }
        ],
    )
    with pytest.raises(DatasetIngestionError, match="Invalid split"):
        audit_dataset_cleaning(
            dataset,
            manifest,
            config=CleaningAuditConfig(
                run_image_inspection=False,
                run_duplicate_detection=False,
            ),
        )


def test_group_split_leakage_is_detected(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _jewelry_jpeg(dataset / "train" / "Bracelet" / "a.jpg")
    _jewelry_jpeg(dataset / "test" / "Bracelet" / "b.jpg")
    manifest = write_manifest(
        dataset / "m.csv",
        [
            {
                "image_id": "a",
                "image_path": "train/Bracelet/a.jpg",
                "group_id": "shared",
                "category": "Bracelet",
                "category_id": "0",
                "split": "train",
                "source": SOURCE_DATASET1,
            },
            {
                "image_id": "b",
                "image_path": "test/Bracelet/b.jpg",
                "group_id": "shared",
                "category": "Bracelet",
                "category_id": "0",
                "split": "test",
                "source": SOURCE_DATASET1,
            },
        ],
    )
    with pytest.raises(DatasetIngestionError, match="appears in multiple splits"):
        audit_dataset_cleaning(
            dataset,
            manifest,
            config=CleaningAuditConfig(
                run_image_inspection=False,
                run_duplicate_detection=False,
            ),
        )


def test_format_and_rgb_policy(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    manifest = _build_mini_dataset(dataset)
    report = audit_dataset_cleaning(
        dataset,
        manifest,
        config=CleaningAuditConfig(run_duplicate_detection=False),
    )
    assert report["format_policy"]["status"] == ALREADY_COMPLIANT
    assert report["format_policy"]["rgb_policy_satisfied"] is True
    assert report["format_policy"]["format_converted"] == 0
    assert report["requirements"]["format_standardization"]["status"] == "PASS"
    assert report["image_quality"]["rgb_convertible"] == 4


def test_report_generation(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    manifest = _build_mini_dataset(dataset)
    report = audit_dataset_cleaning(
        dataset,
        manifest,
        config=CleaningAuditConfig(run_duplicate_detection=False),
    )
    output = tmp_path / "dataset_cleaning_report.json"
    write_cleaning_report(report, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "1.0"
    assert loaded["mode"] == "audit"
    assert loaded["destructive_operations"] is False
    assert "requirements" in loaded
    assert loaded["cleaning_summary"]["duplicates_removed"] == (
        "not_available_from_repository"
    )
    assert loaded["metadata_policy"]["status"] == "compliant"


def test_audit_mode_does_not_modify_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    manifest = _build_mini_dataset(dataset)
    before = _snapshot(dataset)
    audit_dataset_cleaning(dataset, manifest)
    after = _snapshot(dataset)
    assert before == after


def test_audit_is_reproducible(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    manifest = _build_mini_dataset(dataset)
    config = CleaningAuditConfig(run_duplicate_detection=True)
    first = audit_dataset_cleaning(dataset, manifest, config=config)
    second = audit_dataset_cleaning(dataset, manifest, config=config)
    # Drop timestamps; remaining content must match.
    first.pop("generated_at", None)
    second.pop("generated_at", None)
    assert first == second


def test_warning_is_not_invalid(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    path = dataset / "train" / "Bracelet" / "blank.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (2, 2, 2)).save(path, format="JPEG")
    manifest = write_manifest(
        dataset / "m.csv",
        [
            {
                "image_id": "warn",
                "image_path": "train/Bracelet/blank.jpg",
                "group_id": "g1",
                "category": "Bracelet",
                "category_id": "0",
                "split": "train",
                "source": SOURCE_DATASET1,
            }
        ],
    )
    report = audit_dataset_cleaning(
        dataset,
        manifest,
        config=CleaningAuditConfig(run_duplicate_detection=False),
    )
    assert report["image_quality"]["warnings"] == 1
    assert report["image_quality"]["invalid"] == 0
    assert report["image_quality"]["warning_not_treated_as_invalid"] is True
    assert report["requirements"]["invalid_data"]["status"] == "PASS"
