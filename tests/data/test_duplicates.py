"""Unit tests for S1.3 duplicate detection. Uses generated temp images only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageDraw

from data.duplicates import (
    DuplicateDetectionConfig,
    detect_duplicates,
    file_content_hash,
    write_duplicate_report,
)


def _pattern_image(
    path: Path,
    *,
    size: Tuple[int, int] = (48, 48),
    background: Tuple[int, int, int] = (240, 240, 240),
    shape: str = "rect",
    fill: Tuple[int, int, int] = (180, 140, 40),
    extra: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    width, height = size
    if shape == "rect":
        draw.rectangle((8, 8, width - 8, height - 8), fill=fill)
        draw.rectangle((14, 18, width - 14, height - 14), outline=(40, 20, 10), width=2)
    elif shape == "ring":
        draw.ellipse((6, 6, width - 6, height - 6), outline=fill, width=6)
        draw.ellipse((16, 16, width - 16, height - 16), outline=(30, 30, 30), width=2)
    else:
        draw.polygon(
            [(width // 2, 4), (width - 6, height - 6), (6, height - 6)],
            fill=fill,
        )
    if extra:
        draw.rectangle((2, 2, 6, 6), fill=(10, 200, 10))
    image.save(path)
    return path


def _snapshot(root: Path) -> Dict[str, Tuple[int, str]]:
    snapshot: Dict[str, Tuple[int, str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot[path.relative_to(root).as_posix()] = (path.stat().st_size, digest)
    return snapshot


def test_identical_files_are_exact_duplicates(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    first = _pattern_image(dataset / "train" / "Bracelet" / "a.png")
    second = dataset / "train" / "Bracelet" / "b.png"
    second.write_bytes(first.read_bytes())
    report = detect_duplicates(dataset)
    assert report["summary"]["exact_duplicate_groups"] == 1
    group = report["exact_duplicates"][0]
    assert group["duplicate_type"] == "exact"
    assert {item["relative_path"] for item in group["files"]} == {
        "train/Bracelet/a.png",
        "train/Bracelet/b.png",
    }
    assert report["summary"]["perceptual_duplicate_groups"] == 0


def test_same_visual_different_encoding_is_perceptual(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    png = _pattern_image(dataset / "img.png", size=(64, 64))
    bmp = dataset / "img.bmp"
    with Image.open(png) as image:
        image.convert("RGB").save(bmp, format="BMP")
    jpeg = dataset / "img.jpg"
    with Image.open(png) as image:
        image.convert("RGB").save(jpeg, format="JPEG", quality=85)
    assert file_content_hash(png) != file_content_hash(bmp)
    report = detect_duplicates(dataset)
    assert report["summary"]["exact_duplicate_groups"] == 0
    # PNG vs BMP is pixel-identical → perceptual (distance 0). JPEG may join
    # that group or appear as a near-duplicate; it must not be exact.
    assert report["summary"]["perceptual_duplicate_groups"] >= 1
    group = report["perceptual_duplicates"][0]
    assert group["duplicate_type"] == "perceptual"
    assert group["hamming_distance"] == 0
    paths = {item["relative_path"] for item in group["files"]}
    assert "img.png" in paths
    assert "img.bmp" in paths


def test_clearly_different_images_are_not_duplicates(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _pattern_image(dataset / "rect.png", shape="rect", fill=(200, 40, 40))
    _pattern_image(dataset / "ring.png", shape="ring", fill=(40, 40, 200))
    report = detect_duplicates(
        dataset,
        config=DuplicateDetectionConfig(threshold=0),
    )
    assert report["summary"]["exact_duplicate_groups"] == 0
    assert report["summary"]["perceptual_duplicate_groups"] == 0
    assert report["summary"]["near_duplicate_pairs"] == 0


def test_cross_dataset_exact_duplicate(tmp_path: Path) -> None:
    dataset_a = tmp_path / "a"
    dataset_b = tmp_path / "b"
    source = _pattern_image(dataset_a / "one.png")
    target = dataset_b / "two.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    report = detect_duplicates(dataset_a, dataset_b)
    assert report["summary"]["cross_dataset_matches"] >= 1
    match = report["cross_dataset_duplicates"][0]
    assert match["duplicate_type"] == "cross_dataset"
    assert match["match_kind"] == "exact"
    datasets = {item["dataset"] for item in match["matches"]}
    assert datasets == {"dataset_a", "dataset_b"}
    for item in match["matches"]:
        assert ":" not in item["relative_path"] or item["relative_path"].startswith("train")


def test_multiple_files_in_one_exact_group(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    original = _pattern_image(dataset / "copy0.png")
    payload = original.read_bytes()
    for name in ("copy1.png", "copy2.png"):
        (dataset / name).write_bytes(payload)
    report = detect_duplicates(dataset)
    assert report["summary"]["exact_duplicate_groups"] == 1
    assert len(report["exact_duplicates"][0]["files"]) == 3


def test_threshold_behavior_for_near_duplicates(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _pattern_image(dataset / "base.png", extra=False)
    _pattern_image(dataset / "tweaked.png", extra=True)
    strict = detect_duplicates(dataset, config=DuplicateDetectionConfig(threshold=0))
    relaxed = detect_duplicates(dataset, config=DuplicateDetectionConfig(threshold=64))
    assert strict["summary"]["near_duplicate_pairs"] == 0
    # Either perceptual (distance 0) or near (distance > 0), never both.
    if relaxed["summary"]["perceptual_duplicate_groups"]:
        assert relaxed["summary"]["near_duplicate_pairs"] == 0
    else:
        assert relaxed["summary"]["near_duplicate_pairs"] == 1
        assert 1 <= relaxed["near_duplicates"][0]["hamming_distance"] <= 64


def test_empty_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "empty"
    dataset.mkdir()
    report = detect_duplicates(dataset)
    assert report["summary"]["total_images"] == 0
    assert report["summary"]["exact_duplicate_groups"] == 0
    assert report["exact_duplicates"] == []
    assert report["summary"]["near_duplicate_search"]["status"] == "complete"


def test_invalid_and_non_image_files_are_ignored(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "notes.txt").write_text("not an image", encoding="utf-8")
    (dataset / "broken.jpg").write_bytes(b"this is not jpeg data")
    _pattern_image(dataset / "ok.png")
    report = detect_duplicates(dataset)
    assert report["summary"]["total_images"] == 2
    assert report["summary"]["unreadable_images"] == 1
    assert report["summary"]["exact_duplicate_groups"] == 0


def test_output_is_deterministic(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    first = _pattern_image(dataset / "a.png")
    (dataset / "b.png").write_bytes(first.read_bytes())
    first_report = detect_duplicates(dataset)
    second_report = detect_duplicates(dataset)
    first_report.pop("generated_at")
    second_report.pop("generated_at")
    assert first_report == second_report


def test_configured_comparison_limits_are_recorded(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    for index, fill in enumerate(
        [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
    ):
        _pattern_image(
            dataset / f"img_{index}.png",
            shape="rect" if index % 2 == 0 else "ring",
            fill=fill,
        )
    report = detect_duplicates(
        dataset,
        config=DuplicateDetectionConfig(threshold=64, max_pairs=1),
    )
    search = report["summary"]["near_duplicate_search"]
    assert search["status"] == "partial"
    assert search["comparisons_completed"] == 1
    assert search["comparisons_skipped"] > 0
    assert "max_pairs" in search["skip_reasons"]
    assert report["limitations"]
    assert report["limitations"][0]["reason"] == "max_pairs"

    capped = detect_duplicates(
        dataset,
        config=DuplicateDetectionConfig(threshold=64, max_candidates=1),
    )
    assert capped["summary"]["near_duplicate_search"]["status"] == "partial"
    assert "max_candidates" in capped["summary"]["near_duplicate_search"]["skip_reasons"]


def test_report_generation_schema_and_relative_paths(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _pattern_image(dataset / "train" / "Ring" / "x.png")
    output = tmp_path / "duplicate_report.json"
    report = detect_duplicates(dataset)
    write_duplicate_report(report, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "generated_at",
        "configuration",
        "datasets",
        "summary",
        "exact_duplicates",
        "perceptual_duplicates",
        "near_duplicates",
        "cross_dataset_duplicates",
        "limitations",
    ):
        assert key in loaded
    summary = loaded["summary"]
    for key in (
        "total_images",
        "exact_duplicate_groups",
        "perceptual_duplicate_groups",
        "near_duplicate_pairs",
        "cross_dataset_matches",
    ):
        assert key in summary
    assert loaded["datasets"]["dataset_a"]["total_images"] == 1
    assert "\\" not in json.dumps(loaded)


def test_dataset_files_remain_unchanged(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _pattern_image(dataset / "keep.png")
    (dataset / "sidecar.txt").write_text("leave me", encoding="utf-8")
    before = _snapshot(dataset)
    detect_duplicates(dataset)
    after = _snapshot(dataset)
    assert before == after


def test_cli_writes_report(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from detect_duplicates import main

    dataset = tmp_path / "ds"
    _pattern_image(dataset / "ok.png")
    output = tmp_path / "out.json"
    assert main(["--dataset-a", str(dataset), "--output", str(output)]) == 0
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["summary"]["total_images"] == 1
    assert loaded["schema_version"] == "1.0"
