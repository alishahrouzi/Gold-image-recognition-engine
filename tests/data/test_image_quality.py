"""Unit tests for S1.2 image-level validation."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from data.constants import CATEGORY_TO_ID, SOURCE_DATASET1
from data.image_quality import (
    STATUS_INVALID,
    STATUS_VALID,
    STATUS_WARNING,
    build_image_validation_report,
    inspect_sample,
    inspect_samples,
    write_image_validation_report,
)
from data.types import Sample
from tests.data.helpers import write_rgb_image


def _sample(path: Path, image_id: str = "img_001", split: str = "train") -> Sample:
    return Sample(
        image_id=image_id,
        image_path=path,
        group_id="g1",
        category="Bracelet",
        category_id=CATEGORY_TO_ID["Bracelet"],
        split=split,
        source=SOURCE_DATASET1,
    )


def _jewelry_on_background(
    path: Path,
    *,
    size: tuple[int, int],
    background: tuple[int, int, int],
    jewel: tuple[int, int, int] = (180, 140, 40),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    width, height = size
    inset_x = max(2, width // 4)
    inset_y = max(2, height // 4)
    draw.rectangle(
        (inset_x, inset_y, width - inset_x, height - inset_y),
        fill=jewel,
    )
    image.save(path)
    return path


def test_valid_rgb_image(tmp_path: Path) -> None:
    path = _jewelry_on_background(
        tmp_path / "ok.jpg",
        size=(32, 32),
        background=(40, 80, 120),
        jewel=(200, 160, 50),
    )
    result = inspect_sample(_sample(path))
    assert result.status == STATUS_VALID
    assert result.readable is True
    assert result.original_mode == "RGB"
    assert result.rgb_convertible is True
    assert result.abnormal is False
    assert result.width == 32
    assert result.height == 32


def test_grayscale_is_valid_and_rgb_convertible(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    image = Image.new("L", (24, 24), 90)
    ImageDraw.Draw(image).rectangle((6, 6, 18, 18), fill=200)
    image.save(path)
    result = inspect_sample(_sample(path))
    assert result.status == STATUS_VALID
    assert result.original_mode == "L"
    assert result.rgb_convertible is True
    assert result.readable is True


def test_rgba_is_valid_and_not_invalid(tmp_path: Path) -> None:
    path = tmp_path / "alpha.png"
    image = Image.new("RGBA", (20, 20), (10, 20, 30, 128))
    ImageDraw.Draw(image).rectangle((4, 4, 16, 16), fill=(200, 160, 40, 255))
    image.save(path)
    result = inspect_sample(_sample(path))
    assert result.status == STATUS_VALID
    assert result.original_mode == "RGBA"
    assert result.rgb_convertible is True


def test_corrupted_image_is_invalid_not_abnormal(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"this is not an image")
    result = inspect_sample(_sample(path))
    assert result.status == STATUS_INVALID
    assert result.readable is False
    assert result.abnormal is False
    assert result.reason == "decode_error"
    assert result.error


def test_missing_image_is_invalid(tmp_path: Path) -> None:
    missing = tmp_path / "absent.jpg"
    result = inspect_sample(_sample(missing))
    assert result.status == STATUS_INVALID
    assert result.readable is False
    assert result.reason == "missing_file"
    assert result.abnormal is False


def test_near_black_image_is_warning(tmp_path: Path) -> None:
    path = write_rgb_image(tmp_path / "black.png", size=(16, 16), color=(2, 2, 2))
    result = inspect_sample(_sample(path))
    assert result.readable is True
    assert result.status == STATUS_WARNING
    assert result.abnormal is True
    assert "near_black" in result.abnormal_reasons
    assert result.reason is None


def test_near_white_image_is_warning(tmp_path: Path) -> None:
    path = write_rgb_image(tmp_path / "white.png", size=(16, 16), color=(250, 250, 250))
    result = inspect_sample(_sample(path))
    assert result.readable is True
    assert result.status == STATUS_WARNING
    assert result.abnormal is True
    assert "near_white" in result.abnormal_reasons


def test_white_background_jewelry_is_not_blank(tmp_path: Path) -> None:
    path = _jewelry_on_background(
        tmp_path / "white_bg.jpg",
        size=(64, 64),
        background=(255, 255, 255),
    )
    result = inspect_sample(_sample(path))
    assert result.status == STATUS_VALID
    assert result.abnormal is False
    assert result.bright_pixel_ratio is not None
    assert result.bright_pixel_ratio > 0.3


def test_black_background_jewelry_is_not_blank(tmp_path: Path) -> None:
    path = _jewelry_on_background(
        tmp_path / "black_bg.jpg",
        size=(64, 64),
        background=(0, 0, 0),
    )
    result = inspect_sample(_sample(path))
    assert result.status == STATUS_VALID
    assert result.abnormal is False
    assert result.dark_pixel_ratio is not None
    assert result.dark_pixel_ratio > 0.3


def test_very_low_resolution_is_not_invalid(tmp_path: Path) -> None:
    path = _jewelry_on_background(
        tmp_path / "tiny.jpg",
        size=(8, 8),
        background=(240, 240, 240),
    )
    result = inspect_sample(_sample(path))
    assert result.readable is True
    assert result.status != STATUS_INVALID
    assert result.width == 8
    assert result.height == 8


def test_non_square_aspect_ratio(tmp_path: Path) -> None:
    path = _jewelry_on_background(
        tmp_path / "wide.jpg",
        size=(80, 40),
        background=(30, 30, 30),
    )
    result = inspect_sample(_sample(path))
    assert result.width == 80
    assert result.height == 40
    assert result.aspect_ratio == 2.0
    assert result.status == STATUS_VALID


def test_resolution_and_color_mode_statistics(tmp_path: Path) -> None:
    rgb_a = _jewelry_on_background(tmp_path / "a.jpg", size=(10, 12), background=(10, 40, 80))
    rgb_b = _jewelry_on_background(tmp_path / "b.jpg", size=(10, 12), background=(20, 50, 90))
    gray = tmp_path / "c.png"
    image = Image.new("L", (20, 10), 80)
    ImageDraw.Draw(image).rectangle((2, 2, 14, 8), fill=180)
    image.save(gray)

    results = inspect_samples(
        [
            _sample(rgb_a, "a", "train"),
            _sample(rgb_b, "b", "valid"),
            _sample(gray, "c", "test"),
        ]
    )
    report = build_image_validation_report(
        results,
        dataset_root=tmp_path,
        manifest_path=tmp_path / "manifest.csv",
    )
    assert report["resolution"]["min_width"] == 10
    assert report["resolution"]["max_width"] == 20
    assert report["resolution"]["min_height"] == 10
    assert report["resolution"]["max_height"] == 12
    assert report["color_modes"]["RGB"] == 2
    assert report["color_modes"]["L"] == 1
    assert report["summary"]["non_rgb_images"] == 1
    assert report["split_statistics"]["train"]["total"] == 1
    assert report["split_statistics"]["valid"]["total"] == 1
    assert report["split_statistics"]["test"]["total"] == 1


def test_abnormal_reporting_and_json_generation(tmp_path: Path) -> None:
    valid = _jewelry_on_background(
        tmp_path / "ok.jpg",
        size=(32, 32),
        background=(255, 255, 255),
    )
    blank = write_rgb_image(tmp_path / "blank.png", size=(16, 16), color=(255, 255, 255))
    results = inspect_samples([_sample(valid, "ok"), _sample(blank, "blank")])
    report = build_image_validation_report(
        results,
        dataset_root=tmp_path,
        manifest_path=tmp_path / "m.csv",
    )
    assert report["summary"]["abnormal_images"] == 1
    assert report["summary"]["warning_images"] == 1
    assert report["summary"]["valid_images"] == 1
    assert report["abnormal_statistics"]["near_white"] == 1

    output = tmp_path / "dataset_validation_report.json"
    write_image_validation_report(report, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["dataset"] == SOURCE_DATASET1
    assert len(loaded["images"]) == 2
    by_id = {row["image_id"]: row for row in loaded["images"]}
    assert by_id["ok"]["status"] == STATUS_VALID
    assert by_id["blank"]["status"] == STATUS_WARNING
    assert by_id["blank"]["abnormal"] is True
