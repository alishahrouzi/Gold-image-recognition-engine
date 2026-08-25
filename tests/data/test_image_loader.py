"""Unit tests for RGB image loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from data.errors import DatasetIngestionError
from data.loaders.image_loader import load_rgb_image
from tests.data.helpers import write_rgb_image


def test_load_rgb_image(tmp_path: Path) -> None:
    path = write_rgb_image(tmp_path / "ok.jpg", color=(1, 2, 3))
    image = load_rgb_image(path)
    assert image.mode == "RGB"
    assert image.size == (8, 8)


def test_grayscale_converts_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    Image.new("L", (6, 4), 128).save(path)
    image = load_rgb_image(path)
    assert image.mode == "RGB"
    assert image.size == (6, 4)


def test_rgba_converts_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (5, 5), (10, 20, 30, 0)).save(path)
    image = load_rgb_image(path)
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 255, 255)


def test_missing_image_raises(tmp_path: Path) -> None:
    missing = tmp_path / "absent.jpg"
    with pytest.raises(DatasetIngestionError, match="does not exist"):
        load_rgb_image(missing)


def test_unreadable_image_raises(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"this is not an image")
    with pytest.raises(DatasetIngestionError, match="Unable to read image file"):
        load_rgb_image(path)
