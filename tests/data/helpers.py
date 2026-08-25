"""Shared helpers for dataset ingestion tests."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from data.constants import CATEGORY_TO_ID, SOURCE_DATASET1
from data.types import Sample

MANIFEST_FIELDS = [
    "image_id",
    "image_path",
    "group_id",
    "category",
    "category_id",
    "split",
    "source",
]


def write_rgb_image(
    path: Path,
    size: tuple[int, int] = (8, 8),
    color: tuple[int, int, int] = (10, 20, 30),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def make_sample(
    tmp_path: Path,
    *,
    image_id: str = "img_001",
    group_id: str = "bracelet_001",
    category: str = "Bracelet",
    split: str = "train",
    filename: str = "bracelet_001.jpg",
    create_file: bool = True,
) -> Sample:
    image_path = tmp_path / filename
    if create_file:
        write_rgb_image(image_path)
    return Sample(
        image_id=image_id,
        image_path=image_path,
        group_id=group_id,
        category=category,
        category_id=CATEGORY_TO_ID[category],
        split=split,
        source=SOURCE_DATASET1,
    )


def write_manifest(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path
