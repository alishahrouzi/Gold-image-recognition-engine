"""Unit tests for duplicate image_id and cross-split group detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.constants import SOURCE_DATASET1
from data.errors import DatasetIngestionError
from data.loaders.manifest import load_manifest
from data.validation import validate_samples
from tests.data.helpers import make_sample, write_manifest, write_rgb_image


def test_duplicate_image_id_detection(tmp_path: Path) -> None:
    first = make_sample(tmp_path, image_id="dup", filename="one.jpg")
    second = make_sample(
        tmp_path,
        image_id="dup",
        group_id="bracelet_002",
        filename="two.jpg",
    )
    with pytest.raises(DatasetIngestionError, match="Duplicate image_id"):
        validate_samples([first, second], validate_files=True)


def test_cross_split_group_detection(tmp_path: Path) -> None:
    train = make_sample(
        tmp_path,
        image_id="a",
        group_id="shared_group",
        split="train",
        filename="train.jpg",
    )
    test = make_sample(
        tmp_path,
        image_id="b",
        group_id="shared_group",
        split="test",
        filename="test.jpg",
    )
    with pytest.raises(DatasetIngestionError, match="appears in multiple splits"):
        validate_samples([train, test], validate_files=True)


def test_duplicate_image_id_in_manifest(tmp_path: Path) -> None:
    path_a = write_rgb_image(tmp_path / "a.jpg")
    path_b = write_rgb_image(tmp_path / "b.jpg")
    rows = [
        {
            "image_id": "same",
            "image_path": str(path_a),
            "group_id": "g1",
            "category": "Bracelet",
            "category_id": 0,
            "split": "train",
            "source": SOURCE_DATASET1,
        },
        {
            "image_id": "same",
            "image_path": str(path_b),
            "group_id": "g2",
            "category": "Ring",
            "category_id": 4,
            "split": "train",
            "source": SOURCE_DATASET1,
        },
    ]
    manifest = write_manifest(tmp_path / "manifest.csv", rows)
    with pytest.raises(DatasetIngestionError, match="Duplicate image_id"):
        load_manifest(manifest, validate_files=True)
