"""Unit tests for the Sample contract and category mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.constants import CATEGORY_TO_ID, ID_TO_CATEGORY, SOURCE_DATASET1
from data.errors import DatasetIngestionError
from data.types import Sample
from tests.data.helpers import make_sample


def test_sample_creation(tmp_path: Path) -> None:
    sample = make_sample(tmp_path)
    assert sample.image_id == "img_001"
    assert sample.group_id == "bracelet_001"
    assert sample.category == "Bracelet"
    assert sample.category_id == 0
    assert sample.split == "train"
    assert sample.source == SOURCE_DATASET1
    assert sample.image_path.exists()


def test_invalid_category(tmp_path: Path) -> None:
    image_path = tmp_path / "x.jpg"
    image_path.write_text("x")
    with pytest.raises(DatasetIngestionError, match="Unknown category"):
        Sample(
            image_id="img_001",
            image_path=image_path,
            group_id="g1",
            category="Watch",
            category_id=0,
            split="train",
            source=SOURCE_DATASET1,
        )


def test_invalid_category_id(tmp_path: Path) -> None:
    image_path = tmp_path / "x.jpg"
    image_path.write_text("x")
    with pytest.raises(DatasetIngestionError, match="does not match category"):
        Sample(
            image_id="img_001",
            image_path=image_path,
            group_id="g1",
            category="Bracelet",
            category_id=2,
            split="train",
            source=SOURCE_DATASET1,
        )


def test_invalid_split(tmp_path: Path) -> None:
    image_path = tmp_path / "x.jpg"
    image_path.write_text("x")
    with pytest.raises(DatasetIngestionError, match="Invalid split"):
        Sample(
            image_id="img_001",
            image_path=image_path,
            group_id="g1",
            category="Bracelet",
            category_id=0,
            split="validation",
            source=SOURCE_DATASET1,
        )


def test_missing_group_id(tmp_path: Path) -> None:
    image_path = tmp_path / "x.jpg"
    image_path.write_text("x")
    with pytest.raises(DatasetIngestionError, match="Missing group_id"):
        Sample(
            image_id="img_001",
            image_path=image_path,
            group_id="",
            category="Bracelet",
            category_id=0,
            split="train",
            source=SOURCE_DATASET1,
        )


def test_category_mapping_is_canonical() -> None:
    assert CATEGORY_TO_ID == {
        "Bracelet": 0,
        "Earrings": 1,
        "Necklace": 2,
        "Pendant": 3,
        "Ring": 4,
    }
    for name, category_id in CATEGORY_TO_ID.items():
        assert ID_TO_CATEGORY[category_id] == name
