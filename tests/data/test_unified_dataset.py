"""Unit tests for UnifiedDataset length, indexing, and split filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.constants import CATEGORY_TO_ID, SOURCE_DATASET1
from data.datasets.unified_dataset import UnifiedDataset
from data.errors import DatasetIngestionError
from data.types import DatasetItem
from tests.data.helpers import write_manifest, write_rgb_image


def _row(
    tmp_path: Path,
    image_id: str,
    group_id: str,
    category: str,
    split: str,
    filename: str,
) -> dict:
    image_path = write_rgb_image(tmp_path / filename)
    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "group_id": group_id,
        "category": category,
        "category_id": CATEGORY_TO_ID[category],
        "split": split,
        "source": SOURCE_DATASET1,
    }


def test_dataset_length_and_indexing(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, "a", "g1", "Bracelet", "train", "a.jpg"),
        _row(tmp_path, "b", "g2", "Ring", "valid", "b.jpg"),
        _row(tmp_path, "c", "g3", "Pendant", "test", "c.jpg"),
    ]
    manifest = write_manifest(tmp_path / "manifest.csv", rows)
    dataset = UnifiedDataset(manifest, validate_files=True)

    assert len(dataset) == 3
    item = dataset[0]
    assert isinstance(item, DatasetItem)
    assert item.sample.image_id == "a"
    assert item.image.mode == "RGB"
    assert dataset.get_sample(1).image_id == "b"
    assert dataset[-1].sample.image_id == "c"


def test_split_filtering(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, "a", "g1", "Bracelet", "train", "a.jpg"),
        _row(tmp_path, "b", "g2", "Bracelet", "valid", "b.jpg"),
        _row(tmp_path, "c", "g3", "Bracelet", "test", "c.jpg"),
        _row(tmp_path, "d", "g4", "Ring", "train", "d.jpg"),
    ]
    manifest = write_manifest(tmp_path / "manifest.csv", rows)
    train = UnifiedDataset(manifest, split="train")
    valid = UnifiedDataset(manifest, split="valid")
    test = UnifiedDataset(manifest, split="test")

    assert len(train) == 2
    assert {sample.split for sample in train.samples} == {"train"}
    assert len(valid) == 1
    assert valid.get_sample(0).image_id == "b"
    assert len(test) == 1
    assert test.get_sample(0).split == "test"
    text = manifest.read_text(encoding="utf-8")
    assert "valid" in text and "test" in text


def test_invalid_split_filter_raises(tmp_path: Path) -> None:
    rows = [_row(tmp_path, "a", "g1", "Bracelet", "train", "a.jpg")]
    manifest = write_manifest(tmp_path / "manifest.csv", rows)
    with pytest.raises(DatasetIngestionError, match="Invalid split"):
        UnifiedDataset(manifest, split="validation")


def test_category_filtering(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, "a", "g1", "Bracelet", "train", "a.jpg"),
        _row(tmp_path, "b", "g2", "Ring", "train", "b.jpg"),
    ]
    manifest = write_manifest(tmp_path / "manifest.csv", rows)
    rings = UnifiedDataset(manifest, category="Ring")
    assert len(rings) == 1
    assert rings.get_sample(0).category == "Ring"


def test_missing_image_in_manifest(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"
    rows = [
        {
            "image_id": "a",
            "image_path": str(missing),
            "group_id": "g1",
            "category": "Bracelet",
            "category_id": 0,
            "split": "train",
            "source": SOURCE_DATASET1,
        }
    ]
    manifest = write_manifest(tmp_path / "manifest.csv", rows)
    with pytest.raises(DatasetIngestionError, match="does not exist"):
        UnifiedDataset(manifest, validate_files=True)
