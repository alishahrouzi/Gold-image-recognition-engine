"""Unit tests for the S4.8 rotation diagnostic helpers."""

import json
from pathlib import Path

from scripts.run_rotation_diagnostic import _first_category_rank, _load_gallery_categories


def test_first_category_rank_uses_gallery_metadata() -> None:
    categories = {
        "P001": "pendant",
        "P002": "bracelet",
        "P003": "ring",
    }
    results = [
        {"product_id": "P001", "similarity": 99.0},
        {"product_id": "P002", "similarity": 98.0},
        {"product_id": "P003", "similarity": 97.0},
    ]

    assert _first_category_rank(results, categories, "ring") == 3
    assert _first_category_rank(results[:2], categories, "ring") is None


def test_load_gallery_categories_normalizes_values(tmp_path: Path) -> None:
    metadata = tmp_path / "gallery_metadata.json"
    metadata.write_text(
        json.dumps(
            [
                {"product_id": "P001", "category": "Ring"},
                {"product_id": "P002", "category": " Necklace "},
                {"product_id": "P003", "category": "bracelet"},
            ]
        ),
        encoding="utf-8",
    )

    assert _load_gallery_categories(metadata) == {
        "P001": "ring",
        "P002": "necklace",
        "P003": "bracelet",
    }


def test_load_gallery_categories_rejects_empty_mapping(tmp_path: Path) -> None:
    metadata = tmp_path / "gallery_metadata.json"
    metadata.write_text(json.dumps([{"product_id": "P001"}]), encoding="utf-8")

    try:
        _load_gallery_categories(metadata)
    except ValueError as exc:
        assert "no product_id/category pairs" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
