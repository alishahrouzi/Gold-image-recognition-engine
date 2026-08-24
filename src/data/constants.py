"""Canonical dataset vocabulary for Dataset 1 ingestion.

This module is the single source of truth for category IDs, allowed
splits, and the Dataset 1 source label. Do not duplicate these mappings
elsewhere.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

# Canonical jewelry categories for Dataset 1 (stable integer IDs).
CATEGORY_TO_ID: Dict[str, int] = {
    "Bracelet": 0,
    "Earrings": 1,
    "Necklace": 2,
    "Pendant": 3,
    "Ring": 4,
}

ID_TO_CATEGORY: Dict[int, str] = {category_id: name for name, category_id in CATEGORY_TO_ID.items()}

CANONICAL_CATEGORIES: FrozenSet[str] = frozenset(CATEGORY_TO_ID.keys())

# Folder / manifest split labels. Do not rename to training/validation/testing.
SPLIT_ORDER: tuple[str, ...] = ("train", "valid", "test")
ALLOWED_SPLITS: FrozenSet[str] = frozenset(SPLIT_ORDER)

SOURCE_DATASET1: str = "dataset1"

MANIFEST_REQUIRED_FIELDS: FrozenSet[str] = frozenset(
    {
        "image_id",
        "image_path",
        "group_id",
        "category",
        "category_id",
        "split",
        "source",
    }
)
