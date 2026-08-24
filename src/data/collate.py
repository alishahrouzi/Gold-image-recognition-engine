"""Batch collation for UnifiedDataset.

Images stay as a list of RGB PIL images. They are not resized, stacked, or
normalized here because those operations are model-specific preprocessing.
"""

from __future__ import annotations

from typing import List, Sequence, TypedDict

from PIL.Image import Image as PILImage

from .types import DatasetItem


class CollatedBatch(TypedDict):
    """One DataLoader batch.

    Keys:
        image: list of RGB PIL images (possibly different sizes)
        image_id: list of image ids
        group_id: list of product group ids
        category: list of category names
        category_id: list of canonical category ids
        split: list of split labels
        source: list of dataset source labels
    """

    image: List[PILImage]
    image_id: List[str]
    group_id: List[str]
    category: List[str]
    category_id: List[int]
    split: List[str]
    source: List[str]


def collate_samples(items: Sequence[DatasetItem]) -> CollatedBatch:
    """Collate DatasetItem values into lists (no tensor stacking)."""
    if not items:
        raise ValueError("Cannot collate an empty batch.")
    return {
        "image": [item.image for item in items],
        "image_id": [item.sample.image_id for item in items],
        "group_id": [item.sample.group_id for item in items],
        "category": [item.sample.category for item in items],
        "category_id": [item.sample.category_id for item in items],
        "split": [item.sample.split for item in items],
        "source": [item.sample.source for item in items],
    }
