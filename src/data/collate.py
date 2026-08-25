"""Batch collation for UnifiedDataset.

``collate_samples`` keeps RGB PIL images as a list (ingestion / QA).
``collate_preprocessed_samples`` stacks preprocessed tensors for training
and evaluation. Both keep metadata lists aligned with image index ``i``.
"""

from __future__ import annotations

from typing import Any, List, Sequence, TypedDict

from PIL.Image import Image as PILImage

from .errors import PreprocessingError
from .types import DatasetItem


class CollatedBatch(TypedDict):
    """One DataLoader batch of unprocessed RGB images.

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


class PreprocessedBatch(TypedDict):
    """One DataLoader batch after S1.8 preprocessing.

    Keys:
        image: stacked tensor ``[B, 3, H, W]`` (float32)
        image_id / group_id / category / category_id / split / source:
            lists aligned with batch dimension 0
    """

    image: Any
    image_id: List[str]
    group_id: List[str]
    category: List[str]
    category_id: List[int]
    split: List[str]
    source: List[str]


def _metadata_lists(items: Sequence[DatasetItem]) -> dict:
    return {
        "image_id": [item.sample.image_id for item in items],
        "group_id": [item.sample.group_id for item in items],
        "category": [item.sample.category for item in items],
        "category_id": [item.sample.category_id for item in items],
        "split": [item.sample.split for item in items],
        "source": [item.sample.source for item in items],
    }


def collate_samples(items: Sequence[DatasetItem]) -> CollatedBatch:
    """Collate DatasetItem values into lists (no tensor stacking)."""
    if not items:
        raise ValueError("Cannot collate an empty batch.")
    metadata = _metadata_lists(items)
    return {
        "image": [item.image for item in items],
        **metadata,
    }


def collate_preprocessed_samples(items: Sequence[DatasetItem]) -> PreprocessedBatch:
    """Stack preprocessed tensors and keep metadata aligned.

    Each ``item.image`` must already be a CHW tensor (use PreprocessedDataset).
    """
    import torch

    if not items:
        raise ValueError("Cannot collate an empty batch.")

    tensors = []
    for index, item in enumerate(items):
        image = item.image
        if not torch.is_tensor(image):
            raise PreprocessingError(
                "collate_preprocessed_samples expects tensor images. "
                "Wrap UnifiedDataset with PreprocessedDataset before creating "
                "the DataLoader."
            )
        if image.ndim != 3:
            raise PreprocessingError(
                f"Batch item {index} has tensor shape {tuple(image.shape)}; "
                "expected [C, H, W]."
            )
        tensors.append(image)

    stacked = torch.stack(tensors, dim=0)
    metadata = _metadata_lists(items)
    return {
        "image": stacked,
        **metadata,
    }
