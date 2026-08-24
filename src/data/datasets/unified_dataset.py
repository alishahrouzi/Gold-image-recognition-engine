"""PyTorch-compatible dataset over a validated manifest.

Filtering by split or category happens in memory and does not modify the
manifest file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

from torch.utils.data import Dataset as TorchDataset

from ..constants import ALLOWED_SPLITS, CANONICAL_CATEGORIES, SPLIT_ORDER
from ..errors import DatasetIngestionError
from ..loaders.image_loader import load_rgb_image
from ..loaders.manifest import load_manifest
from ..types import DatasetItem, Sample
from ..validation import build_validation_report, validate_samples

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class UnifiedDataset(TorchDataset):
    """Read-only Dataset 1 ingestion view.

    ``__getitem__`` returns a DatasetItem (Sample + RGB PIL image) so a
    PyTorch DataLoader can batch metadata together with images. Use
    ``collate_samples`` as ``collate_fn`` because PIL images cannot be
    stacked without model-specific resizing.

    Example::

        dataset = UnifiedDataset(manifest_path, dataset_root=root, split="train")
        sample = dataset.get_sample(0)
        item = dataset[0]  # DatasetItem
    """

    def __init__(
        self,
        manifest_path: PathLike,
        *,
        dataset_root: Optional[PathLike] = None,
        split: Optional[str] = None,
        category: Optional[str] = None,
        samples: Optional[Sequence[Sample]] = None,
        validate_files: bool = True,
    ) -> None:
        """Create a dataset view.

        Args:
            manifest_path: CSV manifest path. Ignored when ``samples`` is given
                (used by ``filter`` helpers and tests).
            dataset_root: Root used to resolve relative ``image_path`` values.
            split: If set, keep only this split (``train`` / ``valid`` / ``test``).
            category: If set, keep only this canonical category name.
            samples: Optional pre-loaded Samples. When omitted, the manifest
                is loaded and fully validated first, then filtered.
            validate_files: Forwarded to manifest loading / validation.
        """
        if split is not None and split not in ALLOWED_SPLITS:
            allowed = "/".join(SPLIT_ORDER)
            raise DatasetIngestionError(
                f"Invalid split {split!r}. Expected {allowed}."
            )
        if category is not None and category not in CANONICAL_CATEGORIES:
            raise DatasetIngestionError(
                f"Unknown category {category!r}."
            )

        if samples is None:
            all_samples = load_manifest(
                manifest_path,
                dataset_root=dataset_root,
                validate_files=validate_files,
            )
        else:
            all_samples = list(samples)
            validate_samples(all_samples, validate_files=validate_files)

        self._manifest_path = Path(manifest_path) if manifest_path is not None else None
        self._dataset_root = Path(dataset_root) if dataset_root is not None else None
        self._split = split
        self._category = category
        self._samples = _apply_filters(all_samples, split=split, category=category)

        if not self._samples:
            raise DatasetIngestionError(
                _empty_filter_message(split=split, category=category)
            )

        report = build_validation_report(self._samples)
        logger.info(
            "UnifiedDataset ready (manifest=%s split=%s category=%s samples=%s groups=%s)",
            self._manifest_path,
            split,
            category,
            report.total_samples,
            report.total_groups,
        )
        logger.info("Split distribution: %s", report.split_counts)
        logger.info("Category distribution: %s", report.category_counts)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> DatasetItem:
        if index < 0:
            index = len(self._samples) + index
        if index < 0 or index >= len(self._samples):
            raise IndexError(f"Dataset index {index} is out of range.")
        sample = self._samples[index]
        image = load_rgb_image(sample.image_path)
        return DatasetItem(sample=sample, image=image)

    def get_sample(self, index: int) -> Sample:
        """Return Sample metadata without opening the image file."""
        return self._samples[index]

    @property
    def samples(self) -> List[Sample]:
        """Filtered samples (a copy). The underlying manifest is not modified."""
        return list(self._samples)


def _apply_filters(
    samples: Sequence[Sample],
    *,
    split: Optional[str],
    category: Optional[str],
) -> List[Sample]:
    filtered = list(samples)
    if split is not None:
        filtered = [sample for sample in filtered if sample.split == split]
    if category is not None:
        filtered = [sample for sample in filtered if sample.category == category]
    return filtered


def _empty_filter_message(*, split: Optional[str], category: Optional[str]) -> str:
    parts = []
    if split is not None:
        parts.append(f"split={split!r}")
    if category is not None:
        parts.append(f"category={category!r}")
    criterion = " and ".join(parts) if parts else "the given filters"
    return f"No samples remain after filtering by {criterion}."
