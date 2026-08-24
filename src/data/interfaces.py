"""Minimal dataset protocol for ingestion.

UnifiedDataset also subclasses torch.utils.data.Dataset. This Protocol
documents the contract without adding a second abstract hierarchy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import DatasetItem


@runtime_checkable
class Dataset(Protocol):
    """Sequence of DatasetItem values (Sample metadata + RGB image)."""

    def __len__(self) -> int:
        """Return the number of samples in this view of the dataset."""
        ...

    def __getitem__(self, index: int) -> DatasetItem:
        """Return the DatasetItem at ``index``."""
        ...
