"""Explicit errors raised by the dataset ingestion layer."""

from __future__ import annotations


class DatasetIngestionError(ValueError):
    """Raised when a dataset contract or manifest row is invalid."""
