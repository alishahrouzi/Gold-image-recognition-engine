"""Explicit errors raised by dataset ingestion, preprocessing, and pair generation."""

from __future__ import annotations


class DatasetIngestionError(ValueError):
    """Raised when a dataset contract or manifest row is invalid."""


class PreprocessingError(ValueError):
    """Raised when image preprocessing input or configuration is invalid."""


class AugmentationError(PreprocessingError):
    """Raised when training-augmentation input or configuration is invalid."""


class PairGenerationError(ValueError):
    """Raised when pair generation input, sampling, or invariants are invalid."""
