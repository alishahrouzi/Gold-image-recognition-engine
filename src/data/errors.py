"""Explicit errors raised by dataset ingestion, preprocessing, and pair generation."""

from __future__ import annotations


class DatasetIngestionError(ValueError):
    """Raised when a dataset contract or manifest row is invalid."""


class PreprocessingError(ValueError):
    """Raised when image preprocessing input or configuration is invalid."""


class InvalidImageError(PreprocessingError):
    """Raised when input bytes cannot be decoded as a supported image."""


class FileTooLargeError(PreprocessingError):
    """Raised when uploaded image bytes exceed the configured limit."""


class ImageTooSmallError(PreprocessingError):
    """Raised when an image is below the minimum supported dimensions."""


class AugmentationError(PreprocessingError):
    """Raised when training-augmentation input or configuration is invalid."""


class PairGenerationError(ValueError):
    """Raised when pair generation input, sampling, or invariants are invalid."""


class VisualizationError(ValueError):
    """Raised when dataset visualization input, sampling, or QA checks fail."""
