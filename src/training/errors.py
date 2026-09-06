"""Explicit errors raised by the S2.5 training infrastructure."""

from __future__ import annotations


class TrainingError(ValueError):
    """Raised when a training contract is violated."""


class TrainingConfigError(TrainingError):
    """Raised when training configuration is invalid."""
