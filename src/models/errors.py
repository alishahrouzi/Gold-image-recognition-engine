"""Explicit errors raised by encoder configuration and forward contracts."""

from __future__ import annotations


class ModelError(ValueError):
    """Raised when a model contract is violated."""


class EncoderConfigError(ModelError):
    """Raised when encoder configuration is invalid."""


class EncoderInputError(ModelError):
    """Raised when an encoder input tensor does not match the contract."""


class EmbeddingHeadConfigError(ModelError):
    """Raised when embedding-head configuration is invalid."""


class EmbeddingHeadInputError(ModelError):
    """Raised when an embedding-head input tensor does not match the contract."""


class ForwardPassValidationError(ModelError):
    """Raised when S2.4 forward-pass validation fails."""
