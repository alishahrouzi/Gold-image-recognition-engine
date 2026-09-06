"""Training infrastructure for the Gold Image Recognition Engine.

S2.5 provides configuration and training orchestration primitives without
changing the existing data, preprocessing, or model contracts.
"""

from .config import (
    ALLOWED_DEVICES,
    ALLOWED_LOSSES,
    ALLOWED_OPTIMIZERS,
    ALLOWED_SCHEDULERS,
    SUPPORTED_EMBEDDING_DIMS,
    TrainingConfig,
)
from .errors import TrainingConfigError, TrainingError

__all__ = [
    "ALLOWED_DEVICES",
    "ALLOWED_LOSSES",
    "ALLOWED_OPTIMIZERS",
    "ALLOWED_SCHEDULERS",
    "SUPPORTED_EMBEDDING_DIMS",
    "TrainingConfig",
    "TrainingConfigError",
    "TrainingError",
]
