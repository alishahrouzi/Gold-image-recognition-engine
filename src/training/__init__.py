"""S2.5 training infrastructure.

This package owns training orchestration only. Metric-learning losses and
retrieval semantics remain in their dedicated Sprint 3 modules.
"""

from .checkpoint import CheckpointManager
from .config import (
    ALLOWED_DEVICES,
    ALLOWED_LOSSES,
    ALLOWED_MONITORS,
    ALLOWED_OPTIMIZERS,
    ALLOWED_SCHEDULERS,
    SUPPORTED_EMBEDDING_DIMS,
    TrainingConfig,
)
from .early_stopping import EarlyStopping
from .errors import TrainingConfigError, TrainingError
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from .seed import (
    capture_rng_state,
    make_dataloader_generator,
    restore_rng_state,
    seed_worker,
    set_seed,
)
from .trainer import Trainer, resolve_device
from .validation import EpochResult, run_validation_epoch
from .loop import run_training_epoch

__all__ = [
    "ALLOWED_DEVICES",
    "ALLOWED_LOSSES",
    "ALLOWED_MONITORS",
    "ALLOWED_OPTIMIZERS",
    "ALLOWED_SCHEDULERS",
    "SUPPORTED_EMBEDDING_DIMS",
    "CheckpointManager",
    "EarlyStopping",
    "EpochResult",
    "Trainer",
    "TrainingConfig",
    "TrainingConfigError",
    "TrainingError",
    "build_optimizer",
    "build_scheduler",
    "capture_rng_state",
    "make_dataloader_generator",
    "restore_rng_state",
    "resolve_device",
    "run_training_epoch",
    "run_validation_epoch",
    "seed_worker",
    "set_seed",
]
