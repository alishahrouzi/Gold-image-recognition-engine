"""Configuration for the S2.5 training infrastructure.

Training configuration owns optimization, scheduling, checkpointing,
reproducibility, and early-stopping concerns. Model architecture settings live
under ``src.models.config`` and dataset/pair-generation settings remain under
``src.data``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from .errors import TrainingConfigError

DEFAULT_SEED: int = 2026
DEFAULT_EMBEDDING_DIM: int = 128
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_EPOCHS: int = 10
DEFAULT_LEARNING_RATE: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_LOSS_NAME: str = "contrastive"
DEFAULT_MARGIN: float = 1.0
DEFAULT_OPTIMIZER_NAME: str = "adamw"
DEFAULT_SCHEDULER_NAME: str = "cosine"
DEFAULT_NUM_WORKERS: int = 0
DEFAULT_PIN_MEMORY: bool = True
DEFAULT_DEVICE: str = "auto"
DEFAULT_CHECKPOINT_DIR: str = "checkpoints"
DEFAULT_SAVE_BEST: bool = True
DEFAULT_EARLY_STOPPING_ENABLED: bool = True
DEFAULT_EARLY_STOPPING_PATIENCE: int = 3
DEFAULT_EARLY_STOPPING_MIN_DELTA: float = 0.0
DEFAULT_MONITOR: str = "val_loss"
DEFAULT_DETERMINISTIC: bool = True

SUPPORTED_EMBEDDING_DIMS: Tuple[int, ...] = (128, 256)
ALLOWED_LOSSES: Tuple[str, ...] = ("contrastive",)
ALLOWED_OPTIMIZERS: Tuple[str, ...] = ("adamw",)
ALLOWED_SCHEDULERS: Tuple[str, ...] = ("cosine", "none")
ALLOWED_DEVICES: Tuple[str, ...] = ("auto", "cpu", "cuda")
ALLOWED_MONITORS: Tuple[str, ...] = ("val_loss",)


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrainingConfigError(f"{field_name} must be an integer.")
    if value < 1:
        raise TrainingConfigError(f"{field_name} must be a positive integer, got {value}.")
    return value


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrainingConfigError(f"{field_name} must be an integer.")
    if value < 0:
        raise TrainingConfigError(f"{field_name} must be >= 0, got {value}.")
    return value


def _require_finite_float(
    value: Any,
    field_name: str,
    *,
    minimum: float = 0.0,
    strict: bool = False,
) -> float:
    if isinstance(value, bool):
        raise TrainingConfigError(f"{field_name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingConfigError(f"{field_name} must be a finite number.") from exc
    if not isfinite(number):
        raise TrainingConfigError(f"{field_name} must be finite, got {number}.")
    if strict and number <= minimum:
        raise TrainingConfigError(f"{field_name} must be > {minimum}, got {number}.")
    if not strict and number < minimum:
        raise TrainingConfigError(f"{field_name} must be >= {minimum}, got {number}.")
    return number


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingConfigError(f"{field_name} must be a boolean.")
    return value


def _require_choice(value: Any, field_name: str, allowed: Tuple[str, ...]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigError(f"{field_name} must be a non-empty string.")
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise TrainingConfigError(
            f"{field_name} must be one of {list(allowed)}, got {value!r}."
        )
    return normalized


def _require_checkpoint_path(value: Any) -> str:
    if isinstance(value, Path):
        value = value.as_posix()
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigError("checkpoint_dir must be a non-empty path string.")
    return value.strip()


def _require_optional_path(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Path):
        value = value.as_posix()
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigError(f"{field_name} must be a non-empty path string or None.")
    return value.strip()


@dataclass(frozen=True)
class TrainingConfig:
    """Validated S2.5 training-time configuration."""

    seed: int = DEFAULT_SEED
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    loss_name: str = DEFAULT_LOSS_NAME
    margin: float = DEFAULT_MARGIN
    optimizer_name: str = DEFAULT_OPTIMIZER_NAME
    scheduler_name: str = DEFAULT_SCHEDULER_NAME
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = DEFAULT_PIN_MEMORY
    device: str = DEFAULT_DEVICE
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    save_best: bool = DEFAULT_SAVE_BEST
    resume_from: Optional[str] = None
    early_stopping_enabled: bool = DEFAULT_EARLY_STOPPING_ENABLED
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA
    monitor: str = DEFAULT_MONITOR
    deterministic: bool = DEFAULT_DETERMINISTIC

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TrainingConfigError("seed must be an integer.")

        embedding_dim = _require_positive_int(self.embedding_dim, "embedding_dim")
        if embedding_dim not in SUPPORTED_EMBEDDING_DIMS:
            raise TrainingConfigError(
                f"embedding_dim must be one of {list(SUPPORTED_EMBEDDING_DIMS)}, got {embedding_dim}."
            )
        object.__setattr__(self, "embedding_dim", embedding_dim)

        object.__setattr__(self, "batch_size", _require_positive_int(self.batch_size, "batch_size"))
        object.__setattr__(self, "epochs", _require_positive_int(self.epochs, "epochs"))
        object.__setattr__(
            self,
            "learning_rate",
            _require_finite_float(self.learning_rate, "learning_rate", strict=True),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _require_finite_float(self.weight_decay, "weight_decay"),
        )
        object.__setattr__(self, "loss_name", _require_choice(self.loss_name, "loss_name", ALLOWED_LOSSES))
        object.__setattr__(
            self,
            "margin",
            _require_finite_float(self.margin, "margin", strict=True),
        )
        object.__setattr__(
            self,
            "optimizer_name",
            _require_choice(self.optimizer_name, "optimizer_name", ALLOWED_OPTIMIZERS),
        )
        object.__setattr__(
            self,
            "scheduler_name",
            _require_choice(self.scheduler_name, "scheduler_name", ALLOWED_SCHEDULERS),
        )
        object.__setattr__(self, "num_workers", _require_non_negative_int(self.num_workers, "num_workers"))
        object.__setattr__(self, "pin_memory", _require_bool(self.pin_memory, "pin_memory"))
        object.__setattr__(self, "device", _require_choice(self.device, "device", ALLOWED_DEVICES))
        object.__setattr__(self, "checkpoint_dir", _require_checkpoint_path(self.checkpoint_dir))
        object.__setattr__(self, "save_best", _require_bool(self.save_best, "save_best"))
        object.__setattr__(self, "resume_from", _require_optional_path(self.resume_from, "resume_from"))
        object.__setattr__(
            self,
            "early_stopping_enabled",
            _require_bool(self.early_stopping_enabled, "early_stopping_enabled"),
        )
        object.__setattr__(
            self,
            "early_stopping_patience",
            _require_positive_int(self.early_stopping_patience, "early_stopping_patience"),
        )
        object.__setattr__(
            self,
            "early_stopping_min_delta",
            _require_finite_float(
                self.early_stopping_min_delta,
                "early_stopping_min_delta",
            ),
        )
        object.__setattr__(self, "monitor", _require_choice(self.monitor, "monitor", ALLOWED_MONITORS))
        object.__setattr__(self, "deterministic", _require_bool(self.deterministic, "deterministic"))

    def as_loggable_dict(self) -> Mapping[str, Any]:
        """Return a JSON/log-friendly representation of the configuration."""
        payload = asdict(self)
        payload["supported_embedding_dims"] = list(SUPPORTED_EMBEDDING_DIMS)
        payload["policy"] = "s2.5-training-infrastructure-v1"
        return payload
