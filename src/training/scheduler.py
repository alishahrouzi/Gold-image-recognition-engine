"""Learning-rate scheduler factory for S2.5."""

from __future__ import annotations

from typing import Optional

from torch import optim

from .config import TrainingConfig
from .errors import TrainingError


def build_scheduler(
    optimizer: optim.Optimizer,
    config: TrainingConfig,
) -> Optional[optim.lr_scheduler.LRScheduler]:
    """Build the configured epoch-level scheduler.

    The S2.5 baseline uses cosine annealing over the configured number of
    epochs. ``none`` is supported for controlled experiments and debugging.
    """
    if config.scheduler_name == "none":
        return None
    if config.scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=0.0,
        )
    raise TrainingError(f"Unsupported scheduler: {config.scheduler_name!r}.")
