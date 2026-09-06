"""Optimizer factory for S2.5."""

from __future__ import annotations

from typing import Iterable

from torch import nn, optim

from .config import TrainingConfig
from .errors import TrainingError


def build_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """Build the configured optimizer over trainable model parameters only."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise TrainingError("Cannot build an optimizer: model has no trainable parameters.")

    if config.optimizer_name == "adamw":
        return optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    raise TrainingError(f"Unsupported optimizer: {config.optimizer_name!r}.")


def current_learning_rates(optimizer: optim.Optimizer) -> list[float]:
    """Return current learning rates for all optimizer parameter groups."""
    return [float(group["lr"]) for group in optimizer.param_groups]
