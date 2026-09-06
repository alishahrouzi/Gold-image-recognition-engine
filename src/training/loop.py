"""Training-loop primitives for S2.5."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn, optim

from .errors import TrainingError
from .validation import EpochResult, StepFunction, _validate_loss, infer_batch_size, move_batch_to_device


def run_training_epoch(
    model: nn.Module,
    dataloader: Any,
    *,
    optimizer: optim.Optimizer,
    device: torch.device,
    step_fn: StepFunction,
) -> EpochResult:
    """Run one optimization epoch and return a sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    batches = 0

    for batch in dataloader:
        moved_batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _validate_loss(step_fn(model, moved_batch))
        if not loss.requires_grad:
            raise TrainingError("Training step returned a loss without gradients.")
        loss.backward()
        optimizer.step()

        batch_size = infer_batch_size(moved_batch)
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        batches += 1

    if batches == 0 or total_samples == 0:
        raise TrainingError("Training DataLoader produced no batches.")

    return EpochResult(
        loss=total_loss / total_samples,
        batches=batches,
        samples=total_samples,
    )
