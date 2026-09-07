"""Validation-loop primitives for S2.5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn

from .errors import TrainingError

StepFunction = Callable[[nn.Module, Any], Tensor]


@dataclass(frozen=True)
class EpochResult:
    """Aggregate result of one training or validation epoch."""

    loss: float
    batches: int
    samples: int


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensor values in a batch to the target device."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=device.type == "cuda")
    if isinstance(batch, Mapping):
        return type(batch)((key, move_batch_to_device(value, device)) for key, value in batch.items())
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    return batch


def infer_batch_size(batch: Any) -> int:
    """Find a sample dimension in a tensor nested inside a batch."""
    if torch.is_tensor(batch):
        if batch.ndim < 1:
            return 1
        return int(batch.shape[0])
    if isinstance(batch, Mapping):
        for value in batch.values():
            size = infer_batch_size(value)
            if size > 0:
                return size
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        for value in batch:
            size = infer_batch_size(value)
            if size > 0:
                return size
    return 1


def _validate_loss(loss: Tensor) -> Tensor:
    if not torch.is_tensor(loss):
        raise TrainingError(f"Step function must return a torch.Tensor, got {type(loss).__name__}.")
    if loss.numel() != 1:
        raise TrainingError(f"Step function must return a scalar loss, got shape {tuple(loss.shape)}.")
    if not torch.isfinite(loss).all():
        raise TrainingError(f"Non-finite loss encountered: {loss.detach().item()!r}.")
    return loss


def run_validation_epoch(
    model: nn.Module,
    dataloader: Any,
    *,
    device: torch.device,
    step_fn: StepFunction,
) -> EpochResult:
    """Run one no-grad validation pass and return a sample-weighted mean loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            moved_batch = move_batch_to_device(batch, device)
            loss = _validate_loss(step_fn(model, moved_batch))
            batch_size = infer_batch_size(moved_batch)
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            batches += 1

    if batches == 0 or total_samples == 0:
        raise TrainingError("Validation DataLoader produced no batches.")

    return EpochResult(
        loss=total_loss / total_samples,
        batches=batches,
        samples=total_samples,
    )
