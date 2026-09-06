"""S2.6 baseline objective; intentionally separate from S3 metric learning."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn


def _batch_tensors(model: nn.Module, batch: Mapping[str, object]) -> tuple[Tensor, Tensor]:
    images = batch["image"]
    labels = batch["category_id"]
    if not torch.is_tensor(images):
        raise TypeError("Baseline training expects batch['image'] to be a tensor.")
    if not isinstance(labels, (list, tuple)):
        raise TypeError("Baseline training expects batch['category_id'] to be a list or tuple.")
    device = next(model.parameters()).device
    targets = torch.as_tensor(labels, dtype=torch.long, device=device)
    return images.to(device, non_blocking=True), targets


def classification_step(model: nn.Module, batch: Mapping[str, object]) -> Tensor:
    """Return mean category cross-entropy for the injected S2.5 train step."""
    images, targets = _batch_tensors(model, batch)
    logits = model(images)
    if logits.ndim != 2 or logits.shape[0] != targets.shape[0]:
        raise ValueError("Baseline classifier output must have shape [B, num_classes].")
    return nn.functional.cross_entropy(logits, targets)
