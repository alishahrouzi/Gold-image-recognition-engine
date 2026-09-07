"""Metrics for the category-supervised S2.6 baseline."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


def classification_metrics(model: nn.Module, loader: Any) -> Mapping[str, float]:
    """Evaluate category accuracy and embedding norm health on one split."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    correct = total = 0
    norms: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"]
            labels = batch["category_id"]
            if not torch.is_tensor(images):
                raise TypeError("Metric evaluation expects tensor images.")
            targets = torch.as_tensor(labels, dtype=torch.long, device=device)
            images = images.to(device, non_blocking=True)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
            embeddings = model.embed(images)
            norms.append(embeddings.norm(p=2, dim=1).detach().cpu())
    if was_training:
        model.train()
    if total == 0:
        raise ValueError("Cannot compute baseline metrics on an empty loader.")
    all_norms = torch.cat(norms) if norms else torch.empty(0)
    return {
        "accuracy": correct / total,
        "correct": float(correct),
        "samples": float(total),
        "embedding_norm_mean": float(all_norms.mean().item()),
        "embedding_norm_std": float(all_norms.std(unbiased=False).item()),
    }
