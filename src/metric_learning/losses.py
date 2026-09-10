"""Loss functions for metric learning (S3.3)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ContrastiveLoss(nn.Module):
    """Classic pairwise contrastive loss for normalized metric embeddings.

    Labels use the project convention:
    - 0: positive pair (same product)
    - 1: negative pair (different products)

    The loss is:
        (1 - y) * d^2 + y * max(margin - d, 0)^2

    where d is the Euclidean distance between the two embeddings.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        if not isinstance(margin, (int, float)) or isinstance(margin, bool):
            raise ValueError("margin must be a positive finite number.")
        margin_value = float(margin)
        if not torch.isfinite(torch.tensor(margin_value)) or margin_value <= 0.0:
            raise ValueError("margin must be a positive finite number.")
        self.margin = margin_value

    def forward(
        self,
        embedding_a: Tensor,
        embedding_b: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Return the mean contrastive loss for a batch of embedding pairs."""
        if not torch.is_tensor(embedding_a) or not torch.is_tensor(embedding_b):
            raise TypeError("embeddings must be torch.Tensor instances.")
        if embedding_a.ndim != 2 or embedding_b.ndim != 2:
            raise ValueError("embeddings must have shape [B, D].")
        if embedding_a.shape != embedding_b.shape:
            raise ValueError("embedding_a and embedding_b must have identical shapes.")
        if embedding_a.shape[0] == 0:
            raise ValueError("embedding batches must not be empty.")
        if not torch.is_tensor(labels):
            raise TypeError("labels must be a torch.Tensor.")
        if labels.ndim != 1 or labels.shape[0] != embedding_a.shape[0]:
            raise ValueError("labels must have shape [B].")
        if not torch.isfinite(embedding_a).all() or not torch.isfinite(embedding_b).all():
            raise ValueError("embeddings must contain only finite values.")

        labels = labels.to(device=embedding_a.device)
        labels_float = labels.to(dtype=embedding_a.dtype)
        if not torch.all((labels_float == 0) | (labels_float == 1)):
            raise ValueError("labels must contain only 0 (positive) or 1 (negative).")

        diff = embedding_a.float() - embedding_b.float()
        distances = torch.linalg.vector_norm(diff, dim=1)
        distances_squared = (diff * diff).sum(dim=1)

        margin = torch.as_tensor(self.margin, device=distances.device, dtype=distances.dtype)
        negative_term = torch.relu(margin - distances).square()
        losses = (1.0 - labels_float.float()) * distances_squared + labels_float.float() * negative_term

        loss = losses.mean()
        if not torch.isfinite(loss):
            raise ValueError("Contrastive loss became non-finite.")
        return loss
