"""Temporary category-supervision wrapper for the S2.6 baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models import EncoderWithEmbeddingHead


class BaselineClassifier(nn.Module):
    """Train the existing retrieval embedding with category supervision.

    The classifier is baseline-only and is not part of the production
    retrieval architecture. ``embed`` always exposes the same normalized
    embedding produced by S2.3.
    """

    def __init__(self, retrieval_model: EncoderWithEmbeddingHead, num_classes: int = 5) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        self.retrieval_model = retrieval_model
        self.classifier = nn.Linear(retrieval_model.embedding_dim, num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.retrieval_model.embedding_dim

    def embed(self, images: Tensor) -> Tensor:
        return self.retrieval_model(images)

    def forward(self, images: Tensor) -> Tensor:
        embeddings = self.embed(images)
        return self.classifier(embeddings)

    def predict(self, images: Tensor) -> Tensor:
        return self.forward(images).argmax(dim=1)
