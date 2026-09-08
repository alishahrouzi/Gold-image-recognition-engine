"""Distance functions for metric-learning embeddings (S3.2)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor


class DistanceFunction(ABC):
    """Base contract for pairwise and cross-batch embedding distances.

    Implementations return lower-is-better distances. Pairwise mode accepts
    ``[B, D]`` tensors and returns ``[B]``. Matrix mode accepts ``[Q, D]`` and
    ``[G, D]`` tensors and returns ``[Q, G]``.
    """

    @abstractmethod
    def pairwise(self, embedding_a: Tensor, embedding_b: Tensor) -> Tensor:
        """Compute one distance for each aligned pair."""
        raise NotImplementedError

    @abstractmethod
    def matrix(self, embeddings_a: Tensor, embeddings_b: Tensor) -> Tensor:
        """Compute all cross-pair distances between two batches."""
        raise NotImplementedError

    def __call__(self, embedding_a: Tensor, embedding_b: Tensor) -> Tensor:
        """Compute aligned pairwise distances."""
        return self.pairwise(embedding_a, embedding_b)


def _validate_pair_inputs(embedding_a: Tensor, embedding_b: Tensor) -> None:
    if embedding_a.ndim != 2 or embedding_b.ndim != 2:
        raise ValueError("Embedding inputs must have shape [B, D].")
    if embedding_a.shape != embedding_b.shape:
        raise ValueError("Pairwise embedding tensors must have identical shapes.")
    if embedding_a.shape[0] == 0:
        raise ValueError("Embedding batches must contain at least one item.")


def _validate_matrix_inputs(embeddings_a: Tensor, embeddings_b: Tensor) -> None:
    if embeddings_a.ndim != 2 or embeddings_b.ndim != 2:
        raise ValueError("Embedding inputs must have shape [N, D].")
    if embeddings_a.shape[1] != embeddings_b.shape[1]:
        raise ValueError("Embedding dimensions must match.")
    if embeddings_a.shape[0] == 0 or embeddings_b.shape[0] == 0:
        raise ValueError("Embedding batches must contain at least one item.")


class CosineDistance(DistanceFunction):
    """Cosine distance ``1 - cosine_similarity``."""

    def pairwise(self, embedding_a: Tensor, embedding_b: Tensor) -> Tensor:
        _validate_pair_inputs(embedding_a, embedding_b)
        a = F.normalize(embedding_a.float(), p=2, dim=1)
        b = F.normalize(embedding_b.float(), p=2, dim=1)
        similarity = (a * b).sum(dim=1)
        return 1.0 - similarity

    def matrix(self, embeddings_a: Tensor, embeddings_b: Tensor) -> Tensor:
        _validate_matrix_inputs(embeddings_a, embeddings_b)
        a = F.normalize(embeddings_a.float(), p=2, dim=1)
        b = F.normalize(embeddings_b.float(), p=2, dim=1)
        similarity = a @ b.transpose(0, 1)
        return 1.0 - similarity


class EuclideanDistance(DistanceFunction):
    """L2/Euclidean distance between embedding vectors."""

    def pairwise(self, embedding_a: Tensor, embedding_b: Tensor) -> Tensor:
        _validate_pair_inputs(embedding_a, embedding_b)
        return torch.linalg.vector_norm(embedding_a.float() - embedding_b.float(), dim=1)

    def matrix(self, embeddings_a: Tensor, embeddings_b: Tensor) -> Tensor:
        _validate_matrix_inputs(embeddings_a, embeddings_b)
        # torch.cdist computes the L2 distance directly and avoids the
        # cancellation error of the expanded ||a||^2 + ||b||^2 - 2a.b form.
        # That cancellation can turn the theoretical zero distance of
        # identical vectors into a small positive value before sqrt().
        return torch.cdist(embeddings_a.float(), embeddings_b.float(), p=2)
