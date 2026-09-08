"""Vector similarity primitives for S2.7."""

from __future__ import annotations

import torch
from torch import Tensor


def cosine_similarity_matrix(query_embeddings: Tensor, gallery_embeddings: Tensor) -> Tensor:
    """Return pairwise cosine similarities with shape ``[Q, G]``.

    S2.3 embeddings are L2-normalized, so the dot product is equivalent to
    cosine similarity. The function still normalizes defensively so it remains
    correct if called with an unnormalized tensor in isolation.
    """
    if query_embeddings.ndim != 2 or gallery_embeddings.ndim != 2:
        raise ValueError("query_embeddings and gallery_embeddings must be 2-D tensors.")
    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError("Query and gallery embedding dimensions must match.")
    if gallery_embeddings.shape[0] == 0:
        raise ValueError("Gallery must contain at least one embedding.")
    query = torch.nn.functional.normalize(query_embeddings.float(), p=2, dim=1)
    gallery = torch.nn.functional.normalize(gallery_embeddings.float(), p=2, dim=1)
    return query @ gallery.transpose(0, 1)
