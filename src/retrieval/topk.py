"""Top-K retrieval over a materialized gallery."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .result import RetrievalCandidate, RetrievalResult
from .similarity import cosine_similarity_matrix


class TopKRetriever:
    """Rank gallery items by cosine similarity."""

    def __init__(self, gallery_embeddings: Tensor, metadata: Sequence[dict]) -> None:
        if gallery_embeddings.ndim != 2:
            raise ValueError("gallery_embeddings must have shape [N, D].")
        if gallery_embeddings.shape[0] != len(metadata):
            raise ValueError("Gallery metadata must align with gallery embeddings.")
        if gallery_embeddings.shape[0] == 0:
            raise ValueError("Gallery cannot be empty.")
        self.gallery_embeddings = gallery_embeddings.float().cpu()
        self.metadata = tuple(metadata)

    def retrieve(
        self,
        query_embedding: Tensor,
        *,
        query_id: str = "query",
        k: int = 5,
        exclude_image_id: str | None = None,
    ) -> RetrievalResult:
        if k < 1:
            raise ValueError("k must be at least 1.")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise ValueError("query_embedding must have shape [D] or [1, D].")
        if query_embedding.shape[1] != self.gallery_embeddings.shape[1]:
            raise ValueError("Query and gallery embedding dimensions must match.")

        scores = cosine_similarity_matrix(query_embedding.cpu(), self.gallery_embeddings)[0]
        if exclude_image_id is not None:
            for index, item in enumerate(self.metadata):
                if item.get("image_id") == exclude_image_id:
                    scores[index] = float("-inf")

        available = int(torch.isfinite(scores).sum().item())
        if available == 0:
            return RetrievalResult(query_id=query_id, candidates=tuple())
        k = min(k, available)
        values, indices = torch.topk(scores, k=k, largest=True, sorted=True)

        candidates = []
        for rank, (score, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            item = self.metadata[index]
            candidates.append(
                RetrievalCandidate(
                    image_id=str(item["image_id"]),
                    product_group=str(item["product_group"]),
                    category=str(item["category"]),
                    similarity=float(score),
                    rank=rank,
                    image_path=item.get("image_path"),
                )
            )
        return RetrievalResult(query_id=query_id, candidates=tuple(candidates))
