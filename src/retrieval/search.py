"""S3.8 product-level similarity search over a materialized gallery."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import torch
from torch import Tensor

from .result import RetrievalCandidate
from .search_result import ProductSearchCandidate, ProductSearchResult
from .similarity import cosine_similarity_matrix


class SimilaritySearchEngine:
    """Search a gallery and return a bounded set of unique product candidates.

    The gallery remains image-level. S3.8 computes image-to-query cosine
    similarities, excludes an optional query image, groups matching images by
    product, and uses the strongest image match as the product's raw search
    similarity. The product set is capped at ``k`` unique products.

    S3.9 owns the explicit reusable ranking stage; S3.10 owns display-score
    conversion. This class therefore exposes raw similarity evidence and does
    not convert similarity to a probability or display score.
    """

    def __init__(self, gallery_embeddings: Tensor, metadata: Sequence[dict[str, Any]]) -> None:
        if gallery_embeddings.ndim != 2:
            raise ValueError("gallery_embeddings must have shape [N, D].")
        if gallery_embeddings.shape[0] != len(metadata):
            raise ValueError("Gallery metadata must align with gallery embeddings.")
        if gallery_embeddings.shape[0] == 0:
            raise ValueError("Gallery cannot be empty.")
        if not torch.isfinite(gallery_embeddings).all():
            raise ValueError("Gallery embeddings contain non-finite values.")

        self.gallery_embeddings = gallery_embeddings.float().cpu()
        self.metadata = tuple(dict(item) for item in metadata)

    @staticmethod
    def _product_id(item: dict[str, Any]) -> str:
        product_id = item.get("product_id", item.get("product_group"))
        if product_id is None or not str(product_id).strip():
            raise ValueError("Gallery metadata must contain a non-empty product_id.")
        return str(product_id)

    @staticmethod
    def _category(item: dict[str, Any]) -> str:
        category = item.get("category")
        if category is None or not str(category).strip():
            raise ValueError("Gallery metadata must contain a non-empty category.")
        return str(category)

    @staticmethod
    def _image_id(item: dict[str, Any]) -> str:
        image_id = item.get("image_id", item.get("image"))
        if image_id is None or not str(image_id).strip():
            raise ValueError("Gallery metadata must contain a non-empty image_id.")
        return str(image_id)

    @staticmethod
    def _select_top_product_indices(
        candidates: Sequence[ProductSearchCandidate], k: int
    ) -> set[int]:
        """Select the strongest ``k`` products without defining output order.

        S3.8 needs bounded Top-K candidate membership, but S3.9 owns the
        explicit ordering of those candidates. Ties are resolved by product ID
        so candidate membership is deterministic across runs.
        """
        ordered_indices = sorted(
            range(len(candidates)),
            key=lambda index: (-candidates[index].similarity, candidates[index].product_id),
        )
        return set(ordered_indices[: min(k, len(candidates))])

    def search(
        self,
        query_embedding: Tensor,
        *,
        query_id: str = "query",
        k: int = 5,
        exclude_image_id: str | None = None,
    ) -> ProductSearchResult:
        """Return up to ``k`` unique product candidates for one query embedding."""
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("query_id must be a non-empty string.")
        if not isinstance(query_embedding, Tensor):
            raise TypeError("query_embedding must be a torch.Tensor.")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise ValueError("query_embedding must have shape [D] or [1, D].")
        if query_embedding.shape[1] != self.gallery_embeddings.shape[1]:
            raise ValueError("Query and gallery embedding dimensions must match.")
        if not torch.isfinite(query_embedding).all():
            raise ValueError("query_embedding contains non-finite values.")

        scores = cosine_similarity_matrix(query_embedding.cpu(), self.gallery_embeddings)[0]
        if exclude_image_id is not None:
            excluded = str(exclude_image_id)
            for index, item in enumerate(self.metadata):
                if self._image_id(item) == excluded:
                    scores[index] = float("-inf")

        grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for index, score in enumerate(scores.tolist()):
            if math.isfinite(score):
                grouped[self._product_id(self.metadata[index])].append((index, float(score)))

        if not grouped:
            return ProductSearchResult(query_id=query_id, candidates=tuple())

        product_candidates: list[ProductSearchCandidate] = []
        for product_id, matches in grouped.items():
            matches.sort(key=lambda pair: (-pair[1], self._image_id(self.metadata[pair[0]])))
            best_score = matches[0][1]
            best_score_matches = [
                pair
                for pair in matches
                if math.isclose(pair[1], best_score, rel_tol=0.0, abs_tol=1e-12)
            ]
            representative_index = best_score_matches[0][0]
            category = self._category(self.metadata[representative_index])
            product_candidates.append(
                ProductSearchCandidate(
                    product_id=product_id,
                    category=category,
                    similarity=best_score,
                    matched_image_ids=tuple(
                        self._image_id(self.metadata[index]) for index, _ in matches
                    ),
                    matched_image_paths=tuple(
                        self.metadata[index].get("image_path", self.metadata[index].get("image"))
                        for index, _ in matches
                    ),
                )
            )

        selected = self._select_top_product_indices(product_candidates, k)
        # Preserve product discovery order. No rank field is assigned here;
        # S3.9 is responsible for the explicit Similarity-descending ordering.
        selected_candidates = tuple(
            candidate for index, candidate in enumerate(product_candidates) if index in selected
        )
        return ProductSearchResult(query_id=query_id, candidates=selected_candidates)

    def search_from_retrieval_candidates(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        query_id: str = "query",
        k: int = 5,
    ) -> ProductSearchResult:
        """Group already-scored image candidates without recomputing similarity.

        This adapter is useful when a caller already owns an image-level
        retrieval result. It does not mutate or alter the legacy S2.7 result.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("query_id must be a non-empty string.")

        grouped: dict[str, list[RetrievalCandidate]] = defaultdict(list)
        for candidate in candidates:
            if not math.isfinite(float(candidate.similarity)):
                raise ValueError("Retrieval candidates must contain finite similarities.")
            grouped[str(candidate.product_group)].append(candidate)

        products: list[ProductSearchCandidate] = []
        for product_id, matches in grouped.items():
            matches = sorted(matches, key=lambda item: (-float(item.similarity), item.image_id))
            products.append(
                ProductSearchCandidate(
                    product_id=product_id,
                    category=str(matches[0].category),
                    similarity=float(matches[0].similarity),
                    matched_image_ids=tuple(item.image_id for item in matches),
                    matched_image_paths=tuple(item.image_path for item in matches),
                )
            )

        selected = self._select_top_product_indices(products, k)
        selected_products = tuple(
            product for index, product in enumerate(products) if index in selected
        )
        return ProductSearchResult(query_id=query_id, candidates=selected_products)
