"""S3.9 deterministic product ranking for similarity-search results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .search_result import ProductSearchCandidate, ProductSearchResult


@dataclass(frozen=True)
class RankedProductCandidate:
    """A product candidate with its final search rank.

    ``similarity`` remains the raw cosine similarity produced by S3.8. It is
    intentionally not converted into a display score or probability here.
    """

    rank: int
    product_id: str
    category: str
    similarity: float
    matched_image_ids: tuple[str, ...]
    matched_image_paths: tuple[Optional[str], ...]


@dataclass(frozen=True)
class RankedProductSearchResult:
    """S3.9 ranked product-search result."""

    query_id: str
    candidates: tuple[RankedProductCandidate, ...]

    @property
    def top_k(self) -> int:
        return len(self.candidates)


class ProductRanker:
    """Sort product candidates by raw similarity in descending order.

    Ties are resolved by ``product_id`` ascending to make the result fully
    deterministic and independent of gallery/dictionary insertion order.
    """

    def rank(self, result: ProductSearchResult) -> RankedProductSearchResult:
        """Return a new ranked result without mutating the S3.8 result."""
        if not isinstance(result, ProductSearchResult):
            raise TypeError("result must be a ProductSearchResult.")
        if not isinstance(result.query_id, str) or not result.query_id.strip():
            raise ValueError("query_id must be a non-empty string.")

        candidates = tuple(result.candidates)
        seen_products: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, ProductSearchCandidate):
                raise TypeError("All candidates must be ProductSearchCandidate instances.")
            if not candidate.product_id.strip():
                raise ValueError("product_id must be a non-empty string.")
            if candidate.product_id in seen_products:
                raise ValueError("ProductSearchResult contains duplicate product IDs.")
            seen_products.add(candidate.product_id)
            if not math.isfinite(float(candidate.similarity)):
                raise ValueError("Product similarity must be finite.")

        ordered = sorted(
            candidates,
            key=lambda candidate: (-float(candidate.similarity), candidate.product_id),
        )

        ranked = tuple(
            RankedProductCandidate(
                rank=index,
                product_id=candidate.product_id,
                category=candidate.category,
                similarity=float(candidate.similarity),
                matched_image_ids=candidate.matched_image_ids,
                matched_image_paths=candidate.matched_image_paths,
            )
            for index, candidate in enumerate(ordered, start=1)
        )
        return RankedProductSearchResult(query_id=result.query_id, candidates=ranked)
