"""Product-level result contracts for S3.8 similarity search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProductSearchCandidate:
    """A product candidate produced by the S3.8 search stage.

    ``similarity`` is the raw maximum image-to-query cosine similarity observed
    for this product. It is search evidence, not a display score or probability.
    Ranking and display-score conversion are deliberately owned by later sprint
    tasks (S3.9 and S3.10).
    """

    product_id: str
    category: str
    similarity: float
    matched_image_ids: tuple[str, ...]
    matched_image_paths: tuple[Optional[str], ...]


@dataclass(frozen=True)
class ProductSearchResult:
    """Top-K unique product candidates returned by S3.8."""

    query_id: str
    candidates: tuple[ProductSearchCandidate, ...]

    @property
    def top_k(self) -> int:
        return len(self.candidates)
