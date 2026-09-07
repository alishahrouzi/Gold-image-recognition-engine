"""Stable result contracts for baseline retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RetrievalCandidate:
    image_id: str
    product_group: str
    category: str
    similarity: float
    rank: int
    image_path: Optional[str] = None


@dataclass(frozen=True)
class RetrievalResult:
    query_id: str
    candidates: tuple[RetrievalCandidate, ...]

    @property
    def top_k(self) -> int:
        return len(self.candidates)
