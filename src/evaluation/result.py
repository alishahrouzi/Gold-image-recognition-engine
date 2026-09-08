"""Structured S2.8 retrieval evaluation results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


EVALUATION_K = 10
HIT_CUTOFFS = (1, 5, 10)


@dataclass(frozen=True)
class QueryEvaluationRecord:
    """Per-query retrieval metrics for debugging and reporting."""

    query_id: str
    query_group_id: str
    split: str
    gallery_split: str
    first_positive_rank: Optional[int]
    top1_hit: int
    top5_hit: int
    top10_hit: int
    reciprocal_rank: float
    number_of_candidates: int
    number_of_positive_candidates: int
    gallery_positive_images: int = 0
    gallery_positive_products: int = 0
    unique_products_in_ranking: int = 0
    requested_k: int = EVALUATION_K
    short_ranking: bool = False
    excluded: bool = False
    exclusion_reason: Optional[str] = None
    category: Optional[str] = None
    self_match_excluded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalMetricResult:
    """Aggregated Top-K / MRR metrics plus per-query records."""

    top1: float
    top5: float
    top10: float
    mrr: float
    num_queries: int
    num_valid_queries: int
    num_excluded_queries: int
    query_records: tuple[QueryEvaluationRecord, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["query_records"] = [record.to_dict() for record in self.query_records]
        return payload
