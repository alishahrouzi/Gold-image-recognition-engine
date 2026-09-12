"""S3.10 displayable similarity-score conversion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .ranking import RankedProductCandidate, RankedProductSearchResult


COSINE_SIMILARITY = "cosine_similarity"
COSINE_DISTANCE = "cosine_distance"
EUCLIDEAN_DISTANCE = "euclidean_distance"
SUPPORTED_SCORE_METRICS = (
    COSINE_SIMILARITY,
    COSINE_DISTANCE,
    EUCLIDEAN_DISTANCE,
)


@dataclass(frozen=True)
class ScoredProductCandidate:
    """A ranked product candidate enriched with a display Similarity Score.

    ``similarity`` remains the raw S3.8 cosine similarity evidence. The
    ``similarity_score`` is a display value in the inclusive range [0, 100].
    It is a similarity score, not a probability or confidence estimate.
    """

    rank: int
    product_id: str
    category: str
    similarity: float
    similarity_score: float
    matched_image_ids: tuple[str, ...]
    matched_image_paths: tuple[Optional[str], ...]


@dataclass(frozen=True)
class ScoredProductSearchResult:
    """S3.10 ranked result carrying displayable Similarity Scores."""

    query_id: str
    candidates: tuple[ScoredProductCandidate, ...]

    @property
    def top_k(self) -> int:
        return len(self.candidates)


class SimilarityScoreConverter:
    """Convert raw similarity/distance evidence into a 0--100 score.

    The conversion is deterministic and monotonic: a better raw similarity
    always receives a score at least as high as a worse one. The score is
    intended for display and comparison only; it is not calibrated probability.
    """

    def __init__(self, metric: str = COSINE_SIMILARITY) -> None:
        if metric not in SUPPORTED_SCORE_METRICS:
            raise ValueError(
                f"Unsupported score metric: {metric!r}. "
                f"Expected one of {SUPPORTED_SCORE_METRICS}."
            )
        self.metric = metric

    @staticmethod
    def from_cosine_similarity(similarity: float) -> float:
        """Map cosine similarity [-1, 1] linearly to Similarity Score [0, 100]."""
        value = float(similarity)
        if not math.isfinite(value):
            raise ValueError("Cosine similarity must be finite.")
        if value < -1.0 - 1e-6 or value > 1.0 + 1e-6:
            raise ValueError("Cosine similarity must be within [-1, 1].")
        value = min(1.0, max(-1.0, value))
        return (value + 1.0) * 50.0

    @staticmethod
    def from_cosine_distance(distance: float) -> float:
        """Map cosine distance [0, 2] to Similarity Score [100, 0]."""
        value = float(distance)
        if not math.isfinite(value):
            raise ValueError("Cosine distance must be finite.")
        if value < -1e-6 or value > 2.0 + 1e-6:
            raise ValueError("Cosine distance must be within [0, 2].")
        value = min(2.0, max(0.0, value))
        return (1.0 - value / 2.0) * 100.0

    @staticmethod
    def from_euclidean_distance(distance: float) -> float:
        """Map normalized-embedding Euclidean distance [0, 2] to [100, 0]."""
        value = float(distance)
        if not math.isfinite(value):
            raise ValueError("Euclidean distance must be finite.")
        if value < -1e-6 or value > 2.0 + 1e-6:
            raise ValueError(
                "Euclidean distance must be within [0, 2] for normalized embeddings."
            )
        value = min(2.0, max(0.0, value))
        return (1.0 - value / 2.0) * 100.0

    def convert(self, value: float) -> float:
        """Convert one raw metric value into a display Similarity Score."""
        if self.metric == COSINE_SIMILARITY:
            return self.from_cosine_similarity(value)
        if self.metric == COSINE_DISTANCE:
            return self.from_cosine_distance(value)
        return self.from_euclidean_distance(value)

    def score(self, result: RankedProductSearchResult) -> ScoredProductSearchResult:
        """Return a new scored result without mutating the ranked result."""
        if not isinstance(result, RankedProductSearchResult):
            raise TypeError("result must be a RankedProductSearchResult.")
        if not isinstance(result.query_id, str) or not result.query_id.strip():
            raise ValueError("query_id must be a non-empty string.")

        candidates = tuple(result.candidates)
        scored = tuple(
            ScoredProductCandidate(
                rank=self._validate_rank(candidate),
                product_id=self._validate_product_id(candidate),
                category=candidate.category,
                similarity=self._validate_similarity(candidate),
                similarity_score=self.convert(candidate.similarity),
                matched_image_ids=candidate.matched_image_ids,
                matched_image_paths=candidate.matched_image_paths,
            )
            for candidate in candidates
        )
        return ScoredProductSearchResult(query_id=result.query_id, candidates=scored)

    @staticmethod
    def _validate_rank(candidate: RankedProductCandidate) -> int:
        if not isinstance(candidate.rank, int) or isinstance(candidate.rank, bool):
            raise ValueError("rank must be an integer.")
        if candidate.rank < 1:
            raise ValueError("rank must be one-based and positive.")
        return candidate.rank

    @staticmethod
    def _validate_product_id(candidate: RankedProductCandidate) -> str:
        if not isinstance(candidate.product_id, str) or not candidate.product_id.strip():
            raise ValueError("product_id must be a non-empty string.")
        return candidate.product_id

    @staticmethod
    def _validate_similarity(candidate: RankedProductCandidate) -> float:
        value = float(candidate.similarity)
        if not math.isfinite(value):
            raise ValueError("Similarity must be finite.")
        return value
