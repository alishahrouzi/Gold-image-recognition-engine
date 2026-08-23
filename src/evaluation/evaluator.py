"""Product-level image retrieval evaluator.

Responsibilities of this module:
    - Query / Gallery construction from a dataset manifest
    - Ground truth determination via group_id (product identity)
    - Collecting embeddings through an abstract EmbeddingModel interface
    - Cosine similarity computation and deterministic ranking
    - Metric aggregation, both overall and per-category

This module deliberately knows nothing about how embeddings are produced
(CNN, transformer, pretrained, custom -- it doesn't matter). It also knows
nothing about metric formulas; those live in metrics.py. Keeping
similarity/ranking logic here (rather than duplicated per-caller) means the
future Retrieval module can reuse it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from . import metrics

# Fixed by the evaluation protocol. Not user-configurable because
# QueryResult / AggregateMetrics expose named fields (hit_at_1, hit_at_5,
# hit_at_10, ...) for these specific cutoffs.
K_VALUES: Sequence[int] = (1, 5, 10)


@runtime_checkable
class EmbeddingModel(Protocol):
    """Abstract interface the evaluator depends on.

    The evaluator never knows whether the embedding comes from a CNN, a
    transformer, a pretrained model, or a custom model -- it only receives
    a fixed-length embedding vector per image. The actual model will be
    connected later; this evaluator module does not implement it.
    """

    def encode(self, image_path: str) -> np.ndarray:
        """Return a 1-D embedding vector for the image at `image_path`."""
        ...


@dataclass(frozen=True)
class ManifestRecord:
    """A single row of the dataset manifest.

    group_id is treated as pre-computed dataset metadata representing
    product identity. This evaluator never derives group_id from
    filenames or performs group discovery -- it only consumes the
    manifest as given.
    """

    image_id: str
    image_path: str
    group_id: str
    category: str
    split: str
    source: Optional[str] = None


@dataclass
class QueryResult:
    """Per-query evaluation record, retained for reporting/analysis."""

    query_id: str
    group_id: str
    category: str
    excluded: bool
    exclusion_reason: Optional[str] = None
    num_positives: int = 0
    top_1_result: Optional[str] = None
    top_5_results: List[str] = field(default_factory=list)
    top_10_results: List[str] = field(default_factory=list)
    first_positive_rank: Optional[int] = None
    reciprocal_rank: float = 0.0
    hit_at_1: Optional[int] = None
    hit_at_5: Optional[int] = None
    hit_at_10: Optional[int] = None
    precision_at_1: Optional[float] = None
    precision_at_5: Optional[float] = None
    precision_at_10: Optional[float] = None
    recall_at_1: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None

    def to_dict(self) -> Dict:
        """Return a JSON-serializable representation of this result."""
        return {
            "query_id": self.query_id,
            "group_id": self.group_id,
            "category": self.category,
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
            "num_positives": self.num_positives,
            "top_1_result": self.top_1_result,
            "top_5_results": self.top_5_results,
            "top_10_results": self.top_10_results,
            "first_positive_rank": self.first_positive_rank,
            "reciprocal_rank": self.reciprocal_rank,
            "hit_at_1": self.hit_at_1,
            "hit_at_5": self.hit_at_5,
            "hit_at_10": self.hit_at_10,
            "precision_at_1": self.precision_at_1,
            "precision_at_5": self.precision_at_5,
            "precision_at_10": self.precision_at_10,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics computed over a set of valid queries."""

    num_queries: int
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    precision_at_1: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0

    def to_dict(self) -> Dict:
        """Return a JSON-serializable representation of these metrics."""
        return {
            "num_queries": self.num_queries,
            "hit_at_1": self.hit_at_1,
            "hit_at_5": self.hit_at_5,
            "hit_at_10": self.hit_at_10,
            "precision_at_1": self.precision_at_1,
            "precision_at_5": self.precision_at_5,
            "precision_at_10": self.precision_at_10,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
        }


@dataclass
class EvaluationReport:
    """Full evaluation output: query counts, aggregate metrics, and
    query-level results."""

    total_queries: int
    valid_queries: int
    excluded_queries: int
    overall: AggregateMetrics
    per_category: Dict[str, AggregateMetrics]
    query_results: List[QueryResult]

    def to_dict(self) -> Dict:
        """Return a JSON-serializable representation of the full report."""
        return {
            "total_queries": self.total_queries,
            "valid_queries": self.valid_queries,
            "excluded_queries": self.excluded_queries,
            "overall": self.overall.to_dict(),
            "per_category": {
                category: agg.to_dict() for category, agg in self.per_category.items()
            },
            "query_results": [result.to_dict() for result in self.query_results],
        }


class RetrievalEvaluator:
    """Evaluates product-level image retrieval on one manifest split.

    Every image in the split is treated as a Query. Its Gallery is every
    other image in the same split (the Query never appears in its own
    Gallery). Positive/Negative is determined solely by group_id equality
    -- category is never used as product identity.
    """

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        embedding_model: EmbeddingModel,
        split: str = "test",
    ) -> None:
        """Create an evaluator for a given manifest and embedding model.

        Args:
            records: The full dataset manifest (any/all splits). Only rows
                matching `split` are used.
            embedding_model: An object implementing EmbeddingModel.encode.
            split: Which manifest split to evaluate. Defaults to "test",
                the official evaluation split.

        Raises:
            ValueError: If records is empty, no rows match `split`, the
                filtered rows contain duplicate image_id values, or
                embedding_model does not implement `encode`.
        """
        if not records:
            raise ValueError("records must not be empty.")
        if embedding_model is None or not hasattr(embedding_model, "encode"):
            raise ValueError(
                "embedding_model must implement the EmbeddingModel interface "
                "(an `encode(image_path)` method)."
            )

        self._split_records: List[ManifestRecord] = [
            r for r in records if r.split == split
        ]
        if not self._split_records:
            raise ValueError(f"No manifest records found for split={split!r}.")

        image_ids = [r.image_id for r in self._split_records]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(
                f"Duplicate image_id values found in split={split!r} of the manifest."
            )

        self._embedding_model = embedding_model
        self._split = split

    def evaluate(self) -> EvaluationReport:
        """Run the full evaluation protocol and return an EvaluationReport."""
        records = self._split_records
        embeddings = self._compute_embeddings(records)
        similarity = self._cosine_similarity_matrix(embeddings)

        query_results = [
            self._evaluate_query(i, query, records, similarity)
            for i, query in enumerate(records)
        ]

        valid_results = [r for r in query_results if not r.excluded]

        overall = self._aggregate(valid_results)

        categories = sorted({r.category for r in records})
        per_category = {
            category: self._aggregate(
                [r for r in valid_results if r.category == category]
            )
            for category in categories
        }

        return EvaluationReport(
            total_queries=len(query_results),
            valid_queries=len(valid_results),
            excluded_queries=len(query_results) - len(valid_results),
            overall=overall,
            per_category=per_category,
            query_results=query_results,
        )

    def _compute_embeddings(self, records: Sequence[ManifestRecord]) -> np.ndarray:
        """Call the embedding model once per record and stack the vectors."""
        vectors = []
        for record in records:
            raw = self._embedding_model.encode(record.image_path)
            vector = np.asarray(raw, dtype=np.float64)
            if vector.ndim != 1:
                raise ValueError(
                    "embedding_model.encode must return a 1-D vector; got "
                    f"shape {vector.shape} for image_id={record.image_id!r}."
                )
            vectors.append(vector)

        dims = {vector.shape[0] for vector in vectors}
        if len(dims) != 1:
            raise ValueError(
                "All embeddings must have the same dimensionality; got "
                f"dimensions {sorted(dims)}."
            )
        return np.stack(vectors, axis=0)

    @staticmethod
    def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """Return the full pairwise cosine similarity matrix."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError(
                "Encountered a zero-norm embedding; cosine similarity is "
                "undefined for it."
            )
        normalized = embeddings / norms
        return normalized @ normalized.T

    def _evaluate_query(
        self,
        query_index: int,
        query: ManifestRecord,
        records: Sequence[ManifestRecord],
        similarity: np.ndarray,
    ) -> QueryResult:
        """Build the Gallery for one Query, rank it, and compute its metrics."""
        # Gallery = every other record in the split. The Query is never
        # included in its own Gallery.
        gallery_indices = [j for j in range(len(records)) if j != query_index]

        positive_indices = {
            j for j in gallery_indices if records[j].group_id == query.group_id
        }
        num_positives = len(positive_indices)

        if num_positives == 0:
            return QueryResult(
                query_id=query.image_id,
                group_id=query.group_id,
                category=query.category,
                excluded=True,
                exclusion_reason="no_positive_in_gallery",
                num_positives=0,
            )

        # Rank descending by similarity; break ties deterministically by
        # image_id so results are reproducible across runs.
        ranked = sorted(
            gallery_indices,
            key=lambda j: (-similarity[query_index, j], records[j].image_id),
        )
        ranked_ids = [records[j].image_id for j in ranked]
        is_positive_ranked = [j in positive_indices for j in ranked]

        result = QueryResult(
            query_id=query.image_id,
            group_id=query.group_id,
            category=query.category,
            excluded=False,
            num_positives=num_positives,
            top_1_result=ranked_ids[0] if ranked_ids else None,
            top_5_results=ranked_ids[:5],
            top_10_results=ranked_ids[:10],
            first_positive_rank=metrics.first_positive_rank(is_positive_ranked),
            reciprocal_rank=metrics.reciprocal_rank(is_positive_ranked),
        )

        for k in K_VALUES:
            setattr(result, f"hit_at_{k}", metrics.hit_at_k(is_positive_ranked, k))
            setattr(
                result, f"precision_at_{k}", metrics.precision_at_k(is_positive_ranked, k)
            )
            setattr(
                result,
                f"recall_at_{k}",
                metrics.recall_at_k(is_positive_ranked, k, num_positives),
            )

        return result

    @staticmethod
    def _aggregate(results: Sequence[QueryResult]) -> AggregateMetrics:
        """Average per-query metrics over a set of valid QueryResults."""
        n = len(results)
        if n == 0:
            return AggregateMetrics(num_queries=0)

        agg = AggregateMetrics(num_queries=n)
        for k in K_VALUES:
            agg_hit = sum(getattr(r, f"hit_at_{k}") for r in results) / n
            agg_prec = sum(getattr(r, f"precision_at_{k}") for r in results) / n
            agg_rec = sum(getattr(r, f"recall_at_{k}") for r in results) / n
            setattr(agg, f"hit_at_{k}", agg_hit)
            setattr(agg, f"precision_at_{k}", agg_prec)
            setattr(agg, f"recall_at_{k}", agg_rec)

        agg.mrr = metrics.mean_reciprocal_rank([r.reciprocal_rank for r in results])
        return agg
