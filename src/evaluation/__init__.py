"""Product-level image retrieval evaluation module.

Public API:
    EmbeddingModel   -- abstract interface the evaluator depends on
    ManifestRecord   -- one row of the dataset manifest
    RetrievalEvaluator -- runs the evaluation protocol
    EvaluationReport, AggregateMetrics, QueryResult -- result containers
"""

from .evaluator import (
    AggregateMetrics,
    EmbeddingModel,
    EvaluationReport,
    ManifestRecord,
    QueryResult,
    RetrievalEvaluator,
)

__all__ = [
    "AggregateMetrics",
    "EmbeddingModel",
    "EvaluationReport",
    "ManifestRecord",
    "QueryResult",
    "RetrievalEvaluator",
]
