"""Product-level image retrieval evaluation module.

Public API:
    EmbeddingModel   -- abstract interface the S0.5 evaluator depends on
    ManifestRecord   -- one row of the dataset manifest
    RetrievalEvaluator -- runs the S0.5 evaluation protocol
    EvaluationReport, AggregateMetrics, QueryResult -- S0.5 result containers
    evaluate_retrieval / score_retrieval_result -- S2.8 S2.7-backed metrics
    RetrievalMetricResult, QueryEvaluationRecord -- S2.8 result containers
"""

from .errors import EvaluationError
from .evaluator import (
    AggregateMetrics,
    EmbeddingModel,
    EvaluationReport,
    ManifestRecord,
    QueryResult,
    RetrievalEvaluator,
)
from .result import QueryEvaluationRecord, RetrievalMetricResult
from .retrieval_evaluator import evaluate_leave_one_out_gallery, evaluate_retrieval, score_retrieval_result

__all__ = [
    "AggregateMetrics",
    "EmbeddingModel",
    "EvaluationError",
    "EvaluationReport",
    "ManifestRecord",
    "QueryEvaluationRecord",
    "QueryResult",
    "RetrievalEvaluator",
    "RetrievalMetricResult",
    "evaluate_leave_one_out_gallery",
    "evaluate_retrieval",
    "score_retrieval_result",
]
