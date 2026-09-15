"""Product-level image retrieval evaluation module.

Public API:
    EmbeddingModel   -- abstract interface the S0.5 evaluator depends on
    ManifestRecord   -- one row of the dataset manifest
    RetrievalEvaluator -- runs the S0.5 evaluation protocol
    EvaluationReport, AggregateMetrics, QueryResult -- S0.5 result containers
    evaluate_retrieval / score_retrieval_result -- S2.8 S2.7-backed metrics
    RetrievalMetricResult, QueryEvaluationRecord -- S2.8 result containers
    ErrorAnalysisRecord -- S2.9 per-query error evidence
    evaluate_gallery -- S3.11 metric-learning product retrieval evaluation
    build_group_holdout / evaluate_unseen_groups -- S3.13 generalization gate
"""

from .errors import EvaluationError
from .error_analysis import (
    ErrorAnalysisRecord,
    build_error_record,
    build_summary,
    category_confusion,
    category_summary,
    select_representative_examples,
)
from .evaluator import (
    AggregateMetrics,
    EmbeddingModel,
    EvaluationReport,
    ManifestRecord,
    QueryResult,
    RetrievalEvaluator,
)
from .generalization import (
    GroupHoldoutSplit,
    build_gate_pairs,
    build_group_holdout,
    evaluate_unseen_groups,
    select_model,
)
from .metric_learning_retrieval import (
    MetricLearningRetrievalResult,
    RetrievalQueryRecord,
    evaluate_gallery,
    result_to_dict,
)
from .result import QueryEvaluationRecord, RetrievalMetricResult
from .retrieval_evaluator import evaluate_leave_one_out_gallery, evaluate_retrieval, score_retrieval_result

__all__ = [
    "AggregateMetrics",
    "EmbeddingModel",
    "ErrorAnalysisRecord",
    "EvaluationError",
    "EvaluationReport",
    "GroupHoldoutSplit",
    "ManifestRecord",
    "MetricLearningRetrievalResult",
    "QueryEvaluationRecord",
    "QueryResult",
    "RetrievalEvaluator",
    "RetrievalMetricResult",
    "RetrievalQueryRecord",
    "build_error_record",
    "build_gate_pairs",
    "build_group_holdout",
    "build_summary",
    "category_confusion",
    "category_summary",
    "evaluate_gallery",
    "evaluate_leave_one_out_gallery",
    "evaluate_retrieval",
    "evaluate_unseen_groups",
    "result_to_dict",
    "score_retrieval_result",
    "select_model",
    "select_representative_examples",
]
