"""S3.12 comparison of baseline and metric-learning retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.evaluation.errors import EvaluationError


S3_12_POLICY = "s3.12-baseline-vs-metric-learning-v1"
COMPARISON_METRICS = ("top1", "top5", "mrr")


@dataclass(frozen=True)
class ModelComparison:
    """Comparison row for one retrieval model."""

    model: str
    top1: float
    top5: float
    mrr: float


@dataclass(frozen=True)
class MetricImprovement:
    """Absolute and relative change from baseline to metric learning."""

    metric: str
    baseline: float
    metric_learning: float
    absolute_change: float
    relative_change: float | None


@dataclass(frozen=True)
class BaselineVsMetricLearningResult:
    """S3.12 comparison result and decision metadata."""

    baseline: ModelComparison
    siamese: ModelComparison
    selected_siamese_model: str
    improvements: tuple[MetricImprovement, ...]
    siamese_better_metrics: tuple[str, ...]
    diagnostics: Mapping[str, Any]


def _finite_metric(payload: Mapping[str, Any], metric: str, label: str) -> float:
    import math

    value = payload.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{label} is missing numeric metric {metric!r}.")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise EvaluationError(f"{label}.{metric} must be finite and in [0, 1].")
    return value


def _extract_metrics(payload: Mapping[str, Any], label: str) -> ModelComparison:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise EvaluationError(f"{label} does not contain a metrics mapping.")
    return ModelComparison(
        model=label,
        top1=_finite_metric(metrics, "top1", label),
        top5=_finite_metric(metrics, "top5", label),
        mrr=_finite_metric(metrics, "mrr", label),
    )


def _comparison_metrics(comparison: ModelComparison) -> dict[str, float]:
    return {"top1": comparison.top1, "top5": comparison.top5, "mrr": comparison.mrr}


def _improvements(baseline: ModelComparison, siamese: ModelComparison) -> tuple[MetricImprovement, ...]:
    baseline_values = _comparison_metrics(baseline)
    siamese_values = _comparison_metrics(siamese)
    rows: list[MetricImprovement] = []
    for metric in COMPARISON_METRICS:
        base = baseline_values[metric]
        learned = siamese_values[metric]
        absolute = learned - base
        relative = (absolute / base) if base != 0.0 else None
        rows.append(
            MetricImprovement(
                metric=metric,
                baseline=base,
                metric_learning=learned,
                absolute_change=absolute,
                relative_change=relative,
            )
        )
    return tuple(rows)


def compare_results(
    baseline_payload: Mapping[str, Any],
    siamese_payload: Mapping[str, Any],
    *,
    selected_siamese_model: str,
) -> BaselineVsMetricLearningResult:
    """Compare S2.8 baseline metrics with one selected S3.11 Siamese model.

    Only Top-1, Top-5 and MRR are compared, matching the official S3.12 scope.
    No display similarity score is used. The caller must explicitly identify the
    Siamese model selected from S3.11 retrieval evidence.
    """
    if not selected_siamese_model.strip():
        raise EvaluationError("selected_siamese_model must not be empty.")
    baseline = _extract_metrics(baseline_payload, "Baseline")
    siamese = _extract_metrics(siamese_payload, selected_siamese_model)
    improvements = _improvements(baseline, siamese)
    better = tuple(row.metric for row in improvements if row.absolute_change > 0.0)
    diagnostics = {
        "all_compared_metrics_improved": len(better) == len(COMPARISON_METRICS),
        "improved_metrics": list(better),
        "unchanged_metrics": [row.metric for row in improvements if row.absolute_change == 0.0],
        "regressed_metrics": [row.metric for row in improvements if row.absolute_change < 0.0],
        "selection_basis": "S3.11 retrieval performance",
        "selection_priority": ["top1", "top5", "mrr"],
    }
    return BaselineVsMetricLearningResult(
        baseline=baseline,
        siamese=siamese,
        selected_siamese_model=selected_siamese_model,
        improvements=improvements,
        siamese_better_metrics=better,
        diagnostics=diagnostics,
    )


def result_to_dict(result: BaselineVsMetricLearningResult) -> dict[str, Any]:
    """Serialize the S3.12 comparison result."""
    return {
        "policy": S3_12_POLICY,
        "comparison": {
            "baseline": {
                "top1": result.baseline.top1,
                "top5": result.baseline.top5,
                "mrr": result.baseline.mrr,
            },
            "siamese": {
                "model": result.selected_siamese_model,
                "top1": result.siamese.top1,
                "top5": result.siamese.top5,
                "mrr": result.siamese.mrr,
            },
        },
        "improvements": [
            {
                "metric": row.metric,
                "baseline": row.baseline,
                "metric_learning": row.metric_learning,
                "absolute_change": row.absolute_change,
                "relative_change": row.relative_change,
            }
            for row in result.improvements
        ],
        "diagnostics": dict(result.diagnostics),
        "decision": "Metric learning is better than baseline on all compared metrics."
        if result.diagnostics["all_compared_metrics_improved"]
        else "Metric learning is not better than baseline on every compared metric.",
    }
