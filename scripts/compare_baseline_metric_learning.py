"""Generate the S3.12 Baseline vs Metric Learning comparison report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.comparison import BaselineVsMetricLearningResult, compare_results, result_to_dict
from src.evaluation.errors import EvaluationError


DEFAULT_BASELINE = PROJECT_ROOT / "reports" / "evaluation" / "s2.8_baseline_evaluation.json"
DEFAULT_SIAMESE = PROJECT_ROOT / "reports" / "evaluation" / "s3.11_retrieval_evaluation.json"
DEFAULT_JSON = PROJECT_ROOT / "reports" / "evaluation" / "s3.12_baseline_vs_metric_learning.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "reports" / "evaluation" / "s3.12_baseline_vs_metric_learning.md"
DEFAULT_SIAMESE_MODEL = "s3.5_random"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise EvaluationError(f"Report not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Invalid JSON report: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"Report root must be a JSON object: {path}")
    return payload


def _extract_siamese_model(payload: dict[str, Any], model: str) -> dict[str, Any]:
    models = payload.get("models")
    if not isinstance(models, dict):
        raise EvaluationError("S3.11 report does not contain a models mapping.")
    selected = models.get(model)
    if not isinstance(selected, dict):
        available = ", ".join(sorted(models))
        raise EvaluationError(f"Siamese model {model!r} not found. Available: {available}")
    return selected


def _select_model(payload: dict[str, Any], requested: str | None) -> str:
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise EvaluationError("S3.11 report does not contain any models.")
    if requested:
        _extract_siamese_model(payload, requested)
        return requested

    # Deterministic selection: maximize Top-1, then Top-5, then MRR.
    scored: list[tuple[float, float, float, str]] = []
    for name in sorted(models):
        selected = _extract_siamese_model(payload, name)
        metrics = selected.get("metrics")
        if not isinstance(metrics, dict):
            raise EvaluationError(f"S3.11 model {name!r} has no metrics mapping.")
        top1 = float(metrics.get("top1"))
        top5 = float(metrics.get("top5"))
        mrr = float(metrics.get("mrr"))
        scored.append((top1, top5, mrr, name))
    return max(scored)[3]


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _relative(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:+.2f}%"


def _markdown(result: BaselineVsMetricLearningResult) -> str:
    rows = []
    for row in result.improvements:
        rows.append(
            f"| {row.metric.upper()} | {_pct(row.baseline)} | {_pct(row.metric_learning)} | "
            f"{_pct(row.absolute_change)} | {_relative(row.relative_change)} |"
        )
    return "\n".join(
        [
            "# S3.12 — Compare Baseline vs Metric Learning",
            "",
            "## Official comparison",
            "",
            "| Model | Top-1 | Top-5 | MRR |",
            "|---|---:|---:|---:|",
            f"| Baseline | {_pct(result.baseline.top1)} | {_pct(result.baseline.top5)} | {result.baseline.mrr:.4f} |",
            f"| {result.selected_siamese_model} | {_pct(result.siamese.top1)} | {_pct(result.siamese.top5)} | {result.siamese.mrr:.4f} |",
            "",
            "## Improvement vs baseline",
            "",
            "| Metric | Baseline | Metric Learning | Absolute Change | Relative Change |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "## Decision",
            "",
            f"**Selected Siamese model:** `{result.selected_siamese_model}`",
            "",
            "The selected model is chosen from S3.11 retrieval evidence using Top-1, then Top-5, then MRR.",
            "Only the official S3.12 metrics are compared. S3.10 Similarity Score is not used.",
            "",
            f"**Result:** {result_to_dict(result)['decision']}",
            "",
            "## Scope and interpretation",
            "",
            "This is a controlled comparison of the existing S2.8 baseline and the selected S3.11 metric-learning model.",
            "Both results use the current train leave-one-image-out retrieval protocol. The result is not a final test-set generalization score.",
            "",
            "## Policy",
            "",
            "`s3.12-baseline-vs-metric-learning-v1`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate S3.12 baseline vs metric-learning comparison.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--siamese", type=Path, default=DEFAULT_SIAMESE)
    parser.add_argument("--siamese-model", default=None, help="S3.11 model name; default selects by Top-1/Top-5/MRR.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    try:
        baseline = _load_json(args.baseline)
        siamese_report = _load_json(args.siamese)
        selected_model = _select_model(siamese_report, args.siamese_model or DEFAULT_SIAMESE_MODEL)
        siamese = _extract_siamese_model(siamese_report, selected_model)
        result = compare_results(
            baseline,
            siamese,
            selected_siamese_model=selected_model,
        )
        payload = result_to_dict(result)
        payload["sources"] = {
            "baseline_report": str(args.baseline),
            "siamese_report": str(args.siamese),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        args.output_markdown.write_text(_markdown(result), encoding="utf-8")

        print("S3.12 Baseline vs Metric Learning")
        print(f"Baseline: Top1={result.baseline.top1:.6f} Top5={result.baseline.top5:.6f} MRR={result.baseline.mrr:.6f}")
        print(f"Siamese ({selected_model}): Top1={result.siamese.top1:.6f} Top5={result.siamese.top5:.6f} MRR={result.siamese.mrr:.6f}")
        for row in result.improvements:
            print(f"{row.metric}: absolute={row.absolute_change:+.6f} relative={_relative(row.relative_change)}")
        print(f"json: {args.output_json}")
        print(f"markdown: {args.output_markdown}")
        return 0
    except (EvaluationError, OSError, TypeError, ValueError) as exc:
        print(f"S3.12 error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
