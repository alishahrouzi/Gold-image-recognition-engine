from __future__ import annotations

import pytest

from src.evaluation.comparison import compare_results, result_to_dict
from src.evaluation.errors import EvaluationError


def _baseline(top1=0.014140009272137228, top5=0.04381084840055633, mrr=0.02704998859379944):
    return {"metrics": {"top1": top1, "top5": top5, "mrr": mrr}}


def _siamese(top1=0.19448307834955958, top5=0.3891979601298099, mrr=0.27952770970851637):
    return {"metrics": {"top1": top1, "top5": top5, "mrr": mrr}}


def test_compare_results_computes_official_metrics_and_improvements():
    result = compare_results(_baseline(), _siamese(), selected_siamese_model="s3.5_random")

    assert result.baseline.top1 == pytest.approx(0.014140009272137228)
    assert result.siamese.top1 == pytest.approx(0.19448307834955958)
    assert result.siamese_better_metrics == ("top1", "top5", "mrr")
    assert result.diagnostics["all_compared_metrics_improved"] is True


def test_relative_change_is_computed_from_baseline():
    result = compare_results(_baseline(0.1, 0.2, 0.25), _siamese(0.2, 0.3, 0.5), selected_siamese_model="random")
    changes = {row.metric: row for row in result.improvements}

    assert changes["top1"].absolute_change == pytest.approx(0.1)
    assert changes["top1"].relative_change == pytest.approx(1.0)
    assert changes["top5"].relative_change == pytest.approx(0.5)
    assert changes["mrr"].relative_change == pytest.approx(1.0)


def test_zero_baseline_uses_none_relative_change():
    result = compare_results(_baseline(0.0, 0.2, 0.0), _siamese(0.1, 0.2, 0.1), selected_siamese_model="random")
    changes = {row.metric: row for row in result.improvements}

    assert changes["top1"].relative_change is None
    assert changes["mrr"].relative_change is None


def test_regression_is_reported():
    result = compare_results(_baseline(0.2, 0.4, 0.3), _siamese(0.1, 0.5, 0.2), selected_siamese_model="random")

    assert result.diagnostics["regressed_metrics"] == ["top1", "mrr"]
    assert result.diagnostics["improved_metrics"] == ["top5"]
    assert result.diagnostics["all_compared_metrics_improved"] is False


def test_invalid_metric_is_rejected():
    with pytest.raises(EvaluationError):
        compare_results(_baseline(), _siamese(top1=1.5), selected_siamese_model="random")


def test_missing_metrics_mapping_is_rejected():
    with pytest.raises(EvaluationError):
        compare_results({}, _siamese(), selected_siamese_model="random")


def test_empty_model_name_is_rejected():
    with pytest.raises(EvaluationError):
        compare_results(_baseline(), _siamese(), selected_siamese_model=" ")


def test_result_serialization_matches_s3_12_contract():
    result = compare_results(_baseline(), _siamese(), selected_siamese_model="s3.5_random")
    payload = result_to_dict(result)

    assert payload["policy"] == "s3.12-baseline-vs-metric-learning-v1"
    assert set(payload["comparison"]) == {"baseline", "siamese"}
    assert payload["comparison"]["siamese"]["model"] == "s3.5_random"
    assert [row["metric"] for row in payload["improvements"]] == ["top1", "top5", "mrr"]
    assert payload["decision"].startswith("Metric learning is better")
