from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.evaluate_s4_9_human_relevance import aggregate, load_rows, ndcg, score_query


def _rows(labels):
    return [
        {
            "annotation_id": f"q1_r{i}",
            "query_index": "1",
            "query_image_id": "Q1",
            "query_category": "Ring",
            "rank": str(i),
            "relevance": str(label),
        }
        for i, label in enumerate(labels, start=1)
    ]


def test_ndcg_is_one_for_ideal_order():
    assert ndcg([3, 3, 2, 1, 0], 5) == pytest.approx(1.0)


def test_score_query_uses_labels_two_and_three_as_binary_relevant():
    scored = score_query(_rows([3, 2, 1, 0, 0, 0, 0, 0, 0, 0]))
    assert scored["hit_at_1"] == 1
    assert scored["hit_at_5"] == 1
    assert scored["precision_at_5"] == pytest.approx(0.4)
    assert scored["mean_relevance_at_5"] == pytest.approx(1.2)


def test_load_rows_rejects_missing_labels(tmp_path: Path):
    path = tmp_path / "annotations.csv"
    rows = _rows([3] * 9 + [""])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Unannotated"):
        load_rows(path)


def test_aggregate_returns_means():
    values = [
        {"category": "Ring", "hit_at_1": 1.0, "precision_at_5": 0.4},
        {"category": "Ring", "hit_at_1": 0.0, "precision_at_5": 0.8},
    ]
    result = aggregate(values)
    assert result["num_queries"] == 2
    assert result["hit_at_1"] == pytest.approx(0.5)
    assert result["precision_at_5"] == pytest.approx(0.6)
