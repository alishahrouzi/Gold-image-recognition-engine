from __future__ import annotations

from scripts.analyze_s4_9_human_relevance import aggregate, classify, query_result


def _items(labels):
    return [
        {
            "query_index": "1",
            "query_image_id": "Q1",
            "query_category": "Ring",
            "rank": str(i),
            "_rank": i,
            "_label": label,
        }
        for i, label in enumerate(labels, start=1)
    ]


def test_classify_no_relevant():
    assert classify([0] * 10) == "no_relevant_in_top10"


def test_classify_relevant_only_late():
    labels = [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    assert classify(labels) == "relevant_only_in_ranks_6_to_10"


def test_classify_relevant_in_top5_not_top1():
    labels = [0, 1, 2] + [0] * 7
    assert classify(labels) == "relevant_in_top5_but_not_top1"


def test_query_result_tracks_relevant_ranks():
    result = query_result(_items([3, 0, 2, 1, 0, 2, 0, 0, 0, 0]))
    assert result["first_relevant_rank"] == 1
    assert result["relevant_ranks_top10"] == [1, 3, 6]
    assert result["relevant_count_top10"] == 3


def test_aggregate_counts_failure_modes():
    results = [
        query_result(_items([0] * 10)),
        query_result(_items([0, 2] + [0] * 8)),
    ]
    summary = aggregate(results)
    assert summary["queries"] == 2
    assert summary["no_relevant_in_top10"] == 1
    assert summary["relevant_in_top10"] == 1
