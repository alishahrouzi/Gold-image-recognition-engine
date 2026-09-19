from pathlib import Path

from evaluation.s4_9_category_confusion import build_top1_confusion, summarize


def test_top1_confusion_matrix_counts():
    rows = [
        {"query_image_id": "q1", "query_category": "Ring", "rank": "1", "candidate_category": "Ring"},
        {"query_image_id": "q1", "query_category": "Ring", "rank": "2", "candidate_category": "Bracelet"},
        {"query_image_id": "q2", "query_category": "Pendant", "rank": "1", "candidate_category": "Necklace"},
        {"query_image_id": "q2", "query_category": "Pendant", "rank": "2", "candidate_category": "Pendant"},
    ]
    categories, matrix = build_top1_confusion(rows)
    assert "Ring" in categories
    assert matrix["Ring"]["Ring"] == 1
    assert matrix["Pendant"]["Necklace"] == 1


def test_summary_does_not_treat_category_mismatch_as_classifier_error():
    rows = [
        {"query_image_id": "q1", "query_category": "Ring", "rank": "1", "candidate_category": "Bracelet"},
        {"query_image_id": "q1", "query_category": "Ring", "rank": "2", "candidate_category": "Ring"},
        {"query_image_id": "q2", "query_category": "Ring", "rank": "1", "candidate_category": "Ring"},
    ]
    report = summarize(rows)
    assert report["top1"]["total_queries"] == 2
    assert report["top1"]["category_mismatch_queries"] == 1
    assert report["top1"]["category_mismatch_rate"] == 0.5
    assert report["protocol"]["test_used"] is False
