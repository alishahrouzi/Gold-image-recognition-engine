"""Integration tests for src/evaluation/evaluator.py.

Uses a small synthetic manifest and a deterministic dummy embedding model
(no trained neural network or real images required) to verify:
    - a Query never retrieves itself (case 8)
    - same category / different group_id is treated as Negative (case 9)
    - different category / different group_id is treated as Negative (case 10)
    - group_id (not category) determines ground truth end-to-end
    - queries without a positive gallery item are correctly excluded
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.evaluator import ManifestRecord, RetrievalEvaluator  # noqa: E402


class DummyEmbeddingModel:
    """Test-only stand-in for a real embedding model.

    Maps each image_path to a pre-set vector so similarity/ranking is fully
    controlled and predictable in tests. Not a real CNN/embedding model.
    """

    def __init__(self, vectors_by_path):
        self._vectors_by_path = vectors_by_path

    def encode(self, image_path: str) -> np.ndarray:
        return np.asarray(self._vectors_by_path[image_path], dtype=np.float64)


def _record(image_id, group_id, category, split="test"):
    return ManifestRecord(
        image_id=image_id,
        image_path=f"/fake/{image_id}.jpg",
        group_id=group_id,
        category=category,
        split=split,
    )


def test_query_never_retrieves_itself():
    """Case 8: Query must be excluded from its own Gallery."""
    records = [
        _record("q", "group_A", "Bracelet"),
        _record("p1", "group_A", "Bracelet"),
        _record("n1", "group_B", "Bracelet"),
    ]
    # All embeddings identical -> similarity(query, query) would be the
    # highest possible score if self-retrieval were allowed to leak in.
    vectors = {r.image_path: [1.0, 0.0] for r in records}
    model = DummyEmbeddingModel(vectors)

    evaluator = RetrievalEvaluator(records=records, embedding_model=model, split="test")
    report = evaluator.evaluate()

    q_result = next(r for r in report.query_results if r.query_id == "q")
    assert "q" not in q_result.top_10_results
    assert q_result.top_1_result != "q"


def test_same_category_different_group_is_negative():
    """Case 9: same category, different group_id -> Negative."""
    records = [
        _record("q", "group_012", "Bracelet"),
        _record("pos", "group_012", "Bracelet"),  # same group -> Positive
        _record("neg", "group_035", "Bracelet"),  # same category, diff group -> Negative
    ]
    vectors = {
        "/fake/q.jpg": [1.0, 0.0],
        "/fake/pos.jpg": [0.9, 0.1],
        "/fake/neg.jpg": [0.95, 0.05],  # deliberately very similar, but still Negative
    }
    model = DummyEmbeddingModel(vectors)

    evaluator = RetrievalEvaluator(records=records, embedding_model=model, split="test")
    report = evaluator.evaluate()

    q_result = next(r for r in report.query_results if r.query_id == "q")
    assert q_result.num_positives == 1
    # "neg" must never count as a positive despite sharing the category and
    # despite being closer in embedding space than "pos".
    assert q_result.first_positive_rank == 2
    assert q_result.hit_at_1 == 0
    assert q_result.hit_at_5 == 1


def test_different_category_different_group_is_negative():
    """Case 10: different category, different group_id -> Negative."""
    records = [
        _record("q", "group_012", "Bracelet"),
        _record("pos", "group_012", "Bracelet"),
        _record("neg", "group_900", "Earrings"),
    ]
    vectors = {
        "/fake/q.jpg": [1.0, 0.0],
        "/fake/pos.jpg": [0.8, 0.2],
        "/fake/neg.jpg": [0.6, 0.4],
    }
    model = DummyEmbeddingModel(vectors)

    evaluator = RetrievalEvaluator(records=records, embedding_model=model, split="test")
    report = evaluator.evaluate()

    q_result = next(r for r in report.query_results if r.query_id == "q")
    assert q_result.num_positives == 1
    assert q_result.first_positive_rank == 1
    assert q_result.hit_at_1 == 1


def test_query_without_positive_is_excluded():
    """A query whose group_id appears nowhere else in the Gallery must be
    excluded from Recall/MRR/Hit metrics, and counted in excluded_queries."""
    records = [
        _record("q1", "group_A", "Bracelet"),
        _record("p1", "group_A", "Bracelet"),
        _record("lonely", "group_singleton", "Ring"),  # only member of its group
    ]
    vectors = {r.image_path: [1.0, 0.0, 0.0] for r in records}
    model = DummyEmbeddingModel(vectors)

    evaluator = RetrievalEvaluator(records=records, embedding_model=model, split="test")
    report = evaluator.evaluate()

    assert report.total_queries == 3
    assert report.valid_queries == 2
    assert report.excluded_queries == 1

    lonely_result = next(r for r in report.query_results if r.query_id == "lonely")
    assert lonely_result.excluded is True
    assert lonely_result.exclusion_reason == "no_positive_in_gallery"


def test_per_category_metrics_are_separated():
    records = [
        _record("b1", "group_b", "Bracelet"),
        _record("b2", "group_b", "Bracelet"),
        _record("r1", "group_r", "Ring"),
        _record("r2", "group_r", "Ring"),
    ]
    vectors = {
        "/fake/b1.jpg": [1.0, 0.0],
        "/fake/b2.jpg": [0.9, 0.1],
        "/fake/r1.jpg": [0.0, 1.0],
        "/fake/r2.jpg": [0.1, 0.9],
    }
    model = DummyEmbeddingModel(vectors)

    evaluator = RetrievalEvaluator(records=records, embedding_model=model, split="test")
    report = evaluator.evaluate()

    assert set(report.per_category.keys()) == {"Bracelet", "Ring"}
    assert report.per_category["Bracelet"].num_queries == 2
    assert report.per_category["Ring"].num_queries == 2
    assert report.per_category["Bracelet"].hit_at_1 == 1.0
    assert report.per_category["Ring"].hit_at_1 == 1.0


def test_only_requested_split_is_evaluated():
    records = [
        _record("q", "group_A", "Bracelet", split="test"),
        _record("p1", "group_A", "Bracelet", split="test"),
        _record("train_only", "group_A", "Bracelet", split="train"),
    ]
    vectors = {r.image_path: [1.0, 0.0] for r in records}
    model = DummyEmbeddingModel(vectors)

    evaluator = RetrievalEvaluator(records=records, embedding_model=model, split="test")
    report = evaluator.evaluate()

    assert report.total_queries == 2
    for result in report.query_results:
        assert "train_only" not in result.top_10_results


def test_empty_records_raises():
    with pytest.raises(ValueError):
        RetrievalEvaluator(records=[], embedding_model=DummyEmbeddingModel({}), split="test")


def test_no_matching_split_raises():
    records = [_record("q", "group_A", "Bracelet", split="train")]
    model = DummyEmbeddingModel({"/fake/q.jpg": [1.0, 0.0]})
    with pytest.raises(ValueError):
        RetrievalEvaluator(records=records, embedding_model=model, split="test")


def test_duplicate_image_id_raises():
    records = [
        _record("q", "group_A", "Bracelet"),
        _record("q", "group_B", "Ring"),
    ]
    model = DummyEmbeddingModel({r.image_path: [1.0, 0.0] for r in records})
    with pytest.raises(ValueError):
        RetrievalEvaluator(records=records, embedding_model=model, split="test")
