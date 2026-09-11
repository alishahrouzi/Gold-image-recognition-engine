import torch

from src.data.constants import SOURCE_DATASET1
from src.data.types import Sample
from src.retrieval.embedding_evaluation import (
    EvaluationPair,
    build_evaluation_pairs,
    evaluate_pair_geometry,
    nearest_neighbor_accuracy,
)


def _sample(image_id: str, group_id: str, category: str) -> Sample:
    return Sample(
        image_id=image_id,
        image_path=f"{image_id}.jpg",
        group_id=group_id,
        category=category,
        category_id=0,
        split="train",
        source=SOURCE_DATASET1,
        metadata={},
    )


def test_build_evaluation_pairs_is_deterministic_and_balanced() -> None:
    samples = (
        _sample("a1", "g1", "Ring"),
        _sample("a2", "g1", "Ring"),
        _sample("b1", "g2", "Ring"),
        _sample("b2", "g2", "Ring"),
        _sample("c1", "g3", "Earrings"),
        _sample("c2", "g3", "Earrings"),
    )
    pairs_a = build_evaluation_pairs(samples, seed=2026)
    pairs_b = build_evaluation_pairs(samples, seed=2026)
    assert pairs_a == pairs_b
    assert sum(pair.label == 1 for pair in pairs_a) == 3
    assert sum(pair.label == 0 for pair in pairs_a) == 3
    assert all(pair.group_id_1 != pair.group_id_2 for pair in pairs_a if pair.label == 0)


def test_build_evaluation_pairs_never_reuses_self_pair() -> None:
    samples = (
        _sample("a1", "g1", "Ring"),
        _sample("a2", "g1", "Ring"),
        _sample("b1", "g2", "Earrings"),
        _sample("b2", "g2", "Earrings"),
    )
    pairs = build_evaluation_pairs(samples, seed=7)
    assert all(pair.image_id_1 != pair.image_id_2 for pair in pairs)


def test_evaluate_pair_geometry_separates_same_and_different_products() -> None:
    pairs = (
        EvaluationPair("a1", "a2", "g1", "g1", "Ring", "Ring", 1),
        EvaluationPair("a1", "b1", "g1", "g2", "Ring", "Earrings", 0, "cross_category"),
    )
    embeddings = {
        "a1": torch.tensor([1.0, 0.0, 0.0]),
        "a2": torch.tensor([0.99, 0.01, 0.0]),
        "b1": torch.tensor([-1.0, 0.0, 0.0]),
    }
    result = evaluate_pair_geometry(pairs, embeddings)
    assert result["pair_counts"]["same_product"] == 1
    assert result["pair_counts"]["different_product"] == 1
    assert result["classification"]["roc_auc"] == 1.0
    assert result["cosine_distance"]["separation_mean"] > 0.0


def test_nearest_neighbor_excludes_self() -> None:
    samples = (
        _sample("a1", "g1", "Ring"),
        _sample("a2", "g1", "Ring"),
        _sample("b1", "g2", "Earrings"),
    )
    embeddings = {
        "a1": torch.tensor([1.0, 0.0]),
        "a2": torch.tensor([0.99, 0.01]),
        "b1": torch.tensor([-1.0, 0.0]),
    }
    result = nearest_neighbor_accuracy(samples, embeddings)
    assert result["self_exclusions"] == 3
    assert result["correct_same_product"] == 2
    assert result["accuracy"] == 2 / 3
