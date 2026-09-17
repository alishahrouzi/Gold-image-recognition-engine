from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from data.constants import SOURCE_DATASET1
from data.types import Sample
from evaluation.final_test import (
    compare_models,
    evaluate_robustness,
    evaluate_split,
    make_robustness_variants,
)
from retrieval.gallery import Gallery


class FakeExtractor:
    def __init__(self, embedding: torch.Tensor) -> None:
        self.embedding = embedding.float()

    def extract_one(self, image: torch.Tensor) -> torch.Tensor:
        return self.embedding.clone()


def sample(tmp_path: Path, image_id: str, group_id: str, category: str, split: str) -> Sample:
    path = tmp_path / f"{image_id}.jpg"
    Image.new("RGB", (64, 64), (120, 80, 40)).save(path)
    category_id = {"Bracelet": 0, "Earrings": 1, "Necklace": 2, "Pendant": 3, "Ring": 4}[category]
    return Sample(
        image_id=image_id,
        image_path=path,
        group_id=group_id,
        category=category,
        category_id=category_id,
        split=split,
        source=SOURCE_DATASET1,
    )


def gallery() -> Gallery:
    return Gallery(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]),
        metadata=(
            {"product_id": "P1", "category": "Ring", "image": "p1.jpg", "image_id": "G1"},
            {"product_id": "P2", "category": "Necklace", "image": "p2.jpg", "image_id": "G2"},
            {"product_id": "P3", "category": "Bracelet", "image": "p3.jpg", "image_id": "G3"},
        ),
    )


def test_final_evaluation_uses_query_category_and_keeps_ranked_candidates(tmp_path: Path) -> None:
    query = sample(tmp_path, "Q1", "UNSEEN", "Ring", "test")
    result = evaluate_split(
        FakeExtractor(torch.tensor([1.0, 0.0])),
        gallery(),
        [query],
        model="test-model",
        split="test",
    )
    assert result.metrics["top1"] == 1.0
    assert result.metrics["top5"] == 1.0
    assert result.metrics["mrr"] == 1.0
    assert result.diagnostics["exact_product_ground_truth_available"] is False
    assert result.query_records[0]["ranked_candidates"][0]["product_id"] == "P1"


def test_robustness_variants_are_deterministic_and_keep_size_constraints(tmp_path: Path) -> None:
    path = tmp_path / "image.jpg"
    Image.new("RGB", (100, 80), (100, 100, 100)).save(path)
    image = Image.open(path)
    variants = make_robustness_variants(image)
    assert set(variants) == {"original", "rotation", "brightness", "center_crop", "resize"}
    assert all(variant.mode == "RGB" for variant in variants.values())
    assert variants["rotation"].size == image.size
    assert variants["brightness"].size == image.size


def test_compare_models_requires_test_and_selects_by_final_metric_priority(tmp_path: Path) -> None:
    query = sample(tmp_path, "Q1", "UNSEEN", "Ring", "test")
    extractor = FakeExtractor(torch.tensor([1.0, 0.0]))
    result_a = evaluate_split(extractor, gallery(), [query], model="a", split="test")
    result_b = evaluate_split(FakeExtractor(torch.tensor([0.0, 1.0])), gallery(), [query], model="b", split="test")
    comparison = compare_models({"a": result_a, "b": result_b})
    assert comparison["selected_model"] == "a"
    assert comparison["selection_priority"] == ["test_top1_category", "test_top5_category", "test_mrr_category"]


def test_compare_models_rejects_validation_results(tmp_path: Path) -> None:
    query = sample(tmp_path, "Q1", "UNSEEN", "Ring", "valid")
    result = evaluate_split(
        FakeExtractor(torch.tensor([1.0, 0.0])),
        gallery(),
        [query],
        model="model",
        split="valid",
    )
    try:
        compare_models({"model": result})
    except ValueError as exc:
        assert "test" in str(exc)
    else:
        raise AssertionError("Validation results must not be accepted for final model selection.")


def test_robustness_evaluation_reports_all_transforms(tmp_path: Path) -> None:
    queries = [sample(tmp_path, "Q1", "UNSEEN", "Ring", "test")]
    result = evaluate_robustness(
        FakeExtractor(torch.tensor([1.0, 0.0])),
        gallery(),
        queries,
        model="model",
        max_queries=1,
    )
    assert result["transforms"] == ["original", "rotation", "brightness", "center_crop", "resize"]
    assert set(result["results"]) == set(result["transforms"])
