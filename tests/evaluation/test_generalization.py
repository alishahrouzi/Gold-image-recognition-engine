from __future__ import annotations

from pathlib import Path

import torch

from data.types import Sample
from evaluation.generalization import (
    build_gate_pairs,
    build_group_holdout,
    evaluate_unseen_groups,
    select_model,
)
from retrieval.gallery import Gallery


def _sample(image_id: str, group_id: str, category: str, category_id: int) -> Sample:
    return Sample(
        image_id=image_id,
        image_path=Path(f"{image_id}.jpg"),
        group_id=group_id,
        category=category,
        category_id=category_id,
        split="train",
        source="dataset1",
    )


def _dataset_samples() -> list[Sample]:
    return [
        _sample("a1", "A", "Ring", 4),
        _sample("a2", "A", "Ring", 4),
        _sample("b1", "B", "Ring", 4),
        _sample("b2", "B", "Ring", 4),
        _sample("c1", "C", "Necklace", 2),
        _sample("c2", "C", "Necklace", 2),
        _sample("d1", "D", "Necklace", 2),
        _sample("d2", "D", "Necklace", 2),
        _sample("e1", "E", "Bracelet", 0),
        _sample("e2", "E", "Bracelet", 0),
        _sample("singleton", "S", "Bracelet", 0),
    ]


def test_group_holdout_is_disjoint_and_excludes_singletons() -> None:
    split = build_group_holdout(_dataset_samples(), holdout_fraction=0.2, seed=42)

    train_groups = {sample.group_id for sample in split.train_samples}
    holdout_groups = {sample.group_id for sample in split.holdout_samples}
    assert train_groups.isdisjoint(holdout_groups)
    assert "S" not in holdout_groups
    assert all(
        sum(sample.group_id == group_id for sample in split.holdout_samples) >= 2
        for group_id in holdout_groups
    )


def test_gate_pairs_never_reference_holdout_groups() -> None:
    split = build_group_holdout(_dataset_samples(), holdout_fraction=0.2, seed=42)
    pairs, report = build_gate_pairs(split.train_samples, model_name="s3.5_random")
    holdout_groups = set(split.holdout_group_ids)

    assert report["validation_results"]["passed"] is True
    assert all(
        pair.group_id_1 not in holdout_groups and pair.group_id_2 not in holdout_groups
        for pair in pairs
    )


def test_select_model_uses_top1_then_top5_then_mrr() -> None:
    results = {
        "a": {"metrics": {"top1": 0.2, "top5": 0.3, "mrr": 0.4}},
        "b": {"metrics": {"top1": 0.2, "top5": 0.35, "mrr": 0.1}},
        "c": {"metrics": {"top1": 0.19, "top5": 0.9, "mrr": 0.9}},
    }
    assert select_model(results) == "b"


def test_unseen_group_evaluation_finds_true_product() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0],
            [0.7, 0.7], [0.7, 0.7],
        ],
        dtype=torch.float32,
    )
    metadata = tuple(
        {
            "product_id": group,
            "category": category,
            "image": f"{image}.jpg",
            "image_id": image,
        }
        for group, category, image in (
            ("A", "Ring", "a1"),
            ("A", "Ring", "a2"),
            ("B", "Necklace", "b1"),
            ("B", "Necklace", "b2"),
            ("C", "Bracelet", "c1"),
            ("C", "Bracelet", "c2"),
        )
    )
    result = evaluate_unseen_groups(
        Gallery(embeddings=embeddings, metadata=metadata),
        holdout_group_ids={"B"},
    )

    assert result["query_counts"]["total"] == 2
    assert result["metrics"]["top1"] == 1.0
    assert result["metrics"]["top5"] == 1.0
    assert result["metrics"]["mrr"] == 1.0
