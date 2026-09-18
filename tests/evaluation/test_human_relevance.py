from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_s4_9_human_relevance import (
    SELECTED_MODEL,
    load_query_records,
    select_queries,
)


def _record(image_id: str, category: str) -> dict:
    return {
        "query_image_id": image_id,
        "category": category,
        "ranked_candidates": [
            {
                "rank": rank,
                "product_id": f"p{rank}",
                "category": category,
                "similarity": 1.0 - rank / 100.0,
                "matched_image_ids": [f"g{rank}"],
                "matched_image_paths": [f"train/{category}/g{rank}.jpg"],
            }
            for rank in range(1, 11)
        ],
    }


def test_select_queries_is_deterministic_and_stratified():
    records = []
    for category in ["Bracelet", "Earrings", "Necklace", "Pendant", "Ring"]:
        records.extend(_record(f"{category}-{i}", category) for i in range(25))

    first = select_queries(records, per_category=5, seed=42)
    second = select_queries(records, per_category=5, seed=42)

    assert [(r["category"], r["query_image_id"]) for r in first] == [
        (r["category"], r["query_image_id"]) for r in second
    ]
    counts = {}
    for row in first:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert counts == {c: 5 for c in ["Bracelet", "Earrings", "Necklace", "Pendant", "Ring"]}


def test_select_queries_rejects_insufficient_category():
    records = [_record(f"Ring-{i}", "Ring") for i in range(2)]
    with pytest.raises(ValueError, match="cannot sample"):
        select_queries(records, per_category=3, seed=42)


def test_load_query_records_enforces_frozen_model_and_validation(tmp_path: Path):
    path = tmp_path / "evaluation.json"
    path.write_text(
        json.dumps({"model": SELECTED_MODEL, "split": "valid", "query_records": []}),
        encoding="utf-8",
    )
    assert load_query_records(path) == []

    path.write_text(
        json.dumps({"model": "s3.5_random", "split": "valid", "query_records": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen model"):
        load_query_records(path)
