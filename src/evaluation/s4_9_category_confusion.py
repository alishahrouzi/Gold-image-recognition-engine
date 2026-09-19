"""S4.9 category-confusion diagnostics over completed human-relevance results.

This module intentionally analyses existing evidence only. It does not rerun
inference, change the frozen MVP model, or inspect the test split.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("experiments/final_evaluation/s4.9/human_relevance")
DEFAULT_ANNOTATIONS = ROOT / "s4.9_human_relevance_annotations.csv"
DEFAULT_ERROR_ANALYSIS = ROOT / "s4.9_error_analysis.json"
DEFAULT_OUTPUT = ROOT / "s4.9_category_confusion_report.json"
CATEGORIES = ("Bracelet", "Earrings", "Necklace", "Pendant", "Ring")


def _norm(value: str) -> str:
    return str(value).strip().lower()


def load_annotations(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"query_image_id", "query_category", "rank", "candidate_category"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")
    return rows


def build_top1_confusion(rows: list[dict[str, str]]) -> tuple[list[str], dict[str, dict[str, int]]]:
    top1: dict[str, dict[str, str]] = {}
    for row in rows:
        try:
            rank = int(row["rank"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid rank: {row.get('rank')!r}") from exc
        if rank != 1:
            continue
        qid = row["query_image_id"]
        if qid in top1:
            raise ValueError(f"Duplicate Top-1 row for query {qid!r}")
        top1[qid] = {
            "query_category": row["query_category"].strip(),
            "candidate_category": row["candidate_category"].strip(),
        }

    categories = sorted(
        {v["query_category"] for v in top1.values()} |
        {v["candidate_category"] for v in top1.values()},
        key=lambda x: (_norm(x) not in {_norm(c) for c in CATEGORIES}, _norm(x)),
    )
    matrix = {q: {c: 0 for c in categories} for q in categories}
    for value in top1.values():
        matrix[value["query_category"]][value["candidate_category"]] += 1
    return categories, matrix


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    categories, matrix = build_top1_confusion(rows)
    queries = sorted({row["query_image_id"] for row in rows})
    query_category = {}
    for row in rows:
        query_category[row["query_image_id"]] = row["query_category"].strip()

    top1_mismatch = sum(
        count
        for q, candidates in matrix.items()
        for c, count in candidates.items()
        if _norm(q) != _norm(c)
    )
    top1_total = sum(sum(v.values()) for v in matrix.values())

    rank_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if int(row["rank"]) <= 10:
            rank_counts[row["query_image_id"]][row["candidate_category"].strip()] += 1

    category_hit = {}
    for k in (1, 5, 10):
        hits = []
        for qid in queries:
            qcat = _norm(query_category[qid])
            candidates = [
                row for row in rows
                if row["query_image_id"] == qid and int(row["rank"]) <= k
            ]
            hits.append(any(_norm(r["candidate_category"]) == qcat for r in candidates))
        category_hit[f"top{k}"] = sum(hits) / len(hits) if hits else 0.0

    top1_pairs = Counter(
        (_norm(v["query_category"]), _norm(v["candidate_category"]))
        for v in (
            {"query_category": q, "candidate_category": c}
            for q, cs in matrix.items()
            for c, n in cs.items()
            for _ in range(n)
        )
    )

    return {
        "protocol": {
            "source": str(DEFAULT_ANNOTATIONS),
            "split": "valid",
            "model": "s3.5_cross_category",
            "test_used": False,
            "rerun_inference": False,
            "queries": len(queries),
            "judgments": len(rows),
        },
        "top1": {
            "total_queries": top1_total,
            "category_mismatch_queries": top1_mismatch,
            "category_mismatch_rate": top1_mismatch / top1_total if top1_total else 0.0,
            "categories": categories,
            "confusion_matrix": matrix,
        },
        "category_hit_rate_from_annotation": category_hit,
        "most_common_top1_confusions": [
            {"query_category": q, "candidate_category": c, "count": n}
            for (q, c), n in top1_pairs.most_common()
            if q != c
        ],
        "top10_category_distribution": {
            qid: dict(sorted(counts.items()))
            for qid, counts in rank_counts.items()
        },
        "interpretation": [
            "A cross-category Top-1 match is an embedding-space retrieval confusion, not a separate classifier error.",
            "These annotations are a 100-query development sample and should not be presented as a full-population benchmark.",
            "No training/model parameters are changed by this diagnostic.",
        ],
    }


def run(annotations: Path = DEFAULT_ANNOTATIONS, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    rows = load_annotations(annotations)
    report = summarize(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
