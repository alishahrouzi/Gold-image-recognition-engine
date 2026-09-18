#!/usr/bin/env python3
"""Evaluate completed S4.9 human visual-relevance annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

DEFAULT_INPUT = Path("experiments/final_evaluation/s4.9/human_relevance/s4.9_human_relevance_annotations.csv")
DEFAULT_OUTPUT = Path("experiments/final_evaluation/s4.9/human_relevance/s4.9_human_relevance_report.json")
EXPECTED_MODEL = "s3.5_cross_category"
VALID_LABELS = {0, 1, 2, 3}


def dcg(values):
    return sum(value / math.log2(index + 2) for index, value in enumerate(values))


def ndcg(values, k):
    actual = list(values[:k])
    ideal = sorted(values, reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(actual) / ideal_dcg if ideal_dcg else 0.0


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Annotation CSV is empty.")
    required = {"annotation_id", "query_index", "query_image_id", "query_category", "rank", "relevance"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")

    grouped = defaultdict(list)
    for row in rows:
        raw = str(row["relevance"]).strip()
        if raw == "":
            raise ValueError(f"Unannotated judgment: {row['annotation_id']}")
        try:
            label = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid relevance label: {raw!r}") from exc
        if label not in VALID_LABELS:
            raise ValueError(f"Relevance must be 0..3; got {label}.")
        row["_label"] = label
        row["_rank"] = int(row["rank"])
        grouped[(row["query_index"], row["query_image_id"])].append(row)

    for key, items in grouped.items():
        ranks = sorted(row["_rank"] for row in items)
        if ranks != list(range(1, 11)):
            raise ValueError(f"Query {key} must contain exactly ranks 1..10.")
    return grouped


def score_query(rows):
    rows = sorted(rows, key=lambda row: row["_rank"])
    labels = [row["_label"] for row in rows]
    binary = [1 if label >= 2 else 0 for label in labels]
    return {
        "category": rows[0]["query_category"],
        "precision_at_1": sum(binary[:1]) / 1,
        "precision_at_5": sum(binary[:5]) / 5,
        "precision_at_10": sum(binary[:10]) / 10,
        "hit_at_1": int(any(binary[:1])),
        "hit_at_5": int(any(binary[:5])),
        "hit_at_10": int(any(binary[:10])),
        "mean_relevance_at_1": sum(labels[:1]) / 1,
        "mean_relevance_at_5": sum(labels[:5]) / 5,
        "mean_relevance_at_10": sum(labels[:10]) / 10,
        "ndcg_at_5": ndcg(labels, 5),
        "ndcg_at_10": ndcg(labels, 10),
    }


def aggregate(scored):
    if not scored:
        return {"num_queries": 0}
    keys = [key for key in scored[0] if key != "category"]
    result = {"num_queries": len(scored)}
    for key in keys:
        result[key] = sum(float(row[key]) for row in scored) / len(scored)
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate S4.9 human visual relevance.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=EXPECTED_MODEL)
    args = parser.parse_args()

    grouped = load_rows(Path(args.input).expanduser().resolve())
    scored_by_query = {
        key: score_query(rows) for key, rows in grouped.items()
    }
    scored = list(scored_by_query.values())

    by_category = defaultdict(list)
    for row in scored:
        by_category[row["category"]].append(row)

    report = {
        "policy": "s4.9-human-visual-relevance-v1",
        "model": args.model,
        "source_split": "valid",
        "binary_relevance_threshold": 2,
        "relevance_scale": {
            "0": "irrelevant",
            "1": "weakly similar",
            "2": "visually similar",
            "3": "highly similar",
        },
        "annotation": {
            "queries": len(scored),
            "judgments": len(scored) * 10,
            "annotators": 1,
            "inter_annotator_agreement": None,
        },
        "overall": aggregate(scored),
        "per_category": {
            category: aggregate(items)
            for category, items in sorted(by_category.items())
        },
        "query_results": [
            {
                "query_image_id": key[1],
                **values,
            }
            for key, values in sorted(scored_by_query.items(), key=lambda item: int(item[0][0]))
        ],
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "model": args.model,
        "queries": len(scored),
        "judgments": len(scored) * 10,
        "report": str(output),
        "overall": report["overall"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
