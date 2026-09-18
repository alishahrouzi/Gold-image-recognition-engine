#!/usr/bin/env python3
"""Run data-driven S4.9 human visual-relevance error analysis.

This analysis does not retrain or modify the frozen model. It classifies
annotated Top-10 failures by where relevant results occur in the ranking.
Visual root-cause labels (angle, lighting, crop, etc.) are intentionally
not inferred automatically from metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_INPUT = Path(
    "experiments/final_evaluation/s4.9/human_relevance/"
    "s4.9_human_relevance_annotations.csv"
)
DEFAULT_OUTPUT = Path(
    "experiments/final_evaluation/s4.9/human_relevance/"
    "s4.9_error_analysis.json"
)
EXPECTED_MODEL = "s3.5_cross_category"
RELEVANT_THRESHOLD = 2


def load_rows(path: Path) -> dict[tuple[str, str], list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Annotation CSV is empty.")

    required = {
        "query_index",
        "query_image_id",
        "query_category",
        "rank",
        "relevance",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")

    grouped = defaultdict(list)
    for row in rows:
        label = int(str(row["relevance"]).strip())
        if label not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid relevance label: {label}")
        row["_rank"] = int(row["rank"])
        row["_label"] = label
        grouped[(str(row["query_index"]), str(row["query_image_id"]))].append(row)

    for key, items in grouped.items():
        ranks = sorted(row["_rank"] for row in items)
        if ranks != list(range(1, 11)):
            raise ValueError(f"Query {key} must contain exactly ranks 1..10.")
    return grouped


def classify(labels: list[int]) -> str:
    relevant = [i + 1 for i, label in enumerate(labels) if label >= RELEVANT_THRESHOLD]
    if not relevant:
        return "no_relevant_in_top10"
    first = relevant[0]
    if first == 1:
        if sum(label >= RELEVANT_THRESHOLD for label in labels[:5]) >= 2:
            return "relevant_at_top1_with_additional_top5_relevance"
        return "relevant_at_top1_only_or_sparse"
    if first <= 5:
        return "relevant_in_top5_but_not_top1"
    return "relevant_only_in_ranks_6_to_10"


def query_result(items: list[dict]) -> dict:
    items = sorted(items, key=lambda row: row["_rank"])
    labels = [row["_label"] for row in items]
    relevant_ranks = [
        row["_rank"] for row in items if row["_label"] >= RELEVANT_THRESHOLD
    ]
    return {
        "query_index": int(items[0]["query_index"]),
        "query_image_id": items[0]["query_image_id"],
        "category": items[0]["query_category"],
        "classification": classify(labels),
        "relevant_count_top10": len(relevant_ranks),
        "relevant_ranks_top10": relevant_ranks,
        "first_relevant_rank": relevant_ranks[0] if relevant_ranks else None,
        "top1_relevance": labels[0],
        "top5_relevant_count": sum(x >= RELEVANT_THRESHOLD for x in labels[:5]),
        "top10_relevant_count": sum(x >= RELEVANT_THRESHOLD for x in labels),
        "relevance_distribution_top10": {
            str(label): labels.count(label) for label in range(4)
        },
    }


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {"queries": 0}
    first_rank_values = [
        r["first_relevant_rank"] for r in results
        if r["first_relevant_rank"] is not None
    ]
    return {
        "queries": len(results),
        "no_relevant_in_top10": sum(
            r["classification"] == "no_relevant_in_top10" for r in results
        ),
        "relevant_in_top10": sum(bool(r["relevant_ranks_top10"]) for r in results),
        "top1_relevant": sum(r["top1_relevance"] >= RELEVANT_THRESHOLD for r in results),
        "top5_relevant": sum(r["top5_relevant_count"] > 0 for r in results),
        "relevant_first_rank_mean": (
            sum(first_rank_values) / len(first_rank_values)
            if first_rank_values
            else None
        ),
        "relevant_first_rank_median": (
            sorted(first_rank_values)[len(first_rank_values) // 2]
            if first_rank_values
            else None
        ),
        "mean_relevant_count_top10": sum(
            r["relevant_count_top10"] for r in results
        ) / len(results),
        "classification_counts": dict(
            Counter(r["classification"] for r in results)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="S4.9 human relevance error analysis.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=EXPECTED_MODEL)
    args = parser.parse_args()

    if args.model != EXPECTED_MODEL:
        raise ValueError(
            f"Error analysis is frozen to {EXPECTED_MODEL!r}; got {args.model!r}."
        )

    grouped = load_rows(Path(args.input).expanduser().resolve())
    results = [
        query_result(items)
        for _, items in sorted(grouped.items(), key=lambda item: int(item[0][0]))
    ]

    by_category = defaultdict(list)
    for result in results:
        by_category[result["category"]].append(result)

    label_counts = Counter()
    for items in grouped.values():
        label_counts.update(row["_label"] for row in items)

    report = {
        "policy": "s4.9-human-visual-relevance-error-analysis-v1",
        "model": EXPECTED_MODEL,
        "source_split": "valid",
        "binary_relevance_threshold": RELEVANT_THRESHOLD,
        "annotation": {
            "queries": len(results),
            "judgments": len(results) * 10,
            "annotators": 1,
        },
        "overall": aggregate(results),
        "overall_label_distribution": {
            str(label): label_counts[label] for label in range(4)
        },
        "per_category": {
            category: aggregate(items)
            for category, items in sorted(by_category.items())
        },
        "query_results": results,
        "interpretation": {
            "no_relevant_in_top10": "No candidate with relevance >= 2 in Top-10; candidate-generation/retrieval failure at this cutoff.",
            "relevant_only_in_ranks_6_to_10": "A relevant candidate exists but ranking places it below Top-5.",
            "relevant_in_top5_but_not_top1": "A relevant candidate is retrievable but Top-1 ranking is incorrect.",
            "relevant_at_top1_only_or_sparse": "Top-1 is relevant, but additional Top-5 relevance is sparse.",
            "relevant_at_top1_with_additional_top5_relevance": "Top-1 is relevant and at least one additional relevant result is present in Top-5.",
        },
        "limitations": [
            "These categories describe ranking behavior, not visual root causes.",
            "Angle, lighting, crop, background, and fine-grained design causes require image-level inspection.",
            "One annotator means inter-annotator agreement is unavailable.",
            "This analysis uses validation annotations and does not use test labels for model selection.",
        ],
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "model": EXPECTED_MODEL,
        "queries": len(results),
        "judgments": len(results) * 10,
        "report": str(output),
        "overall": report["overall"],
        "overall_label_distribution": report["overall_label_distribution"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
