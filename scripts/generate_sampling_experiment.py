"""Generate one controlled S3.5 pair dataset for a negative strategy.

The script reuses S1.10 positive generation and validation. It does not train
models; its output is a strategy-specific pair CSV plus an audit report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.constants import SOURCE_DATASET1
from data.loaders.manifest import load_manifest
from data.pairs.generator import generate_positive_pairs, occupied_keys, write_pairs_csv
from data.pairs.strategies import (
    CROSS_CATEGORY_NEGATIVE,
    RANDOM_NEGATIVE,
    SAME_CATEGORY_NEGATIVE,
    sample_negatives_by_strategy,
)
from data.pairs.types import PAIR_TYPE_NEGATIVE, PAIR_TYPE_POSITIVE, Pair, sort_pairs
from data.pairs.validation import validate_pairs

STRATEGIES = (RANDOM_NEGATIVE, SAME_CATEGORY_NEGATIVE, CROSS_CATEGORY_NEGATIVE)
DEFAULT_SEED = 2026
DEFAULT_NEGATIVE_RATIO = 1.0


def build_experiment_pairs(samples, *, strategy: str, seed: int, negative_ratio: float):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}. Choose from {STRATEGIES}.")
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be >= 0.")

    train_samples = [sample for sample in samples if sample.split == "train"]
    positives = generate_positive_pairs(train_samples)
    negative_count = int(round(len(positives) * negative_ratio))
    negatives = sample_negatives_by_strategy(
        train_samples,
        Random(seed),
        strategy=strategy,
        count=negative_count,
        occupied=occupied_keys(positives),
    )
    pairs = sort_pairs([*positives, *negatives], ("train", "valid", "test"))
    checks = validate_pairs(pairs, samples)
    if not all(checks.values()):
        raise ValueError(f"S3.5 pair validation failed: {checks}")
    return pairs, checks


def build_report(pairs, checks, *, strategy, seed, negative_ratio, manifest):
    positives = [pair for pair in pairs if pair.pair_type == PAIR_TYPE_POSITIVE]
    negatives = [pair for pair in pairs if pair.pair_type == PAIR_TYPE_NEGATIVE]
    same = [pair for pair in negatives if pair.negative_type == SAME_CATEGORY_NEGATIVE]
    cross = [pair for pair in negatives if pair.negative_type == CROSS_CATEGORY_NEGATIVE]
    random = [pair for pair in negatives if pair.negative_type == RANDOM_NEGATIVE]
    return {
        "dataset": SOURCE_DATASET1,
        "manifest": str(Path(manifest)),
        "strategy": strategy,
        "seed": seed,
        "positive_negative_ratio": negative_ratio,
        "total_pairs": len(pairs),
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
        "negative_counts": {
            "random": len(random),
            "same_category": len(same),
            "cross_category": len(cross),
        },
        "split_counts": {
            split: sum(pair.split == split for pair in pairs)
            for split in ("train", "valid", "test")
        },
        "validation_results": {"passed": all(checks.values()), "checks": dict(checks)},
        "policy": "s3.5-controlled-negative-strategy-v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a controlled S3.5 sampling experiment.")
    parser.add_argument("--manifest", default="reports/dataset/dataset1_manifest.csv")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--negative-ratio", type=float, default=DEFAULT_NEGATIVE_RATIO)
    parser.add_argument("--output-dir", default="experiments/sampling")
    args = parser.parse_args()

    samples = load_manifest(args.manifest, dataset_root=args.dataset_root, validate_files=True)
    pairs, checks = build_experiment_pairs(
        samples, strategy=args.strategy, seed=args.seed, negative_ratio=args.negative_ratio
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.strategy}_pairs.csv"
    report_path = output_dir / f"{args.strategy}_report.json"
    write_pairs_csv(pairs, csv_path)
    report = build_report(
        pairs, checks, strategy=args.strategy, seed=args.seed,
        negative_ratio=args.negative_ratio, manifest=args.manifest,
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
