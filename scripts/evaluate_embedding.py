#!/usr/bin/env python3
"""Run S3.6 independent embedding-geometry evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from retrieval.embedding_evaluation import (  # noqa: E402
    build_evaluation_pairs,
    evaluate_model,
    load_evaluation_samples,
)

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "embedding_evaluation" / "s3.6_report.json"
DEFAULT_HYBRID = PROJECT_ROOT / "experiments" / "siamese" / "checkpoints" / "best.pt"
DEFAULT_RANDOM = PROJECT_ROOT / "experiments" / "siamese" / "sampling_experiments" / "random" / "checkpoints" / "best.pt"
DEFAULT_SAME = PROJECT_ROOT / "experiments" / "siamese" / "sampling_experiments" / "same_category" / "checkpoints" / "best.pt"
DEFAULT_CROSS = PROJECT_ROOT / "experiments" / "siamese" / "sampling_experiments" / "cross_category" / "checkpoints" / "best.pt"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Siamese embedding geometry (S3.6).")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="train", choices=("train", "valid", "test"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hybrid-checkpoint", default=str(DEFAULT_HYBRID))
    parser.add_argument("--random-checkpoint", default=str(DEFAULT_RANDOM))
    parser.add_argument("--same-category-checkpoint", default=str(DEFAULT_SAME))
    parser.add_argument("--cross-category-checkpoint", default=str(DEFAULT_CROSS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def resolve_device(value: str) -> str:
    if value == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
    return value


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    device = resolve_device(args.device)
    samples = load_evaluation_samples(
        Path(args.manifest),
        dataset_root=Path(args.dataset_root),
        split=args.split,
    )
    pairs = build_evaluation_pairs(
        samples,
        seed=args.seed,
        negative_ratio=args.negative_ratio,
    )

    checkpoints = {
        "s3.4_hybrid": Path(args.hybrid_checkpoint),
        "s3.5_random": Path(args.random_checkpoint),
        "s3.5_same_category": Path(args.same_category_checkpoint),
        "s3.5_cross_category": Path(args.cross_category_checkpoint),
    }
    results: dict[str, object] = {}
    for name, checkpoint in checkpoints.items():
        logging.info("Evaluating %s: %s", name, checkpoint)
        results[name] = evaluate_model(
            checkpoint,
            samples,
            pairs,
            device=device,
            batch_size=args.batch_size,
        )

    report = {
        "task": "S3.6-Embedding-Evaluation",
        "policy": "s3.6-independent-embedding-geometry-v1",
        "dataset": "dataset1",
        "manifest": str(Path(args.manifest)),
        "dataset_root": str(Path(args.dataset_root)),
        "split": args.split,
        "seed": args.seed,
        "negative_ratio": args.negative_ratio,
        "pair_counts": {
            "total": len(pairs),
            "same_product": sum(pair.label == 1 for pair in pairs),
            "different_product": sum(pair.label == 0 for pair in pairs),
        },
        "device": device,
        "models": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("S3.6 Embedding Evaluation")
    print("=" * 72)
    print(f"split: {args.split}")
    print(f"pairs: {len(pairs)} (same={report['pair_counts']['same_product']}, different={report['pair_counts']['different_product']})")
    print(f"device: {device}")
    for name, result in results.items():
        classification = result["classification"]
        cosine = result["cosine_distance"]
        print(
            f"{name}: ROC-AUC={classification['roc_auc']:.6f} "
            f"PR-AUC={classification['pr_auc']:.6f} "
            f"same_mean={cosine['same_product']['mean']:.6f} "
            f"different_mean={cosine['different_product']['mean']:.6f} "
            f"separation={cosine['separation_mean']:.6f} "
            f"NN={result['nearest_neighbor']['accuracy']:.6f}"
        )
    print(f"report: {output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
