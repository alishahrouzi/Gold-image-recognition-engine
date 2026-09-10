#!/usr/bin/env python3
"""Train the Dataset 1 Siamese metric-learning model (S3.4)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metric_learning.training import train_siamese  # noqa: E402
from training.config import TrainingConfig  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_PAIRS = PROJECT_ROOT / "reports" / "dataset" / "dataset1_pairs.csv"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "siamese" / "checkpoints"
DEFAULT_SUMMARY = PROJECT_ROOT / "experiments" / "siamese" / "training_summary.json"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dataset 1 Siamese model (S3.4).")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--no-early-stopping", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    config = TrainingConfig(
        seed=args.seed,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_name="contrastive",
        margin=args.margin,
        optimizer_name="adamw",
        scheduler_name=args.scheduler,
        num_workers=args.num_workers,
        pin_memory=True,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_best=True,
        early_stopping_enabled=not args.no_early_stopping,
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
        monitor="val_loss",
        deterministic=True,
    )

    summary = train_siamese(
        Path(args.manifest),
        Path(args.pairs),
        config=config,
    )

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    print("=" * 60)
    print("Siamese training (S3.4)")
    print("=" * 60)
    print(f"device: {summary['device']}")
    print(f"completed_epoch: {summary['completed_epoch']}")
    print(f"best_val_loss: {summary['best_val_loss']}")
    print(f"stopped_early: {summary['stopped_early']}")
    print(f"pair_counts: {summary['pair_counts']}")
    print(f"checkpoint_best: {summary['checkpoint_best']}")
    print(f"summary: {summary_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
