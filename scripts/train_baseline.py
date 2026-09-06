"""CLI for the first real S2.6 Dataset 1 baseline training run."""

from __future__ import annotations

import argparse
import json

from baselines import BaselineConfig, run_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the S2.6 Gold baseline.")
    parser.add_argument("--manifest", required=True, help="Dataset 1 CSV manifest path.")
    parser.add_argument("--dataset-root", default=None, help="Root for relative image paths in the manifest.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, choices=(128, 256), default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="experiments/baseline/checkpoints")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--no-early-stopping", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BaselineConfig(
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_enabled=not args.no_early_stopping,
    )
    from data.preprocessing import AugmentationConfig
    augmentation = AugmentationConfig.disabled() if args.no_augmentation else AugmentationConfig(seed=args.seed)
    report = run_baseline(
        args.manifest,
        dataset_root=args.dataset_root,
        config=config,
        augmentation=augmentation,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
