#!/usr/bin/env python3
"""Run the S3.13 group-disjoint Generalization / Model Selection Gate.

The runner creates one deterministic product-group holdout, retrains the four
current Siamese candidates without those groups, builds a runtime-like gallery
from the resulting checkpoint, and evaluates only held-out product queries.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from data.loaders.manifest import load_manifest
from evaluation.generalization import (
    S3_13_K,
    S3_13_MODELS,
    S3_13_POLICY,
    build_gate_pairs,
    build_group_holdout,
    evaluate_unseen_groups,
    select_model,
    write_pairs,
)
from metric_learning.training import train_siamese
from retrieval.embedding import load_siamese_embedding_model
from retrieval.gallery import GalleryBuilder, build_gallery_loader
from training.config import TrainingConfig

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "dataset" / "ai-tool-pool-pool-jewelry-vision"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "generalization" / "s3.13"

MODEL_CHECKPOINT_PATHS = {
    "s3.4_hybrid": "checkpoints/best.pt",
    "s3.5_random": "checkpoints/best.pt",
    "s3.5_same_category": "checkpoints/best.pt",
    "s3.5_cross_category": "checkpoints/best.pt",
}


def resolve_dataset_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("ZARGAR_DATASET1_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    candidate = PROJECT_ROOT.parent / "dataset" / "ai-tool-pool-jewelry-vision"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        "Dataset 1 root could not be resolved. Provide --dataset-root or set "
        "ZARGAR_DATASET1_ROOT."
    )


def resolve_device(requested: str) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def model_training_config(args: argparse.Namespace, checkpoint_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        seed=args.training_seed,
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
        checkpoint_dir=checkpoint_dir,
        save_best=True,
        early_stopping_enabled=not args.no_early_stopping,
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
        monitor="val_loss",
        deterministic=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S3.13 Generalization / Model Selection Gate.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--pair-seed", type=int, default=2026)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--models", nargs="+", choices=S3_13_MODELS, default=list(S3_13_MODELS))
    parser.add_argument("--no-early-stopping", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    manifest = Path(args.manifest).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    requested_device = resolve_device(args.device)
    samples = load_manifest(manifest, dataset_root=dataset_root, validate_files=True)
    split = build_group_holdout(
        samples,
        holdout_fraction=args.holdout_fraction,
        seed=args.split_seed,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    split_report_path = output_root / "s3.13_group_split.json"
    split_report_path.write_text(
        json.dumps(split.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    common = {
        "experiment": "EXP-S3.13",
        "policy": S3_13_POLICY,
        "manifest": str(manifest),
        "dataset_root": str(dataset_root),
        "device": requested_device,
        "split_seed": args.split_seed,
        "pair_seed": args.pair_seed,
        "training_seed": args.training_seed,
        "holdout_fraction": args.holdout_fraction,
        "k": S3_13_K,
        "group_split": split.to_dict(),
        "models": {},
    }

    for model_name in args.models:
        started = time.perf_counter()
        model_dir = output_root / model_name
        pair_path = model_dir / "pairs.csv"
        pair_report_path = model_dir / "pair_report.json"
        checkpoint_dir = model_dir / "checkpoints"
        summary_path = model_dir / "training_summary.json"
        gallery_embeddings_path = model_dir / "gallery_embeddings.pt"
        gallery_metadata_path = model_dir / "gallery_metadata.json"
        evaluation_path = model_dir / "evaluation.json"

        pairs, pair_report = build_gate_pairs(
            split.train_samples,
            model_name=model_name,
            seed=args.pair_seed,
        )
        write_pairs(pairs, str(pair_path))
        model_dir.mkdir(parents=True, exist_ok=True)
        pair_report_path.write_text(
            json.dumps(pair_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        config = model_training_config(args, checkpoint_dir)
        training_summary = train_siamese(
            manifest,
            pair_path,
            dataset_root=dataset_root,
            config=config,
        )
        summary_path.write_text(
            json.dumps(training_summary, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

        checkpoint_path = Path(training_summary["checkpoint_best"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Best checkpoint for {model_name} was not produced: {checkpoint_path}"
            )

        extractor = load_siamese_embedding_model(checkpoint_path, device=requested_device)
        gallery_loader = build_gallery_loader(
            manifest,
            dataset_root=dataset_root,
            split="train",
            batch_size=args.batch_size,
            seed=args.training_seed,
            num_workers=args.num_workers,
            pin_memory=requested_device == "cuda",
        )
        gallery = GalleryBuilder(extractor).build(gallery_loader)
        gallery.save(gallery_embeddings_path, gallery_metadata_path)

        evaluation = evaluate_unseen_groups(
            gallery,
            holdout_group_ids=set(split.holdout_group_ids),
            k=S3_13_K,
        )
        evaluation["model"] = model_name
        evaluation["checkpoint"] = str(checkpoint_path)
        evaluation["training_summary"] = {
            "best_val_loss": training_summary.get("best_val_loss"),
            "completed_epoch": training_summary.get("completed_epoch"),
            "stopped_early": training_summary.get("stopped_early"),
        }
        evaluation_path.write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

        common["models"][model_name] = {
            "pair_report": pair_report,
            "training_summary": evaluation["training_summary"],
            "checkpoint": str(checkpoint_path),
            "gallery": {
                "size": gallery.size,
                "embedding_dim": gallery.embedding_dim,
                "product_groups": len({item["product_id"] for item in gallery.metadata}),
                "embeddings": str(gallery_embeddings_path),
                "metadata": str(gallery_metadata_path),
            },
            "evaluation": evaluation,
            "duration_seconds": time.perf_counter() - started,
        }

    selected = select_model(
        {name: payload["evaluation"] for name, payload in common["models"].items()}
    )
    common["decision"] = {
        "selected_model": selected,
        "selection_priority": ["top1", "top5", "mrr"],
        "selection_basis": "unseen-product-group retrieval",
        "runtime_architecture_changed": False,
        "quality_threshold_applied": False,
        "note": (
            "S3.13 selects the strongest candidate under a group-disjoint protocol. "
            "It does not declare a universal production-accuracy threshold; model-v2 "
            "work remains a separate decision if absolute generalization is insufficient."
        ),
    }
    common["status"] = "completed"
    common["duration_seconds"] = sum(
        payload["duration_seconds"] for payload in common["models"].values()
    )

    report_path = output_root / "s3.13_generalization_gate.json"
    report_path.write_text(
        json.dumps(common, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(common["decision"], indent=2, ensure_ascii=False))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
