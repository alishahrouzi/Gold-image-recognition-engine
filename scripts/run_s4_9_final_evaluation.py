#!/usr/bin/env python3
"""Run S4.9 final unseen-image generalization and robustness evaluation.

The script evaluates frozen S3.13 checkpoints without retraining or tuning.
The gallery is always Dataset 1 train; validation/test images are query-only.
For Dataset 1 valid/test singleton product groups, category-aware retrieval is
used as the automatic ground truth. Ranked candidates are exported for a later
human visual-relevance benchmark.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from data.loaders.manifest import load_manifest
from data.types import Sample
from evaluation.final_test import compare_models, evaluate_robustness, evaluate_split
from retrieval.embedding import load_siamese_embedding_model
from retrieval.gallery import GalleryBuilder, build_gallery_loader

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "final_evaluation" / "s4.9"
DEFAULT_MODELS = (
    "s3.4_hybrid",
    "s3.5_random",
    "s3.5_same_category",
    "s3.5_cross_category",
)


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
        "Dataset 1 root could not be resolved. Provide --dataset-root or set ZARGAR_DATASET1_ROOT."
    )


def resolve_device(requested: str) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def parse_checkpoint_specs(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint must use MODEL=PATH syntax: {value!r}")
        model, raw_path = value.split("=", 1)
        if model not in DEFAULT_MODELS:
            raise ValueError(f"Unsupported S4.9 model: {model!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint for {model} does not exist: {path}")
        parsed[model] = path
    if not parsed:
        raise ValueError("Provide at least one --checkpoint MODEL=PATH.")
    return parsed


def split_samples(samples: list[Sample], split: str) -> list[Sample]:
    selected = [sample for sample in samples if sample.split == split]
    if not selected:
        raise ValueError(f"No samples found for split={split!r}.")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S4.9 final unseen-image evaluation.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint", action="append", required=True, help="MODEL=CHECKPOINT_PATH; repeat for models")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--robustness-queries", type=int, default=100)
    parser.add_argument("--skip-robustness", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    dataset_root = resolve_dataset_root(args.dataset_root)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoints = parse_checkpoint_specs(args.checkpoint)

    samples = load_manifest(manifest, dataset_root=dataset_root, validate_files=True)
    queries = split_samples(samples, args.split)
    preprocessor = None
    model_results = {}

    for model_name, checkpoint in checkpoints.items():
        logging.info("Evaluating %s from frozen checkpoint %s", model_name, checkpoint)
        model_dir = output_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        extractor = load_siamese_embedding_model(checkpoint, device=device)
        gallery_loader = build_gallery_loader(
            manifest,
            dataset_root=dataset_root,
            split="train",
            batch_size=args.batch_size,
            seed=42,
            num_workers=args.num_workers,
            pin_memory=device == "cuda",
        )
        gallery = GalleryBuilder(extractor).build(gallery_loader)
        gallery.save(model_dir / "train_gallery_embeddings.pt", model_dir / "train_gallery_metadata.json")

        result = evaluate_split(
            extractor,
            gallery,
            queries,
            model=model_name,
            split=args.split,
            preprocessor=preprocessor,
        )
        (model_dir / f"{args.split}_evaluation.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        model_results[model_name] = result

        if not args.skip_robustness:
            robustness = evaluate_robustness(
                extractor,
                gallery,
                queries,
                model=model_name,
                max_queries=args.robustness_queries,
                preprocessor=preprocessor,
            )
            (model_dir / "robustness.json").write_text(
                json.dumps(robustness, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    comparison = compare_models(model_results)
    comparison["manifest"] = str(manifest)
    comparison["dataset_root"] = str(dataset_root)
    comparison["gallery_split"] = "train"
    comparison["query_split"] = args.split
    comparison["device"] = device
    comparison["checkpoint_paths"] = {name: str(path) for name, path in checkpoints.items()}
    (output_root / f"s4.9_{args.split}_model_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
