#!/usr/bin/env python3
"""Run S4.9 final unseen-image generalization and robustness evaluation.

Model selection is frozen on the development ``valid`` split. The untouched
``test`` split is confirmation only and is never used to select a model.
Dataset 1 valid/test product groups are singleton groups, so category-aware
retrieval is the automatic ground truth. Ranked candidates are exported for a
later human visual-relevance benchmark.
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
from evaluation.final_test import (
    build_model_selection_report,
    evaluate_robustness,
    evaluate_split,
    select_model_from_validation,
)
from retrieval.embedding import load_siamese_embedding_model
from retrieval.gallery import Gallery, GalleryBuilder, build_gallery_loader

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "final_evaluation" / "s4.9"
DEFAULT_GALLERY_ROOT = PROJECT_ROOT / "experiments" / "generalization" / "s3.13"
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


def load_or_build_gallery(
    model_name: str,
    checkpoint: Path,
    manifest: Path,
    dataset_root: Path,
    gallery_root: Path,
    *,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Gallery:
    """Reuse the S3.13 train gallery when present; otherwise rebuild it."""
    model_dir = gallery_root / model_name
    embedding_path = model_dir / "gallery_embeddings.pt"
    metadata_path = model_dir / "gallery_metadata.json"
    if embedding_path.is_file() and metadata_path.is_file():
        logging.info("Reusing S3.13 gallery for %s", model_name)
        gallery = Gallery.load(embedding_path, metadata_path)
        if gallery.size != 4328 or gallery.embedding_dim != 128:
            raise ValueError(
                f"Unexpected S3.13 gallery shape for {model_name}: "
                f"{gallery.size}x{gallery.embedding_dim}; expected 4328x128."
            )
        return gallery

    logging.info("S3.13 gallery not found for %s; rebuilding from checkpoint", model_name)
    extractor = load_siamese_embedding_model(checkpoint, device=device)
    loader = build_gallery_loader(
        manifest,
        dataset_root=dataset_root,
        split="train",
        batch_size=batch_size,
        seed=42,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )
    gallery = GalleryBuilder(extractor).build(loader)
    model_dir.mkdir(parents=True, exist_ok=True)
    gallery.save(embedding_path, metadata_path)
    return gallery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S4.9 final unseen-image evaluation.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--gallery-root", default=str(DEFAULT_GALLERY_ROOT))
    parser.add_argument("--checkpoint", action="append", required=True, help="MODEL=CHECKPOINT_PATH; repeat for models")
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
    gallery_root = Path(args.gallery_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoints = parse_checkpoint_specs(args.checkpoint)

    samples = load_manifest(manifest, dataset_root=dataset_root, validate_files=True)
    valid_queries = split_samples(samples, "valid")
    test_queries = split_samples(samples, "test")

    validation_results = {}
    test_results = {}
    robustness_results = {}

    for model_name, checkpoint in checkpoints.items():
        model_dir = output_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        extractor = load_siamese_embedding_model(checkpoint, device=device)
        gallery = load_or_build_gallery(
            model_name,
            checkpoint,
            manifest,
            dataset_root,
            gallery_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        valid_result = evaluate_split(
            extractor, gallery, valid_queries, model=model_name, split="valid"
        )
        validation_results[model_name] = valid_result
        (model_dir / "valid_evaluation.json").write_text(
            json.dumps(valid_result.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        test_result = evaluate_split(
            extractor, gallery, test_queries, model=model_name, split="test"
        )
        test_results[model_name] = test_result
        (model_dir / "test_evaluation.json").write_text(
            json.dumps(test_result.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        if not args.skip_robustness:
            robustness = evaluate_robustness(
                extractor,
                gallery,
                valid_queries,
                model=model_name,
                max_queries=args.robustness_queries,
            )
            robustness_results[model_name] = robustness
            (model_dir / "valid_robustness.json").write_text(
                json.dumps(robustness, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    selected_model = select_model_from_validation(validation_results)
    comparison = build_model_selection_report(
        validation_results,
        test_results,
        selected_model=selected_model,
    )
    comparison.update(
        {
            "manifest": str(manifest),
            "dataset_root": str(dataset_root),
            "gallery_split": "train",
            "validation_query_split": "valid",
            "final_query_split": "test",
            "device": device,
            "checkpoint_paths": {name: str(path) for name, path in checkpoints.items()},
            "robustness": {
                "query_split": "valid",
                "num_queries_per_model": args.robustness_queries,
                "results": robustness_results,
            },
        }
    )
    report_path = output_root / "s4.9_final_evaluation_report.json"
    report_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_model": selected_model, "report": str(report_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
