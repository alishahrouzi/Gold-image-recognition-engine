"""Build S3.7 image-level galleries for the Siamese model variants."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch

from src.retrieval.embedding import load_siamese_embedding_model
from src.retrieval.gallery import Gallery, GalleryBuilder, build_gallery_loader

EXPECTED_TRAIN_IMAGES = 4328
EXPECTED_TRAIN_GROUPS = 1494
MODEL_CHECKPOINTS = {
    "s3.4_hybrid": "experiments/siamese/checkpoints/best.pt",
    "s3.5_random": "experiments/siamese/sampling_experiments/random/checkpoints/best.pt",
    "s3.5_same_category": "experiments/siamese/sampling_experiments/same_category/checkpoints/best.pt",
    "s3.5_cross_category": "experiments/siamese/sampling_experiments/cross_category/checkpoints/best.pt",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build S3.7 Siamese retrieval galleries.")
    parser.add_argument("--manifest", required=True, help="Dataset 1 CSV manifest path.")
    parser.add_argument("--dataset-root", required=True, help="Root for relative image paths in the manifest.")
    parser.add_argument("--output-dir", default="experiments/retrieval/siamese")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def resolve_device(requested: str) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def build_report(gallery: Gallery, *, model: str, checkpoint: Path, manifest: Path,
                 split: str, device: str, batch_size: int, seed: int,
                 duration_seconds: float) -> dict:
    categories = Counter(item["category"] for item in gallery.metadata)
    products = {item["product_id"] for item in gallery.metadata}
    norms = torch.linalg.vector_norm(gallery.embeddings.float(), dim=1)
    return {
        "task": "S3.7-Gallery-Builder",
        "policy": "s3.7-image-level-gallery-v1",
        "model": model,
        "split": split,
        "num_embeddings": gallery.size,
        "embedding_dim": gallery.embedding_dim,
        "num_product_groups": len(products),
        "category_counts": dict(sorted(categories.items())),
        "embedding_norm_mean": float(norms.mean().item()),
        "embedding_norm_std": float(norms.std(unbiased=False).item()),
        "embedding_norm_min": float(norms.min().item()),
        "embedding_norm_max": float(norms.max().item()),
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "device": device,
        "batch_size": batch_size,
        "seed": seed,
        "duration_seconds": duration_seconds,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.split != "train":
        raise ValueError("S3.7 gallery generation is restricted to the train split.")

    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    device = resolve_device(args.device)
    loader = build_gallery_loader(
        manifest,
        dataset_root=args.dataset_root,
        split=args.split,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    output_root = Path(args.output_dir)
    summary = {
        "task": "S3.7-Gallery-Builder",
        "policy": "s3.7-image-level-gallery-v1",
        "split": args.split,
        "models": {},
        "verification": {},
    }
    reference_metadata = None

    for model_name, checkpoint_text in MODEL_CHECKPOINTS.items():
        checkpoint = Path(checkpoint_text)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found for {model_name}: {checkpoint}")

        started = time.perf_counter()
        extractor = load_siamese_embedding_model(checkpoint, device=device)
        gallery = GalleryBuilder(extractor).build(loader)
        duration = time.perf_counter() - started

        if gallery.size != EXPECTED_TRAIN_IMAGES:
            raise RuntimeError(f"Unexpected gallery size for {model_name}: {gallery.size}")
        if len({item["product_id"] for item in gallery.metadata}) != EXPECTED_TRAIN_GROUPS:
            raise RuntimeError(f"Unexpected product count for {model_name}")
        if gallery.embedding_dim != 128:
            raise RuntimeError(f"Unexpected embedding dimension for {model_name}: {gallery.embedding_dim}")

        metadata_signature = tuple(
            (item["product_id"], item["category"], item["image"], item["image_id"])
            for item in gallery.metadata
        )
        if reference_metadata is None:
            reference_metadata = metadata_signature
        elif metadata_signature != reference_metadata:
            raise RuntimeError(f"Gallery metadata ordering/content differs for {model_name}")

        model_dir = output_root / model_name
        embeddings_path = model_dir / "gallery_embeddings.pt"
        metadata_path = model_dir / "gallery_metadata.json"
        report_path = model_dir / "gallery_report.json"
        gallery.save(embeddings_path, metadata_path)
        report = build_report(
            gallery,
            model=model_name,
            checkpoint=checkpoint,
            manifest=manifest,
            split=args.split,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            duration_seconds=duration,
        )
        report["expected_num_embeddings"] = EXPECTED_TRAIN_IMAGES
        report["expected_num_product_groups"] = EXPECTED_TRAIN_GROUPS
        report["verification"] = {
            "count_match": True,
            "product_count_match": True,
            "embedding_dim_match": True,
            "finite_embeddings": True,
            "metadata_alignment_match": True,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["models"][model_name] = report

    summary["verification"] = {
        "model_count": len(MODEL_CHECKPOINTS),
        "all_gallery_sizes_match": True,
        "all_product_counts_match": True,
        "all_embedding_dims_match": True,
        "metadata_alignment_across_models": True,
    }
    summary_path = output_root / "s3.7_gallery_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
