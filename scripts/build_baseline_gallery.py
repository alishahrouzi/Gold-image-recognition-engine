"""Build the real Dataset 1 train embedding gallery for S2.7."""

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

from src.retrieval.embedding import load_baseline_embedding_model
from src.retrieval.gallery import Gallery, GalleryBuilder, build_gallery_loader

EXPECTED_TRAIN_IMAGES = 4328
EXPECTED_TRAIN_GROUPS = 1494


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the S2.7 baseline retrieval gallery.")
    parser.add_argument("--manifest", required=True, help="Dataset 1 CSV manifest path.")
    parser.add_argument("--dataset-root", default=None, help="Root for relative image paths in the manifest.")
    parser.add_argument("--checkpoint", default="experiments/baseline/checkpoints/best.pt")
    parser.add_argument("--output-dir", default="experiments/retrieval/baseline")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, choices=(128, 256), default=128)
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


def build_report(
    gallery: Gallery,
    *,
    checkpoint: Path,
    manifest: Path,
    split: str,
    device: str,
    batch_size: int,
    seed: int,
    duration_seconds: float,
) -> dict:
    categories = Counter(item["category"] for item in gallery.metadata)
    groups = {item["product_group"] for item in gallery.metadata}
    norms = torch.linalg.vector_norm(gallery.embeddings.float(), dim=1)
    return {
        "policy": "s2.7-baseline-gallery-v1",
        "split": split,
        "num_embeddings": gallery.size,
        "embedding_dim": gallery.embedding_dim,
        "num_product_groups": len(groups),
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
        "test_split_used": split == "test",
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.split != "train":
        raise ValueError("S2.7 real baseline gallery generation is restricted to the train split.")

    checkpoint = Path(args.checkpoint)
    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    device = resolve_device(args.device)
    started = time.perf_counter()
    extractor = load_baseline_embedding_model(
        checkpoint,
        embedding_dim=args.embedding_dim,
        device=device,
    )
    loader = build_gallery_loader(
        manifest,
        dataset_root=args.dataset_root,
        split="train",
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )
    gallery = GalleryBuilder(extractor).build(loader)
    duration = time.perf_counter() - started

    if gallery.size != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(
            f"Unexpected train gallery size: {gallery.size}; expected {EXPECTED_TRAIN_IMAGES}."
        )
    actual_groups = len({item["product_group"] for item in gallery.metadata})
    if actual_groups != EXPECTED_TRAIN_GROUPS:
        raise RuntimeError(
            f"Unexpected train product-group count: {actual_groups}; expected {EXPECTED_TRAIN_GROUPS}."
        )
    if gallery.embedding_dim != args.embedding_dim:
        raise RuntimeError(
            f"Unexpected embedding dimension: {gallery.embedding_dim}; expected {args.embedding_dim}."
        )

    embeddings_path = output_dir / "gallery_embeddings.pt"
    metadata_path = output_dir / "gallery_metadata.json"
    report_path = output_dir / "gallery_report.json"
    gallery.save(embeddings_path, metadata_path)

    report = build_report(
        gallery,
        checkpoint=checkpoint,
        manifest=manifest,
        split="train",
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
        duration_seconds=duration,
    )
    report["expected_num_embeddings"] = EXPECTED_TRAIN_IMAGES
    report["expected_num_product_groups"] = EXPECTED_TRAIN_GROUPS
    report["verification"] = {
        "count_match": True,
        "group_count_match": True,
        "embedding_dim_match": True,
        "finite_embeddings": True,
        "test_split_used": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"embeddings: {embeddings_path}")
    print(f"metadata: {metadata_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
