"""Validate the local S4.3 runtime gallery artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.gallery import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_GALLERY_ROOT,
    GalleryRuntimeConfig,
    GalleryLoader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and load an MVP runtime gallery.")
    parser.add_argument("--model", default="s3.5_random", help="Gallery/model experiment name.")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_GALLERY_ROOT),
        help="Root directory containing per-model gallery directories.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="Expected embedding dimension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GalleryRuntimeConfig(
        model_name=args.model,
        root=Path(args.root),
        expected_embedding_dim=args.embedding_dim,
    )
    runtime = GalleryLoader(config).load()

    categories = sorted(
        {str(item["category"]) for item in runtime.gallery.metadata}
    )
    print("S4.3 Runtime Gallery")
    print(f"model: {runtime.model_name}")
    print(f"embeddings: {runtime.size}")
    print(f"embedding_dim: {runtime.embedding_dim}")
    print(f"products: {runtime.product_count}")
    print(f"categories: {', '.join(categories)}")
    print(f"embedding_path: {runtime.embedding_path}")
    print(f"metadata_path: {runtime.metadata_path}")
    print("status: VALID")


if __name__ == "__main__":
    main()
