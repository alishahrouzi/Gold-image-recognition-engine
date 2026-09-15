"""Run the S4.4 local FastAPI application with an explicit model checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import uvicorn

from inference.gallery import GalleryRuntimeConfig, GalleryLoader
from inference.pipeline import InferencePipeline
from inference.query import QueryProcessor
from retrieval.embedding import load_siamese_embedding_model
from api.app import SearchService, create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Gold visual search API.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained S3.5 random Siamese checkpoint (.pt).",
    )
    parser.add_argument(
        "--gallery-root",
        type=Path,
        default=Path("experiments/retrieval/siamese"),
        help="Root containing the selected S3.7 gallery directory.",
    )
    parser.add_argument(
        "--model",
        default="s3.5_random",
        help="Model/gallery experiment name.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. cuda or cpu.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def build_service(args: argparse.Namespace) -> SearchService:
    embedder = load_siamese_embedding_model(args.checkpoint, device=args.device)
    gallery = GalleryLoader(
        GalleryRuntimeConfig(
            model_name=args.model,
            root=args.gallery_root,
            expected_embedding_dim=128,
        )
    ).load()
    pipeline = InferencePipeline(embedder, gallery.search_engine)
    return SearchService(QueryProcessor(), pipeline)


def main() -> None:
    args = parse_args()
    service = build_service(args)
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
