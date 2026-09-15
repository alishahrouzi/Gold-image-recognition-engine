"""Run the S4.4 local FastAPI application with S4.6 startup error handling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import uvicorn

from api.app import SearchService, create_app
from api.errors import GalleryUnavailableError, ModelUnavailableError
from inference.gallery import GalleryLoader, GalleryRuntimeConfig
from inference.pipeline import InferencePipeline
from inference.query import QueryProcessor
from retrieval.embedding import load_siamese_embedding_model


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
    """Load the model and gallery with explicit S4.6 startup failure types."""
    try:
        embedder = load_siamese_embedding_model(args.checkpoint, device=args.device)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ModelUnavailableError(
            "The search model could not be loaded. Check the checkpoint path and compatibility."
        ) from exc

    try:
        gallery = GalleryLoader(
            GalleryRuntimeConfig(
                model_name=args.model,
                root=args.gallery_root,
                expected_embedding_dim=128,
            )
        ).load()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise GalleryUnavailableError(
            "The search gallery could not be loaded. Build or restore the configured gallery."
        ) from exc

    pipeline = InferencePipeline(embedder, gallery.search_engine)
    return SearchService(QueryProcessor(), pipeline)


def main() -> None:
    args = parse_args()
    try:
        service = build_service(args)
    except ModelUnavailableError as exc:
        raise SystemExit(f"MODEL_UNAVAILABLE: {exc.message}") from exc
    except GalleryUnavailableError as exc:
        raise SystemExit(f"GALLERY_UNAVAILABLE: {exc.message}") from exc

    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
