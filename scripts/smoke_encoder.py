#!/usr/bin/env python3
"""Lightweight encoder sanity check (parameters, size, CPU/CUDA smoke).

Not a training benchmark. Does not modify Dataset 1 files or the manifest.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models import (  # noqa: E402
    CustomCNNEncoder,
    EncoderConfig,
    count_parameters,
    estimate_parameter_bytes,
)

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S2.1 custom CNN encoder smoke test.")
    parser.add_argument("--embedding-dim", type=int, default=EncoderConfig().embedding_dim)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args(argv)


def _forward_smoke(encoder: CustomCNNEncoder, device: torch.device, batch_size: int) -> str:
    images = torch.randn(
        batch_size,
        encoder.config.input_channels,
        encoder.config.input_height,
        encoder.config.input_width,
        dtype=torch.float32,
        device=device,
    )
    encoder = encoder.to(device)
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(images)
    if embeddings.shape != (batch_size, encoder.embedding_dim):
        raise RuntimeError(
            f"Unexpected embedding shape {tuple(embeddings.shape)} on {device}."
        )
    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        raise RuntimeError(f"Non-finite embeddings on {device}.")
    return (
        f"{device.type}: ok shape={tuple(embeddings.shape)} dtype={embeddings.dtype}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    config = EncoderConfig(embedding_dim=args.embedding_dim)
    encoder = CustomCNNEncoder(config)
    n_params = count_parameters(encoder)
    size_bytes = estimate_parameter_bytes(encoder)
    size_mb = size_bytes / (1024 * 1024)

    logger.info("architecture=CustomCNNEncoder")
    logger.info("embedding_dim=%s", encoder.embedding_dim)
    logger.info("input_shape=[B, %s, %s, %s]", *config.input_shape)
    logger.info("block_channels=%s", list(config.block_channels))
    logger.info("trainable_parameters=%s", n_params)
    logger.info("parameter_bytes=%s (%.3f MiB)", size_bytes, size_mb)
    logger.info(_forward_smoke(encoder, torch.device("cpu"), args.batch_size))
    if torch.cuda.is_available():
        logger.info(_forward_smoke(encoder, torch.device("cuda"), args.batch_size))
    else:
        logger.info("cuda: skipped (unavailable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
