#!/usr/bin/env python3
"""S2.2 Custom CNN v1 smoke test (parameters, shapes, CPU/CUDA, timing).

Not a training benchmark and not a retrieval-quality evaluation.
Does not modify Dataset 1 files or the manifest.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models import (  # noqa: E402
    ARCHITECTURE_ID,
    CustomCNNEncoder,
    EncoderConfig,
    count_parameters,
    estimate_parameter_bytes,
    format_encoder_summary,
    trace_encoder_shapes,
)

logger = logging.getLogger(__name__)

FORWARD_BATCH_SIZES: Sequence[int] = (1, 8, 16, 32)
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
_LEGACY_DATASET_ROOT = Path(
    r"e:\Privat File\Projects\Zargar Interview\dataset\ai-tool-pool-jewelry-vision"
)
_LOCAL_DATASET_ROOT = PROJECT_ROOT.parent / "dataset" / "ai-tool-pool-jewelry-vision"


def _dataset_root() -> Path:
    env = os.environ.get("ZARGAR_DATASET1_ROOT")
    if env:
        return Path(env)
    if _LEGACY_DATASET_ROOT.is_dir():
        return _LEGACY_DATASET_ROOT
    return _LOCAL_DATASET_ROOT


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S2.2 Custom CNN v1 encoder smoke test.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--skip-dataset1",
        action="store_true",
        help="Skip Dataset 1 DataLoader integration even if the root is present.",
    )
    return parser.parse_args(argv)


def _make_images(encoder: CustomCNNEncoder, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randn(
        batch_size,
        encoder.config.input_channels,
        encoder.config.input_height,
        encoder.config.input_width,
        dtype=torch.float32,
        device=device,
    )


def _forward_once(encoder: CustomCNNEncoder, images: torch.Tensor) -> torch.Tensor:
    encoder.eval()
    with torch.no_grad():
        features = encoder(images)
    if features.shape != (images.shape[0], encoder.feature_dim):
        raise RuntimeError(f"Unexpected feature shape {tuple(features.shape)}.")
    if features.dtype != torch.float32 and images.dtype == torch.float32:
        raise RuntimeError(f"Unexpected feature dtype {features.dtype}.")
    if torch.isnan(features).any() or torch.isinf(features).any():
        raise RuntimeError("Non-finite features.")
    return features


def _dataset1_forward_report(encoder: CustomCNNEncoder) -> str:
    from torch.utils.data import DataLoader, Subset

    from data.collate import collate_preprocessed_samples
    from data.datasets.unified_dataset import UnifiedDataset
    from data.preprocessing import ImagePreprocessor, PreprocessedDataset

    dataset_root = _dataset_root()
    if not DEFAULT_MANIFEST.is_file() or not (dataset_root / "train").is_dir():
        return "dataset1: skipped (root or manifest unavailable)"

    base = UnifiedDataset(
        DEFAULT_MANIFEST,
        dataset_root=dataset_root,
        split="valid",
        validate_files=False,
    )
    dataset = PreprocessedDataset(
        Subset(base, list(range(min(4, len(base))))),
        ImagePreprocessor(),
        role="valid",
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_preprocessed_samples,
    )
    images = next(iter(loader))["image"]
    features = _forward_once(encoder.cpu(), images)
    return (
        f"dataset1: ok images={tuple(images.shape)} features={tuple(features.shape)} "
        f"dtype={features.dtype} finite=true"
    )


def _timed_cpu_forward(encoder: CustomCNNEncoder, batch_size: int) -> float:
    images = _make_images(encoder, batch_size, torch.device("cpu"))
    _forward_once(encoder, images)
    started = time.perf_counter()
    _forward_once(encoder, images)
    return time.perf_counter() - started


def _cuda_forward_report(encoder: CustomCNNEncoder, batch_size: int) -> str:
    device = torch.device("cuda")
    encoder = encoder.to(device)
    images = _make_images(encoder, batch_size, device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    _forward_once(encoder, images)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    _forward_once(encoder, images)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    return (
        f"cuda batch={batch_size}: ok time={elapsed * 1000:.2f} ms "
        f"peak_allocated={allocated / (1024 * 1024):.2f} MiB "
        f"peak_reserved={reserved / (1024 * 1024):.2f} MiB"
    )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    config = EncoderConfig()
    encoder = CustomCNNEncoder(config)
    n_params = count_parameters(encoder, trainable_only=False)
    n_trainable = count_parameters(encoder, trainable_only=True)
    size_bytes = estimate_parameter_bytes(encoder)
    size_mb = size_bytes / (1024 * 1024)

    logger.info("architecture_id=%s", ARCHITECTURE_ID)
    logger.info("architecture=CustomCNNEncoder v1 (S2.2 backbone; S2.3 projection on EmbeddingHead)")
    logger.info("feature_dim=%s", encoder.feature_dim)
    logger.info("input_shape=[B, %s, %s, %s]", *config.input_shape)
    logger.info("block_channels=%s", list(config.block_channels))
    logger.info("convs_per_stage=%s", config.convs_per_stage)
    logger.info("activation=%s normalization=%s downsample=%s", config.activation, config.normalization, config.downsample)
    logger.info("total_parameters=%s", n_params)
    logger.info("trainable_parameters=%s", n_trainable)
    logger.info("parameter_bytes=%s (%.3f MiB)", size_bytes, size_mb)
    logger.info("embeddings_l2_normalized=false")

    logger.info("shape_trace:\n%s", "\n".join(f"  {name}: {list(shape)}" for name, shape in trace_encoder_shapes(encoder)))
    logger.info("summary:\n%s", format_encoder_summary(encoder))

    for batch_size in FORWARD_BATCH_SIZES:
        cpu_time = _timed_cpu_forward(encoder, batch_size)
        logger.info(
            "cpu batch=%s: ok shape=(%s, %s) dtype=torch.float32 finite=true time=%.2f ms",
            batch_size,
            batch_size,
            encoder.feature_dim,
            cpu_time * 1000,
        )

    if not args.skip_dataset1:
        logger.info(_dataset1_forward_report(encoder))

    if torch.cuda.is_available():
        for batch_size in FORWARD_BATCH_SIZES:
            try:
                logger.info(_cuda_forward_report(CustomCNNEncoder(config), batch_size))
            except torch.cuda.OutOfMemoryError:
                logger.warning("cuda batch=%s: OOM (forward-only; not a training batch-size decision)", batch_size)
                torch.cuda.empty_cache()
    else:
        logger.info("cuda: skipped (unavailable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
