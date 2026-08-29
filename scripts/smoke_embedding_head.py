#!/usr/bin/env python3
"""S2.3 EmbeddingHead smoke + 128 vs 256 engineering comparison.

Not a retrieval-quality evaluation. No training. Does not modify Dataset 1
files or the manifest. Model/head timing excludes DataLoader time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models import (  # noqa: E402
    CustomCNNEncoder,
    EmbeddingHead,
    EncoderWithEmbeddingHead,
    count_parameters,
    estimate_parameter_bytes,
)

logger = logging.getLogger(__name__)

FORWARD_BATCH_SIZES: Sequence[int] = (1, 8, 16, 32)
EMBEDDING_DIMS: Sequence[int] = (128, 256)
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "benchmark" / "embedding_head"
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
    parser = argparse.ArgumentParser(description="S2.3 EmbeddingHead smoke and 128 vs 256 engineering comparison.")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--skip-dataset1", action="store_true")
    parser.add_argument("--write-report", action="store_true", help="Write JSON/MD under reports/benchmark/embedding_head/.")
    return parser.parse_args(argv)


def _device_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "vram_total_mib": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_total_mib"] = props.total_memory / (1024 * 1024)
    return info


def _make_images(encoder: CustomCNNEncoder, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randn(
        batch_size,
        encoder.config.input_channels,
        encoder.config.input_height,
        encoder.config.input_width,
        dtype=torch.float32,
        device=device,
    )


def _l2_ok(embeddings: torch.Tensor, atol: float = 1e-5) -> bool:
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1)
    return bool(torch.allclose(norms, torch.ones_like(norms), atol=atol))


def _time_cpu(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - started) / max(iters, 1)


def _time_cuda(fn, warmup: int, iters: int, device: torch.device) -> tuple[float, int, int]:
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - started) / max(iters, 1)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    return elapsed, allocated, reserved


def _dataset1_report(encoder: CustomCNNEncoder) -> List[Dict[str, Any]]:
    from torch.utils.data import DataLoader, Subset

    from data.collate import collate_preprocessed_samples
    from data.datasets.unified_dataset import UnifiedDataset
    from data.preprocessing import ImagePreprocessor, PreprocessedDataset

    dataset_root = _dataset_root()
    if not DEFAULT_MANIFEST.is_file() or not (dataset_root / "train").is_dir():
        logger.info("dataset1: skipped (root or manifest unavailable)")
        return []

    base = UnifiedDataset(
        DEFAULT_MANIFEST,
        dataset_root=dataset_root,
        split="valid",
        validate_files=False,
    )
    rows = []
    for batch_size in (1, 8, 32):
        if len(base) < batch_size:
            continue
        dataset = PreprocessedDataset(
            Subset(base, list(range(batch_size))),
            ImagePreprocessor(),
            role="valid",
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_preprocessed_samples,
        )
        images = next(iter(loader))["image"]
        encoder.eval()
        with torch.no_grad():
            features = encoder.encode_features(images)
        for dim in EMBEDDING_DIMS:
            head = EmbeddingHead(embedding_dim=dim)
            head.eval()
            with torch.no_grad():
                embeddings = head(features)
            row = {
                "batch_size": batch_size,
                "embedding_dim": dim,
                "image_shape": list(images.shape),
                "feature_shape": list(features.shape),
                "embedding_shape": list(embeddings.shape),
                "finite": bool(torch.isfinite(embeddings).all().item()),
                "l2_unit": _l2_ok(embeddings),
            }
            rows.append(row)
            logger.info(
                "dataset1 batch=%s dim=%s images=%s features=%s embeddings=%s finite=%s l2_unit=%s",
                batch_size,
                dim,
                tuple(images.shape),
                tuple(features.shape),
                tuple(embeddings.shape),
                row["finite"],
                row["l2_unit"],
            )
    return rows


def _write_reports(payload: Dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "s2.3_embedding_head_benchmark.json"
    md_path = report_dir / "s2.3_embedding_head_benchmark.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    hw = payload["hardware"]
    lines = [
        "# S2.3 Embedding Head engineering comparison (128 vs 256)",
        "",
        "Untrained model. **No retrieval metrics.** Timing is encoder/head forward only (synthetic tensors unless labeled Dataset 1).",
        "",
        "## Hardware",
        "",
        f"- Python: {hw['python']}",
        f"- PyTorch: {hw['pytorch']}",
        f"- Platform: {hw['platform']}",
        f"- CUDA available: {hw['cuda_available']}",
        f"- GPU: {hw['gpu_name']}",
        f"- VRAM total (MiB): {hw['vram_total_mib']}",
        "",
        "GPU utilization is not reported (not measured).",
        "",
        "## Parameter counts",
        "",
        "| Module | embedding_dim | params | bytes |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload["parameters"]:
        lines.append(
            f"| {row['module']} | {row.get('embedding_dim', '—')} | {row['params']} | {row['bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Embedding storage (float32)",
            "",
            "| embedding_dim | bytes / sample | bytes / batch 32 |",
            "| ---: | ---: | ---: |",
            "| 128 | 512 | 16384 |",
            "| 256 | 1024 | 32768 |",
            "",
            "## CPU forward (ms / iter, after warmup)",
            "",
            "| batch | dim | encoder_ms | head_ms | composed_ms | feature_bytes | embedding_bytes |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["cpu"]:
        lines.append(
            f"| {row['batch_size']} | {row['embedding_dim']} | {row['encoder_ms']:.3f} | "
            f"{row['head_ms']:.3f} | {row['composed_ms']:.3f} | {row['feature_bytes']} | "
            f"{row['embedding_bytes']} |"
        )
    lines.extend(["", "## CUDA forward (if available)", "", "| batch | dim | encoder_ms | head_ms | composed_ms | peak_alloc_MiB | peak_reserved_MiB |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    if payload["cuda"]:
        for row in payload["cuda"]:
            lines.append(
                f"| {row['batch_size']} | {row['embedding_dim']} | {row['encoder_ms']:.3f} | "
                f"{row['head_ms']:.3f} | {row['composed_ms']:.3f} | {row['peak_allocated_mib']:.2f} | "
                f"{row['peak_reserved_mib']:.2f} |"
            )
    else:
        lines.append("| — | — | skipped | skipped | skipped | — | — |")
    lines.extend(["", "## Dataset 1 integration", ""])
    if payload["dataset1"]:
        for row in payload["dataset1"]:
            lines.append(
                f"- batch={row['batch_size']} dim={row['embedding_dim']} "
                f"images={row['image_shape']} features={row['feature_shape']} "
                f"embeddings={row['embedding_shape']} finite={row['finite']} l2_unit={row['l2_unit']}"
            )
    else:
        lines.append("Skipped (Dataset 1 unavailable).")
    lines.extend(
        [
            "",
            "Dataset 1 manifest and source images were not modified.",
            "No 128 vs 256 winner is declared; retrieval evaluation is deferred until the model is trained.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", json_path)
    logger.info("wrote %s", md_path)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    hardware = _device_info()
    logger.info("hardware=%s", hardware)

    encoder = CustomCNNEncoder()
    encoder.eval()
    encoder_params = count_parameters(encoder, trainable_only=False)
    encoder_bytes = estimate_parameter_bytes(encoder)
    param_rows = [
        {
            "module": "CustomCNNEncoder",
            "embedding_dim": None,
            "params": encoder_params,
            "bytes": encoder_bytes,
        }
    ]
    for dim in EMBEDDING_DIMS:
        head = EmbeddingHead(embedding_dim=dim)
        param_rows.append(
            {
                "module": "EmbeddingHead",
                "embedding_dim": dim,
                "params": count_parameters(head, trainable_only=False),
                "bytes": estimate_parameter_bytes(head),
            }
        )
        composed = EncoderWithEmbeddingHead(CustomCNNEncoder(), EmbeddingHead(embedding_dim=dim))
        param_rows.append(
            {
                "module": "EncoderWithEmbeddingHead",
                "embedding_dim": dim,
                "params": count_parameters(composed, trainable_only=False),
                "bytes": estimate_parameter_bytes(composed),
            }
        )
        storage = dim * 4
        logger.info(
            "embedding_dim=%s head_params=%s head_bytes=%s embedding_storage_per_sample_bytes=%s",
            dim,
            count_parameters(head, trainable_only=False),
            estimate_parameter_bytes(head),
            storage,
        )

    cpu_rows: List[Dict[str, Any]] = []
    cuda_rows: List[Dict[str, Any]] = []
    cpu = torch.device("cpu")

    for batch_size in FORWARD_BATCH_SIZES:
        images = _make_images(encoder, batch_size, cpu)
        with torch.no_grad():
            shared_features = encoder.encode_features(images)
        for dim in EMBEDDING_DIMS:
            head = EmbeddingHead(embedding_dim=dim)
            head.eval()

            def encoder_fn() -> None:
                with torch.no_grad():
                    encoder.encode_features(images)

            def head_fn() -> None:
                with torch.no_grad():
                    head(shared_features)

            def composed_fn() -> None:
                with torch.no_grad():
                    head(encoder.encode_features(images))

            encoder_s = _time_cpu(encoder_fn, args.warmup, args.iters)
            head_s = _time_cpu(head_fn, args.warmup, args.iters)
            composed_s = _time_cpu(composed_fn, args.warmup, args.iters)
            with torch.no_grad():
                embeddings = head(shared_features)
            if embeddings.shape != (batch_size, dim) or not torch.isfinite(embeddings).all() or not _l2_ok(embeddings):
                raise RuntimeError(f"CPU check failed batch={batch_size} dim={dim}")
            row = {
                "batch_size": batch_size,
                "embedding_dim": dim,
                "encoder_ms": encoder_s * 1000,
                "head_ms": head_s * 1000,
                "composed_ms": composed_s * 1000,
                "feature_bytes": batch_size * 256 * 4,
                "embedding_bytes": batch_size * dim * 4,
            }
            cpu_rows.append(row)
            logger.info(
                "cpu batch=%s dim=%s encoder=%.3f ms head=%.3f ms composed=%.3f ms",
                batch_size,
                dim,
                row["encoder_ms"],
                row["head_ms"],
                row["composed_ms"],
            )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("cuda device=%s vram_mib=%.1f", hardware["gpu_name"], hardware["vram_total_mib"] or 0)
        for batch_size in FORWARD_BATCH_SIZES:
            try:
                enc_gpu = CustomCNNEncoder().to(device)
                enc_gpu.eval()
                images = _make_images(enc_gpu, batch_size, device)
                with torch.no_grad():
                    shared_features = enc_gpu.encode_features(images)
                for dim in EMBEDDING_DIMS:
                    head = EmbeddingHead(embedding_dim=dim).to(device)
                    head.eval()

                    def encoder_fn() -> None:
                        with torch.no_grad():
                            enc_gpu.encode_features(images)

                    def head_fn() -> None:
                        with torch.no_grad():
                            head(shared_features)

                    def composed_fn() -> None:
                        with torch.no_grad():
                            head(enc_gpu.encode_features(images))

                    encoder_s, _, _ = _time_cuda(encoder_fn, args.warmup, args.iters, device)
                    head_s, _, _ = _time_cuda(head_fn, args.warmup, args.iters, device)
                    composed_s, allocated, reserved = _time_cuda(composed_fn, args.warmup, args.iters, device)
                    with torch.no_grad():
                        embeddings = head(enc_gpu.encode_features(images))
                    if embeddings.shape != (batch_size, dim) or not torch.isfinite(embeddings).all() or not _l2_ok(embeddings):
                        raise RuntimeError(f"CUDA check failed batch={batch_size} dim={dim}")
                    row = {
                        "batch_size": batch_size,
                        "embedding_dim": dim,
                        "encoder_ms": encoder_s * 1000,
                        "head_ms": head_s * 1000,
                        "composed_ms": composed_s * 1000,
                        "peak_allocated_mib": allocated / (1024 * 1024),
                        "peak_reserved_mib": reserved / (1024 * 1024),
                    }
                    cuda_rows.append(row)
                    logger.info(
                        "cuda batch=%s dim=%s encoder=%.3f ms head=%.3f ms composed=%.3f ms "
                        "peak_alloc=%.2f MiB peak_reserved=%.2f MiB",
                        batch_size,
                        dim,
                        row["encoder_ms"],
                        row["head_ms"],
                        row["composed_ms"],
                        row["peak_allocated_mib"],
                        row["peak_reserved_mib"],
                    )
            except torch.cuda.OutOfMemoryError:
                logger.warning("cuda batch=%s: OOM (forward-only; not a training batch-size decision)", batch_size)
                torch.cuda.empty_cache()
    else:
        logger.info("cuda: skipped (unavailable)")

    dataset1_rows: List[Dict[str, Any]] = []
    if not args.skip_dataset1:
        dataset1_rows = _dataset1_report(CustomCNNEncoder())

    payload = {
        "hardware": hardware,
        "warmup": args.warmup,
        "iters": args.iters,
        "parameters": param_rows,
        "cpu": cpu_rows,
        "cuda": cuda_rows,
        "dataset1": dataset1_rows,
        "notes": [
            "No training, loss, similarity, or retrieval.",
            "128 vs 256 is an engineering comparison only.",
            "Dataset 1 files and manifest were not modified.",
        ],
    }
    if args.write_report:
        _write_reports(payload, DEFAULT_REPORT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
