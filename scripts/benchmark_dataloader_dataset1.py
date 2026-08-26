#!/usr/bin/env python3
"""Benchmark Dataset 1 DataLoader throughput (S1.12).

Measures the existing data pipeline only. Does not modify images or the
manifest. Does not run an Encoder or a training loop.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.benchmark.dataloader_benchmark import (  # noqa: E402
    DEFAULT_BATCH_SIZES,
    DEFAULT_MEASUREMENT_BATCHES,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    DEFAULT_WARMUP_BATCHES,
    build_practical_configuration_matrix,
    build_smoke_configuration_matrix,
    run_dataloader_benchmark,
    write_dataloader_benchmark_json,
    write_dataloader_benchmark_markdown,
)

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
_LEGACY_DATASET_ROOT = Path(
    r"e:\Privat File\Projects\Zargar Interview\dataset\ai-tool-pool-jewelry-vision"
)
_LOCAL_DATASET_ROOT = (
    PROJECT_ROOT.parent / "dataset" / "ai-tool-pool-jewelry-vision"
)


def _default_dataset_root() -> Path:
    env = os.environ.get("ZARGAR_DATASET1_ROOT")
    if env:
        return Path(env)
    if _LEGACY_DATASET_ROOT.exists():
        return _LEGACY_DATASET_ROOT
    return _LOCAL_DATASET_ROOT


DEFAULT_DATASET_ROOT = _default_dataset_root()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "benchmark" / "dataloader"


def _parse_int_list(raw: str) -> List[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    try:
        parsed = [int(part) for part in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return parsed


def _parse_splits(raw: str) -> List[str]:
    splits = [part.strip() for part in raw.split(",") if part.strip()]
    if not splits:
        raise argparse.ArgumentTypeError("Expected at least one split.")
    return splits


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S1.12 Dataset 1 DataLoader benchmark (data pipeline only)."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--split",
        default="train,valid,test",
        help="Comma-separated splits. Default: train,valid,test.",
    )
    parser.add_argument(
        "--batch-sizes",
        default=",".join(str(size) for size in DEFAULT_BATCH_SIZES),
        help="Comma-separated batch sizes. Default: 8,16,32,64.",
    )
    parser.add_argument(
        "--num-workers",
        default=",".join(str(count) for count in DEFAULT_NUM_WORKERS),
        help="Comma-separated worker counts. Default: 0,2,4.",
    )
    parser.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES)
    parser.add_argument(
        "--measurement-batches",
        type=int,
        default=DEFAULT_MEASUREMENT_BATCHES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or cuda. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Reduced matrix for fast validation.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip dataloader_gpu configurations.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    splits = _parse_splits(args.split)
    batch_sizes = _parse_int_list(args.batch_sizes)
    num_workers = _parse_int_list(args.num_workers)
    include_gpu = not args.cpu_only and args.device.strip().lower() != "cpu"

    if args.smoke:
        configurations = build_smoke_configuration_matrix(include_gpu=include_gpu)
        if splits != ["train", "valid", "test"]:
            configurations = [item for item in configurations if item.split in splits]
    else:
        configurations = build_practical_configuration_matrix(
            splits=splits,
            batch_sizes=batch_sizes,
            num_workers=num_workers,
            include_gpu=include_gpu,
        )

    logger.info("Running %s DataLoader configurations", len(configurations))
    report = run_dataloader_benchmark(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        configurations=configurations,
        warmup_batches=args.warmup_batches,
        measurement_batches=args.measurement_batches,
        seed=args.seed,
        device=args.device if not args.cpu_only else "cpu",
    )

    output_dir = Path(args.output).expanduser().resolve()
    json_path = output_dir / "dataset1_dataloader_benchmark.json"
    md_path = output_dir / "dataset1_dataloader_benchmark.md"
    write_dataloader_benchmark_json(report, json_path)
    write_dataloader_benchmark_markdown(report, md_path)

    rec = report["recommendation"]
    statuses = {}
    for row in report["configurations"]:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1

    print("=" * 60)
    print("S1.12 Dataset 1 DataLoader benchmark")
    print(f"Manifest:     {report['dataset']['manifest']}")
    print(f"Dataset root: {report['dataset']['path']}")
    print(f"Device:       {report['benchmark']['device']}")
    print(f"Mode:         {'smoke' if args.smoke else 'full'}")
    print(f"Configs:      {len(report['configurations'])} ({statuses})")
    print(f"JSON:         {json_path}")
    print(f"Markdown:     {md_path}")
    print(
        "Recommended DataLoader batch: "
        f"{rec.get('recommended_dataloader_batch_size')}"
    )
    print(rec.get("dataloader_safe_batch_size_note"))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
