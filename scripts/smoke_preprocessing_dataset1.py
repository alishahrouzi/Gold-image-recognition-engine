#!/usr/bin/env python3
"""Smoke-test S1.8 preprocessing against Dataset 1 (representative + optional full).

Does not modify images or the manifest.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.collate import collate_preprocessed_samples  # noqa: E402
from data.datasets.unified_dataset import UnifiedDataset  # noqa: E402
from data.loaders.manifest import load_manifest  # noqa: E402
from data.preprocessing import ImagePreprocessor, PreprocessedDataset  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "ZARGAR_DATASET1_ROOT",
        r"e:\Privat File\Projects\Zargar Interview\dataset\ai-tool-pool-jewelry-vision",
    )
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S1.8 Dataset 1 preprocessing smoke test.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process all manifest rows instead of a representative subset.",
    )
    return parser.parse_args(argv)


def representative_indices(samples) -> list[int]:
    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    group_sizes: dict[str, int] = defaultdict(int)
    for index, sample in enumerate(samples):
        by_key[(sample.split, sample.category)].append(index)
        group_sizes[sample.group_id] += 1

    selected = [by_key[key][0] for key in sorted(by_key)]
    for predicate in (
        lambda s: group_sizes[s.group_id] >= 2,
        lambda s: group_sizes[s.group_id] == 1,
    ):
        for index, sample in enumerate(samples):
            if predicate(sample) and index not in selected:
                selected.append(index)
                break
    return selected


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    manifest = Path(args.manifest).expanduser().resolve()
    root = Path(args.dataset_root).expanduser().resolve()

    samples = load_manifest(manifest, dataset_root=root, validate_files=True)
    indices = list(range(len(samples))) if args.full else representative_indices(samples)

    base = UnifiedDataset(manifest, dataset_root=root, validate_files=False)
    base._samples = samples  # noqa: SLF001
    dataset = PreprocessedDataset(Subset(base, indices), ImagePreprocessor())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_preprocessed_samples,
        num_workers=0,
    )

    total = 0
    for batch in loader:
        images = batch["image"]
        if images.shape[1:] != (3, 224, 224):
            raise RuntimeError(f"Unexpected batch shape {tuple(images.shape)}")
        if torch.isnan(images).any() or torch.isinf(images).any():
            raise RuntimeError("NaN/Inf detected in preprocessed batch.")
        if len(batch["image_id"]) != images.shape[0]:
            raise RuntimeError("Metadata length does not match batch size.")
        total += images.shape[0]

    print("=" * 60)
    print("S1.8 Dataset 1 preprocessing smoke test")
    print(f"Manifest:       {manifest}")
    print(f"Dataset root:   {root}")
    print(f"Mode:           {'full' if args.full else 'representative'}")
    print(f"Images checked: {total}")
    print("Result:         PASS")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
