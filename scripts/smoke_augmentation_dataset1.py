#!/usr/bin/env python3
"""Smoke-test S1.9 training augmentation + S1.8 preprocessing on Dataset 1.

Uses a representative subset. Does not modify images or the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.collate import collate_preprocessed_samples  # noqa: E402
from data.datasets.unified_dataset import UnifiedDataset  # noqa: E402
from data.loaders.manifest import load_manifest  # noqa: E402
from data.preprocessing import (  # noqa: E402
    AugmentationConfig,
    ImagePreprocessor,
    build_preprocessed_dataset,
)

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "ZARGAR_DATASET1_ROOT",
        r"e:\Privat File\Projects\Zargar Interview\dataset\ai-tool-pool-jewelry-vision",
    )
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S1.9 Dataset 1 augmentation smoke test.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Process all train rows instead of a representative subset.",
    )
    return parser.parse_args(argv)


def representative_train_indices(samples) -> list[int]:
    by_category: dict[str, list[int]] = defaultdict(list)
    group_sizes: dict[str, int] = defaultdict(int)
    for index, sample in enumerate(samples):
        if sample.split != "train":
            continue
        by_category[sample.category].append(index)
        group_sizes[sample.group_id] += 1

    selected = [by_category[key][0] for key in sorted(by_category)]
    for predicate in (
        lambda s: s.split == "train" and group_sizes[s.group_id] >= 2,
        lambda s: s.split == "train" and group_sizes[s.group_id] == 1,
    ):
        for index, sample in enumerate(samples):
            if predicate(sample) and index not in selected:
                selected.append(index)
                break
    return selected


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    manifest = Path(args.manifest).expanduser().resolve()
    root = Path(args.dataset_root).expanduser().resolve()

    samples = load_manifest(manifest, dataset_root=root, validate_files=True)
    if args.full_train:
        indices = [i for i, sample in enumerate(samples) if sample.split == "train"]
    else:
        indices = representative_train_indices(samples)

    checksums = {samples[i].image_path: _digest(samples[i].image_path) for i in indices}

    base = UnifiedDataset(manifest, dataset_root=root, validate_files=False)
    base._samples = [samples[i] for i in indices]  # noqa: SLF001
    dataset = build_preprocessed_dataset(
        base,
        role="train",
        preprocessor=ImagePreprocessor(),
        augmentation=AugmentationConfig(seed=args.seed),
    )
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
        if images.dtype != torch.float32:
            raise RuntimeError(f"Unexpected dtype {images.dtype}")
        if torch.isnan(images).any() or torch.isinf(images).any():
            raise RuntimeError("NaN/Inf detected in augmented+preprocessed batch.")
        if len(batch["group_id"]) != images.shape[0]:
            raise RuntimeError("Metadata length does not match batch size.")
        total += images.shape[0]

    for path, digest in checksums.items():
        if _digest(path) != digest:
            raise RuntimeError(f"Source file was modified: {path}")

    valid_item = next(sample for sample in samples if sample.split == "valid")
    test_item = next(sample for sample in samples if sample.split == "test")
    valid_view = UnifiedDataset(manifest, dataset_root=root, split="valid", validate_files=False)
    test_view = UnifiedDataset(manifest, dataset_root=root, split="test", validate_files=False)
    valid_ds = build_preprocessed_dataset(valid_view, role="valid")
    test_ds = build_preprocessed_dataset(test_view, role="test")
    if valid_ds.augmentor is not None or test_ds.augmentor is not None:
        raise RuntimeError("Valid/test views must not attach a training augmentor.")
    _ = valid_item
    _ = test_item
    _ = valid_ds[0]
    _ = test_ds[0]

    print("=" * 60)
    print("S1.9 Dataset 1 augmentation smoke test")
    print(f"Manifest:       {manifest}")
    print(f"Dataset root:   {root}")
    print(f"Mode:           {'full-train' if args.full_train else 'representative-train'}")
    print(f"Seed:           {args.seed}")
    print(f"Images checked: {total}")
    print("Valid/test:     deterministic (no augmentation)")
    print("Source files:   unchanged")
    print("Result:         PASS")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
