#!/usr/bin/env python3
"""Visualize Dataset 1 samples and pairs for data QA (S1.11).

Reads the existing manifest and pair CSV. Does not modify the dataset,
manifest, pair CSV, or source images. Does not regenerate pairs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.visualization import (  # noqa: E402
    DEFAULT_AUGMENTATION_SAMPLES,
    DEFAULT_NEGATIVE_PAIRS,
    DEFAULT_POSITIVE_PAIRS,
    DEFAULT_TEST_SAMPLES,
    DEFAULT_TRAIN_SAMPLES,
    DEFAULT_VALID_SAMPLES,
    DEFAULT_VISUALIZATION_SEED,
    VisualizationConfig,
    generate_dataset_visualizations,
    load_visualization_inputs,
)

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_PAIRS = PROJECT_ROOT / "reports" / "dataset" / "dataset1_pairs.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "visualization" / "dataset1"
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "ZARGAR_DATASET1_ROOT",
        r"e:\Privat File\Projects\Zargar Interview\dataset\ai-tool-pool-jewelry-vision",
    )
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Dataset 1 for data QA (S1.11).")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=DEFAULT_VISUALIZATION_SEED)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--valid-samples", type=int, default=DEFAULT_VALID_SAMPLES)
    parser.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--positive-pairs", type=int, default=DEFAULT_POSITIVE_PAIRS)
    parser.add_argument("--negative-pairs", type=int, default=DEFAULT_NEGATIVE_PAIRS)
    parser.add_argument(
        "--augmentation-samples",
        type=int,
        default=DEFAULT_AUGMENTATION_SAMPLES,
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().resolve()
    pairs_path = Path(args.pairs).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    config = VisualizationConfig(
        seed=args.seed,
        train_samples=args.train_samples,
        valid_samples=args.valid_samples,
        test_samples=args.test_samples,
        positive_pairs=args.positive_pairs,
        negative_pairs=args.negative_pairs,
        augmentation_samples=args.augmentation_samples,
    )
    samples, pairs = load_visualization_inputs(
        manifest_path,
        pairs_path,
        dataset_root=dataset_root,
        validate_files=True,
    )
    result = generate_dataset_visualizations(
        samples=samples,
        pairs=pairs,
        output_dir=output_dir,
        config=config,
        dataset="dataset1",
        manifest=manifest_path,
        pair_source=pairs_path,
    )
    report = result.report
    print("=" * 60)
    print("Dataset 1 visualization (S1.11)")
    print("=" * 60)
    print(f"manifest: {manifest_path}")
    print(f"pair_source: {pairs_path}")
    print(f"seed: {report['seed']}")
    print(
        f"train groups: requested={report['train']['requested']} "
        f"selected={report['train']['selected']} "
        f"(images_shown={report['train']['images_shown']}, "
        f"multi_image_groups={report['train']['multi_image_groups']})"
    )
    print(
        f"valid images: requested={report['valid']['requested']} "
        f"selected={report['valid']['selected']}"
    )
    print(
        f"test images: requested={report['test']['requested']} "
        f"selected={report['test']['selected']}"
    )
    print(
        f"positive pairs: requested={report['positive_pairs']['requested']} "
        f"selected={report['positive_pairs']['selected']} "
        f"validated={report['positive_pairs']['validated']}"
    )
    print(
        f"negative pairs: requested={report['negative_pairs']['requested']} "
        f"selected={report['negative_pairs']['selected']} "
        f"validated={report['negative_pairs']['validated']} "
        f"(same_category={report['negative_pairs']['same_category_selected']}, "
        f"cross_category={report['negative_pairs']['cross_category_selected']})"
    )
    print(
        f"augmentation: requested={report['augmentation']['requested']} "
        f"selected={report['augmentation']['selected']} "
        f"(train only)"
    )
    print(f"validation_errors: {len(report['validation_errors'])}")
    print(f"source_files_unchanged: {report['source_files_unchanged']}")
    print(f"output_dir: {output_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
