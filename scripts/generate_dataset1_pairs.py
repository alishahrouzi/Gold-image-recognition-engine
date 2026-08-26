#!/usr/bin/env python3
"""Generate the Dataset 1 pair dataset (S1.10).

Reads reports/dataset/dataset1_manifest.csv. Does not modify the manifest
or image files. Writes a separate pair CSV and JSON audit report.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.loaders.manifest import load_manifest  # noqa: E402
from data.pairs import (  # noqa: E402
    DEFAULT_PAIR_SEED,
    DEFAULT_POSITIVE_NEGATIVE_RATIO,
    DEFAULT_SAME_CATEGORY_NEGATIVE_RATIO,
    PairGenerationConfig,
    generate_pair_dataset,
    write_pair_generation_report,
    write_pairs_csv,
)

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_PAIRS = PROJECT_ROOT / "reports" / "dataset" / "dataset1_pairs.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "dataset" / "dataset1_pair_generation_report.json"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Dataset 1 pair dataset (S1.10).")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_PAIRS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--seed", type=int, default=DEFAULT_PAIR_SEED)
    parser.add_argument(
        "--positive-negative-ratio",
        type=float,
        default=DEFAULT_POSITIVE_NEGATIVE_RATIO,
    )
    parser.add_argument(
        "--same-category-negative-ratio",
        type=float,
        default=DEFAULT_SAME_CATEGORY_NEGATIVE_RATIO,
    )
    parser.add_argument("--max-positive-pairs", type=int, default=None)
    parser.add_argument("--max-negative-pairs", type=int, default=None)
    parser.add_argument("--exclude-train", action="store_true")
    parser.add_argument("--exclude-valid", action="store_true")
    parser.add_argument("--exclude-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().resolve()

    samples = load_manifest(manifest_path, validate_files=False)
    config = PairGenerationConfig(
        seed=args.seed,
        positive_negative_ratio=args.positive_negative_ratio,
        same_category_negative_ratio=args.same_category_negative_ratio,
        max_positive_pairs=args.max_positive_pairs,
        max_negative_pairs=args.max_negative_pairs,
        include_train=not args.exclude_train,
        include_valid=not args.exclude_valid,
        include_test=not args.exclude_test,
    )
    result = generate_pair_dataset(
        samples,
        config,
        dataset="dataset1",
        manifest=manifest_path,
    )
    write_pairs_csv(result.pairs, args.output)
    write_pair_generation_report(result.report, args.report)

    report = result.report
    print("=" * 60)
    print("Dataset 1 pair generation (S1.10)")
    print("=" * 60)
    print(f"manifest: {manifest_path}")
    print(f"seed: {report['seed']}")
    print(f"available_positive_pairs: {report['available_positive_pairs']}")
    print(f"selected_positive_pairs: {report['selected_positive_pairs']}")
    print(f"selected_negative_pairs: {report['selected_negative_pairs']}")
    print(f"same_category_negative_count: {report['same_category_negative_count']}")
    print(f"cross_category_negative_count: {report['cross_category_negative_count']}")
    print(f"total_pairs: {report['total_pairs']}")
    print(f"validation_passed: {report['validation_results']['passed']}")
    print(f"pairs_csv: {args.output}")
    print(f"report_json: {args.report}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
