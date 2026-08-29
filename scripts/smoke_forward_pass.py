#!/usr/bin/env python3
"""S2.4 forward-pass smoke: Dataset 1 → preprocess → encoder → embedding.

Validates shapes, finite values, and L2 unit embeddings for D=128 and D=256.
Not a training run and not a retrieval-quality evaluation.
Does not modify Dataset 1 files or the manifest.
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

from models.forward_pass import (  # noqa: E402
    DEFAULT_REPORT_DIR,
    REQUIRED_BATCH_SIZES,
    run_validation_suite,
    write_forward_pass_reports,
)

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S2.4 forward-pass validation smoke.")
    parser.add_argument(
        "--skip-dataset1",
        action="store_true",
        help="Skip Dataset 1 DataLoader integration even if the root is present.",
    )
    parser.add_argument(
        "--skip-optional-batch64",
        action="store_true",
        help="Do not attempt optional batch_size=64.",
    )
    parser.add_argument(
        "--no-write-report",
        action="store_true",
        help="Skip writing reports/benchmark/forward_pass/.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    payload = run_validation_suite(
        batch_sizes=REQUIRED_BATCH_SIZES,
        include_dataset1=not args.skip_dataset1,
        include_optional_batch64=not args.skip_optional_batch64,
        include_edge_cases=True,
    )
    hw = payload["hardware"]
    logger.info(
        "device=%s cuda_available=%s gpu=%s status=%s runs=%s failures=%s",
        hw["detected_device"],
        hw["cuda_available"],
        hw["gpu_name"],
        payload["status"],
        payload["run_count"],
        payload["failure_count"],
    )
    for warning in payload["warnings"]:
        logger.warning("%s", warning)
    for row in payload["runs"]:
        mark = "ok" if row["passed"] else "FAIL"
        emb = row["embedding"]
        logger.info(
            "%s device=%s split=%s source=%s batch=%s D=%s images=%s features=%s "
            "embeddings=%s l2_min=%.8f l2_max=%.8f max_err=%.2e",
            mark,
            row["device"],
            row["split"],
            row["source"],
            row["batch_size"],
            row["embedding_dim"],
            row["input"]["shape"],
            row["encoder_features"]["shape"],
            emb["shape"],
            emb["l2_norm_min"],
            emb["l2_norm_max"],
            emb["max_l2_error"],
        )
        if row["failures"]:
            logger.error("failures=%s", row["failures"])
    if not args.no_write_report:
        json_path, md_path = write_forward_pass_reports(payload, DEFAULT_REPORT_DIR)
        logger.info("wrote %s", json_path)
        logger.info("wrote %s", md_path)
    logger.info("final_status=%s", payload["status"])
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
