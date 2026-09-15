"""Run a controlled rotation robustness diagnostic against the MVP API.

This experiment does not change the model or inference pipeline. It compares
transformations that match the training augmentation policy with the original
S4.8 angle transformation so we can distinguish a model weakness from a test
construction issue.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GALLERY_METADATA = PROJECT_ROOT / "experiments/retrieval/siamese/s3.5_random/gallery_metadata.json"
DEFAULT_RING = (
    PROJECT_ROOT.parent
    / "dataset"
    / "ai-tool-pool-jewelry-vision"
    / "test"
    / "Ring"
    / "ring_161_jpg.rf.ad1f0fa2a46757f02686ee9abd024646.jpg"
)

TRANSFORMS: tuple[dict[str, Any], ...] = (
    {"name": "baseline_0deg", "degrees": 0.0, "resample": "bilinear", "expand": False},
    {"name": "train_like_plus5deg", "degrees": 5.0, "resample": "bilinear", "expand": False},
    {"name": "train_like_plus10deg", "degrees": 10.0, "resample": "bilinear", "expand": False},
    {"name": "controlled_plus12deg", "degrees": 12.0, "resample": "bilinear", "expand": False},
    {"name": "controlled_minus12deg", "degrees": -12.0, "resample": "bilinear", "expand": False},
    {"name": "s48_original_plus12deg", "degrees": 12.0, "resample": "bicubic", "expand": True},
)


def _load_gallery_categories(metadata_path: Path) -> dict[str, str]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Gallery metadata must contain a list: {metadata_path}")

    categories: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        product_id = item.get("product_id")
        category = item.get("category")
        if product_id is None or category is None:
            continue
        categories[str(product_id)] = str(category).strip().lower()
    if not categories:
        raise ValueError(f"Gallery metadata contains no product_id/category pairs: {metadata_path}")
    return categories


def _write_rotation(source: Path, destination: Path, degrees: float, resample: str, expand: bool) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        filter_mode = Image.Resampling.BILINEAR if resample == "bilinear" else Image.Resampling.BICUBIC
        if degrees == 0.0:
            transformed = image.copy()
        else:
            transformed = image.rotate(degrees, resample=filter_mode, expand=expand, fillcolor=(255, 255, 255))
        transformed.save(destination, format="JPEG", quality=95)


def _search(client: httpx.Client, base_url: str, image_path: Path, k: int) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    with image_path.open("rb") as handle:
        response = client.post(
            f"{base_url}/search",
            params={"k": k},
            files={"file": (image_path.name, handle, "image/jpeg")},
        )
    response.raise_for_status()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise AssertionError(f"Invalid search response: {body!r}")
    results = body["results"]
    if not 1 <= len(results) <= k:
        raise AssertionError(f"Unexpected result count: {len(results)}")
    return body, elapsed_ms


def _first_category_rank(results: list[dict[str, Any]], categories: dict[str, str], target: str) -> int | None:
    for rank, item in enumerate(results, start=1):
        product_id = str(item.get("product_id", ""))
        if categories.get(product_id) == target:
            return rank
    return None


def _run_transform(
    client: httpx.Client,
    base_url: str,
    image_path: Path,
    transform: dict[str, Any],
    categories: dict[str, str],
    k: int,
) -> dict[str, Any]:
    body, latency_ms = _search(client, base_url, image_path, k)
    results = body["results"]
    category = str(body.get("category", "")).lower()
    ring_rank = _first_category_rank(results, categories, "ring")
    return {
        "name": transform["name"],
        "degrees": transform["degrees"],
        "resample": transform["resample"],
        "expand": transform["expand"],
        "status": "passed" if category == "ring" else "failed",
        "category": category,
        "expected_category": "ring",
        "first_ring_rank_in_top_k": ring_rank,
        "top_result": results[0],
        "results": results,
        "latency_ms": round(latency_ms, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose S4.8 rotation robustness without changing the model.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ring-image", type=Path, default=DEFAULT_RING)
    parser.add_argument("--gallery-metadata", type=Path, default=DEFAULT_GALLERY_METADATA)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("experiments/e2e/s4.8_rotation_diagnostic.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.k <= 50:
        raise SystemExit("--k must be between 1 and 50.")
    if not args.ring_image.is_file():
        raise SystemExit(f"Ring image does not exist: {args.ring_image}")
    if not args.gallery_metadata.is_file():
        raise SystemExit(f"Gallery metadata does not exist: {args.gallery_metadata}")

    categories = _load_gallery_categories(args.gallery_metadata)
    report: dict[str, Any] = {
        "experiment": "EXP-S4.8-ROTATION-DIAGNOSTIC",
        "policy": "s4.8-rotation-diagnostic-v1",
        "base_url": args.base_url.rstrip("/"),
        "k": args.k,
        "source_image": str(args.ring_image),
        "target_category": "ring",
        "transforms": [
            "Training-like rotations use BILINEAR and expand=False, matching the S1.9 training augmentation implementation.",
            "The original S4.8 angle test uses BICUBIC and expand=True and is included explicitly for comparison.",
            "This diagnostic does not retrain or modify the model, encoder, embedding, gallery, search, ranking, or API.",
        ],
        "status": "failed",
        "passed_count": 0,
        "failed_count": 0,
        "results": [],
    }

    with tempfile.TemporaryDirectory(prefix="gold-s4.8-rotation-") as temp_dir, httpx.Client(timeout=30.0) as client:
        try:
            health = client.get(f"{args.base_url.rstrip('/')}/health")
            health.raise_for_status()
            if health.json() != {"status": "ok"}:
                raise AssertionError(f"Unexpected health response: {health.text}")
        except (AssertionError, httpx.HTTPError, OSError) as exc:
            report["setup_error"] = str(exc)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 1

        for index, transform in enumerate(TRANSFORMS):
            image_path = Path(temp_dir) / f"rotation_{index}.jpg"
            _write_rotation(
                args.ring_image,
                image_path,
                transform["degrees"],
                transform["resample"],
                transform["expand"],
            )
            try:
                result = _run_transform(client, args.base_url.rstrip("/"), image_path, transform, categories, args.k)
            except (AssertionError, OSError, httpx.HTTPError, ValueError) as exc:
                result = {
                    "name": transform["name"],
                    "degrees": transform["degrees"],
                    "resample": transform["resample"],
                    "expand": transform["expand"],
                    "status": "failed",
                    "error": str(exc),
                }
            report["results"].append(result)
            print(
                f"{transform['name']}: {result['status'].upper()} — "
                f"category={result.get('category', 'ERROR')}, "
                f"first_ring_rank={result.get('first_ring_rank_in_top_k')}, "
                f"latency_ms={result.get('latency_ms')}"
            )

    report["passed_count"] = sum(1 for item in report["results"] if item["status"] == "passed")
    report["failed_count"] = len(report["results"]) - report["passed_count"]
    report["status"] = "passed" if report["failed_count"] == 0 else "failed"

    # Explicit interpretation hints are based only on the controlled outcomes.
    by_name = {item["name"]: item for item in report["results"]}
    train_like = [by_name[name] for name in ("train_like_plus5deg", "train_like_plus10deg") if name in by_name]
    original = by_name.get("s48_original_plus12deg")
    if train_like and all(item.get("category") == "ring" for item in train_like) and original and original.get("category") != "ring":
        report["diagnostic_interpretation"] = "strong_evidence_test_construction_issue"
    elif train_like and any(item.get("category") != "ring" for item in train_like):
        report["diagnostic_interpretation"] = "evidence_model_rotation_robustness_issue"
    else:
        report["diagnostic_interpretation"] = "inconclusive_requires_further_rotation_analysis"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"S4.8 rotation diagnostic: {'PASS' if report['status'] == 'passed' else 'FAIL'} — "
        f"{report['passed_count']}/{len(report['results'])} variants classified as ring; "
        f"interpretation={report['diagnostic_interpretation']}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, OSError, ValueError, httpx.HTTPError) as exc:
        print(f"S4.8 rotation diagnostic: FAIL — {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
