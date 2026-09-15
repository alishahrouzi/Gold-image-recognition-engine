"""Run real S4.8 end-to-end scenarios against the running MVP API."""

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
from PIL import Image, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
DEFAULT_GALLERY_METADATA = PROJECT_ROOT / "experiments/retrieval/siamese/s3.5_random/gallery_metadata.json"
DEFAULT_DATASET1 = DATASET_ROOT / "ai-tool-pool-jewelry-vision"
DEFAULT_DATASET2 = DATASET_ROOT / "jewelry-design-dataset"


def _images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _find_category_image(root: Path, category: str) -> Path:
    candidates = _images(root / "test" / category)
    if not candidates:
        candidates = [path for path in _images(root) if category.lower() in {part.lower() for part in path.parts}]
    if not candidates:
        raise FileNotFoundError(
            f"Could not discover a Dataset 1 test image for category '{category}'. "
            f"Pass --{category.lower()}-image explicitly."
        )
    return candidates[0]


def _gallery_paths(metadata_path: Path) -> set[str]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Gallery metadata must contain a list: {metadata_path}")
    paths: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw = item.get("image", item.get("image_path"))
        if raw:
            paths.add(str(Path(str(raw)).expanduser().resolve()).lower())
    return paths


def _find_out_of_gallery_image(root: Path, gallery_paths: set[str]) -> Path:
    for candidate in _images(root):
        if str(candidate.resolve()).lower() not in gallery_paths:
            return candidate
    raise FileNotFoundError(
        "Could not find an image outside the configured runtime gallery. "
        "Pass --out-of-gallery-image explicitly."
    )


def _write_transform(source: Path, destination: Path, kind: str) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        if kind == "angle":
            transformed = image.rotate(12.0, resample=Image.Resampling.BICUBIC, expand=True)
            transformed.save(destination, format="JPEG", quality=95)
        elif kind == "light":
            transformed = ImageEnhance.Brightness(image).enhance(0.55)
            transformed = ImageEnhance.Contrast(transformed).enhance(1.15)
            transformed.save(destination, format="JPEG", quality=95)
        else:
            raise ValueError(f"Unknown transform kind: {kind}")


def _search(client: httpx.Client, base_url: str, image_path: Path, k: int) -> dict[str, Any]:
    with image_path.open("rb") as handle:
        response = client.post(
            f"{base_url}/search",
            params={"k": k},
            files={"file": (image_path.name, handle, "image/jpeg")},
        )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise AssertionError(f"Invalid search response for {image_path}: {body!r}")
    results = body["results"]
    if not 1 <= len(results) <= k:
        raise AssertionError(f"Unexpected result count for {image_path}: {len(results)}")
    product_ids = [str(item.get("product_id", "")) for item in results]
    if any(not product_id for product_id in product_ids):
        raise AssertionError(f"Search returned an empty product_id: {body!r}")
    if len(product_ids) != len(set(product_ids)):
        raise AssertionError(f"Search returned duplicate products: {body!r}")
    for item in results:
        similarity = float(item["similarity"])
        if not 0.0 <= similarity <= 100.0:
            raise AssertionError(f"Similarity outside [0,100]: {similarity}")

    top_product = quote(product_ids[0], safe="")
    image_response = client.get(f"{base_url}/result-image/{top_product}")
    if image_response.status_code != 200:
        raise AssertionError(
            f"Top result image could not be served: status={image_response.status_code}, "
            f"product_id={product_ids[0]}"
        )
    return body


def _run_scenario(
    client: httpx.Client,
    base_url: str,
    name: str,
    image_path: Path,
    k: int,
    expected_category: str | None = None,
    out_of_gallery: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        body = _search(client, base_url, image_path, k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        category = str(body["category"])
        if expected_category is not None and category != expected_category:
            return {
                "name": name,
                "status": "failed",
                "query_image": str(image_path),
                "expected_category": expected_category,
                "category": category,
                "results": body["results"],
                "latency_ms": round(elapsed_ms, 2),
                "out_of_gallery_query": out_of_gallery,
                "error": f"expected category '{expected_category}', got '{category}'.",
            }
        return {
            "name": name,
            "status": "passed",
            "query_image": str(image_path),
            "expected_category": expected_category,
            "category": category,
            "results": body["results"],
            "latency_ms": round(elapsed_ms, 2),
            "out_of_gallery_query": out_of_gallery,
        }
    except (AssertionError, OSError, httpx.HTTPError, ValueError) as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "name": name,
            "status": "failed",
            "query_image": str(image_path),
            "expected_category": expected_category,
            "latency_ms": round(elapsed_ms, 2),
            "out_of_gallery_query": out_of_gallery,
            "error": str(exc),
        }


def _health(client: httpx.Client, base_url: str) -> None:
    response = client.get(f"{base_url}/health")
    response.raise_for_status()
    if response.json() != {"status": "ok"}:
        raise AssertionError(f"Unexpected health response: {response.text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S4.8 real-scenario end-to-end tests.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dataset1", type=Path, default=DEFAULT_DATASET1)
    parser.add_argument("--dataset2", type=Path, default=DEFAULT_DATASET2)
    parser.add_argument("--gallery-metadata", type=Path, default=DEFAULT_GALLERY_METADATA)
    parser.add_argument("--ring-image", type=Path)
    parser.add_argument("--necklace-image", type=Path)
    parser.add_argument("--angle-image", type=Path)
    parser.add_argument("--light-image", type=Path)
    parser.add_argument("--out-of-gallery-image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiments/e2e/s4.8_e2e_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.k < 1 or args.k > 50:
        raise SystemExit("--k must be between 1 and 50.")

    base_url = args.base_url.rstrip("/")
    ring = args.ring_image or _find_category_image(args.dataset1, "Ring")
    necklace = args.necklace_image or _find_category_image(args.dataset1, "Necklace")
    gallery_paths = _gallery_paths(args.gallery_metadata)
    out_of_gallery = args.out_of_gallery_image or _find_out_of_gallery_image(args.dataset2, gallery_paths)

    for path in (ring, necklace, out_of_gallery):
        if not path.is_file():
            raise FileNotFoundError(f"Scenario image does not exist: {path}")
    if str(out_of_gallery.resolve()).lower() in gallery_paths:
        raise AssertionError(f"Test 5 image is present in the runtime gallery: {out_of_gallery}")

    report: dict[str, Any] = {
        "experiment": "EXP-S4.8",
        "policy": "s4.8-real-scenarios-v1",
        "base_url": base_url,
        "k": args.k,
        "status": "failed",
        "scenario_count": 5,
        "passed_count": 0,
        "failed_count": 0,
        "scenarios": [],
        "notes": [
            "Tests 1 and 2 use Dataset 1 test images, which are outside the train runtime gallery.",
            "Tests 3 and 4 use explicit real images when supplied; otherwise controlled angle/lighting transforms are derived from the ring test image.",
            "Test 5 verifies graceful retrieval for a query image absent from the runtime gallery. The MVP has no OOD detector, so nearest gallery products are expected rather than NO_RESULTS.",
            "All scenarios are executed independently. A failed scenario is recorded and does not prevent later scenarios from running.",
        ],
    }

    with tempfile.TemporaryDirectory(prefix="gold-s4.8-") as temp_dir, httpx.Client(timeout=30.0) as client:
        try:
            _health(client, base_url)
        except (AssertionError, httpx.HTTPError, OSError) as exc:
            report["setup_error"] = str(exc)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2, ensure_ascii=False))
            print(f"S4.8 E2E: FAIL — API health check failed; report written to {args.output}")
            return 1

        temp_root = Path(temp_dir)
        angle = args.angle_image
        light = args.light_image
        if angle is None:
            angle = temp_root / "ring_angle.jpg"
            _write_transform(ring, angle, "angle")
        if light is None:
            light = temp_root / "ring_light.jpg"
            _write_transform(ring, light, "light")

        scenarios = [
            ("Test 1 — ring to similar rings", ring, "ring", False),
            ("Test 2 — necklace to similar necklaces", necklace, "necklace", False),
            ("Test 3 — different angle", angle, "ring", False),
            ("Test 4 — different lighting", light, "ring", False),
            ("Test 5 — product absent from gallery", out_of_gallery, None, True),
        ]
        for name, image_path, expected_category, is_ood in scenarios:
            result = _run_scenario(
                client,
                base_url,
                name,
                image_path,
                args.k,
                expected_category=expected_category,
                out_of_gallery=is_ood,
            )
            report["scenarios"].append(result)
            print(f"{name}: {result['status'].upper()}")
            if result["status"] == "failed":
                print(f"  error: {result['error']}")

    report["passed_count"] = sum(1 for item in report["scenarios"] if item["status"] == "passed")
    report["failed_count"] = len(report["scenarios"]) - report["passed_count"]
    report["status"] = "passed" if report["failed_count"] == 0 else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"S4.8 E2E: {'PASS' if report['status'] == 'passed' else 'FAIL'} — "
        f"{report['passed_count']}/{report['scenario_count']} scenarios passed; "
        f"report written to {args.output}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, OSError, ValueError, httpx.HTTPError) as exc:
        print(f"S4.8 E2E: FAIL — {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
