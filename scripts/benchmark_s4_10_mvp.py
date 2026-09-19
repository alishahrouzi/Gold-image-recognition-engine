"""S4.10 warm-runtime HTTP performance benchmark for the MVP API.

The benchmark measures the deployed API path, not model-training time. Start
the real MVP API first, then run this script against representative query
images. Results are written as a machine-readable JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import httpx


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def benchmark(url: str, images: list[Path], k: int, warmup: int, repeats: int, timeout: float) -> dict:
    if not images:
        raise ValueError("At least one image is required.")
    latencies: list[float] = []
    measured_errors = 0
    warmup_errors = 0
    result_counts: list[int] = []
    categories: list[str] = []

    with httpx.Client(timeout=timeout) as client:
        for image in images:
            for _ in range(warmup):
                with image.open("rb") as handle:
                    response = client.post(
                        f"{url.rstrip('/')}/search?k={k}",
                        files={"file": (image.name, handle, "image/jpeg")},
                    )
                if response.status_code != 200:
                    warmup_errors += 1

            for _ in range(repeats):
                started = time.perf_counter()
                try:
                    with image.open("rb") as handle:
                        response = client.post(
                            f"{url.rstrip('/')}/search?k={k}",
                            files={"file": (image.name, handle, "image/jpeg")},
                        )
                    elapsed = (time.perf_counter() - started) * 1000.0
                    latencies.append(elapsed)
                    if response.status_code != 200:
                        measured_errors += 1
                        continue
                    body = response.json()
                    result_counts.append(len(body.get("results", [])))
                    if body.get("category"):
                        categories.append(str(body["category"]))
                except httpx.RequestError as exc:
                    measured_errors += 1
                    if total == 0:
                        raise RuntimeError(
                            f"Could not connect to the MVP API at {url!r}. "
                            "Start it first with scripts/run_api.py and the selected checkpoint."
                        ) from exc
                except (OSError, ValueError) as exc:
                    measured_errors += 1
                    raise RuntimeError(f"Benchmark request failed for {image}: {exc}") from exc

    total = len(latencies)
    total_seconds = sum(latencies) / 1000.0
    return {
        "protocol": {
            "warmup_per_image": warmup,
            "measured_repeats_per_image": repeats,
            "k": k,
            "images": [str(p) for p in images],
            "cold_start_included": False,
            "scope": "warm HTTP /search path",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "results": {
            "requests_measured": total,
            "errors": measured_errors,
            "warmup_errors": warmup_errors,
            "error_rate": measured_errors / total if total else 0.0,
            "latency_ms": {
                "mean": statistics.fmean(latencies) if latencies else 0.0,
                "median_p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
                "min": min(latencies) if latencies else 0.0,
                "max": max(latencies) if latencies else 0.0,
            },
            "sequential_requests_per_second": (
                1000.0 / statistics.fmean(latencies) if latencies else 0.0
            ),
            "mean_result_count": statistics.fmean(result_counts) if result_counts else 0.0,
            "returned_categories": dict(sorted({c: categories.count(c) for c in set(categories)}.items())),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmark/s4.10_mvp_performance.json"),
    )
    args = parser.parse_args()

    report = benchmark(args.url, args.image, args.k, args.warmup, args.repeats, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
