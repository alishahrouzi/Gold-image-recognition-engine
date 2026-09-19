from pathlib import Path

from scripts.benchmark_s4_10_mvp import benchmark, percentile


def test_percentile_interpolates():
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.5) == 25.0


def test_benchmark_requires_images():
    try:
        benchmark("http://127.0.0.1:1", [], 5, 0, 1, 1.0)
    except ValueError as exc:
        assert "image" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")
