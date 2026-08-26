"""Unit tests for S1.12 DataLoader benchmark helpers.

Uses generated temp images. Does not require CUDA. Does not modify Dataset 1.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from data.benchmark.dataloader_benchmark import (
    CONFIGURATION_RESULT_FIELDS,
    DataLoaderBenchmarkConfig,
    DataLoaderBenchmarkError,
    GPU_UTILIZATION_UNAVAILABLE_REASON,
    build_practical_configuration_matrix,
    build_recommendation,
    build_smoke_configuration_matrix,
    cap_batch_counts,
    collect_hardware_snapshot,
    compute_latency_metrics,
    configuration_result,
    non_blocking_transfer,
    percentile,
    process_rss_mb,
    run_dataloader_benchmark,
    validate_benchmark_config,
    write_dataloader_benchmark_json,
    write_dataloader_benchmark_markdown,
)
from data.constants import CATEGORY_TO_ID, SOURCE_DATASET1
from tests.data.helpers import write_manifest, write_rgb_image


def _tiny_manifest(tmp_path: Path, *, n_train: int = 12, n_valid: int = 4, n_test: int = 4) -> Path:
    rows = []
    index = 0
    for split, count in (("train", n_train), ("valid", n_valid), ("test", n_test)):
        for local in range(count):
            index += 1
            category = "Bracelet" if local % 2 == 0 else "Ring"
            image_path = write_rgb_image(tmp_path / f"{split}_{index}.jpg", size=(32, 24))
            rows.append(
                {
                    "image_id": f"img_{index}",
                    "image_path": str(image_path),
                    "group_id": f"{category.lower()}_{local // 2:03d}_{split}",
                    "category": category,
                    "category_id": CATEGORY_TO_ID[category],
                    "split": split,
                    "source": SOURCE_DATASET1,
                }
            )
    return write_manifest(tmp_path / "manifest.csv", rows)


def _pass_row(**overrides):
    payload = configuration_result(
        split="train",
        stage="dataloader",
        batch_size=16,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        status="PASS",
        augmentation_applied=True,
        warmup_batches_used=5,
        measurement_batches_used=30,
        images_measured=480,
        first_batch_ms=40.0,
        mean_batch_ms=20.0,
        median_batch_ms=19.0,
        p95_batch_ms=28.0,
        batches_per_sec=50.0,
        images_per_sec=800.0,
        ram_before_mb=200.0,
        ram_after_warmup_mb=260.0,
        ram_peak_mb=280.0,
        ram_delta_mb=80.0,
    )
    payload.update(overrides)
    return payload


def test_configuration_result_schema_is_deterministic() -> None:
    row = configuration_result(
        split="train",
        stage="dataloader",
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        status="PASS",
    )
    assert tuple(row.keys()) == CONFIGURATION_RESULT_FIELDS
    assert list(row.keys())[0] == "split"
    assert list(row.keys())[-1] == "gpu_utilization_percent"


def test_percentile_and_latency_metrics() -> None:
    values = [0.10, 0.20, 0.30, 0.40]
    assert percentile(values, 0) == pytest.approx(0.10)
    assert percentile(values, 100) == pytest.approx(0.40)
    assert percentile(values, 50) == pytest.approx(0.25)
    metrics = compute_latency_metrics([0.2, 0.2, 0.1], images_measured=24)
    assert metrics["mean_batch_ms"] == pytest.approx(166.6666, rel=1e-3)
    assert metrics["batches_per_sec"] == pytest.approx(6.0)
    assert metrics["images_per_sec"] == pytest.approx(48.0)
    assert metrics["p95_batch_ms"] == pytest.approx(percentile([0.2, 0.2, 0.1], 95.0) * 1000.0)


def test_throughput_uses_actual_image_count() -> None:
    metrics = compute_latency_metrics([1.0, 1.0], images_measured=24)
    assert metrics["images_per_sec"] == pytest.approx(12.0)
    assert metrics["batches_per_sec"] == pytest.approx(1.0)


def test_invalid_batch_size_rejected() -> None:
    with pytest.raises(DataLoaderBenchmarkError, match="batch_size"):
        validate_benchmark_config(split="train", batch_size=0, num_workers=0)
    with pytest.raises(DataLoaderBenchmarkError, match="batch_size"):
        validate_benchmark_config(split="train", batch_size=-4, num_workers=0)
    with pytest.raises(DataLoaderBenchmarkError, match="batch_size"):
        DataLoaderBenchmarkConfig(
            split="train",
            batch_size=0,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )


def test_worker_configuration_validation() -> None:
    validate_benchmark_config(split="train", batch_size=8, num_workers=0, persistent_workers=False)
    validate_benchmark_config(
        split="valid",
        batch_size=8,
        num_workers=2,
        persistent_workers=True,
    )
    with pytest.raises(DataLoaderBenchmarkError, match="num_workers"):
        validate_benchmark_config(split="train", batch_size=8, num_workers=-1)
    with pytest.raises(DataLoaderBenchmarkError, match="persistent_workers"):
        validate_benchmark_config(
            split="train",
            batch_size=8,
            num_workers=0,
            persistent_workers=True,
        )
    with pytest.raises(DataLoaderBenchmarkError, match="split"):
        validate_benchmark_config(split="gallery", batch_size=8, num_workers=0)


def test_cap_batch_counts_for_small_splits() -> None:
    assert cap_batch_counts(212, 64, warmup_batches=5, measurement_batches=30) == (3, 1)
    assert cap_batch_counts(12, 8, warmup_batches=5, measurement_batches=30) == (1, 1)
    assert cap_batch_counts(4328, 16, warmup_batches=5, measurement_batches=30) == (5, 30)


def test_non_blocking_follows_pin_memory_on_cuda() -> None:
    assert non_blocking_transfer(True, "cuda") is True
    assert non_blocking_transfer(False, "cuda") is False
    assert non_blocking_transfer(True, "cpu") is False


def test_oom_result_is_not_recommended() -> None:
    oom = _pass_row(
        status="OOM",
        error="CUDA out of memory",
        batch_size=64,
        stage="dataloader_gpu",
        images_per_sec=None,
        mean_batch_ms=None,
    )
    ok_small = _pass_row(batch_size=8, images_per_sec=100.0, stage="dataloader_gpu")
    ok_mid = _pass_row(
        batch_size=16,
        num_workers=2,
        pin_memory=True,
        images_per_sec=400.0,
        stage="dataloader_gpu",
        ram_peak_mb=220.0,
    )
    recommendation = build_recommendation(
        [oom, ok_small, ok_mid], cuda_available=True
    )
    assert recommendation["largest_successful_dataloader_batch"] == 16
    assert recommendation["recommended_dataloader"]["batch_size"] == 16
    assert recommendation["oom_configurations"][0]["batch_size"] == 64
    assert "DataLoader-safe batch size" in recommendation["dataloader_safe_batch_size_note"]
    assert "not the final model-training batch size" in recommendation[
        "dataloader_safe_batch_size_note"
    ]


def test_recommendation_prefers_stable_throughput() -> None:
    slow = _pass_row(batch_size=8, num_workers=0, images_per_sec=50.0, ram_peak_mb=100.0)
    fast_unstable = _pass_row(
        batch_size=32,
        num_workers=4,
        images_per_sec=900.0,
        mean_batch_ms=10.0,
        p95_batch_ms=80.0,
        ram_peak_mb=500.0,
    )
    stable = _pass_row(
        batch_size=16,
        num_workers=2,
        images_per_sec=700.0,
        mean_batch_ms=20.0,
        p95_batch_ms=24.0,
        ram_peak_mb=250.0,
    )
    recommendation = build_recommendation(
        [slow, fast_unstable, stable], cuda_available=False
    )
    assert recommendation["fastest_configuration"]["batch_size"] == 32
    assert recommendation["most_memory_efficient_configuration"]["batch_size"] == 8
    assert recommendation["stable_configuration"]["batch_size"] == 16
    assert recommendation["recommended_dataloader_batch_size"] == 16


def test_process_rss_is_process_level() -> None:
    rss = process_rss_mb()
    assert rss > 0.0
    hardware = collect_hardware_snapshot(torch.device("cpu"))
    assert hardware["process_rss_mb"] > 0.0
    assert hardware["gpu_utilization_measurable"] is False
    assert "null" in hardware["gpu_utilization_note"].lower() or "not reported" in (
        hardware["gpu_utilization_note"].lower()
    )


def test_tiny_cpu_benchmark_and_reports(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    configs = [
        DataLoaderBenchmarkConfig(
            split="train",
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            stage="dataloader",
        ),
        DataLoaderBenchmarkConfig(
            split="valid",
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            stage="dataloader",
        ),
        DataLoaderBenchmarkConfig(
            split="test",
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            stage="dataloader",
        ),
    ]
    report = run_dataloader_benchmark(
        manifest_path=manifest,
        dataset_root=tmp_path,
        configurations=configs,
        warmup_batches=1,
        measurement_batches=2,
        seed=2026,
        device="cpu",
    )
    assert report["benchmark"]["seed"] == 2026
    assert report["preprocessing"]["image_size"] == 224
    assert report["augmentation"]["train"]["enabled"] is True
    assert report["augmentation"]["valid"]["enabled"] is False
    assert report["augmentation"]["test"]["deterministic"] is True
    by_split = {row["split"]: row for row in report["configurations"]}
    assert by_split["train"]["status"] == "PASS"
    assert by_split["train"]["augmentation_applied"] is True
    assert by_split["valid"]["augmentation_applied"] is False
    assert by_split["test"]["augmentation_applied"] is False
    assert by_split["train"]["images_per_sec"] > 0
    assert tuple(by_split["train"].keys()) == CONFIGURATION_RESULT_FIELDS
    assert "DataLoader-safe batch size" in report["recommendation"]["dataloader_safe_batch_size_note"]

    json_path = write_dataloader_benchmark_json(report, tmp_path / "out.json")
    md_path = write_dataloader_benchmark_markdown(report, tmp_path / "out.md")
    text = md_path.read_text(encoding="utf-8")
    assert "Recommended DataLoader configuration" in text
    assert "not the final model-training batch size" in text
    assert json_path.read_text(encoding="utf-8").strip().startswith("{")


def test_gpu_stage_skipped_on_cpu(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path, n_train=8, n_valid=2, n_test=2)
    configs = build_smoke_configuration_matrix(include_gpu=True)
    report = run_dataloader_benchmark(
        manifest_path=manifest,
        dataset_root=tmp_path,
        configurations=configs,
        warmup_batches=0,
        measurement_batches=1,
        seed=2026,
        device="cpu",
    )
    stages = {row["stage"] for row in report["configurations"]}
    assert "dataloader" in stages
    gpu_rows = [row for row in report["configurations"] if row["stage"] == "dataloader_gpu"]
    assert gpu_rows, "GPU-stage rows must remain in the report as SKIPPED when CUDA is off."
    assert all(row["status"] == "SKIPPED" for row in gpu_rows)


def test_practical_matrix_is_not_full_cartesian() -> None:
    matrix = build_practical_configuration_matrix(include_gpu=True)
    train_cpu = [
        row
        for row in matrix
        if row.split == "train" and row.stage == "dataloader"
    ]
    full = 4 * 3 * 2 * 2
    assert len(train_cpu) < full
    assert {row.batch_size for row in train_cpu} == {8, 16, 32, 64}
    assert {row.num_workers for row in train_cpu} >= {0, 2, 4}
    assert any(row.pin_memory for row in matrix)
    assert any(row.persistent_workers for row in matrix)
    assert any(row.split == "valid" for row in matrix)
    assert any(row.split == "test" for row in matrix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_transfer_stage_when_cuda_present(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path, n_train=8, n_valid=2, n_test=2)
    configs = [
        DataLoaderBenchmarkConfig(
            split="train",
            batch_size=4,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            stage="dataloader_gpu",
        )
    ]
    report = run_dataloader_benchmark(
        manifest_path=manifest,
        dataset_root=tmp_path,
        configurations=configs,
        warmup_batches=0,
        measurement_batches=1,
        seed=2026,
        device="cuda",
    )
    row = report["configurations"][0]
    assert row["status"] == "PASS"
    assert row["gpu_peak_allocated_mb"] is not None
    assert row["gpu_utilization_percent"] is None
    assert GPU_UTILIZATION_UNAVAILABLE_REASON in report["benchmark"]["notes"]
