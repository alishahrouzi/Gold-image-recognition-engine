"""DataLoader performance benchmark for Dataset 1 (S1.12).

Measures the existing ingestion → (optional S1.9 augmentation) → S1.8
preprocessing → collation pipeline. Does not implement an encoder or
training loop. Does not modify the manifest or source images.
"""

from .dataloader_benchmark import (
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

__all__ = [
    "CONFIGURATION_RESULT_FIELDS",
    "DataLoaderBenchmarkConfig",
    "DataLoaderBenchmarkError",
    "GPU_UTILIZATION_UNAVAILABLE_REASON",
    "build_practical_configuration_matrix",
    "build_recommendation",
    "build_smoke_configuration_matrix",
    "cap_batch_counts",
    "collect_hardware_snapshot",
    "compute_latency_metrics",
    "configuration_result",
    "non_blocking_transfer",
    "percentile",
    "process_rss_mb",
    "run_dataloader_benchmark",
    "validate_benchmark_config",
    "write_dataloader_benchmark_json",
    "write_dataloader_benchmark_markdown",
]
