"""Dataset 1 integration checks for S1.11 visualization.

Skipped automatically when the Dataset 1 root, manifest, or pair CSV is
unavailable. Override the root with env var ``ZARGAR_DATASET1_ROOT``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from data.loaders.manifest import load_manifest
from data.pairs import load_pairs_csv
from data.visualization import (
    VisualizationConfig,
    generate_dataset_visualizations,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_PAIRS = PROJECT_ROOT / "reports" / "dataset" / "dataset1_pairs.csv"
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "ZARGAR_DATASET1_ROOT",
        r"e:\Privat File\Projects\Zargar Interview\dataset\ai-tool-pool-jewelry-vision",
    )
)


def _dataset_available() -> bool:
    return (
        DEFAULT_MANIFEST.is_file()
        and DEFAULT_PAIRS.is_file()
        and (DEFAULT_DATASET_ROOT / "train").is_dir()
    )


pytestmark = pytest.mark.skipif(
    not _dataset_available(),
    reason="Dataset 1 root, manifest, or pair CSV not available",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dataset1_visualization_smoke(tmp_path: Path) -> None:
    manifest_digest = _digest(DEFAULT_MANIFEST)
    pairs_digest = _digest(DEFAULT_PAIRS)
    samples = load_manifest(
        DEFAULT_MANIFEST,
        dataset_root=DEFAULT_DATASET_ROOT,
        validate_files=True,
    )
    pairs = load_pairs_csv(DEFAULT_PAIRS)
    config = VisualizationConfig(
        seed=2026,
        train_samples=4,
        valid_samples=4,
        test_samples=4,
        positive_pairs=4,
        negative_pairs=4,
        augmentation_samples=2,
    )
    watched = [sample.image_path for sample in samples[:8]]
    watched_digests = {path: _digest(path) for path in watched if path.is_file()}

    result = generate_dataset_visualizations(
        samples=samples,
        pairs=pairs,
        output_dir=tmp_path / "dataset1",
        config=config,
        dataset="dataset1",
        manifest=DEFAULT_MANIFEST,
        pair_source=DEFAULT_PAIRS,
    )
    report = result.report
    assert report["seed"] == 2026
    assert report["train"]["selected"] == 4
    assert report["train"]["multi_image_groups"] >= 1
    assert report["valid"]["selected"] == 4
    assert report["test"]["selected"] == 4
    assert report["positive_pairs"]["selected"] == 4
    assert report["positive_pairs"]["validated"] is True
    assert report["negative_pairs"]["selected"] == 4
    assert report["negative_pairs"]["same_category_selected"] >= 1
    assert report["negative_pairs"]["cross_category_selected"] >= 1
    assert report["validation_errors"] == []
    assert report["source_files_unchanged"] is True
    assert all(sample.split == "train" for sample in result.selection.augmentation_samples)
    assert (tmp_path / "dataset1" / "visualization_report.json").is_file()
    assert _digest(DEFAULT_MANIFEST) == manifest_digest
    assert _digest(DEFAULT_PAIRS) == pairs_digest
    for path, digest in watched_digests.items():
        assert _digest(path) == digest
