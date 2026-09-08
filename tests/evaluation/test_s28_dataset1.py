"""Dataset 1 contract checks for S2.8 (no trained-model metrics)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.leakage import DATASET1_EXPECTED, manifest_sha256, verify_dataset1_contract
from src.data.loaders.manifest import load_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"


pytestmark = pytest.mark.skipif(not MANIFEST.is_file(), reason="dataset1_manifest.csv is not present")


def test_dataset1_counts_and_split_isolation() -> None:
    samples = load_manifest(MANIFEST, validate_files=False)
    report = verify_dataset1_contract(samples, validate_files=False)
    assert report["cross_split_groups"] == []
    assert report["counts"] == DATASET1_EXPECTED

    train = [sample for sample in samples if sample.split == "train"]
    groups: dict[str, int] = {}
    for sample in train:
        groups[sample.group_id] = groups.get(sample.group_id, 0) + 1
    multi_image_groups = sum(1 for count in groups.values() if count >= 2)
    assert multi_image_groups > 0
    assert len(train) == DATASET1_EXPECTED["train_images"]


def test_manifest_hash_is_stable() -> None:
    digest = manifest_sha256(MANIFEST)
    assert len(digest) == 64
    assert digest == manifest_sha256(MANIFEST)
