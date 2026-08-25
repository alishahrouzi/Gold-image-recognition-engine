"""Dataset 1 integration / smoke checks for S1.8 preprocessing.

Skipped automatically when the Dataset 1 root is unavailable.
Override the root with env var ``ZARGAR_DATASET1_ROOT``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from data.collate import collate_preprocessed_samples
from data.datasets.unified_dataset import UnifiedDataset
from data.loaders.manifest import load_manifest
from data.preprocessing import ImagePreprocessor, PreprocessedDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "ZARGAR_DATASET1_ROOT",
        r"e:\Privat File\Projects\Zargar Interview\dataset\ai-tool-pool-jewelry-vision",
    )
)


def _dataset_available() -> bool:
    return DEFAULT_MANIFEST.is_file() and (DEFAULT_DATASET_ROOT / "train").is_dir()


pytestmark = pytest.mark.skipif(
    not _dataset_available(),
    reason="Dataset 1 root or manifest not available",
)


def _representative_indices(samples) -> list[int]:
    """Pick a small diverse subset across splits, categories, and group sizes."""
    by_split_category: dict[tuple[str, str], list[int]] = defaultdict(list)
    group_sizes: dict[str, int] = defaultdict(int)
    for index, sample in enumerate(samples):
        by_split_category[(sample.split, sample.category)].append(index)
        group_sizes[sample.group_id] += 1

    selected: list[int] = []
    for key in sorted(by_split_category):
        selected.append(by_split_category[key][0])

    # Prefer at least one multi-image group and one singleton.
    multi = next((i for i, s in enumerate(samples) if group_sizes[s.group_id] >= 2), None)
    single = next((i for i, s in enumerate(samples) if group_sizes[s.group_id] == 1), None)
    for index in (multi, single):
        if index is not None and index not in selected:
            selected.append(index)

    return selected


def test_real_dataset1_representative_smoke() -> None:
    samples = load_manifest(
        DEFAULT_MANIFEST,
        dataset_root=DEFAULT_DATASET_ROOT,
        validate_files=True,
    )
    indices = _representative_indices(samples)
    assert len(indices) >= 5

    base = UnifiedDataset(
        DEFAULT_MANIFEST,
        dataset_root=DEFAULT_DATASET_ROOT,
        validate_files=False,
    )
    # Reuse already-validated samples to avoid a second full filesystem walk.
    base._samples = samples  # noqa: SLF001 — test-only narrow view
    dataset = PreprocessedDataset(Subset(base, indices))
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_preprocessed_samples,
    )

    seen = 0
    for batch in loader:
        images = batch["image"]
        assert images.ndim == 4
        assert images.shape[1:] == (3, 224, 224)
        assert images.dtype == torch.float32
        assert not torch.isnan(images).any()
        assert not torch.isinf(images).any()
        batch_size = images.shape[0]
        assert len(batch["image_id"]) == batch_size
        assert len(batch["group_id"]) == batch_size
        assert len(batch["category_id"]) == batch_size
        for index in range(batch_size):
            assert batch["source"][index] == "dataset1"
            assert batch["split"][index] in {"train", "valid", "test"}
        seen += batch_size

    assert seen == len(indices)


def test_real_dataset1_standalone_query_path() -> None:
    """Query/gallery path: preprocess a loaded PIL image without a manifest row."""
    from data.loaders.image_loader import load_rgb_image

    samples = load_manifest(
        DEFAULT_MANIFEST,
        dataset_root=DEFAULT_DATASET_ROOT,
        validate_files=False,
    )
    # One sample from each split.
    by_split = {}
    for sample in samples:
        by_split.setdefault(sample.split, sample)
    assert set(by_split) == {"train", "valid", "test"}

    preprocessor = ImagePreprocessor()
    for sample in by_split.values():
        image = load_rgb_image(sample.image_path)
        tensor = preprocessor(image)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()
