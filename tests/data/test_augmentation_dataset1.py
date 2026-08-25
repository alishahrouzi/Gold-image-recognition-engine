"""Dataset 1 integration / smoke checks for S1.9 training augmentation.

Skipped automatically when the Dataset 1 root is unavailable.
Override the root with env var ``ZARGAR_DATASET1_ROOT``.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from data.collate import collate_preprocessed_samples
from data.datasets.unified_dataset import UnifiedDataset
from data.loaders.image_loader import load_rgb_image
from data.loaders.manifest import load_manifest
from data.preprocessing import (
    AugmentationConfig,
    ImagePreprocessor,
    TrainingAugmentor,
    build_preprocessed_dataset,
)

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


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _representative_indices(samples) -> list[int]:
    by_split_category: dict[tuple[str, str], list[int]] = defaultdict(list)
    group_sizes: dict[str, int] = defaultdict(int)
    for index, sample in enumerate(samples):
        by_split_category[(sample.split, sample.category)].append(index)
        group_sizes[sample.group_id] += 1

    selected: list[int] = []
    for key in sorted(by_split_category):
        selected.append(by_split_category[key][0])

    multi = next((i for i, s in enumerate(samples) if group_sizes[s.group_id] >= 2), None)
    single = next((i for i, s in enumerate(samples) if group_sizes[s.group_id] == 1), None)
    for index in (multi, single):
        if index is not None and index not in selected:
            selected.append(index)
    return selected


def test_dataset1_augmentation_and_preprocessing_smoke() -> None:
    samples = load_manifest(
        DEFAULT_MANIFEST,
        dataset_root=DEFAULT_DATASET_ROOT,
        validate_files=True,
    )
    indices = _representative_indices(samples)
    train_indices = [i for i in indices if samples[i].split == "train"]
    assert train_indices

    checksums = {samples[i].image_path: _file_digest(samples[i].image_path) for i in train_indices}

    base = UnifiedDataset(
        DEFAULT_MANIFEST,
        dataset_root=DEFAULT_DATASET_ROOT,
        validate_files=False,
    )
    base._samples = [samples[i] for i in train_indices]  # noqa: SLF001

    dataset = build_preprocessed_dataset(
        base,
        role="train",
        preprocessor=ImagePreprocessor(),
        augmentation=AugmentationConfig(seed=2026),
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_preprocessed_samples,
    )

    seen = 0
    for batch in loader:
        images = batch["image"]
        assert images.shape[1:] == (3, 224, 224)
        assert images.dtype == torch.float32
        assert not torch.isnan(images).any()
        assert not torch.isinf(images).any()
        for index in range(images.shape[0]):
            assert batch["source"][index] == "dataset1"
            assert batch["group_id"][index]
            assert batch["split"][index] == "train"
        seen += images.shape[0]

    assert seen == len(train_indices)
    for path, digest in checksums.items():
        assert _file_digest(path) == digest


def test_dataset1_query_gallery_are_deterministic() -> None:
    samples = load_manifest(
        DEFAULT_MANIFEST,
        dataset_root=DEFAULT_DATASET_ROOT,
        validate_files=False,
    )
    by_split = {}
    for sample in samples:
        by_split.setdefault(sample.split, sample)

    preprocessor = ImagePreprocessor()
    for sample in by_split.values():
        image = load_rgb_image(sample.image_path)
        query = preprocessor(image)
        gallery = preprocessor(load_rgb_image(sample.image_path))
        assert torch.equal(query, gallery)
        assert query.shape == (3, 224, 224)


def test_dataset1_metadata_unchanged_after_augmentation() -> None:
    samples = load_manifest(
        DEFAULT_MANIFEST,
        dataset_root=DEFAULT_DATASET_ROOT,
        validate_files=False,
    )
    train = next(sample for sample in samples if sample.split == "train")
    image = load_rgb_image(train.image_path)
    TrainingAugmentor(AugmentationConfig(seed=1))(image)
    assert train.group_id
    assert train.image_id
    assert train.split == "train"
