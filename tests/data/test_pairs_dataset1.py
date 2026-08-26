"""Integration tests: S1.10 pair generation against Dataset 1 manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.constants import SOURCE_DATASET1, SPLIT_ORDER
from data.loaders.manifest import load_manifest
from data.pairs import (
    PAIR_CSV_FIELDS,
    PairGenerationConfig,
    generate_pair_dataset,
    write_pair_generation_report,
    write_pairs_csv,
)
from data.pairs.sampler import count_available_positive_pairs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"

pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file(),
    reason="Dataset 1 manifest is not present.",
)


@pytest.fixture(scope="module")
def dataset1_samples():
    return load_manifest(MANIFEST, validate_files=False)


def test_available_positives_are_derived_from_manifest(dataset1_samples) -> None:
    available = count_available_positive_pairs(dataset1_samples)
    sizes: dict[str, int] = {}
    for sample in dataset1_samples:
        sizes[sample.group_id] = sizes.get(sample.group_id, 0) + 1
    expected = sum(size * (size - 1) // 2 for size in sizes.values() if size >= 2)
    assert available == expected
    assert available > 0


def test_default_generation_on_dataset1(dataset1_samples, tmp_path: Path) -> None:
    result = generate_pair_dataset(
        dataset1_samples,
        PairGenerationConfig(),
        dataset=SOURCE_DATASET1,
        manifest=MANIFEST,
    )
    report = result.report
    available = report["available_positive_pairs"]
    assert report["validation_results"]["passed"] is True
    assert report["seed"] == 2026
    assert report["selected_positive_pairs"] == available
    assert report["positive_pairs"] == report["negative_pairs"]
    assert report["same_category_negative_count"] + report["cross_category_negative_count"] == report[
        "negative_pairs"
    ]
    # Default 0.5 same-category ratio should be exact when the count is even.
    if report["negative_pairs"] % 2 == 0:
        assert report["same_category_negative_count"] == report["negative_pairs"] // 2
        assert report["cross_category_negative_count"] == report["negative_pairs"] // 2

    by_id = {sample.image_id: sample for sample in dataset1_samples}
    seen = set()
    for pair in result.pairs:
        assert pair.split in SPLIT_ORDER
        assert pair.image_id_1 < pair.image_id_2
        assert pair.image_id_1 in by_id and pair.image_id_2 in by_id
        left = by_id[pair.image_id_1]
        right = by_id[pair.image_id_2]
        assert left.split == right.split == pair.split
        key = (pair.image_id_1, pair.image_id_2)
        assert key not in seen
        seen.add(key)
        if pair.pair_type == "positive":
            assert pair.label == 1
            assert pair.group_id_1 == pair.group_id_2
        else:
            assert pair.label == 0
            assert pair.group_id_1 != pair.group_id_2

    csv_path = write_pairs_csv(result.pairs, tmp_path / "dataset1_pairs.csv")
    json_path = write_pair_generation_report(report, tmp_path / "report.json")
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(PAIR_CSV_FIELDS)
    assert json_path.is_file()

    # Valid/test in Dataset 1 are singleton groups, so they contribute 0 positives.
    for split in ("valid", "test"):
        assert report["per_split_statistics"][split]["available_positive_pairs"] == 0
        assert report["per_split_statistics"][split]["positive_pairs"] == 0
        assert report["per_split_statistics"][split]["negative_pairs"] == 0


def test_dataset1_generation_is_deterministic(dataset1_samples) -> None:
    config = PairGenerationConfig(seed=2026)
    first = generate_pair_dataset(dataset1_samples, config, manifest=MANIFEST)
    second = generate_pair_dataset(dataset1_samples, config, manifest=MANIFEST)
    assert [pair.pair_id for pair in first.pairs] == [pair.pair_id for pair in second.pairs]


def test_dataset1_different_seed_changes_negatives(dataset1_samples) -> None:
    first = generate_pair_dataset(dataset1_samples, PairGenerationConfig(seed=2026))
    second = generate_pair_dataset(dataset1_samples, PairGenerationConfig(seed=7))
    first_pos = [pair.pair_id for pair in first.pairs if pair.pair_type == "positive"]
    second_pos = [pair.pair_id for pair in second.pairs if pair.pair_type == "positive"]
    assert first_pos == second_pos
    first_neg = {pair.pair_id for pair in first.pairs if pair.pair_type == "negative"}
    second_neg = {pair.pair_id for pair in second.pairs if pair.pair_type == "negative"}
    assert first_neg != second_neg
