"""Unit tests for S1.11 read-only Dataset 1 visualization."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from random import Random

import pytest

from data.constants import CATEGORY_TO_ID, SOURCE_DATASET1
from data.errors import VisualizationError
from data.pairs import (
    NEGATIVE_TYPE_CROSS_CATEGORY,
    NEGATIVE_TYPE_SAME_CATEGORY,
    PAIR_TYPE_NEGATIVE,
    PAIR_TYPE_POSITIVE,
    Pair,
    load_pairs_csv,
    pair_from_samples,
    write_pairs_csv,
)
from data.types import Sample
from data.visualization import (
    VisualizationConfig,
    generate_dataset_visualizations,
    sample_train_groups,
)
from data.visualization.renderer import build_train_augmentor, render_augmentation_panels
from data.visualization.types import VisualizationSelection
from data.visualization.validation import (
    validate_image_ready,
    validate_negative_pair,
    validate_positive_pair,
    validate_selection,
)
from tests.data.helpers import make_sample


def _sample(
    tmp_path: Path,
    image_id: str,
    group_id: str,
    *,
    category: str = "Bracelet",
    split: str = "train",
    create_file: bool = True,
) -> Sample:
    return make_sample(
        tmp_path,
        image_id=image_id,
        group_id=group_id,
        category=category,
        split=split,
        filename=f"{split}/{image_id}.jpg",
        create_file=create_file,
    )


def _positive(a: Sample, b: Sample) -> Pair:
    return pair_from_samples(
        a, b, label=1, pair_type=PAIR_TYPE_POSITIVE, negative_type=None
    )


def _negative(a: Sample, b: Sample, negative_type: str) -> Pair:
    return pair_from_samples(
        a,
        b,
        label=0,
        pair_type=PAIR_TYPE_NEGATIVE,
        negative_type=negative_type,
    )


def _mini_dataset(tmp_path: Path) -> tuple[list[Sample], list[Pair]]:
    """Train groups of size 3/2/1, valid/test singletons, both negative types."""
    g1 = [
        _sample(tmp_path, "t1a", "g1"),
        _sample(tmp_path, "t1b", "g1"),
        _sample(tmp_path, "t1c", "g1"),
    ]
    g2 = [
        _sample(tmp_path, "t2a", "g2"),
        _sample(tmp_path, "t2b", "g2"),
    ]
    g3 = [_sample(tmp_path, "t3a", "g3")]
    g4 = [
        _sample(tmp_path, "r1a", "g4", category="Ring"),
        _sample(tmp_path, "r1b", "g4", category="Ring"),
    ]
    valid = [
        _sample(tmp_path, "v1", "vg1", split="valid"),
        _sample(tmp_path, "v2", "vg2", split="valid", category="Ring"),
    ]
    test = [
        _sample(tmp_path, "s1", "sg1", split="test"),
        _sample(tmp_path, "s2", "sg2", split="test", category="Ring"),
    ]
    samples = g1 + g2 + g3 + g4 + valid + test
    pairs = [
        _positive(g1[0], g1[1]),
        _positive(g1[0], g1[2]),
        _positive(g1[1], g1[2]),
        _positive(g2[0], g2[1]),
        _positive(g4[0], g4[1]),
        _negative(g1[0], g2[0], NEGATIVE_TYPE_SAME_CATEGORY),
        _negative(g1[1], g2[1], NEGATIVE_TYPE_SAME_CATEGORY),
        _negative(g1[0], g4[0], NEGATIVE_TYPE_CROSS_CATEGORY),
        _negative(g2[0], g4[1], NEGATIVE_TYPE_CROSS_CATEGORY),
    ]
    return samples, pairs


def _config(**kwargs) -> VisualizationConfig:
    defaults = {
        "seed": 2026,
        "train_samples": 3,
        "valid_samples": 2,
        "test_samples": 2,
        "positive_pairs": 3,
        "negative_pairs": 4,
        "augmentation_samples": 2,
    }
    defaults.update(kwargs)
    return VisualizationConfig(**defaults)


def test_sampling_is_deterministic(tmp_path: Path) -> None:
    samples, pairs = _mini_dataset(tmp_path)
    first = generate_dataset_visualizations(
        samples=samples,
        pairs=pairs,
        output_dir=tmp_path / "viz_a",
        config=_config(),
        manifest=tmp_path / "manifest.csv",
        pair_source=tmp_path / "pairs.csv",
    )
    second = generate_dataset_visualizations(
        samples=samples,
        pairs=pairs,
        output_dir=tmp_path / "viz_b",
        config=_config(),
        manifest=tmp_path / "manifest.csv",
        pair_source=tmp_path / "pairs.csv",
    )
    assert [panel.group_id for panel in first.selection.train_groups] == [
        panel.group_id for panel in second.selection.train_groups
    ]
    assert [sample.image_id for sample in first.selection.valid_samples] == [
        sample.image_id for sample in second.selection.valid_samples
    ]
    assert [sample.image_id for sample in first.selection.test_samples] == [
        sample.image_id for sample in second.selection.test_samples
    ]
    assert [panel.pair.pair_id for panel in first.selection.positive_pairs] == [
        panel.pair.pair_id for panel in second.selection.positive_pairs
    ]
    assert [panel.pair.pair_id for panel in first.selection.same_category_negatives] == [
        panel.pair.pair_id for panel in second.selection.same_category_negatives
    ]
    assert [panel.pair.pair_id for panel in first.selection.cross_category_negatives] == [
        panel.pair.pair_id for panel in second.selection.cross_category_negatives
    ]


def test_does_not_modify_global_rng(tmp_path: Path) -> None:
    samples, pairs = _mini_dataset(tmp_path)
    random.seed(12345)
    before = [random.random() for _ in range(5)]
    random.seed(12345)
    generate_dataset_visualizations(
        samples=samples,
        pairs=pairs,
        output_dir=tmp_path / "viz",
        config=_config(seed=99),
    )
    after = [random.random() for _ in range(5)]
    assert before == after


def test_train_sampling_is_group_aware(tmp_path: Path) -> None:
    samples, _pairs = _mini_dataset(tmp_path)
    panels = sample_train_groups(samples, count=3, rng=Random(2026))
    assert panels
    by_id = {sample.group_id: [] for sample in samples if sample.split == "train"}
    for sample in samples:
        if sample.split == "train":
            by_id[sample.group_id].append(sample.image_id)
    for panel in panels:
        assert panel.split == "train"
        shown = [member.image_id for member in panel.samples]
        expected = sorted(by_id[panel.group_id])
        assert shown == expected
    assert any(len(panel.samples) >= 2 for panel in panels)


def test_valid_and_test_sampling_use_correct_splits(tmp_path: Path) -> None:
    samples, pairs = _mini_dataset(tmp_path)
    result = generate_dataset_visualizations(
        samples=samples,
        pairs=pairs,
        output_dir=tmp_path / "viz",
        config=_config(),
    )
    assert all(sample.split == "valid" for sample in result.selection.valid_samples)
    assert all(sample.split == "test" for sample in result.selection.test_samples)
    assert result.report["valid"]["augmentation"] is False
    assert result.report["test"]["augmentation"] is False
    assert result.report["augmentation"]["valid_test_augmented"] is False


def test_positive_pair_validation_accepts_same_group(tmp_path: Path) -> None:
    a = _sample(tmp_path, "a", "g1")
    b = _sample(tmp_path, "b", "g1")
    pair = _positive(a, b)
    validate_positive_pair(pair, a, b)


def test_positive_pair_validation_rejects_different_group(tmp_path: Path) -> None:
    a = _sample(tmp_path, "a", "g1")
    b = _sample(tmp_path, "b", "g2")
    pair = Pair(
        pair_id="a__b",
        image_id_1="a",
        image_id_2="b",
        group_id_1="g1",
        group_id_2="g2",
        category_id_1=CATEGORY_TO_ID["Bracelet"],
        category_id_2=CATEGORY_TO_ID["Bracelet"],
        category_1="Bracelet",
        category_2="Bracelet",
        split="train",
        label=1,
        pair_type=PAIR_TYPE_POSITIVE,
        negative_type=None,
    )
    with pytest.raises(VisualizationError, match="same group_id"):
        validate_positive_pair(pair, a, b)


def test_positive_pair_validation_rejects_different_category(tmp_path: Path) -> None:
    a = _sample(tmp_path, "a", "g1")
    b = _sample(tmp_path, "b", "g1", category="Ring")
    pair = Pair(
        pair_id="a__b",
        image_id_1="a",
        image_id_2="b",
        group_id_1="g1",
        group_id_2="g1",
        category_id_1=CATEGORY_TO_ID["Bracelet"],
        category_id_2=CATEGORY_TO_ID["Ring"],
        category_1="Bracelet",
        category_2="Ring",
        split="train",
        label=1,
        pair_type=PAIR_TYPE_POSITIVE,
        negative_type=None,
    )
    with pytest.raises(VisualizationError, match="same category"):
        validate_positive_pair(pair, a, b)


def test_negative_pair_validation_requires_different_groups(tmp_path: Path) -> None:
    a = _sample(tmp_path, "a", "g1")
    b = _sample(tmp_path, "b", "g1")
    pair = Pair(
        pair_id="a__b",
        image_id_1="a",
        image_id_2="b",
        group_id_1="g1",
        group_id_2="g1",
        category_id_1=0,
        category_id_2=0,
        category_1="Bracelet",
        category_2="Bracelet",
        split="train",
        label=0,
        pair_type=PAIR_TYPE_NEGATIVE,
        negative_type=NEGATIVE_TYPE_SAME_CATEGORY,
    )
    with pytest.raises(VisualizationError, match="different group"):
        validate_negative_pair(pair, a, b)


def test_same_category_negative_validation(tmp_path: Path) -> None:
    a = _sample(tmp_path, "a", "g1")
    b = _sample(tmp_path, "b", "g2")
    pair = _negative(a, b, NEGATIVE_TYPE_SAME_CATEGORY)
    validate_negative_pair(pair, a, b)

    other = _sample(tmp_path, "c", "g3", category="Ring")
    bad = _negative(a, other, NEGATIVE_TYPE_SAME_CATEGORY)
    with pytest.raises(VisualizationError, match="different categories"):
        validate_negative_pair(bad, a, other)


def test_cross_category_negative_validation(tmp_path: Path) -> None:
    a = _sample(tmp_path, "a", "g1")
    b = _sample(tmp_path, "b", "g2", category="Ring")
    pair = _negative(a, b, NEGATIVE_TYPE_CROSS_CATEGORY)
    validate_negative_pair(pair, a, b)

    other = _sample(tmp_path, "c", "g3")
    bad = _negative(a, other, NEGATIVE_TYPE_CROSS_CATEGORY)
    with pytest.raises(VisualizationError, match="same category"):
        validate_negative_pair(bad, a, other)


def test_missing_image_handling(tmp_path: Path) -> None:
    sample = _sample(tmp_path, "missing", "g1", create_file=False)
    with pytest.raises(VisualizationError, match="failed QA"):
        validate_image_ready(sample)


def test_invalid_pair_fails_loudly_without_figures(tmp_path: Path) -> None:
    samples, pairs = _mini_dataset(tmp_path)
    bad = Pair(
        pair_id="t1a__t2a",
        image_id_1="t1a",
        image_id_2="t2a",
        group_id_1="g1",
        group_id_2="g2",
        category_id_1=0,
        category_id_2=0,
        category_1="Bracelet",
        category_2="Bracelet",
        split="train",
        label=1,
        pair_type=PAIR_TYPE_POSITIVE,
        negative_type=None,
    )
    output_dir = tmp_path / "viz_bad"
    with pytest.raises(VisualizationError, match="Visualization QA failed"):
        generate_dataset_visualizations(
            samples=samples,
            pairs=[bad],
            output_dir=output_dir,
            config=_config(positive_pairs=1, negative_pairs=0),
            manifest=tmp_path / "manifest.csv",
            pair_source=tmp_path / "pairs.csv",
        )
    assert (output_dir / "visualization_report.json").is_file()
    assert not (output_dir / "positive_pairs.png").exists()


def test_output_directory_and_report_generation(tmp_path: Path) -> None:
    samples, pairs = _mini_dataset(tmp_path)
    output_dir = tmp_path / "nested" / "viz"
    result = generate_dataset_visualizations(
        samples=samples,
        pairs=pairs,
        output_dir=output_dir,
        config=_config(),
        dataset=SOURCE_DATASET1,
        manifest=tmp_path / "manifest.csv",
        pair_source=tmp_path / "pairs.csv",
    )
    assert output_dir.is_dir()
    expected = [
        "train_samples.png",
        "valid_samples.png",
        "test_samples.png",
        "positive_pairs.png",
        "negative_pairs.png",
        "negative_pairs_same_category.png",
        "negative_pairs_cross_category.png",
        "augmentation_samples.png",
        "visualization_report.json",
    ]
    for name in expected:
        assert (output_dir / name).is_file()
        assert name in result.output_files
    report = result.report
    assert report["dataset"] == SOURCE_DATASET1
    assert report["seed"] == 2026
    assert report["train"]["requested"] == 3
    assert report["train"]["selected"] == len(result.selection.train_groups)
    assert report["valid"]["requested"] == 2
    assert report["test"]["requested"] == 2
    assert report["positive_pairs"]["validated"] is True
    assert report["negative_pairs"]["validated"] is True
    assert report["negative_pairs"]["same_category_selected"] >= 1
    assert report["negative_pairs"]["cross_category_selected"] >= 1
    assert report["validation_errors"] == []
    assert report["source_files_unchanged"] is True


def test_source_immutability(tmp_path: Path) -> None:
    samples, pairs = _mini_dataset(tmp_path)
    pair_csv = write_pairs_csv(pairs, tmp_path / "pairs.csv")
    checksums = {sample.image_path: hashlib.sha256(sample.image_path.read_bytes()).hexdigest() for sample in samples}
    pair_digest = hashlib.sha256(pair_csv.read_bytes()).hexdigest()
    generate_dataset_visualizations(
        samples=samples,
        pairs=load_pairs_csv(pair_csv),
        output_dir=tmp_path / "viz",
        config=_config(),
        pair_source=pair_csv,
    )
    for path, digest in checksums.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert hashlib.sha256(pair_csv.read_bytes()).hexdigest() == pair_digest


def test_augmentation_visualization_is_train_only(tmp_path: Path) -> None:
    samples, pairs = _mini_dataset(tmp_path)
    result = generate_dataset_visualizations(
        samples=samples,
        pairs=pairs,
        output_dir=tmp_path / "viz",
        config=_config(),
    )
    assert all(sample.split == "train" for sample in result.selection.augmentation_samples)
    valid = next(sample for sample in samples if sample.split == "valid")
    with pytest.raises(ValueError, match="train-only"):
        render_augmentation_panels(
            [valid],
            tmp_path / "bad_aug.png",
            build_train_augmentor(2026),
            "should fail",
        )


def test_cross_split_pair_is_rejected(tmp_path: Path) -> None:
    train = _sample(tmp_path, "a", "g1", split="train")
    valid = _sample(tmp_path, "b", "g2", split="valid")
    pair = Pair(
        pair_id="a__b",
        image_id_1="a",
        image_id_2="b",
        group_id_1="g1",
        group_id_2="g2",
        category_id_1=0,
        category_id_2=0,
        category_1="Bracelet",
        category_2="Bracelet",
        split="train",
        label=0,
        pair_type=PAIR_TYPE_NEGATIVE,
        negative_type=NEGATIVE_TYPE_SAME_CATEGORY,
    )
    with pytest.raises(VisualizationError, match="crosses splits"):
        validate_negative_pair(pair, train, valid)


def test_selection_collects_missing_image_errors(tmp_path: Path) -> None:
    missing = _sample(tmp_path, "gone", "g1", create_file=False)
    present = _sample(tmp_path, "here", "g1")
    selection = VisualizationSelection(train_groups=[], valid_samples=[missing, present])
    errors = validate_selection(selection, [missing, present])
    assert errors
    assert any("gone" in message for message in errors)

