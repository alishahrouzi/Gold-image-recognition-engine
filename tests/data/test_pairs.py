"""Unit tests for S1.10 group-aware pair generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.constants import CATEGORY_TO_ID, SOURCE_DATASET1
from data.errors import DatasetIngestionError, PairGenerationError
from data.loaders.manifest import load_manifest
from data.pairs import (
    Pair,
    PairGenerationConfig,
    canonicalize_image_ids,
    generate_pair_dataset,
    generate_positive_pairs,
    make_pair_id,
    pair_from_samples,
    validate_pairs,
)
from data.types import Sample
from tests.data.helpers import make_sample, write_manifest, write_rgb_image


def _cfg(**kwargs) -> PairGenerationConfig:
    defaults = {
        "include_train": True,
        "include_valid": False,
        "include_test": False,
        "seed": 2026,
    }
    defaults.update(kwargs)
    return PairGenerationConfig(**defaults)


def _sample(
    tmp_path: Path,
    image_id: str,
    group_id: str,
    *,
    category: str = "Bracelet",
    split: str = "train",
) -> Sample:
    return make_sample(
        tmp_path,
        image_id=image_id,
        group_id=group_id,
        category=category,
        split=split,
        filename=f"{image_id}.jpg",
    )


def test_group_of_three_yields_three_positive_pairs(tmp_path: Path) -> None:
    samples = [
        _sample(tmp_path, "a", "g1"),
        _sample(tmp_path, "b", "g1"),
        _sample(tmp_path, "c", "g1"),
    ]
    pairs = generate_positive_pairs(samples)
    assert len(pairs) == 3
    keys = {pair.unordered_key() for pair in pairs}
    assert keys == {("a", "b"), ("a", "c"), ("b", "c")}
    assert all(pair.label == 1 and pair.pair_type == "positive" for pair in pairs)
    assert all(pair.negative_type is None for pair in pairs)
    assert all(pair.image_id_1 != pair.image_id_2 for pair in pairs)


def test_group_of_two_yields_one_positive_pair(tmp_path: Path) -> None:
    samples = [_sample(tmp_path, "a", "g1"), _sample(tmp_path, "b", "g1")]
    pairs = generate_positive_pairs(samples)
    assert len(pairs) == 1
    assert pairs[0].unordered_key() == ("a", "b")


def test_singleton_yields_zero_positive_pairs(tmp_path: Path) -> None:
    samples = [_sample(tmp_path, "a", "g1"), _sample(tmp_path, "b", "g2")]
    assert generate_positive_pairs(samples) == []


def test_no_self_pairs(tmp_path: Path) -> None:
    samples = [
        _sample(tmp_path, "a", "g1"),
        _sample(tmp_path, "b", "g1"),
        _sample(tmp_path, "c", "g1"),
    ]
    pairs = generate_positive_pairs(samples)
    assert all(pair.image_id_1 != pair.image_id_2 for pair in pairs)
    with pytest.raises(PairGenerationError, match="Self-pair"):
        canonicalize_image_ids("a", "a")


def test_same_category_and_cross_category_negatives(tmp_path: Path) -> None:
    samples = [
        _sample(tmp_path, "a1", "ga", category="Bracelet"),
        _sample(tmp_path, "a2", "ga", category="Bracelet"),
        _sample(tmp_path, "b1", "gb", category="Bracelet"),
        _sample(tmp_path, "r1", "gr", category="Ring"),
    ]
    result = generate_pair_dataset(samples, _cfg(same_category_negative_ratio=0.5))
    negatives = [pair for pair in result.pairs if pair.pair_type == "negative"]
    assert len(negatives) == 1  # 1 positive => 1 negative at 1:1, ratio 0.5 rounds to 1 same / 0 cross
    # With n_neg=1 and ratio 0.5, round(1*0.5)=0 same and 1 cross, or 1 same and 0 cross.
    # int(round(1 * 0.5)) == 0, so this default yields 0 same + 1 cross.
    assert negatives[0].label == 0
    assert negatives[0].group_id_1 != negatives[0].group_id_2

    same_only = generate_pair_dataset(samples, _cfg(same_category_negative_ratio=1.0))
    same_neg = [pair for pair in same_only.pairs if pair.pair_type == "negative"]
    assert len(same_neg) == 1
    assert same_neg[0].negative_type == "same_category"
    assert same_neg[0].category_1 == same_neg[0].category_2 == "Bracelet"

    cross_only = generate_pair_dataset(samples, _cfg(same_category_negative_ratio=0.0))
    cross_neg = [pair for pair in cross_only.pairs if pair.pair_type == "negative"]
    assert len(cross_neg) == 1
    assert cross_neg[0].negative_type == "cross_category"
    assert cross_neg[0].category_1 != cross_neg[0].category_2


def test_unordered_pair_uniqueness(tmp_path: Path) -> None:
    samples = [
        _sample(tmp_path, "img_b", "g1"),
        _sample(tmp_path, "img_a", "g1"),
    ]
    pairs = generate_positive_pairs(samples)
    assert len(pairs) == 1
    assert pairs[0].image_id_1 == "img_a"
    assert pairs[0].image_id_2 == "img_b"
    assert pairs[0].pair_id == make_pair_id("img_b", "img_a")
    assert canonicalize_image_ids("img_b", "img_a") == ("img_a", "img_b")

    duplicate = Pair(
        pair_id=pairs[0].pair_id,
        image_id_1="img_b",
        image_id_2="img_a",
        group_id_1="g1",
        group_id_2="g1",
        category_id_1=0,
        category_id_2=0,
        category_1="Bracelet",
        category_2="Bracelet",
        split="train",
        label=1,
        pair_type="positive",
        negative_type=None,
    )
    with pytest.raises(PairGenerationError, match="not canonical"):
        validate_pairs([duplicate], samples)


def test_cross_split_pair_is_rejected(tmp_path: Path) -> None:
    train = _sample(tmp_path, "a", "g_train", split="train")
    valid = _sample(tmp_path, "b", "g_valid", split="valid", category="Ring")
    with pytest.raises(PairGenerationError, match="Cross-split pair"):
        pair_from_samples(
            train,
            valid,
            label=0,
            pair_type="negative",
            negative_type="cross_category",
        )

    leaked = Pair(
        pair_id=make_pair_id("a", "b"),
        image_id_1="a",
        image_id_2="b",
        group_id_1="g_train",
        group_id_2="g_valid",
        category_id_1=0,
        category_id_2=4,
        category_1="Bracelet",
        category_2="Ring",
        split="train",
        label=0,
        pair_type="negative",
        negative_type="cross_category",
    )
    with pytest.raises(PairGenerationError, match="Cross-split pair"):
        validate_pairs([leaked], [train, valid])


def test_positive_and_negative_labels(tmp_path: Path) -> None:
    samples = [
        _sample(tmp_path, "a1", "ga", category="Bracelet"),
        _sample(tmp_path, "a2", "ga", category="Bracelet"),
        _sample(tmp_path, "b1", "gb", category="Ring"),
        _sample(tmp_path, "b2", "gb", category="Ring"),
    ]
    result = generate_pair_dataset(samples, _cfg(same_category_negative_ratio=0.0))
    positives = [pair for pair in result.pairs if pair.pair_type == "positive"]
    negatives = [pair for pair in result.pairs if pair.pair_type == "negative"]
    assert len(positives) == 2
    assert all(pair.label == 1 and pair.group_id_1 == pair.group_id_2 for pair in positives)
    assert all(pair.label == 0 and pair.group_id_1 != pair.group_id_2 for pair in negatives)


def test_determinism_same_seed(tmp_path: Path) -> None:
    samples = _diverse_samples(tmp_path)
    first = generate_pair_dataset(samples, _cfg(seed=2026))
    second = generate_pair_dataset(samples, _cfg(seed=2026))
    assert [pair.pair_id for pair in first.pairs] == [pair.pair_id for pair in second.pairs]
    assert [pair.unordered_key() for pair in first.pairs] == [
        pair.unordered_key() for pair in second.pairs
    ]


def test_different_seed_can_change_negatives(tmp_path: Path) -> None:
    samples = _diverse_samples(tmp_path)
    first = generate_pair_dataset(samples, _cfg(seed=1))
    second = generate_pair_dataset(samples, _cfg(seed=2))
    first_pos = {pair.pair_id for pair in first.pairs if pair.pair_type == "positive"}
    second_pos = {pair.pair_id for pair in second.pairs if pair.pair_type == "positive"}
    assert first_pos == second_pos
    first_neg = {pair.pair_id for pair in first.pairs if pair.pair_type == "negative"}
    second_neg = {pair.pair_id for pair in second.pairs if pair.pair_type == "negative"}
    assert first_neg != second_neg


def test_balanced_ratio(tmp_path: Path) -> None:
    samples = _diverse_samples(tmp_path)
    result = generate_pair_dataset(
        samples,
        _cfg(positive_negative_ratio=1.0, same_category_negative_ratio=0.5),
    )
    n_pos = result.report["positive_pairs"]
    n_neg = result.report["negative_pairs"]
    assert n_pos == n_neg
    assert result.report["same_category_negative_count"] == n_neg // 2
    assert result.report["cross_category_negative_count"] == n_neg - n_neg // 2


def test_category_and_category_id_match_manifest(tmp_path: Path) -> None:
    samples = _diverse_samples(tmp_path)
    by_id = {sample.image_id: sample for sample in samples}
    result = generate_pair_dataset(samples, _cfg())
    for pair in result.pairs:
        left = by_id[pair.image_id_1]
        right = by_id[pair.image_id_2]
        assert pair.category_1 == left.category
        assert pair.category_2 == right.category
        assert pair.category_id_1 == left.category_id == CATEGORY_TO_ID[left.category]
        assert pair.category_id_2 == right.category_id == CATEGORY_TO_ID[right.category]


def test_singleton_groups_participate_in_negatives(tmp_path: Path) -> None:
    samples = [
        _sample(tmp_path, "a1", "ga", category="Bracelet"),
        _sample(tmp_path, "a2", "ga", category="Bracelet"),
        _sample(tmp_path, "solo", "g_solo", category="Bracelet"),
    ]
    result = generate_pair_dataset(samples, _cfg(same_category_negative_ratio=1.0))
    negatives = [pair for pair in result.pairs if pair.pair_type == "negative"]
    assert len(negatives) == 1
    involved = {negatives[0].image_id_1, negatives[0].image_id_2}
    assert "solo" in involved


def test_missing_image_id_is_rejected(tmp_path: Path) -> None:
    samples = [_sample(tmp_path, "a", "g1"), _sample(tmp_path, "b", "g1")]
    pair = generate_positive_pairs(samples)[0]
    broken = Pair(
        pair_id=make_pair_id("a", "missing"),
        image_id_1="a",
        image_id_2="missing",
        group_id_1="g1",
        group_id_2="g1",
        category_id_1=0,
        category_id_2=0,
        category_1="Bracelet",
        category_2="Bracelet",
        split="train",
        label=1,
        pair_type="positive",
        negative_type=None,
    )
    with pytest.raises(PairGenerationError, match="missing image_id"):
        validate_pairs([broken], samples)
    del pair


def test_invalid_group_is_rejected(tmp_path: Path) -> None:
    samples = [_sample(tmp_path, "a", "g1"), _sample(tmp_path, "b", "g1")]
    pair = generate_positive_pairs(samples)[0]
    broken = Pair(
        **{
            **pair.__dict__,
            "group_id_1": "not_in_manifest",
        }
    )
    with pytest.raises(PairGenerationError, match="does not match manifest group_id"):
        validate_pairs([broken], samples)


def test_invalid_split_is_rejected(tmp_path: Path) -> None:
    samples = [_sample(tmp_path, "a", "g1"), _sample(tmp_path, "b", "g1")]
    pair = generate_positive_pairs(samples)[0]
    broken = Pair(**{**pair.__dict__, "split": "holdout"})
    with pytest.raises(PairGenerationError, match="Invalid pair split"):
        validate_pairs([broken], samples)


def test_malformed_manifest_is_rejected(tmp_path: Path) -> None:
    path_a = write_rgb_image(tmp_path / "a.jpg")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,split\nimg,train\n", encoding="utf-8")
    with pytest.raises(DatasetIngestionError, match="missing required columns"):
        load_manifest(manifest, validate_files=False)
    del path_a


def test_configured_split_missing_from_manifest(tmp_path: Path) -> None:
    samples = [_sample(tmp_path, "a", "g1"), _sample(tmp_path, "b", "g1")]
    with pytest.raises(PairGenerationError, match="missing from the manifest"):
        generate_pair_dataset(
            samples,
            PairGenerationConfig(include_train=True, include_valid=True, include_test=False),
        )


def test_pairs_are_generated_independently_per_split(tmp_path: Path) -> None:
    samples = [
        _sample(tmp_path, "t1", "gt", split="train"),
        _sample(tmp_path, "t2", "gt", split="train"),
        _sample(tmp_path, "v1", "gv", split="valid", category="Ring"),
        _sample(tmp_path, "v2", "gv", split="valid", category="Ring"),
        _sample(tmp_path, "tx", "gtx", split="train", category="Necklace"),
        _sample(tmp_path, "vx", "gvx", split="valid", category="Necklace"),
    ]
    result = generate_pair_dataset(
        samples,
        PairGenerationConfig(
            include_train=True,
            include_valid=True,
            include_test=False,
            same_category_negative_ratio=0.0,
        ),
    )
    assert all(pair.split in {"train", "valid"} for pair in result.pairs)
    train_pairs = [pair for pair in result.pairs if pair.split == "train"]
    valid_pairs = [pair for pair in result.pairs if pair.split == "valid"]
    assert train_pairs and valid_pairs
    assert {pair.split for pair in train_pairs} == {"train"}
    assert {pair.split for pair in valid_pairs} == {"valid"}


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(PairGenerationError, match="At least one"):
        PairGenerationConfig(include_train=False, include_valid=False, include_test=False)
    with pytest.raises(PairGenerationError, match="must be in \\[0, 1\\]"):
        PairGenerationConfig(same_category_negative_ratio=1.5)
    with pytest.raises(PairGenerationError, match="must be >= 0"):
        PairGenerationConfig(positive_negative_ratio=-1.0)


def test_generation_does_not_rewrite_manifest(tmp_path: Path) -> None:
    samples = _diverse_samples(tmp_path)
    rows = [
        {
            "image_id": sample.image_id,
            "image_path": str(sample.image_path),
            "group_id": sample.group_id,
            "category": sample.category,
            "category_id": str(sample.category_id),
            "split": sample.split,
            "source": SOURCE_DATASET1,
        }
        for sample in samples
    ]
    manifest = write_manifest(tmp_path / "manifest.csv", rows)
    before = manifest.read_bytes()
    generate_pair_dataset(samples, _cfg())
    assert manifest.read_bytes() == before


def test_generator_does_not_touch_global_random(tmp_path: Path) -> None:
    import random

    samples = _diverse_samples(tmp_path)
    random.seed(12345)
    before = [random.random() for _ in range(5)]
    random.seed(12345)
    generate_pair_dataset(samples, _cfg(seed=99))
    after = [random.random() for _ in range(5)]
    assert before == after


def _diverse_samples(tmp_path: Path) -> list[Sample]:
    """Enough groups/categories for 1:1 sampling with both negative types."""
    samples: list[Sample] = []
    # 4 Bracelet groups of size 2 => 4 positives
    for index in range(4):
        group_id = f"brace_{index}"
        samples.append(_sample(tmp_path, f"b{index}a", group_id, category="Bracelet"))
        samples.append(_sample(tmp_path, f"b{index}b", group_id, category="Bracelet"))
    # 4 Ring groups of size 2 => 4 positives
    for index in range(4):
        group_id = f"ring_{index}"
        samples.append(_sample(tmp_path, f"r{index}a", group_id, category="Ring"))
        samples.append(_sample(tmp_path, f"r{index}b", group_id, category="Ring"))
    # Singletons that expand the negative pool
    for index in range(6):
        samples.append(
            _sample(tmp_path, f"solo_b{index}", f"solo_b_{index}", category="Bracelet")
        )
        samples.append(
            _sample(tmp_path, f"solo_r{index}", f"solo_r_{index}", category="Ring")
        )
    return samples
