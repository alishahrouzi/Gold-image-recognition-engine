"""Fail-loud validation for generated pairs.

Does not repair pairs. Any invariant failure raises PairGenerationError.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Sequence, Set, Tuple

from ..constants import ALLOWED_SPLITS, CATEGORY_TO_ID, SPLIT_ORDER
from ..errors import PairGenerationError
from ..types import Sample
from .types import (
    NEGATIVE_TYPE_CROSS_CATEGORY,
    NEGATIVE_TYPE_SAME_CATEGORY,
    PAIR_TYPE_NEGATIVE,
    PAIR_TYPE_POSITIVE,
    Pair,
    make_pair_id,
)

CheckMap = Dict[str, bool]


def validate_pairs(
    pairs: Sequence[Pair],
    samples: Sequence[Sample],
) -> CheckMap:
    """Validate pair invariants against the image-level manifest samples."""
    if not samples:
        raise PairGenerationError("Cannot validate pairs against an empty sample list.")

    by_image = _index_samples(samples)
    groups_to_splits = _group_splits(samples)

    checks: CheckMap = {
        "no_self_pairs": True,
        "no_duplicate_unordered_pairs": True,
        "no_duplicate_pair_ids": True,
        "positive_correctness": True,
        "negative_correctness": True,
        "split_isolation": True,
        "valid_image_ids": True,
        "valid_groups": True,
        "category_consistency": True,
        "no_cross_split_group_leakage": True,
        "canonical_pair_ids": True,
    }

    seen_keys: Set[Tuple[str, str]] = set()
    seen_ids: Set[str] = set()

    for pair in pairs:
        _check_self_pair(pair)
        _check_split_vocabulary(pair)
        _check_duplicate_key(pair, seen_keys)
        _check_duplicate_id(pair, seen_ids)
        _check_pair_id(pair)
        left, right = _resolve_images(pair, by_image)
        _check_split_isolation(pair, left, right)
        _check_groups(pair, left, right)
        _check_categories(pair, left, right)
        _check_labels(pair)
        _check_group_split_leakage(pair, left, right, groups_to_splits)

    return checks


def _index_samples(samples: Sequence[Sample]) -> Dict[str, Sample]:
    by_image: Dict[str, Sample] = {}
    for sample in samples:
        if sample.image_id in by_image:
            raise PairGenerationError(f"Duplicate image_id {sample.image_id!r} in samples.")
        by_image[sample.image_id] = sample
    return by_image


def _group_splits(samples: Sequence[Sample]) -> Mapping[str, Set[str]]:
    splits_by_group: Dict[str, Set[str]] = defaultdict(set)
    for sample in samples:
        splits_by_group[sample.group_id].add(sample.split)
    return splits_by_group


def _check_self_pair(pair: Pair) -> None:
    if pair.image_id_1 == pair.image_id_2:
        raise PairGenerationError(f"Self-pair is invalid: {pair.image_id_1!r}.")


def _check_split_vocabulary(pair: Pair) -> None:
    if pair.split not in ALLOWED_SPLITS:
        allowed = "/".join(SPLIT_ORDER)
        raise PairGenerationError(
            f"Invalid pair split {pair.split!r}. Expected {allowed}."
        )


def _check_duplicate_key(pair: Pair, seen_keys: Set[Tuple[str, str]]) -> None:
    key = pair.unordered_key()
    swapped = (pair.image_id_2, pair.image_id_1)
    if key in seen_keys or swapped in seen_keys:
        raise PairGenerationError(
            f"Duplicate unordered pair {pair.image_id_1!r} <-> {pair.image_id_2!r}."
        )
    if pair.image_id_1 > pair.image_id_2:
        raise PairGenerationError(
            f"Pair {pair.pair_id} is not canonical; image_id_1 must be <= image_id_2."
        )
    seen_keys.add(key)


def _check_duplicate_id(pair: Pair, seen_ids: Set[str]) -> None:
    if pair.pair_id in seen_ids:
        raise PairGenerationError(f"Duplicate pair_id {pair.pair_id!r}.")
    seen_ids.add(pair.pair_id)


def _check_pair_id(pair: Pair) -> None:
    expected = make_pair_id(pair.image_id_1, pair.image_id_2)
    if pair.pair_id != expected:
        raise PairGenerationError(
            f"pair_id {pair.pair_id!r} does not match canonical id {expected!r}."
        )


def _resolve_images(pair: Pair, by_image: Mapping[str, Sample]) -> Tuple[Sample, Sample]:
    try:
        left = by_image[pair.image_id_1]
        right = by_image[pair.image_id_2]
    except KeyError as exc:
        missing = exc.args[0]
        raise PairGenerationError(
            f"Pair {pair.pair_id} references missing image_id {missing!r}."
        ) from exc
    return left, right


def _check_split_isolation(pair: Pair, left: Sample, right: Sample) -> None:
    if left.split != right.split:
        raise PairGenerationError(
            f"Cross-split pair is invalid: {left.image_id!r} ({left.split}) "
            f"<-> {right.image_id!r} ({right.split})."
        )
    if pair.split != left.split:
        raise PairGenerationError(
            f"Pair {pair.pair_id} split {pair.split!r} does not match "
            f"image split {left.split!r}."
        )


def _check_groups(pair: Pair, left: Sample, right: Sample) -> None:
    if not pair.group_id_1 or not pair.group_id_2:
        raise PairGenerationError(f"Pair {pair.pair_id} has an empty group_id.")
    if pair.group_id_1 != left.group_id:
        raise PairGenerationError(
            f"Pair {pair.pair_id} group_id_1 {pair.group_id_1!r} does not match "
            f"manifest group_id {left.group_id!r}."
        )
    if pair.group_id_2 != right.group_id:
        raise PairGenerationError(
            f"Pair {pair.pair_id} group_id_2 {pair.group_id_2!r} does not match "
            f"manifest group_id {right.group_id!r}."
        )


def _check_categories(pair: Pair, left: Sample, right: Sample) -> None:
    for sample, category, category_id in (
        (left, pair.category_1, pair.category_id_1),
        (right, pair.category_2, pair.category_id_2),
    ):
        if category != sample.category:
            raise PairGenerationError(
                f"Pair {pair.pair_id} category {category!r} does not match "
                f"manifest category {sample.category!r} for {sample.image_id!r}."
            )
        if category not in CATEGORY_TO_ID:
            raise PairGenerationError(
                f"Pair {pair.pair_id} has unknown category {category!r}."
            )
        expected_id = CATEGORY_TO_ID[category]
        if category_id != expected_id or category_id != sample.category_id:
            raise PairGenerationError(
                f"Pair {pair.pair_id} category_id {category_id} does not match "
                f"category {category!r} (expected {expected_id}) for "
                f"{sample.image_id!r}."
            )


def _check_labels(pair: Pair) -> None:
    if pair.pair_type == PAIR_TYPE_POSITIVE:
        if pair.label != 1:
            raise PairGenerationError(
                f"Positive pair {pair.pair_id} must have label=1, got {pair.label}."
            )
        if pair.group_id_1 != pair.group_id_2:
            raise PairGenerationError(
                f"Positive pair {pair.pair_id} must have the same group_id."
            )
        if pair.negative_type is not None:
            raise PairGenerationError(
                f"Positive pair {pair.pair_id} must have negative_type=None."
            )
        return
    if pair.pair_type == PAIR_TYPE_NEGATIVE:
        if pair.label != 0:
            raise PairGenerationError(
                f"Negative pair {pair.pair_id} must have label=0, got {pair.label}."
            )
        if pair.group_id_1 == pair.group_id_2:
            raise PairGenerationError(
                f"Negative pair {pair.pair_id} must have different group_ids."
            )
        if pair.negative_type == NEGATIVE_TYPE_SAME_CATEGORY:
            if pair.category_1 != pair.category_2:
                raise PairGenerationError(
                    f"same_category negative {pair.pair_id} has different categories."
                )
            return
        if pair.negative_type == NEGATIVE_TYPE_CROSS_CATEGORY:
            if pair.category_1 == pair.category_2:
                raise PairGenerationError(
                    f"cross_category negative {pair.pair_id} has the same category."
                )
            return
        raise PairGenerationError(
            f"Negative pair {pair.pair_id} has invalid negative_type "
            f"{pair.negative_type!r}."
        )
    raise PairGenerationError(
        f"Pair {pair.pair_id} has invalid pair_type {pair.pair_type!r}."
    )


def _check_group_split_leakage(
    pair: Pair,
    left: Sample,
    right: Sample,
    groups_to_splits: Mapping[str, Set[str]],
) -> None:
    for group_id in (left.group_id, right.group_id):
        splits = groups_to_splits.get(group_id, set())
        if len(splits) > 1:
            split_list = ", ".join(name for name in SPLIT_ORDER if name in splits)
            raise PairGenerationError(
                f"Group {group_id!r} appears in multiple splits: {split_list}."
            )
        if splits and pair.split not in splits:
            raise PairGenerationError(
                f"Pair {pair.pair_id} uses group {group_id!r} from splits "
                f"{sorted(splits)} rather than {pair.split!r}."
            )
