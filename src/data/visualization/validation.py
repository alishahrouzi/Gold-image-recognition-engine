"""Fail-loud QA checks for selected visualization samples and pairs."""

from __future__ import annotations

from typing import List, Sequence

from ..errors import DatasetIngestionError, PairGenerationError, VisualizationError
from ..loaders.image_loader import load_rgb_image
from ..pairs.types import (
    NEGATIVE_TYPE_CROSS_CATEGORY,
    NEGATIVE_TYPE_SAME_CATEGORY,
    PAIR_TYPE_NEGATIVE,
    PAIR_TYPE_POSITIVE,
    Pair,
)
from ..pairs.validation import validate_pairs
from ..types import Sample
from .types import PairPanel, VisualizationSelection


def validate_image_ready(sample: Sample) -> None:
    """Require the file to exist, decode, and convert to RGB. Read-only."""
    try:
        image = load_rgb_image(sample.image_path)
    except DatasetIngestionError as exc:
        raise VisualizationError(
            f"Image {sample.image_id!r} failed QA: {exc}"
        ) from exc
    if image.mode != "RGB":
        raise VisualizationError(
            f"Image {sample.image_id!r} did not convert to RGB (mode={image.mode!r})."
        )


def validate_positive_pair(pair: Pair, sample_a: Sample, sample_b: Sample) -> None:
    if pair.pair_type != PAIR_TYPE_POSITIVE or pair.label != 1:
        raise VisualizationError(
            f"Pair {pair.pair_id} is not a positive pair (type={pair.pair_type!r}, "
            f"label={pair.label})."
        )
    if pair.group_id_1 != pair.group_id_2 or sample_a.group_id != sample_b.group_id:
        raise VisualizationError(
            f"Positive pair {pair.pair_id} must have the same group_id "
            f"({pair.group_id_1!r} vs {pair.group_id_2!r})."
        )
    if pair.category_1 != pair.category_2 or sample_a.category != sample_b.category:
        raise VisualizationError(
            f"Positive pair {pair.pair_id} must have the same category "
            f"({pair.category_1!r} vs {pair.category_2!r})."
        )
    _assert_split_isolated(pair, sample_a, sample_b)


def validate_negative_pair(pair: Pair, sample_a: Sample, sample_b: Sample) -> None:
    if pair.pair_type != PAIR_TYPE_NEGATIVE or pair.label != 0:
        raise VisualizationError(
            f"Pair {pair.pair_id} is not a negative pair (type={pair.pair_type!r}, "
            f"label={pair.label})."
        )
    if pair.group_id_1 == pair.group_id_2 or sample_a.group_id == sample_b.group_id:
        raise VisualizationError(
            f"Negative pair {pair.pair_id} must have different group_ids."
        )
    if pair.negative_type == NEGATIVE_TYPE_SAME_CATEGORY:
        if pair.category_1 != pair.category_2 or sample_a.category != sample_b.category:
            raise VisualizationError(
                f"same_category negative {pair.pair_id} has different categories."
            )
    elif pair.negative_type == NEGATIVE_TYPE_CROSS_CATEGORY:
        if pair.category_1 == pair.category_2 or sample_a.category == sample_b.category:
            raise VisualizationError(
                f"cross_category negative {pair.pair_id} has the same category."
            )
    else:
        raise VisualizationError(
            f"Negative pair {pair.pair_id} has invalid negative_type "
            f"{pair.negative_type!r}."
        )
    _assert_split_isolated(pair, sample_a, sample_b)


def validate_selection(
    selection: VisualizationSelection,
    samples: Sequence[Sample],
) -> List[str]:
    """Validate selected images and pairs. Returns collected error messages.

    Callers must fail loudly when the returned list is non-empty. Invalid
    pair metadata is never silently visualized.
    """
    errors: List[str] = []

    for sample in _unique_samples(selection):
        try:
            validate_image_ready(sample)
        except VisualizationError as exc:
            errors.append(str(exc))

    for panel in selection.train_groups:
        if panel.split != "train":
            errors.append(f"Train panel {panel.group_id!r} has split {panel.split!r}.")
        sizes = {member.split for member in panel.samples}
        if sizes != {"train"}:
            errors.append(f"Train group {panel.group_id!r} is not split-isolated.")

    for sample in selection.valid_samples:
        if sample.split != "valid":
            errors.append(f"Valid sample {sample.image_id!r} has split {sample.split!r}.")
    for sample in selection.test_samples:
        if sample.split != "test":
            errors.append(f"Test sample {sample.image_id!r} has split {sample.split!r}.")
    for sample in selection.augmentation_samples:
        if sample.split != "train":
            errors.append(
                f"Augmentation sample {sample.image_id!r} is split {sample.split!r}; "
                "augmentation is train-only."
            )

    selected_pairs = [panel.pair for panel in selection.all_pair_panels()]
    if selected_pairs:
        try:
            validate_pairs(selected_pairs, samples)
        except PairGenerationError as exc:
            errors.append(str(exc))

    for panel in selection.positive_pairs:
        try:
            validate_positive_pair(panel.pair, panel.sample_a, panel.sample_b)
        except VisualizationError as exc:
            errors.append(str(exc))
    for panel in selection.same_category_negatives:
        try:
            validate_negative_pair(panel.pair, panel.sample_a, panel.sample_b)
            if panel.pair.negative_type != NEGATIVE_TYPE_SAME_CATEGORY:
                errors.append(
                    f"Pair {panel.pair.pair_id} was selected as same-category "
                    f"but has negative_type={panel.pair.negative_type!r}."
                )
        except VisualizationError as exc:
            errors.append(str(exc))
    for panel in selection.cross_category_negatives:
        try:
            validate_negative_pair(panel.pair, panel.sample_a, panel.sample_b)
            if panel.pair.negative_type != NEGATIVE_TYPE_CROSS_CATEGORY:
                errors.append(
                    f"Pair {panel.pair.pair_id} was selected as cross-category "
                    f"but has negative_type={panel.pair.negative_type!r}."
                )
        except VisualizationError as exc:
            errors.append(str(exc))

    return errors


def raise_if_invalid(errors: Sequence[str]) -> None:
    if not errors:
        return
    details = "\n".join(f"- {message}" for message in errors)
    raise VisualizationError(
        "Visualization QA failed. Invalid pair or image metadata will not be "
        f"rendered.\n{details}"
    )


def _assert_split_isolated(pair: Pair, sample_a: Sample, sample_b: Sample) -> None:
    if sample_a.split != sample_b.split:
        raise VisualizationError(
            f"Pair {pair.pair_id} crosses splits "
            f"{sample_a.split!r} and {sample_b.split!r}."
        )
    if pair.split != sample_a.split:
        raise VisualizationError(
            f"Pair {pair.pair_id} split {pair.split!r} does not match "
            f"image split {sample_a.split!r}."
        )


def _unique_samples(selection: VisualizationSelection) -> List[Sample]:
    seen = set()
    unique: List[Sample] = []
    for sample in selection.all_samples():
        if sample.image_id in seen:
            continue
        seen.add(sample.image_id)
        unique.append(sample)
    return unique
