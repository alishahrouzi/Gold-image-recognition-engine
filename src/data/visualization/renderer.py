"""Read-only matplotlib rendering for Dataset 1 QA figures.

Displays original RGB pixels (and in-memory S1.9 augmented RGB). Source
files are never written. Normalized tensors are not saved as images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..loaders.image_loader import load_rgb_image
from ..preprocessing.augmentation import AugmentationConfig, TrainingAugmentor
from ..types import Sample
from .config import DISPLAY_MAX_EDGE
from .types import GroupPanel, PairPanel

PathLike = Union[str, Path]


def render_group_panels(panels: Sequence[GroupPanel], output_path: PathLike, title: str) -> Path:
    if not panels:
        return _empty_figure(output_path, title)
    max_cols = max(len(panel.samples) for panel in panels)
    fig, axes = plt.subplots(
        len(panels),
        max_cols,
        figsize=(3.2 * max_cols, 3.4 * len(panels)),
        squeeze=False,
    )
    for row, panel in enumerate(panels):
        for col in range(max_cols):
            axis = axes[row][col]
            if col >= len(panel.samples):
                axis.axis("off")
                continue
            sample = panel.samples[col]
            _show_rgb(axis, _load_display_image(sample), _sample_caption(sample))
        axes[row][0].set_ylabel(
            f"GROUP: {panel.group_id}\n{panel.category} / {panel.split}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
            labelpad=48,
        )
    fig.suptitle(title, fontsize=12)
    return _save(fig, output_path)


def render_sample_grid(
    samples: Sequence[Sample],
    output_path: PathLike,
    title: str,
    columns: int = 5,
) -> Path:
    if not samples:
        return _empty_figure(output_path, title)
    columns = max(1, columns)
    rows = (len(samples) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.1 * columns, 3.4 * rows),
        squeeze=False,
    )
    for index, sample in enumerate(samples):
        axis = axes[index // columns][index % columns]
        _show_rgb(axis, _load_display_image(sample), _sample_caption(sample))
    for index in range(len(samples), rows * columns):
        axes[index // columns][index % columns].axis("off")
    fig.suptitle(title, fontsize=12)
    return _save(fig, output_path)


def render_pair_panels(
    panels: Sequence[PairPanel],
    output_path: PathLike,
    title: str,
) -> Path:
    if not panels:
        return _empty_figure(output_path, title)
    fig, axes = plt.subplots(
        len(panels),
        2,
        figsize=(7.2, 3.5 * len(panels)),
        squeeze=False,
    )
    for row, panel in enumerate(panels):
        pair = panel.pair
        caption_a = (
            f"A  {pair.image_id_1}\n"
            f"group={pair.group_id_1}  cat={pair.category_1}"
        )
        caption_b = (
            f"B  {pair.image_id_2}\n"
            f"group={pair.group_id_2}  cat={pair.category_2}"
        )
        _show_rgb(axes[row][0], _load_display_image(panel.sample_a), caption_a)
        _show_rgb(axes[row][1], _load_display_image(panel.sample_b), caption_b)
        axes[row][0].set_ylabel(
            f"{pair.pair_id}\nlabel={pair.label} {pair.pair_type}\n"
            f"{pair.negative_type or ''} split={pair.split}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=7,
            labelpad=64,
        )
    fig.suptitle(title, fontsize=12)
    return _save(fig, output_path)


def render_augmentation_panels(
    samples: Sequence[Sample],
    output_path: PathLike,
    augmentor: TrainingAugmentor,
    title: str,
) -> Path:
    if not samples:
        return _empty_figure(output_path, title)
    for sample in samples:
        if sample.split != "train":
            raise ValueError(
                f"Augmentation visualization is train-only, got {sample.image_id!r} "
                f"split={sample.split!r}."
            )
    fig, axes = plt.subplots(
        len(samples),
        2,
        figsize=(7.2, 3.5 * len(samples)),
        squeeze=False,
    )
    for row, sample in enumerate(samples):
        original = load_rgb_image(sample.image_path)
        augmented = augmentor(original)
        _show_rgb(axes[row][0], _for_display(original), f"Original\n{_sample_caption(sample)}")
        _show_rgb(axes[row][1], _for_display(augmented), f"Augmented (S1.9)\n{_sample_caption(sample)}")
    fig.suptitle(title, fontsize=12)
    return _save(fig, output_path)


def build_train_augmentor(seed: int) -> TrainingAugmentor:
    """Reuse the S1.9 default jewelry-safe training contract."""
    return TrainingAugmentor(AugmentationConfig(seed=seed), seed=seed)


def _load_display_image(sample: Sample) -> Image.Image:
    return _for_display(load_rgb_image(sample.image_path))


def _for_display(image: Image.Image) -> Image.Image:
    display = image.copy()
    display.thumbnail((DISPLAY_MAX_EDGE, DISPLAY_MAX_EDGE))
    return display


def _sample_caption(sample: Sample) -> str:
    return (
        f"{sample.image_id}\n"
        f"group={sample.group_id}  {sample.category}  {sample.split}"
    )


def _show_rgb(axis, image: Image.Image, caption: str) -> None:
    axis.imshow(np.asarray(image.convert("RGB")))
    axis.set_title(caption, fontsize=7)
    axis.set_xticks([])
    axis.set_yticks([])


def _empty_figure(output_path: PathLike, title: str) -> Path:
    fig, axis = plt.subplots(figsize=(8, 2))
    axis.axis("off")
    axis.text(0.5, 0.5, "No samples selected", ha="center", va="center")
    fig.suptitle(title, fontsize=12)
    return _save(fig, output_path)


def _save(fig, output_path: PathLike) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
