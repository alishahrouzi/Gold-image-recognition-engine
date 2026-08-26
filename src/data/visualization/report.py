"""Write the S1.11 visualization audit report. Does not modify source data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

from .config import VisualizationConfig
from .types import VisualizationSelection

PathLike = Union[str, Path]


def build_visualization_report(
    *,
    dataset: str,
    manifest: PathLike,
    pair_source: PathLike,
    output_dir: PathLike,
    config: VisualizationConfig,
    selection: VisualizationSelection,
    validation_errors: Sequence[str],
    source_files_unchanged: bool,
    output_files: Mapping[str, PathLike],
) -> Dict[str, Any]:
    errors = list(validation_errors)
    pairs_valid = not errors
    return {
        "dataset": dataset,
        "manifest": str(Path(manifest)),
        "pair_source": str(Path(pair_source)),
        "output_dir": str(Path(output_dir)),
        "seed": config.seed,
        "sampling_configuration": dict(config.as_loggable_dict()),
        "train": {
            "requested": config.train_samples,
            "selected": len(selection.train_groups),
            "unit": "groups",
            "images_shown": sum(len(panel.samples) for panel in selection.train_groups),
            "multi_image_groups": sum(
                1 for panel in selection.train_groups if len(panel.samples) >= 2
            ),
        },
        "valid": {
            "requested": config.valid_samples,
            "selected": len(selection.valid_samples),
            "unit": "images",
            "augmentation": False,
        },
        "test": {
            "requested": config.test_samples,
            "selected": len(selection.test_samples),
            "unit": "images",
            "augmentation": False,
        },
        "positive_pairs": {
            "requested": config.positive_pairs,
            "selected": len(selection.positive_pairs),
            "validated": pairs_valid,
        },
        "negative_pairs": {
            "requested": config.negative_pairs,
            "selected": len(selection.same_category_negatives)
            + len(selection.cross_category_negatives),
            "validated": pairs_valid,
            "same_category_selected": len(selection.same_category_negatives),
            "cross_category_selected": len(selection.cross_category_negatives),
        },
        "augmentation": {
            "requested": config.augmentation_samples,
            "selected": len(selection.augmentation_samples),
            "role": "train",
            "implementation": "data.preprocessing.augmentation.TrainingAugmentor",
            "valid_test_augmented": False,
        },
        "validation_errors": errors,
        "source_files_unchanged": source_files_unchanged,
        "output_files": {name: str(Path(path)) for name, path in output_files.items()},
    }


def write_visualization_report(report: Mapping[str, Any], output_path: PathLike) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path
