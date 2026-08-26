"""Read-only Dataset 1 visualization pipeline (S1.11).

Loads the existing manifest and pair CSV, samples deterministically,
validates selected rows, and writes QA figures. Does not modify the
dataset, manifest, pair CSV, or source images.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from ..constants import SOURCE_DATASET1
from ..errors import VisualizationError
from ..loaders.manifest import load_manifest
from ..pairs.generator import load_pairs_csv
from ..pairs.types import Pair
from ..types import Sample
from .config import VisualizationConfig
from .renderer import (
    build_train_augmentor,
    render_augmentation_panels,
    render_group_panels,
    render_pair_panels,
    render_sample_grid,
)
from .report import build_visualization_report, write_visualization_report
from .sampler import build_selection
from .types import VisualizationSelection
from .validation import raise_if_invalid, validate_selection

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass(frozen=True)
class VisualizationResult:
    """Rendered figures plus the audit report that describes them."""

    selection: VisualizationSelection
    report: Dict[str, Any]
    output_files: Dict[str, Path]
    config: VisualizationConfig


def generate_dataset_visualizations(
    *,
    samples: Sequence[Sample],
    pairs: Sequence[Pair],
    output_dir: PathLike,
    config: Optional[VisualizationConfig] = None,
    dataset: str = SOURCE_DATASET1,
    manifest: Optional[PathLike] = None,
    pair_source: Optional[PathLike] = None,
) -> VisualizationResult:
    """Select, validate, and render visualization figures.

    Raises VisualizationError if selected pair metadata is invalid. Figures
    are not written in that case; the JSON report is still written so the
    failure is auditable.
    """
    config = config or VisualizationConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = Random(config.seed)
    selection = build_selection(samples, pairs, config, rng)
    errors = validate_selection(selection, samples)

    checksums = _snapshot_sources(selection, manifest=manifest, pair_source=pair_source)

    report_path = output_path / "visualization_report.json"
    if errors:
        report = build_visualization_report(
            dataset=dataset,
            manifest=manifest or "",
            pair_source=pair_source or "",
            output_dir=output_path,
            config=config,
            selection=selection,
            validation_errors=errors,
            source_files_unchanged=_unchanged(checksums),
            output_files={},
        )
        write_visualization_report(report, report_path)
        raise_if_invalid(errors)

    output_files = _render_all(selection, output_path, config.seed)
    source_files_unchanged = _unchanged(checksums)
    if not source_files_unchanged:
        raise VisualizationError(
            "Source files, manifest, or pair CSV changed during visualization."
        )

    report = build_visualization_report(
        dataset=dataset,
        manifest=manifest or "",
        pair_source=pair_source or "",
        output_dir=output_path,
        config=config,
        selection=selection,
        validation_errors=[],
        source_files_unchanged=True,
        output_files=output_files,
    )
    output_files["visualization_report.json"] = write_visualization_report(report, report_path)
    report["output_files"] = {name: str(path) for name, path in output_files.items()}
    write_visualization_report(report, report_path)

    logger.info("Wrote dataset visualization to %s", output_path)
    return VisualizationResult(
        selection=selection,
        report=report,
        output_files=output_files,
        config=config,
    )


def load_visualization_inputs(
    manifest_path: PathLike,
    pairs_path: PathLike,
    *,
    dataset_root: Optional[PathLike] = None,
    validate_files: bool = True,
) -> tuple[list[Sample], list[Pair]]:
    """Load existing manifest + pair CSV. Does not regenerate pairs."""
    samples = load_manifest(
        manifest_path,
        dataset_root=dataset_root,
        validate_files=validate_files,
    )
    pairs = load_pairs_csv(pairs_path)
    return samples, pairs


def _render_all(
    selection: VisualizationSelection,
    output_dir: Path,
    seed: int,
) -> Dict[str, Path]:
    files: Dict[str, Path] = {
        "train_samples.png": render_group_panels(
            selection.train_groups,
            output_dir / "train_samples.png",
            "Train groups (group-aware views)",
        ),
        "valid_samples.png": render_sample_grid(
            selection.valid_samples,
            output_dir / "valid_samples.png",
            "Validation samples (preprocessing only, no augmentation)",
        ),
        "test_samples.png": render_sample_grid(
            selection.test_samples,
            output_dir / "test_samples.png",
            "Test samples (preprocessing only, no augmentation)",
        ),
        "positive_pairs.png": render_pair_panels(
            selection.positive_pairs,
            output_dir / "positive_pairs.png",
            "Positive pairs (same group_id / same product)",
        ),
        "negative_pairs.png": render_pair_panels(
            list(selection.same_category_negatives)
            + list(selection.cross_category_negatives),
            output_dir / "negative_pairs.png",
            "Negative pairs (same-category first, then cross-category)",
        ),
        "negative_pairs_same_category.png": render_pair_panels(
            selection.same_category_negatives,
            output_dir / "negative_pairs_same_category.png",
            "Same-category negatives (same category, different group_id)",
        ),
        "negative_pairs_cross_category.png": render_pair_panels(
            selection.cross_category_negatives,
            output_dir / "negative_pairs_cross_category.png",
            "Cross-category negatives (different category, different group_id)",
        ),
        "augmentation_samples.png": render_augmentation_panels(
            selection.augmentation_samples,
            output_dir / "augmentation_samples.png",
            build_train_augmentor(seed),
            "Training augmentation (S1.9) — original vs augmented",
        ),
    }
    return files


def _snapshot_sources(
    selection: VisualizationSelection,
    *,
    manifest: Optional[PathLike],
    pair_source: Optional[PathLike],
) -> Dict[Path, str]:
    paths = []
    if manifest:
        paths.append(Path(manifest))
    if pair_source:
        paths.append(Path(pair_source))
    for sample in selection.all_samples():
        paths.append(Path(sample.image_path))
    checksums: Dict[Path, str] = {}
    for path in paths:
        if path.is_file():
            checksums[path] = _digest(path)
    return checksums


def _unchanged(checksums: Mapping[Path, str]) -> bool:
    return all(path.is_file() and _digest(path) == digest for path, digest in checksums.items())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
