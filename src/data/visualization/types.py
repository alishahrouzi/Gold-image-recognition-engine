"""Selection records for S1.11 visualization. Metadata only until rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from ..pairs.types import Pair
from ..types import Sample


@dataclass(frozen=True)
class GroupPanel:
    """One product group and all of its selected images."""

    group_id: str
    category: str
    split: str
    samples: tuple[Sample, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("GroupPanel must contain at least one sample.")


@dataclass(frozen=True)
class PairPanel:
    """One pair CSV row resolved against manifest samples."""

    pair: Pair
    sample_a: Sample
    sample_b: Sample


@dataclass
class VisualizationSelection:
    """Deterministic sample chosen for QA figures."""

    train_groups: List[GroupPanel] = field(default_factory=list)
    valid_samples: List[Sample] = field(default_factory=list)
    test_samples: List[Sample] = field(default_factory=list)
    positive_pairs: List[PairPanel] = field(default_factory=list)
    same_category_negatives: List[PairPanel] = field(default_factory=list)
    cross_category_negatives: List[PairPanel] = field(default_factory=list)
    augmentation_samples: List[Sample] = field(default_factory=list)

    def all_samples(self) -> List[Sample]:
        samples: List[Sample] = []
        samples.extend(member for panel in self.train_groups for member in panel.samples)
        samples.extend(self.valid_samples)
        samples.extend(self.test_samples)
        samples.extend(self.augmentation_samples)
        for panel in (
            self.positive_pairs
            + self.same_category_negatives
            + self.cross_category_negatives
        ):
            samples.append(panel.sample_a)
            samples.append(panel.sample_b)
        return samples

    def all_pair_panels(self) -> Sequence[PairPanel]:
        return (
            list(self.positive_pairs)
            + list(self.same_category_negatives)
            + list(self.cross_category_negatives)
        )
