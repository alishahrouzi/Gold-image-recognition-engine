"""PyTorch Dataset adapter for the existing S1.10 Pair contract.

This module does not generate pairs. It resolves the image IDs already stored
in Pair records to manifest Samples, loads both images, and applies the shared
preprocessing pipeline. The S1.10 pair label convention is preserved in the
stored Pair objects: label=1 means positive and label=0 means negative.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..loaders.image_loader import load_rgb_image
from ..preprocessing.augmentation import AugmentationConfig
from ..preprocessing.pipeline import ImagePreprocessor
from ..types import Sample
from .types import Pair, PAIR_TYPE_NEGATIVE, PAIR_TYPE_POSITIVE


class PairDataset(Dataset):
    """Load image tensors for an existing collection of validated pairs.

    The dataset is intentionally read-only with respect to the manifest and
    pair records. Both images are passed through the same ImagePreprocessor.
    Training augmentation is optional and should only be enabled for the
    training split; validation remains deterministic by default.
    """

    def __init__(
        self,
        pairs: Sequence[Pair],
        samples: Sequence[Sample],
        *,
        preprocessor: Optional[ImagePreprocessor] = None,
        augmentation: Optional[AugmentationConfig] = None,
    ) -> None:
        if not pairs:
            raise ValueError("PairDataset requires at least one pair.")
        if not samples:
            raise ValueError("PairDataset requires at least one manifest sample.")

        self.pairs = tuple(pairs)
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.augmentation = augmentation
        self._samples_by_id: Mapping[str, Sample] = {
            sample.image_id: sample for sample in samples
        }
        if len(self._samples_by_id) != len(samples):
            raise ValueError("Manifest samples contain duplicate image_id values.")

        missing = sorted(
            {
                image_id
                for pair in self.pairs
                for image_id in (pair.image_id_1, pair.image_id_2)
                if image_id not in self._samples_by_id
            }
        )
        if missing:
            raise ValueError(
                f"PairDataset cannot resolve image IDs from the manifest: {missing[:10]}"
            )

        for pair in self.pairs:
            if pair.pair_type not in {PAIR_TYPE_POSITIVE, PAIR_TYPE_NEGATIVE}:
                raise ValueError(f"Unsupported pair_type {pair.pair_type!r}.")
            if pair.label not in {0, 1}:
                raise ValueError(f"Pair label must be 0 or 1, got {pair.label!r}.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.pairs[index]
        sample_a = self._samples_by_id[pair.image_id_1]
        sample_b = self._samples_by_id[pair.image_id_2]

        image_a = load_rgb_image(sample_a.image_path)
        image_b = load_rgb_image(sample_b.image_path)

        if self.augmentation is not None:
            image_a = self.augmentation(image_a)
            image_b = self.augmentation(image_b)

        tensor_a = self.preprocessor(image_a)
        tensor_b = self.preprocessor(image_b)

        # S1.10 uses 1=positive / 0=negative. S3.3 ContrastiveLoss uses
        # 0=positive / 1=negative, so the training step performs the explicit
        # semantic conversion rather than mutating the source Pair contract.
        loss_label = 1 - pair.label
        return {
            "image_a": tensor_a,
            "image_b": tensor_b,
            "label": torch.tensor(loss_label, dtype=torch.float32),
            "pair_label": torch.tensor(pair.label, dtype=torch.int64),
            "pair_id": pair.pair_id,
            "split": pair.split,
        }
