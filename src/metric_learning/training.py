"""S3.4 Siamese training integration.

This module connects the existing PairDataset, SiameseNetwork, S3.2 distance
abstraction, S3.3 ContrastiveLoss, and S2.5 Trainer without changing any of
their ownership boundaries.
"""

from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Callable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data.loaders.manifest import load_manifest
from data.pairs.dataset import PairDataset
from data.pairs.generator import load_pairs_csv
from data.pairs.types import Pair
from data.preprocessing.augmentation import AugmentationConfig
from data.preprocessing.pipeline import ImagePreprocessor
from metric_learning.distance import DistanceFunction, EuclideanDistance
from metric_learning.losses import ContrastiveLoss
from metric_learning.siamese import SiameseNetwork
from training.config import TrainingConfig
from training.trainer import Trainer


Batch = Mapping[str, object]
StepFunction = Callable[[nn.Module, Batch], Tensor]
DEFAULT_PAIR_VALIDATION_FRACTION = 0.2


def _split_train_pairs_for_validation(
    train_pairs: Sequence[Pair],
    *,
    validation_fraction: float,
    seed: int,
) -> Tuple[Tuple[Pair, ...], Tuple[Pair, ...]]:
    """Create a deterministic pair-level holdout from the train pair pool.

    Dataset 1 valid/test splits contain one image per product group, so S1.10
    correctly generates no positive pairs for those splits. Contrastive
    validation therefore uses a deterministic holdout of the existing train
    pair pool rather than inventing positives or changing the dataset split.
    The split is stratified by the S1.10 pair label to preserve the 1:1
    positive/negative balance.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if not train_pairs:
        raise ValueError("Cannot split an empty train pair pool.")

    rng = Random(seed)
    by_label = {0: [], 1: []}
    for pair in train_pairs:
        if pair.label not in by_label:
            raise ValueError(f"Unexpected S1.10 pair label: {pair.label!r}")
        by_label[pair.label].append(pair)

    train_selected = []
    valid_selected = []
    for label in (0, 1):
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        n_valid = max(1, int(round(len(candidates) * validation_fraction)))
        if n_valid >= len(candidates):
            n_valid = len(candidates) - 1
        valid_selected.extend(candidates[:n_valid])
        train_selected.extend(candidates[n_valid:])

    if not train_selected or not valid_selected:
        raise ValueError("Pair validation split produced an empty partition.")

    return tuple(train_selected), tuple(valid_selected)


def build_pair_dataloaders(
    manifest_path: str | Path,
    pairs_path: str | Path,
    *,
    dataset_root: Optional[str | Path] = None,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    preprocessor: Optional[ImagePreprocessor] = None,
    train_augmentation: Optional[AugmentationConfig] = None,
    validation_fraction: float = DEFAULT_PAIR_VALIDATION_FRACTION,
    split_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Mapping[str, int]]:
    """Build train/validation DataLoaders from the existing pair CSV.

    Relative image_path values in the manifest are resolved against
    ``dataset_root``. Dataset 1's valid/test image splits are not used for
    contrastive validation because each group has only one image there.
    Instead, a deterministic, label-stratified holdout is taken from the
    existing train pair pool. Test pairs remain excluded from training.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0.")
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise ValueError("split_seed must be an integer.")

    samples = load_manifest(
        manifest_path,
        dataset_root=dataset_root,
        validate_files=True,
    )
    pairs = load_pairs_csv(pairs_path)
    source_train_pairs = tuple(pair for pair in pairs if pair.split == "train")
    if not source_train_pairs:
        raise ValueError("Pair CSV contains no train pairs.")

    train_pairs, valid_pairs = _split_train_pairs_for_validation(
        source_train_pairs,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )

    processor = preprocessor or ImagePreprocessor()
    train_dataset = PairDataset(
        train_pairs,
        samples,
        preprocessor=processor,
        augmentation=train_augmentation,
    )
    valid_dataset = PairDataset(
        valid_pairs,
        samples,
        preprocessor=processor,
        augmentation=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    counts = {
        "source_train_pairs": len(source_train_pairs),
        "train_pairs": len(train_pairs),
        "valid_pairs": len(valid_pairs),
        "test_pairs": sum(pair.split == "test" for pair in pairs),
    }
    return train_loader, valid_loader, counts


def contrastive_step(
    model: nn.Module,
    batch: Batch,
    *,
    distance: DistanceFunction,
    loss_fn: ContrastiveLoss,
) -> Tensor:
    """Run one Siamese forward → distance → contrastive-loss step."""
    if not isinstance(model, SiameseNetwork):
        raise TypeError("contrastive_step expects a SiameseNetwork model.")
    try:
        image_a = batch["image_a"]
        image_b = batch["image_b"]
        labels = batch["label"]
    except KeyError as exc:
        raise ValueError(f"Pair batch is missing required key: {exc.args[0]}") from exc
    if not torch.is_tensor(image_a) or not torch.is_tensor(image_b):
        raise TypeError("Pair batch image_a and image_b must be tensors.")
    if not torch.is_tensor(labels):
        raise TypeError("Pair batch label must be a tensor.")

    embedding_a, embedding_b = model(image_a, image_b)
    distances = distance.pairwise(embedding_a, embedding_b)
    positive = labels == 0
    negative = labels == 1
    if not torch.all(positive | negative):
        raise ValueError("Converted contrastive labels must contain only 0 or 1.")

    distances_squared = distances.square()
    margin = torch.as_tensor(loss_fn.margin, device=distances.device, dtype=distances.dtype)
    losses = (1.0 - labels) * distances_squared + labels * torch.relu(margin - distances).square()
    loss = losses.mean()
    if not torch.isfinite(loss):
        raise ValueError("Siamese contrastive loss became non-finite.")
    return loss


def make_contrastive_step(
    *,
    distance: Optional[DistanceFunction] = None,
    loss_fn: Optional[ContrastiveLoss] = None,
) -> StepFunction:
    """Create a Trainer-compatible closure for S3.4."""
    selected_distance = distance or EuclideanDistance()
    selected_loss = loss_fn or ContrastiveLoss()

    def step(model: nn.Module, batch: Batch) -> Tensor:
        return contrastive_step(
            model,
            batch,
            distance=selected_distance,
            loss_fn=selected_loss,
        )

    return step


def build_siamese_trainer(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    config: Optional[TrainingConfig] = None,
    model: Optional[SiameseNetwork] = None,
    distance: Optional[DistanceFunction] = None,
    loss_fn: Optional[ContrastiveLoss] = None,
) -> Trainer:
    """Construct the existing S2.5 Trainer for Siamese contrastive training."""
    selected_config = config or TrainingConfig()
    selected_model = model or SiameseNetwork()
    selected_loss = loss_fn or ContrastiveLoss(margin=selected_config.margin)
    if abs(selected_loss.margin - selected_config.margin) > 0.0:
        raise ValueError(
            "ContrastiveLoss margin must match TrainingConfig.margin for S3.4."
        )
    step = make_contrastive_step(
        distance=distance or EuclideanDistance(),
        loss_fn=selected_loss,
    )
    return Trainer(
        selected_model,
        train_loader,
        valid_loader,
        config=selected_config,
        train_step=step,
        valid_step=step,
    )


def train_siamese(
    manifest_path: str | Path,
    pairs_path: str | Path,
    *,
    dataset_root: Optional[str | Path] = None,
    config: Optional[TrainingConfig] = None,
    train_augmentation: Optional[AugmentationConfig] = None,
) -> Mapping[str, object]:
    """Build the S3.4 data path and train a Siamese model to completion."""
    selected_config = config or TrainingConfig()
    train_loader, valid_loader, counts = build_pair_dataloaders(
        manifest_path,
        pairs_path,
        dataset_root=dataset_root,
        batch_size=selected_config.batch_size,
        num_workers=selected_config.num_workers,
        pin_memory=selected_config.pin_memory,
        train_augmentation=train_augmentation,
        validation_fraction=DEFAULT_PAIR_VALIDATION_FRACTION,
        split_seed=selected_config.seed,
    )
    trainer = build_siamese_trainer(
        train_loader,
        valid_loader,
        config=selected_config,
    )
    summary = dict(trainer.fit())
    summary["pair_counts"] = dict(counts)
    summary["manifest"] = str(Path(manifest_path))
    summary["dataset_root"] = str(Path(dataset_root)) if dataset_root is not None else None
    summary["pairs_csv"] = str(Path(pairs_path))
    summary["pair_validation_fraction"] = DEFAULT_PAIR_VALIDATION_FRACTION
    summary["pair_validation_seed"] = selected_config.seed
    summary["distance"] = "euclidean"
    summary["loss"] = "contrastive"
    return summary
