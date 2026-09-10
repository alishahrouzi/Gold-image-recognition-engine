"""S3.4 Siamese training integration.

This module connects the existing PairDataset, SiameseNetwork, S3.2 distance
abstraction, S3.3 ContrastiveLoss, and S2.5 Trainer without changing any of
their ownership boundaries.
"""

from __future__ import annotations

from pathlib import Path
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
) -> Tuple[DataLoader, DataLoader, Mapping[str, int]]:
    """Build train/valid DataLoaders from the existing manifest and pair CSV.

    Relative image_path values in the manifest are resolved against
    ``dataset_root``. Pair records are loaded as-is; no pair generation occurs
    here. Test pairs are intentionally ignored because S3.4 trains on train
    and validates on valid only. Test evaluation belongs to a later task.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0.")

    samples = load_manifest(
        manifest_path,
        dataset_root=dataset_root,
        validate_files=True,
    )
    pairs = load_pairs_csv(pairs_path)
    train_pairs = tuple(pair for pair in pairs if pair.split == "train")
    valid_pairs = tuple(pair for pair in pairs if pair.split == "valid")
    if not train_pairs:
        raise ValueError("Pair CSV contains no train pairs.")
    if not valid_pairs:
        raise ValueError("Pair CSV contains no valid pairs.")

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
    # ContrastiveLoss consumes distances semantically but also owns its
    # numerical formula. Keep the distance calculation explicit here so the
    # S3.2 abstraction remains part of the training pipeline.
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
    summary["distance"] = "euclidean"
    summary["loss"] = "contrastive"
    return summary
