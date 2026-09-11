"""Embedding extraction for baseline and Siamese retrieval models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.baselines.classifier import BaselineClassifier
from src.models.config import EncoderConfig
from src.models.embedding_head import EmbeddingHead, EmbeddingHeadConfig, CustomCNNEncoder, EncoderWithEmbeddingHead
from src.metric_learning.siamese import SiameseNetwork


class EmbeddingExtractor:
    """Extract normalized embeddings without exposing training-specific heads."""

    def __init__(self, model: EncoderWithEmbeddingHead, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def extract(self, images: Tensor) -> Tensor:
        if not torch.is_tensor(images) or images.ndim != 4:
            raise ValueError(f"images must have shape [B, C, H, W], got {getattr(images, 'shape', None)}")
        embeddings = self.model(images.to(self.device, non_blocking=True))
        if embeddings.ndim != 2:
            raise RuntimeError(f"Embedding model returned shape {tuple(embeddings.shape)}; expected [B, D].")
        return embeddings.detach().cpu()

    @torch.inference_mode()
    def extract_one(self, image: Tensor) -> Tensor:
        if image.ndim != 3:
            raise ValueError(f"image must have shape [C, H, W], got {tuple(image.shape)}")
        return self.extract(image.unsqueeze(0))[0]


def _build_retrieval_model(embedding_dim: int = 128) -> EncoderWithEmbeddingHead:
    encoder = CustomCNNEncoder(EncoderConfig())
    head = EmbeddingHead(
        EmbeddingHeadConfig(
            feature_dim=encoder.feature_dim,
            embedding_dim=embedding_dim,
        )
    )
    return EncoderWithEmbeddingHead(encoder, head)


def load_baseline_embedding_model(
    checkpoint_path: str | Path,
    *,
    embedding_dim: int = 128,
    num_classes: int = 5,
    device: torch.device | str = "cpu",
) -> EmbeddingExtractor:
    """Load the S2.6 best checkpoint and expose only its retrieval path."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    retrieval_model = _build_retrieval_model(embedding_dim=embedding_dim)
    wrapper = BaselineClassifier(retrieval_model, num_classes=num_classes)
    payload: Any = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Invalid S2.6 checkpoint: missing model_state_dict.")
    wrapper.load_state_dict(payload["model_state_dict"])
    return EmbeddingExtractor(wrapper.retrieval_model, device=device)


def load_siamese_embedding_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> EmbeddingExtractor:
    """Load a complete S3.4/S3.5 Siamese checkpoint for embedding extraction.

    The checkpoint is produced by the shared S2.5 CheckpointManager and stores
    the SiameseNetwork ``model_state_dict``. No optimizer, scheduler, RNG state,
    or training-only object is restored for evaluation.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    payload: Any = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid Siamese checkpoint: missing model_state_dict: {path}")
    model = SiameseNetwork()
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Siamese checkpoint architecture does not match the current model: {path}"
        ) from exc
    return EmbeddingExtractor(model.backbone, device=device)
