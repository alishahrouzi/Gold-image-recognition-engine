"""S2.3 embedding head: raw CNN features → Linear projection → L2-normalized embedding."""

from __future__ import annotations

from typing import Optional, Union

import torch.nn.functional as F
from torch import Tensor, nn

from .config import EmbeddingHeadConfig
from .encoder import CustomCNNEncoder
from .errors import EmbeddingHeadConfigError
from .interfaces import validate_embedding_head_input


class EmbeddingHead(nn.Module):
    """Project ``[B, feature_dim]`` to ``[B, embedding_dim]`` and L2-normalize.

    Baseline S2.3 structure (no MLP, dropout, BatchNorm, LayerNorm, or loss)::

        Linear(feature_dim → embedding_dim)
            → F.normalize(p=2, dim=1, eps=l2_eps)

    Device placement is external: ``head.to(device)`` then ``head(features)``.
    This module does not call ``.cuda()``.
    """

    def __init__(
        self,
        embedding_dim: Union[int, EmbeddingHeadConfig] = 128,
        *,
        feature_dim: int = 256,
        config: Optional[EmbeddingHeadConfig] = None,
        l2_eps: Optional[float] = None,
    ) -> None:
        super().__init__()
        if isinstance(embedding_dim, EmbeddingHeadConfig):
            if config is not None:
                raise EmbeddingHeadConfigError(
                    "Pass either EmbeddingHead(config) or EmbeddingHead(EmbeddingHeadConfig(...)), not both."
                )
            self._config = embedding_dim
        elif config is not None:
            self._config = config
        else:
            kwargs: dict = {"embedding_dim": embedding_dim, "feature_dim": feature_dim}
            if l2_eps is not None:
                kwargs["l2_eps"] = l2_eps
            self._config = EmbeddingHeadConfig(**kwargs)
        self.projection = nn.Linear(self._config.feature_dim, self._config.embedding_dim)

    @property
    def config(self) -> EmbeddingHeadConfig:
        return self._config

    @property
    def embedding_dim(self) -> int:
        return self._config.embedding_dim

    @property
    def feature_dim(self) -> int:
        return self._config.feature_dim

    def forward(self, features: Tensor) -> Tensor:
        validate_embedding_head_input(features, self._config)
        projected = self.projection(features)
        return F.normalize(projected, p=2, dim=1, eps=self._config.l2_eps)


class EncoderWithEmbeddingHead(nn.Module):
    """Compose encoder features with an embedding head.

    ``forward(images)`` returns L2-normalized embeddings. Raw GAP features
    remain available via ``encode_features``.
    """

    def __init__(self, encoder: CustomCNNEncoder, head: EmbeddingHead) -> None:
        super().__init__()
        if encoder.feature_dim != head.feature_dim:
            raise EmbeddingHeadConfigError(
                "Encoder feature_dim "
                f"({encoder.feature_dim}) must match EmbeddingHead feature_dim "
                f"({head.feature_dim})."
            )
        self.encoder = encoder
        self.head = head

    @property
    def embedding_dim(self) -> int:
        return self.head.embedding_dim

    @property
    def feature_dim(self) -> int:
        return self.encoder.feature_dim

    def encode_features(self, images: Tensor) -> Tensor:
        return self.encoder.encode_features(images)

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.encoder.encode_features(images))
