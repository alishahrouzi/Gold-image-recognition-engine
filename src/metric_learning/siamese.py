"""S3.1 Siamese architecture for shared image embedding extraction.

The network deliberately contains no loss, pair sampler, optimizer, or
retrieval logic. Both inputs are encoded by the exact same
``EncoderWithEmbeddingHead`` instance so gradients and learned parameters are
shared across the two branches.
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from models import CustomCNNEncoder, EmbeddingHead, EmbeddingHeadConfig, EncoderConfig
from models.embedding_head import EncoderWithEmbeddingHead


class SiameseNetwork(nn.Module):
    """Two-input Siamese network with a single shared embedding backbone.

    Parameters
    ----------
    encoder:
        Optional custom CNN encoder. If omitted, ``CustomCNNEncoder()`` is
        created using the project's current S2 architecture defaults.
    head:
        Optional embedding head. If omitted, a 128-D head is created against
        the encoder's feature width.
    encoder_config:
        Optional encoder configuration used only when ``encoder`` is omitted.
    embedding_config:
        Optional embedding-head configuration used only when ``head`` is
        omitted. Its feature dimension must match the encoder output.

    Notes
    -----
    The same ``backbone`` object processes both images. There are intentionally
    no duplicated encoder/head modules and no loss computation here; future S3
    tasks can attach Contrastive Loss, Triplet Loss, or other objectives to the
    returned embedding pair.
    """

    def __init__(
        self,
        encoder: Optional[CustomCNNEncoder] = None,
        head: Optional[EmbeddingHead] = None,
        *,
        encoder_config: Optional[EncoderConfig] = None,
        embedding_config: Optional[EmbeddingHeadConfig] = None,
    ) -> None:
        super().__init__()

        if encoder is None:
            encoder = CustomCNNEncoder(config=encoder_config)
        elif encoder_config is not None:
            raise ValueError("Pass either encoder or encoder_config, not both.")

        if head is None:
            if embedding_config is None:
                head = EmbeddingHead(feature_dim=encoder.feature_dim, embedding_dim=128)
            else:
                head = EmbeddingHead(config=embedding_config)
        elif embedding_config is not None:
            raise ValueError("Pass either head or embedding_config, not both.")

        self.backbone = EncoderWithEmbeddingHead(encoder, head)

    @property
    def embedding_dim(self) -> int:
        """Width of each output embedding."""
        return self.backbone.embedding_dim

    @property
    def feature_dim(self) -> int:
        """Width of the shared encoder's raw feature vector."""
        return self.backbone.feature_dim

    def encode(self, images: Tensor) -> Tensor:
        """Encode one image batch through the shared backbone."""
        return self.backbone(images)

    def forward(self, image_a: Tensor, image_b: Tensor) -> tuple[Tensor, Tensor]:
        """Return embeddings for two image batches using shared parameters."""
        embedding_a = self.backbone(image_a)
        embedding_b = self.backbone(image_b)
        return embedding_a, embedding_b
