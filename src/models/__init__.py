"""Custom CNN encoder module (S2.1).

Preprocessed image tensor → embedding. No training loop, loss, similarity,
ranking, or pretrained backbones.
"""

from .config import (
    DEFAULT_BLOCK_CHANNELS,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    EncoderConfig,
)
from .encoder import CustomCNNEncoder, count_parameters, estimate_parameter_bytes
from .errors import EncoderConfigError, EncoderInputError, ModelError
from .interfaces import Encoder, validate_encoder_input

__all__ = [
    "DEFAULT_BLOCK_CHANNELS",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_INPUT_CHANNELS",
    "DEFAULT_INPUT_HEIGHT",
    "DEFAULT_INPUT_WIDTH",
    "CustomCNNEncoder",
    "Encoder",
    "EncoderConfig",
    "EncoderConfigError",
    "EncoderInputError",
    "ModelError",
    "count_parameters",
    "estimate_parameter_bytes",
    "validate_encoder_input",
]
