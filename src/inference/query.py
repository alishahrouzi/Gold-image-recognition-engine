"""S4.2 query validation and deterministic preprocessing for uploaded images."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Union
import sys
import torch
from PIL import Image, UnidentifiedImageError

<<<<<<< HEAD
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

=======
>>>>>>> 808d42c4040238a2233de45b911820f3c369c86b
from data.errors import PreprocessingError
from data.loaders.image_loader import load_rgb_image, to_rgb_image
from data.preprocessing.config import ImagePreprocessingConfig
from data.preprocessing.pipeline import ImagePreprocessor


QuerySource = Union[bytes, bytearray, memoryview, Image.Image, str, Path]


@dataclass(frozen=True)
class ProcessedQuery:
    """Validated query information plus the tensor consumed by S4.1."""

    tensor: torch.Tensor
    original_size: tuple[int, int]
    original_mode: str
    format: str | None
    source_bytes: int | None


class QueryProcessor:
    """Validate an uploaded image and apply the project's deterministic pipeline.

    This component is transport-agnostic: S4.4 can pass ``UploadFile.file``
    bytes or decoded image data without making S4.2 depend on FastAPI.
    """

    DEFAULT_MAX_BYTES = 10 * 1024 * 1024
    DEFAULT_MIN_WIDTH = 32
    DEFAULT_MIN_HEIGHT = 32

    def __init__(
        self,
        preprocessor: ImagePreprocessor | None = None,
        *,
        config: ImagePreprocessingConfig | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        min_width: int = DEFAULT_MIN_WIDTH,
        min_height: int = DEFAULT_MIN_HEIGHT,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive.")
        if min_width < 1 or min_height < 1:
            raise ValueError("min_width and min_height must be positive.")
        if preprocessor is not None and config is not None:
            raise ValueError("Provide either preprocessor or config, not both.")

        self.preprocessor = preprocessor or ImagePreprocessor(config)
        self.max_bytes = max_bytes
        self.min_width = min_width
        self.min_height = min_height

    def process(self, source: QuerySource) -> ProcessedQuery:
        """Validate, decode, RGB-convert, resize, normalize, and return one query."""
        image, source_bytes, image_format = self._load_source(source)
        original_size = image.size
        original_mode = image.mode

        self._validate_dimensions(original_size)
        rgb = to_rgb_image(image)
        tensor = self.preprocessor(rgb)

        if tensor.shape != torch.Size(self.preprocessor.config.output_shape):
            raise PreprocessingError(
                f"Query tensor has unexpected shape {tuple(tensor.shape)}."
            )
        if tensor.dtype != torch.float32:
            raise PreprocessingError(
                f"Query tensor has unexpected dtype {tensor.dtype}."
            )
        if not torch.isfinite(tensor).all():
            raise PreprocessingError("Query tensor contains non-finite values.")

        return ProcessedQuery(
            tensor=tensor,
            original_size=original_size,
            original_mode=original_mode,
            format=image_format,
            source_bytes=source_bytes,
        )

    def process_tensor(self, source: QuerySource) -> torch.Tensor:
        """Convenience method returning only the tensor needed by S4.1."""
        return self.process(source).tensor

    def _load_source(
        self, source: QuerySource
    ) -> tuple[Image.Image, int | None, str | None]:
        if isinstance(source, Image.Image):
            try:
                image_format = source.format
                image = source.copy()
                image.load()
            except Exception as exc:
                raise PreprocessingError("Unable to load the provided PIL image.") from exc
            return image, None, image_format

        if isinstance(source, (bytes, bytearray, memoryview)):
            payload = bytes(source)
            self._validate_byte_size(len(payload))
            if not payload:
                raise PreprocessingError("Uploaded image is empty.")
            try:
                with Image.open(BytesIO(payload)) as image:
                    image.load()
                    image_format = image.format
                    return image.copy(), len(payload), image_format
            except UnidentifiedImageError as exc:
                raise PreprocessingError("Uploaded content is not a supported image.") from exc
            except OSError as exc:
                raise PreprocessingError("Uploaded image could not be decoded.") from exc

        if isinstance(source, (str, Path)):
            image = load_rgb_image(source)
            return image, None, image.format

        raise PreprocessingError(
            f"Unsupported query source type {type(source)!r}. "
            "Expected image bytes, PIL.Image.Image, or filesystem path."
        )

    def _validate_byte_size(self, size: int) -> None:
        if size > self.max_bytes:
            raise PreprocessingError(
                f"Uploaded image exceeds the maximum size of {self.max_bytes} bytes."
            )

    def _validate_dimensions(self, size: tuple[int, int]) -> None:
        width, height = size
        if width < self.min_width or height < self.min_height:
            raise PreprocessingError(
                f"Image dimensions {(width, height)} are below the minimum "
                f"{self.min_width}x{self.min_height}."
            )
