"""Tests for S4.2 query validation and preprocessing."""

from __future__ import annotations

from io import BytesIO

import pytest
import torch
from PIL import Image

from data.errors import PreprocessingError
from inference.query import QueryProcessor


def _image_bytes(mode: str = "RGB", size: tuple[int, int] = (128, 96), fmt: str = "PNG") -> bytes:
    image = Image.new(mode, size)
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_process_bytes_returns_expected_query_tensor() -> None:
    result = QueryProcessor().process(_image_bytes())

    assert result.tensor.shape == (3, 224, 224)
    assert result.tensor.dtype == torch.float32
    assert torch.isfinite(result.tensor).all()
    assert result.original_size == (128, 96)
    assert result.original_mode == "RGB"
    assert result.format == "PNG"
    assert result.source_bytes is not None


def test_process_tensor_returns_same_tensor_contract() -> None:
    tensor = QueryProcessor().process_tensor(_image_bytes())

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_rgba_query_is_converted_to_rgb() -> None:
    result = QueryProcessor().process(_image_bytes(mode="RGBA"))

    assert result.tensor.shape == (3, 224, 224)
    assert result.original_mode == "RGBA"


def test_grayscale_query_is_converted_to_rgb() -> None:
    result = QueryProcessor().process(_image_bytes(mode="L"))

    assert result.tensor.shape == (3, 224, 224)
    assert result.original_mode == "L"


def test_empty_upload_is_rejected() -> None:
    with pytest.raises(
        PreprocessingError,
        match="The uploaded file is empty or contains no image data",
    ):
        QueryProcessor().process(b"")


def test_oversized_upload_is_rejected_before_decode() -> None:
    with pytest.raises(PreprocessingError, match="maximum size"):
        QueryProcessor(max_bytes=10).process(b"0" * 11)


def test_tiny_image_is_rejected() -> None:
    with pytest.raises(PreprocessingError, match="below the minimum"):
        QueryProcessor().process(_image_bytes(size=(16, 16)))


def test_invalid_image_bytes_are_rejected() -> None:
    with pytest.raises(PreprocessingError, match="not a supported image"):
        QueryProcessor().process(b"not-an-image")


def test_unsupported_source_type_is_rejected() -> None:
    with pytest.raises(PreprocessingError, match="Unsupported query source type"):
        QueryProcessor().process(123)  # type: ignore[arg-type]


def test_custom_preprocessing_config_is_used() -> None:
    from data.preprocessing.config import ImagePreprocessingConfig

    processor = QueryProcessor(config=ImagePreprocessingConfig(image_size=128))
    result = processor.process(_image_bytes())

    assert result.tensor.shape == (3, 128, 128)


def test_path_source_is_supported(tmp_path) -> None:
    path = tmp_path / "query.png"
    path.write_bytes(_image_bytes())

    result = QueryProcessor().process(path)

    assert result.tensor.shape == (3, 224, 224)
    assert result.source_bytes is None
