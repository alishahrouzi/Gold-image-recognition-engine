"""Unit tests for S1.8 deterministic image preprocessing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.constants import CATEGORY_TO_ID, SOURCE_DATASET1
from data.collate import collate_preprocessed_samples
from data.datasets.unified_dataset import UnifiedDataset
from data.errors import PreprocessingError
from data.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MEAN,
    DEFAULT_STD,
    ImagePreprocessingConfig,
    ImagePreprocessor,
    PreprocessedDataset,
)
from tests.data.helpers import write_manifest, write_rgb_image


def _solid_rgb(size: tuple[int, int], color=(10, 20, 30)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_rgb_arbitrary_size_to_chw() -> None:
    preprocessor = ImagePreprocessor()
    tensor = preprocessor(_solid_rgb((183, 91)))
    assert tensor.shape == (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    assert tensor.dtype == torch.float32


def test_grayscale_to_three_channels() -> None:
    preprocessor = ImagePreprocessor()
    tensor = preprocessor(Image.new("L", (64, 40), 128))
    assert tensor.shape == (3, 224, 224)


def test_rgba_to_three_channels() -> None:
    preprocessor = ImagePreprocessor()
    rgba = Image.new("RGBA", (50, 50), (10, 20, 30, 0))
    tensor = preprocessor(rgba)
    assert tensor.shape == (3, 224, 224)


def test_palette_to_three_channels() -> None:
    preprocessor = ImagePreprocessor()
    palette = Image.new("P", (32, 48))
    palette.putpalette([i % 256 for i in range(768)])
    tensor = preprocessor(palette)
    assert tensor.shape == (3, 224, 224)


@pytest.mark.parametrize(
    "size",
    [
        (183, 183),
        (262, 262),
        (512, 512),
        (1600, 1600),
        (1536, 2048),
        (2048, 2048),
        (7, 300),
    ],
)
def test_variable_resolutions(size: tuple[int, int]) -> None:
    preprocessor = ImagePreprocessor()
    tensor = preprocessor(_solid_rgb(size))
    assert tensor.shape == (3, 224, 224)


def test_dtype_float32() -> None:
    tensor = ImagePreprocessor()(_solid_rgb((16, 16)))
    assert tensor.dtype == torch.float32


def test_range_before_normalization() -> None:
    preprocessor = ImagePreprocessor()
    tensor = preprocessor.to_float_tensor(_solid_rgb((16, 16), color=(0, 128, 255)))
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0
    assert torch.isclose(tensor[0, 0, 0], torch.tensor(0.0))
    assert torch.isclose(tensor[2, 0, 0], torch.tensor(1.0))


def test_normalization_applies_configured_mean_std() -> None:
    mean = (0.1, 0.2, 0.3)
    std = (0.5, 0.5, 0.5)
    config = ImagePreprocessingConfig(image_size=8, mean=mean, std=std)
    preprocessor = ImagePreprocessor(config)
    # Pure white → 1.0 before normalize.
    image = Image.new("RGB", (8, 8), (255, 255, 255))
    tensor = preprocessor(image)
    expected = torch.tensor([(1.0 - m) / s for m, s in zip(mean, std)], dtype=torch.float32)
    for channel in range(3):
        assert torch.allclose(tensor[channel], expected[channel].expand_as(tensor[channel]))


def test_determinism() -> None:
    preprocessor = ImagePreprocessor()
    image = _solid_rgb((101, 77), color=(44, 55, 66))
    first = preprocessor(image)
    second = preprocessor(image)
    assert torch.equal(first, second)


def test_no_nan_or_inf() -> None:
    tensor = ImagePreprocessor()(_solid_rgb((224, 224)))
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_batch_processing(tmp_path: Path) -> None:
    rows = []
    for index in range(4):
        path = write_rgb_image(tmp_path / f"{index}.jpg", size=(20 + index, 30 + index))
        rows.append(
            {
                "image_id": f"img_{index}",
                "image_path": str(path),
                "group_id": f"g_{index}",
                "category": "Ring",
                "category_id": CATEGORY_TO_ID["Ring"],
                "split": "train",
                "source": SOURCE_DATASET1,
            }
        )
    manifest = write_manifest(tmp_path / "manifest.csv", rows)
    dataset = PreprocessedDataset(UnifiedDataset(manifest, split="train"))
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_preprocessed_samples,
    )
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["image"].dtype == torch.float32


def test_metadata_alignment(tmp_path: Path) -> None:
    rows = []
    specs = [
        ("img_a", "group_a", "Bracelet"),
        ("img_b", "group_b", "Earrings"),
        ("img_c", "group_c", "Necklace"),
    ]
    for index, (image_id, group_id, category) in enumerate(specs):
        path = write_rgb_image(tmp_path / f"{index}.jpg", size=(40, 50 + index))
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(path),
                "group_id": group_id,
                "category": category,
                "category_id": CATEGORY_TO_ID[category],
                "split": "valid",
                "source": SOURCE_DATASET1,
            }
        )
    manifest = write_manifest(tmp_path / "manifest.csv", rows)
    dataset = PreprocessedDataset(UnifiedDataset(manifest))
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=collate_preprocessed_samples,
    )
    batch = next(iter(loader))
    for index, (image_id, group_id, category) in enumerate(specs):
        assert batch["image_id"][index] == image_id
        assert batch["group_id"][index] == group_id
        assert batch["category"][index] == category
        assert batch["category_id"][index] == CATEGORY_TO_ID[category]
        assert batch["image"][index].shape == (3, 224, 224)


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(PreprocessingError, match="positive"):
        ImagePreprocessingConfig(image_size=0)
    with pytest.raises(PreprocessingError, match="non-zero"):
        ImagePreprocessingConfig(std=(0.0, 0.1, 0.1))
    with pytest.raises(PreprocessingError, match="Unsupported interpolation"):
        ImagePreprocessingConfig(interpolation="warp")


def test_unsupported_input_type() -> None:
    with pytest.raises(PreprocessingError, match="Unsupported input type"):
        ImagePreprocessor()(object())  # type: ignore[arg-type]


def test_collate_requires_tensors(tmp_path: Path) -> None:
    path = write_rgb_image(tmp_path / "a.jpg")
    rows = [
        {
            "image_id": "a",
            "image_path": str(path),
            "group_id": "g",
            "category": "Ring",
            "category_id": CATEGORY_TO_ID["Ring"],
            "split": "train",
            "source": SOURCE_DATASET1,
        }
    ]
    manifest = write_manifest(tmp_path / "manifest.csv", rows)
    dataset = UnifiedDataset(manifest)
    with pytest.raises(PreprocessingError, match="expects tensor images"):
        collate_preprocessed_samples([dataset[0]])


def test_query_gallery_same_preprocessor() -> None:
    """Standalone API used by future query and gallery paths."""
    config = ImagePreprocessingConfig()
    preprocessor = ImagePreprocessor(config)
    query = preprocessor(_solid_rgb((100, 200), color=(1, 2, 3)))
    gallery = preprocessor(_solid_rgb((300, 50), color=(1, 2, 3)))
    assert query.shape == gallery.shape == (3, 224, 224)
    # Same solid color after stretch should match closely (resize can differ
    # slightly for non-square sources, so compare channel means only for equal sizes).
    same = preprocessor(_solid_rgb((64, 64), color=(10, 20, 30)))
    again = preprocessor(_solid_rgb((64, 64), color=(10, 20, 30)))
    assert torch.equal(same, again)


def test_defaults_match_imagenet_contract() -> None:
    assert DEFAULT_IMAGE_SIZE == 224
    assert DEFAULT_MEAN == (0.485, 0.456, 0.406)
    assert DEFAULT_STD == (0.229, 0.224, 0.225)
