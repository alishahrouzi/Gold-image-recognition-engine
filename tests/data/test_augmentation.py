"""Unit tests for S1.9 identity-preserving training augmentation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from random import Random

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from data.constants import CATEGORY_TO_ID, SOURCE_DATASET1
from data.collate import collate_preprocessed_samples
from data.datasets.unified_dataset import UnifiedDataset
from data.errors import AugmentationError
from data.preprocessing import (
    AugmentationConfig,
    BrightnessConfig,
    ColorConfig,
    ContrastConfig,
    HorizontalFlipConfig,
    ImagePreprocessor,
    PreprocessedDataset,
    RandomCropConfig,
    RotationConfig,
    TrainingAugmentor,
    build_preprocessed_dataset,
)
from tests.data.helpers import write_manifest, write_rgb_image


def _pattern_rgb(size: tuple[int, int] = (48, 32)) -> Image.Image:
    image = Image.new("RGB", size)
    pixels = image.load()
    width, height = size
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                int(255 * x / max(width - 1, 1)),
                int(255 * y / max(height - 1, 1)),
                40 if x < width // 2 else 200,
            )
    return image


def _config(
    *,
    enabled: bool = True,
    seed: int | None = 0,
    flip: bool = False,
    flip_p: float = 0.5,
    rotation: bool = False,
    degrees: float = 10.0,
    brightness: bool = False,
    brightness_factor: float = 0.15,
    contrast: bool = False,
    contrast_factor: float = 0.15,
    color: bool = False,
    saturation: float = 0.08,
    hue: float = 0.02,
    crop: bool = False,
) -> AugmentationConfig:
    return AugmentationConfig(
        enabled=enabled,
        seed=seed,
        horizontal_flip=HorizontalFlipConfig(enabled=flip, probability=flip_p),
        rotation=RotationConfig(enabled=rotation, degrees=degrees),
        brightness=BrightnessConfig(enabled=brightness, factor=brightness_factor),
        contrast=ContrastConfig(enabled=contrast, factor=contrast_factor),
        color=ColorConfig(enabled=color, saturation=saturation, hue=hue),
        random_crop=RandomCropConfig(enabled=crop),
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pixels(image: Image.Image) -> bytes:
    """Stable RGB byte payload for equality checks."""
    return image.convert("RGB").tobytes()


def _manifest_for_splits(tmp_path: Path) -> Path:
    rows = []
    for index, split in enumerate(("train", "valid", "test")):
        path = tmp_path / f"{split}.jpg"
        _pattern_rgb().save(path)
        rows.append(
            {
                "image_id": f"img_{split}",
                "image_path": str(path),
                "group_id": f"g_{split}",
                "category": "Ring",
                "category_id": CATEGORY_TO_ID["Ring"],
                "split": split,
                "source": SOURCE_DATASET1,
            }
        )
        del index
    return write_manifest(tmp_path / "manifest.csv", rows)


def test_augmentor_can_be_instantiated() -> None:
    augmentor = TrainingAugmentor()
    assert augmentor.config.enabled is True
    assert augmentor.config.random_crop.enabled is False


def test_disabled_augmentation_is_preprocessing_compatible() -> None:
    image = _pattern_rgb()
    output = TrainingAugmentor(AugmentationConfig.disabled())(image)
    preprocessor = ImagePreprocessor()
    tensor = preprocessor(output)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.equal(tensor, preprocessor(image))


def test_horizontal_flip_works() -> None:
    image = _pattern_rgb()
    config = _config(flip=True, flip_p=1.0)
    output = TrainingAugmentor(config, rng=Random(0))(image)
    assert _pixels(output) == _pixels(F.hflip(image))
    assert _pixels(output) != _pixels(image)


def test_rotation_works() -> None:
    image = _pattern_rgb()
    config = _config(rotation=True, degrees=10.0)
    rng = Random()
    rng.uniform = lambda a, b: 10.0  # type: ignore[method-assign]
    output = TrainingAugmentor(config, rng=rng)(image)
    expected = image.rotate(
        10.0,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=(255, 255, 255),
    )
    assert _pixels(output) == _pixels(expected)
    assert _pixels(output) != _pixels(image)


def test_brightness_augmentation_works() -> None:
    image = _pattern_rgb()
    config = _config(brightness=True, brightness_factor=0.15)
    rng = Random()
    rng.uniform = lambda a, b: 1.15  # type: ignore[method-assign]
    output = TrainingAugmentor(config, rng=rng)(image)
    expected = F.adjust_brightness(image, 1.15)
    assert _pixels(output) == _pixels(expected)
    assert _pixels(output) != _pixels(image)


def test_contrast_augmentation_works() -> None:
    image = _pattern_rgb()
    config = _config(contrast=True, contrast_factor=0.15)
    rng = Random()
    rng.uniform = lambda a, b: 0.85  # type: ignore[method-assign]
    output = TrainingAugmentor(config, rng=rng)(image)
    expected = F.adjust_contrast(image, 0.85)
    assert _pixels(output) == _pixels(expected)
    assert _pixels(output) != _pixels(image)


def test_color_variation_works() -> None:
    image = _pattern_rgb()
    config = _config(color=True, saturation=0.08, hue=0.02)
    uniforms = [1.08, 0.02]

    class _Rng:
        def random(self) -> float:
            return 1.0

        def uniform(self, low: float, high: float) -> float:
            return uniforms.pop(0)

    output = TrainingAugmentor(config, rng=_Rng())(image)
    expected = F.adjust_hue(F.adjust_saturation(image, 1.08), 0.02)
    assert _pixels(output) == _pixels(expected)
    assert _pixels(output) != _pixels(image)


def test_random_crop_disabled_by_default() -> None:
    config = AugmentationConfig()
    assert config.random_crop.enabled is False
    image = _pattern_rgb()
    identity = _config(enabled=True)
    output = TrainingAugmentor(identity, rng=Random(0))(image)
    assert _pixels(output) == _pixels(image)


def test_does_not_modify_source_file(tmp_path: Path) -> None:
    path = tmp_path / "source.jpg"
    _pattern_rgb().save(path)
    before = _file_digest(path)
    mtime = path.stat().st_mtime_ns
    image = Image.open(path)
    image.load()
    TrainingAugmentor(_config(flip=True, flip_p=1.0, rotation=True, brightness=True))(image)
    image.close()
    assert _file_digest(path) == before
    assert path.stat().st_mtime_ns == mtime


def test_output_compatible_with_preprocessor_contract() -> None:
    image = _pattern_rgb((183, 91))
    augmented = TrainingAugmentor(_config(flip=True, flip_p=1.0, rotation=True, brightness=True, contrast=True, color=True))(image)
    tensor = ImagePreprocessor()(augmented)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_seed_reproduces_first_call_sequence() -> None:
    image = _pattern_rgb()
    config = AugmentationConfig(seed=123)
    first = TrainingAugmentor(config)(image)
    second = TrainingAugmentor(AugmentationConfig(seed=123))(image)
    assert _pixels(first) == _pixels(second)
    first_again = TrainingAugmentor(AugmentationConfig(seed=123))(image)
    assert _pixels(first) == _pixels(first_again)
    augmentor = TrainingAugmentor(config)
    augmentor.reseed(123)
    assert _pixels(augmentor(image)) == _pixels(first)


def test_loggable_config_is_serializable() -> None:
    payload = AugmentationConfig(seed=7).as_loggable_dict()
    assert payload["enabled"] is True
    assert payload["seed"] == 7
    assert payload["random_crop"]["enabled"] is False
    assert payload["policy"] == "s1.9-identity-preserving"
    assert "horizontal_flip" in payload


def test_rejects_aggressive_rotation() -> None:
    with pytest.raises(AugmentationError, match="identity-preserving cap"):
        RotationConfig(degrees=90)


def test_rejects_enabled_augmentation_for_non_train_roles() -> None:
    image = _pattern_rgb()
    preprocessor = ImagePreprocessor()
    for role in ("valid", "test", "query", "gallery"):
        with pytest.raises(AugmentationError, match="only allowed for role='train'"):
            PreprocessedDataset(
                _FakeDataset(image, split="train"),
                preprocessor,
                role=role,
                augmentation=AugmentationConfig(enabled=True),
            )


def test_validation_test_query_gallery_remain_deterministic(tmp_path: Path) -> None:
    manifest = _manifest_for_splits(tmp_path)
    preprocessor = ImagePreprocessor()
    for role, split in (
        ("valid", "valid"),
        ("test", "test"),
        ("query", "test"),
        ("gallery", "valid"),
    ):
        dataset = build_preprocessed_dataset(
            UnifiedDataset(manifest, split=split),
            role=role,
            preprocessor=preprocessor,
        )
        assert dataset.augmentor is None
        item = dataset[0]
        raw = UnifiedDataset(manifest, split=split)[0]
        expected = preprocessor(raw.image)
        assert torch.equal(item.image, expected)
        assert item.sample.group_id == raw.sample.group_id
        assert item.sample.split == split


def test_train_role_applies_augmentation(tmp_path: Path) -> None:
    manifest = _manifest_for_splits(tmp_path)
    preprocessor = ImagePreprocessor()
    config = _config(flip=True, flip_p=1.0)
    dataset = build_preprocessed_dataset(
        UnifiedDataset(manifest, split="train"),
        role="train",
        preprocessor=preprocessor,
        augmentation=config,
    )
    assert dataset.augmentor is not None
    raw = UnifiedDataset(manifest, split="train")[0]
    item = dataset[0]
    expected = preprocessor(F.hflip(raw.image))
    assert torch.equal(item.image, expected)
    assert item.sample.group_id == raw.sample.group_id
    assert item.image.shape == (3, 224, 224)
    assert item.image.dtype == torch.float32


def test_default_preprocessed_dataset_has_no_augmentation(tmp_path: Path) -> None:
    path = write_rgb_image(tmp_path / "a.jpg", size=(20, 30))
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
    dataset = PreprocessedDataset(UnifiedDataset(manifest, split="train"))
    assert dataset.role == "valid"
    assert dataset.augmentor is None
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_preprocessed_samples)
    batch = next(iter(loader))
    assert batch["image"].shape == (1, 3, 224, 224)


class _FakeDataset:
    def __init__(self, image: Image.Image, split: str) -> None:
        from data.types import DatasetItem, Sample

        self._item = DatasetItem(
            sample=Sample(
                image_id="q",
                image_path=Path("q.jpg"),
                group_id="g",
                category="Ring",
                category_id=CATEGORY_TO_ID["Ring"],
                split=split,
                source=SOURCE_DATASET1,
            ),
            image=image,
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return self._item
