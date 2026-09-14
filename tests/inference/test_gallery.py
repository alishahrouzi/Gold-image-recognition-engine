"""Tests for S4.3 runtime gallery loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from retrieval.gallery import Gallery
from src.inference.gallery import GalleryLoader, GalleryRuntimeConfig, load_runtime_gallery


def _metadata(image_id: str, product_id: str = "p1", category: str = "Ring") -> dict:
    return {
        "product_id": product_id,
        "category": category,
        "image": f"{image_id}.jpg",
        "image_id": image_id,
        "product_group": product_id,
        "image_path": f"{image_id}.jpg",
    }


def _gallery() -> Gallery:
    embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )
    metadata = (
        _metadata("ring-1", "p-ring", "Ring"),
        _metadata("necklace-1", "p-necklace", "Necklace"),
        _metadata("bracelet-1", "p-bracelet", "Bracelet"),
    )
    return Gallery(embeddings=embeddings, metadata=metadata)


def _write_gallery(root: Path, model_name: str = "s3.5_random") -> tuple[Path, Path]:
    directory = root / model_name
    directory.mkdir(parents=True)
    embedding_path = directory / "gallery_embeddings.pt"
    metadata_path = directory / "gallery_metadata.json"
    gallery = _gallery()
    gallery.save(embedding_path, metadata_path)
    return embedding_path, metadata_path


def test_load_reads_embeddings_and_metadata_into_memory(tmp_path: Path) -> None:
    embedding_path, metadata_path = _write_gallery(tmp_path)
    loader = GalleryLoader(
        GalleryRuntimeConfig(model_name="s3.5_random", root=tmp_path, expected_embedding_dim=3)
    )

    runtime = loader.load()

    assert runtime.size == 3
    assert runtime.embedding_dim == 3
    assert runtime.product_count == 3
    assert runtime.model_name == "s3.5_random"
    assert runtime.embedding_path == embedding_path
    assert runtime.metadata_path == metadata_path
    assert runtime.search_engine.gallery_embeddings.shape == (3, 3)
    assert runtime.search_engine.metadata == runtime.gallery.metadata


def test_loaded_search_engine_is_ready_for_product_search(tmp_path: Path) -> None:
    _write_gallery(tmp_path)
    runtime = GalleryLoader(
        GalleryRuntimeConfig(model_name="s3.5_random", root=tmp_path, expected_embedding_dim=3)
    ).load()

    result = runtime.search_engine.search(torch.tensor([1.0, 0.0, 0.0]), k=2)

    assert [candidate.product_id for candidate in result.candidates] == ["p-ring", "p-necklace"]
    assert result.candidates[0].similarity == pytest.approx(1.0)


def test_from_gallery_supports_in_memory_runtime(tmp_path: Path) -> None:
    gallery = _gallery()
    loader = GalleryLoader(
        GalleryRuntimeConfig(model_name="memory", root=tmp_path, expected_embedding_dim=3)
    )

    runtime = loader.from_gallery(gallery)

    assert runtime.embedding_path is None
    assert runtime.metadata_path is None
    assert runtime.gallery is gallery
    assert torch.equal(runtime.search_engine.gallery_embeddings, gallery.embeddings)


def test_missing_embeddings_are_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "s3.5_random"
    directory.mkdir(parents=True)
    (directory / "gallery_metadata.json").write_text("[]", encoding="utf-8")

    loader = GalleryLoader(GalleryRuntimeConfig(root=tmp_path, expected_embedding_dim=3))

    with pytest.raises(FileNotFoundError, match="Gallery embeddings"):
        loader.load()


def test_missing_metadata_are_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "s3.5_random"
    directory.mkdir(parents=True)
    torch.save(torch.tensor([[1.0, 0.0, 0.0]]), directory / "gallery_embeddings.pt")

    loader = GalleryLoader(GalleryRuntimeConfig(root=tmp_path, expected_embedding_dim=3))

    with pytest.raises(FileNotFoundError, match="Gallery metadata"):
        loader.load()


def test_embedding_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "s3.5_random"
    directory.mkdir(parents=True)
    torch.save(torch.tensor([[1.0, 0.0]]), directory / "gallery_embeddings.pt")
    (directory / "gallery_metadata.json").write_text(
        json.dumps([_metadata("ring-1")]), encoding="utf-8"
    )

    loader = GalleryLoader(GalleryRuntimeConfig(root=tmp_path, expected_embedding_dim=3))

    with pytest.raises(ValueError, match="embedding dimension"):
        loader.load()


def test_duplicate_image_ids_are_rejected() -> None:
    gallery = Gallery(
        embeddings=torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1
        ),
        metadata=(_metadata("same"), _metadata("same", "p2", "Necklace")),
    )
    loader = GalleryLoader(
        GalleryRuntimeConfig(model_name="memory", expected_embedding_dim=2)
    )

    with pytest.raises(ValueError, match="duplicate image_id"):
        loader.from_gallery(gallery)


def test_non_unit_embeddings_are_rejected_by_default() -> None:
    gallery = Gallery(
        embeddings=torch.tensor([[2.0, 0.0]]),
        metadata=(_metadata("ring-1"),),
    )
    loader = GalleryLoader(
        GalleryRuntimeConfig(model_name="memory", expected_embedding_dim=2)
    )

    with pytest.raises(ValueError, match="not L2-normalized"):
        loader.from_gallery(gallery)


def test_unit_norm_validation_can_be_disabled() -> None:
    gallery = Gallery(
        embeddings=torch.tensor([[2.0, 0.0]]),
        metadata=(_metadata("ring-1"),),
    )
    loader = GalleryLoader(
        GalleryRuntimeConfig(
            model_name="memory",
            expected_embedding_dim=2,
            require_unit_norm=False,
        )
    )

    runtime = loader.from_gallery(gallery)

    assert runtime.size == 1


def test_convenience_loader_uses_selected_model_name(tmp_path: Path) -> None:
    _write_gallery(tmp_path, model_name="s3.5_random")

    runtime = load_runtime_gallery(
        model_name="s3.5_random",
        root=tmp_path,
        expected_embedding_dim=3,
    )

    assert runtime.model_name == "s3.5_random"
    assert runtime.size == 3
