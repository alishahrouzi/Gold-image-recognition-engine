from pathlib import Path

import torch

from src.retrieval.gallery import Gallery
from scripts.build_baseline_gallery import build_report, resolve_device


def _metadata(product_id: str, category: str, image: str) -> dict:
    return {
        "product_id": product_id,
        "category": category,
        "image": image,
        "image_id": image,
        "product_group": product_id,
    }


def test_gallery_save_and_load_round_trip(tmp_path: Path) -> None:
    gallery = Gallery(
        embeddings=torch.nn.functional.normalize(torch.tensor([[3.0, 4.0], [1.0, 0.0]]), dim=1),
        metadata=(
            _metadata("group-1", "Ring", "ring-1.jpg"),
            _metadata("group-2", "Necklace", "necklace-1.jpg"),
        ),
    )
    embedding_path = tmp_path / "gallery_embeddings.pt"
    metadata_path = tmp_path / "gallery_metadata.json"

    gallery.save(embedding_path, metadata_path)
    loaded = Gallery.load(embedding_path, metadata_path)

    assert loaded.size == 2
    assert loaded.embedding_dim == 2
    assert torch.allclose(loaded.embeddings, gallery.embeddings)
    assert loaded.metadata == gallery.metadata


def test_gallery_requires_core_metadata() -> None:
    with torch.no_grad():
        try:
            Gallery(embeddings=torch.tensor([[1.0, 0.0]]), metadata=({"category": "Ring"},))
        except ValueError as exc:
            assert "missing fields" in str(exc)
        else:
            raise AssertionError("Gallery must reject incomplete metadata")


def test_build_report_records_gallery_contract(tmp_path: Path) -> None:
    gallery = Gallery(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        metadata=(
            _metadata("g1", "Ring", "ring-1.jpg"),
            _metadata("g1", "Ring", "ring-2.jpg"),
        ),
    )

    report = build_report(
        gallery,
        checkpoint=tmp_path / "best.pt",
        manifest=tmp_path / "manifest.csv",
        split="train",
        device="cpu",
        batch_size=16,
        seed=42,
        duration_seconds=1.5,
    )

    assert report["policy"] == "s2.7-baseline-gallery-v1"
    assert report["num_embeddings"] == 2
    assert report["embedding_dim"] == 2
    assert report["num_product_groups"] == 1
    assert report["category_counts"] == {"Ring": 2}
    assert report["test_split_used"] is False
    assert report["embedding_norm_mean"] == 1.0


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == "cpu"
