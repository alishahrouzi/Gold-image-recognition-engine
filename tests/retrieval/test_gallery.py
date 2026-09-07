from pathlib import Path

import torch

from src.retrieval.gallery import Gallery
from scripts.build_baseline_gallery import build_report, resolve_device


def test_gallery_save_and_load_round_trip(tmp_path: Path) -> None:
    gallery = Gallery(
        embeddings=torch.nn.functional.normalize(torch.tensor([[3.0, 4.0], [1.0, 0.0]]), dim=1),
        metadata=(
            {
                "image_id": "img-1",
                "product_group": "group-1",
                "category": "Ring",
                "category_id": 4,
                "split": "train",
                "source": "dataset-1",
            },
            {
                "image_id": "img-2",
                "product_group": "group-2",
                "category": "Necklace",
                "category_id": 2,
                "split": "train",
                "source": "dataset-1",
            },
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


def test_build_report_records_gallery_contract(tmp_path: Path) -> None:
    gallery = Gallery(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        metadata=(
            {"product_group": "g1", "category": "Ring"},
            {"product_group": "g1", "category": "Ring"},
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
