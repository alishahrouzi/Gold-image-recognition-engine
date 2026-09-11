import torch
from torch import nn

from src.metric_learning.siamese import SiameseNetwork
from src.retrieval.embedding import EmbeddingExtractor, load_siamese_embedding_model


class DummyEmbeddingModel(nn.Module):
    embedding_dim = 3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(images.mean(dim=(2, 3)), p=2, dim=1)


def test_embedding_extractor_returns_cpu_embeddings() -> None:
    extractor = EmbeddingExtractor(DummyEmbeddingModel(), device="cpu")
    images = torch.ones(2, 3, 4, 4)

    embeddings = extractor.extract(images)

    assert embeddings.shape == (2, 3)
    assert embeddings.device.type == "cpu"
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(2))


def test_embedding_extractor_rejects_non_batched_input() -> None:
    extractor = EmbeddingExtractor(DummyEmbeddingModel())

    try:
        extractor.extract(torch.ones(3, 4, 4))
    except ValueError as exc:
        assert "[B, C, H, W]" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_load_siamese_embedding_model(tmp_path) -> None:
    model = SiameseNetwork()
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "checkpoint_version": "s2.5-checkpoint-v1",
            "epoch": 10,
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )

    extractor = load_siamese_embedding_model(checkpoint, device="cpu")
    images = torch.randn(2, 3, 224, 224)
    embeddings = extractor.extract(images)

    assert embeddings.shape == (2, 128)
    assert torch.allclose(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(2),
        atol=1e-5,
    )
