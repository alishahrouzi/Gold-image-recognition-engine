import torch
from torch import nn

from src.retrieval.gallery import Gallery
from src.retrieval.embedding import EmbeddingExtractor
from src.retrieval.pipeline import BaselineRetrievalPipeline


class DummyEmbeddingModel(nn.Module):
    embedding_dim = 3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Encode each image by its channel-wise mean, then normalize.
        return torch.nn.functional.normalize(images.mean(dim=(2, 3)), p=2, dim=1)


def _build_pipeline() -> BaselineRetrievalPipeline:
    extractor = EmbeddingExtractor(DummyEmbeddingModel(), device="cpu")
    gallery = Gallery(
        embeddings=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.7, 0.7, 0.0],
            ]
        ),
        metadata=(
            {"image_id": "ring-1", "product_group": "ring-a", "category": "Ring"},
            {"image_id": "necklace-1", "product_group": "necklace-a", "category": "Necklace"},
            {"image_id": "bracelet-1", "product_group": "bracelet-a", "category": "Bracelet"},
        ),
    )
    return BaselineRetrievalPipeline(extractor, gallery)


def test_pipeline_extracts_query_and_returns_ranked_candidates() -> None:
    pipeline = _build_pipeline()
    query = torch.tensor([[[1.0]], [[0.0]], [[0.0]]])

    result = pipeline.query(query, query_id="query-1", k=2)

    assert result.query_id == "query-1"
    assert result.top_k == 2
    assert [candidate.image_id for candidate in result.candidates] == [
        "ring-1",
        "bracelet-1",
    ]
    assert [candidate.rank for candidate in result.candidates] == [1, 2]
    assert result.candidates[0].similarity == 1.0


def test_pipeline_propagates_self_match_exclusion() -> None:
    pipeline = _build_pipeline()
    query = torch.tensor([[[1.0]], [[0.0]], [[0.0]]])

    result = pipeline.query(
        query,
        query_id="ring-1",
        k=2,
        exclude_image_id="ring-1",
    )

    assert [candidate.image_id for candidate in result.candidates] == [
        "bracelet-1",
        "necklace-1",
    ]
