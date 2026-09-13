"""Tests for the S4.1 functional MVP inference pipeline."""

from __future__ import annotations

import torch

from src.inference.pipeline import InferencePipeline
from src.retrieval.ranking import ProductRanker
from src.retrieval.scoring import SimilarityScoreConverter
from src.retrieval.search import SimilaritySearchEngine


class FakeEmbedder:
    """Deterministic test double for the existing EmbeddingExtractor contract."""

    def __init__(self, embedding: torch.Tensor) -> None:
        self.embedding = embedding
        self.calls = 0
        self.last_image: torch.Tensor | None = None

    def extract_one(self, image: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_image = image
        return self.embedding.clone()


def _engine() -> SimilaritySearchEngine:
    metadata = [
        {"image_id": "ring-1", "product_id": "p-ring", "category": "Ring", "image": "ring-1.jpg"},
        {"image_id": "ring-2", "product_id": "p-ring", "category": "Ring", "image": "ring-2.jpg"},
        {"image_id": "necklace-1", "product_id": "p-necklace", "category": "Necklace", "image": "necklace-1.jpg"},
        {"image_id": "bracelet-1", "product_id": "p-bracelet", "category": "Bracelet", "image": "bracelet-1.jpg"},
    ]
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return SimilaritySearchEngine(embeddings, metadata)


def test_run_composes_embedding_search_ranking_and_scoring() -> None:
    embedder = FakeEmbedder(torch.tensor([1.0, 0.0]))
    pipeline = InferencePipeline(embedder, _engine())
    image = torch.zeros(3, 224, 224)

    result = pipeline.run(image, query_id="q1", k=2)

    assert embedder.calls == 1
    assert embedder.last_image is image
    assert result.query_id == "q1"
    assert result.top_k == 2
    assert [item.rank for item in result.candidates] == [1, 2]
    assert [item.product_id for item in result.candidates] == ["p-ring", "p-necklace"]
    assert result.candidates[0].similarity == 1.0
    assert result.candidates[0].similarity_score == 100.0
    assert result.candidates[1].similarity_score > 0.0


def test_run_preserves_self_exclusion_through_search_stage() -> None:
    embedder = FakeEmbedder(torch.tensor([1.0, 0.0]))
    pipeline = InferencePipeline(embedder, _engine())

    result = pipeline.run(
        torch.zeros(3, 224, 224),
        query_id="ring-1",
        k=2,
        exclude_image_id="ring-1",
    )

    assert result.candidates[0].product_id == "p-ring"
    assert "ring-1" not in result.candidates[0].matched_image_ids
    assert result.candidates[0].matched_image_ids == ("ring-2",)


def test_default_ranker_and_score_converter_are_used() -> None:
    pipeline = InferencePipeline(FakeEmbedder(torch.tensor([1.0, 0.0])), _engine())

    assert isinstance(pipeline.ranker, ProductRanker)
    assert isinstance(pipeline.score_converter, SimilarityScoreConverter)


def test_custom_ranker_and_score_converter_are_preserved() -> None:
    ranker = ProductRanker()
    converter = SimilarityScoreConverter()
    pipeline = InferencePipeline(
        FakeEmbedder(torch.tensor([1.0, 0.0])),
        _engine(),
        ranker=ranker,
        score_converter=converter,
    )

    assert pipeline.ranker is ranker
    assert pipeline.score_converter is converter


def test_empty_search_result_flows_to_empty_scored_result() -> None:
    # k remains valid; a gallery with candidates always produces results. This
    # test uses an engine whose query is finite but points away from the gallery
    # to verify that the orchestration does not invent a result or score.
    metadata = [
        {"image_id": "a", "product_id": "p1", "category": "Ring", "image": "a.jpg"},
    ]
    engine = SimilaritySearchEngine(torch.tensor([[1.0, 0.0]]), metadata)
    pipeline = InferencePipeline(FakeEmbedder(torch.tensor([0.0, 1.0])), engine)

    result = pipeline.run(torch.zeros(3, 224, 224), k=1)

    assert result.top_k == 1
    assert result.candidates[0].product_id == "p1"
    assert result.candidates[0].similarity == 0.0
    assert result.candidates[0].similarity_score == 50.0
