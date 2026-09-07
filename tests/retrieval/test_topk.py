import pytest
import torch

from src.retrieval.topk import TopKRetriever


@pytest.fixture
def retriever() -> TopKRetriever:
    metadata = [
        {"image_id": "a", "product_group": "p1", "category": "Ring"},
        {"image_id": "b", "product_group": "p2", "category": "Necklace"},
        {"image_id": "c", "product_group": "p3", "category": "Bracelet"},
    ]
    embeddings = torch.tensor([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    return TopKRetriever(embeddings, metadata)


def test_topk_is_sorted_descending(retriever: TopKRetriever) -> None:
    result = retriever.retrieve(torch.tensor([1.0, 0.0]), k=2, query_id="q")

    assert [candidate.image_id for candidate in result.candidates] == ["a", "b"]
    assert [candidate.rank for candidate in result.candidates] == [1, 2]
    assert result.candidates[0].similarity == pytest.approx(1.0)


def test_self_match_can_be_excluded(retriever: TopKRetriever) -> None:
    result = retriever.retrieve(
        torch.tensor([1.0, 0.0]), k=2, query_id="a", exclude_image_id="a"
    )

    assert [candidate.image_id for candidate in result.candidates] == ["b", "c"]


def test_k_cannot_be_zero(retriever: TopKRetriever) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        retriever.retrieve(torch.tensor([1.0, 0.0]), k=0)
