import pytest
import torch

from src.retrieval.search import SimilaritySearchEngine
from src.retrieval.search_result import ProductSearchResult


@pytest.fixture
def engine() -> SimilaritySearchEngine:
    metadata = [
        {"image_id": "a1", "product_id": "p1", "category": "Ring", "image": "a1.jpg"},
        {"image_id": "a2", "product_id": "p1", "category": "Ring", "image": "a2.jpg"},
        {"image_id": "b1", "product_id": "p2", "category": "Necklace", "image": "b1.jpg"},
        {"image_id": "c1", "product_id": "p3", "category": "Bracelet", "image": "c1.jpg"},
    ]
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.7, 0.7],
            [0.0, 1.0],
        ]
    )
    return SimilaritySearchEngine(embeddings, metadata)


def test_search_returns_unique_product_candidates(engine: SimilaritySearchEngine) -> None:
    result = engine.search(torch.tensor([1.0, 0.0]), k=2, query_id="q")

    assert isinstance(result, ProductSearchResult)
    assert result.query_id == "q"
    assert result.top_k == 2
    assert [candidate.product_id for candidate in result.candidates] == ["p1", "p2"]
    assert result.candidates[0].matched_image_ids == ("a1", "a2")


def test_search_uses_strongest_image_match_as_raw_product_similarity(
    engine: SimilaritySearchEngine,
) -> None:
    result = engine.search(torch.tensor([1.0, 0.0]), k=3)

    assert result.candidates[0].product_id == "p1"
    assert result.candidates[0].similarity == pytest.approx(1.0)


def test_self_image_can_be_excluded(engine: SimilaritySearchEngine) -> None:
    result = engine.search(
        torch.tensor([1.0, 0.0]), k=2, exclude_image_id="a1"
    )

    assert "a1" not in result.candidates[0].matched_image_ids
    assert result.candidates[0].product_id == "p1"


def test_k_is_capped_by_available_unique_products(engine: SimilaritySearchEngine) -> None:
    result = engine.search(torch.tensor([1.0, 0.0]), k=99)

    assert result.top_k == 3
    assert len({candidate.product_id for candidate in result.candidates}) == 3


def test_invalid_query_shape_is_rejected(engine: SimilaritySearchEngine) -> None:
    with pytest.raises(ValueError, match="shape"):
        engine.search(torch.ones(2, 2), k=1)


def test_dimension_mismatch_is_rejected(engine: SimilaritySearchEngine) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        engine.search(torch.ones(3), k=1)


def test_non_finite_query_is_rejected(engine: SimilaritySearchEngine) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        engine.search(torch.tensor([float("nan"), 0.0]), k=1)


def test_invalid_k_is_rejected(engine: SimilaritySearchEngine) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        engine.search(torch.ones(2), k=0)


def test_legacy_product_group_metadata_is_supported() -> None:
    metadata = [
        {"image_id": "a", "product_group": "p1", "category": "Ring", "image_path": "a.jpg"},
        {"image_id": "b", "product_group": "p2", "category": "Necklace", "image_path": "b.jpg"},
    ]
    engine = SimilaritySearchEngine(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), metadata)

    result = engine.search(torch.tensor([1.0, 0.0]), k=1)

    assert result.candidates[0].product_id == "p1"


def test_search_from_retrieval_candidates_groups_products() -> None:
    from src.retrieval.result import RetrievalCandidate

    candidates = [
        RetrievalCandidate("a1", "p1", "Ring", 0.9, 1, "a1.jpg"),
        RetrievalCandidate("a2", "p1", "Ring", 0.8, 2, "a2.jpg"),
        RetrievalCandidate("b1", "p2", "Necklace", 0.7, 3, "b1.jpg"),
    ]
    engine = SimilaritySearchEngine(torch.tensor([[1.0, 0.0]]), [{"image_id": "x", "product_id": "x", "category": "Ring"}])

    result = engine.search_from_retrieval_candidates(candidates, k=2)

    assert [candidate.product_id for candidate in result.candidates] == ["p1", "p2"]
    assert result.candidates[0].matched_image_ids == ("a1", "a2")
