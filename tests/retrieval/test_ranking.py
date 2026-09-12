import pytest

from src.retrieval.ranking import ProductRanker, RankedProductSearchResult
from src.retrieval.search_result import ProductSearchCandidate, ProductSearchResult


def candidate(product_id: str, similarity: float, category: str = "Ring") -> ProductSearchCandidate:
    return ProductSearchCandidate(
        product_id=product_id,
        category=category,
        similarity=similarity,
        matched_image_ids=(f"{product_id}-1",),
        matched_image_paths=(f"{product_id}.jpg",),
    )


def test_rank_sorts_similarity_descending() -> None:
    result = ProductSearchResult(
        query_id="q1",
        candidates=(candidate("p1", 0.72), candidate("p2", 0.94), candidate("p3", 0.81)),
    )

    ranked = ProductRanker().rank(result)

    assert isinstance(ranked, RankedProductSearchResult)
    assert [item.product_id for item in ranked.candidates] == ["p2", "p3", "p1"]
    assert [item.similarity for item in ranked.candidates] == pytest.approx([0.94, 0.81, 0.72])


def test_rank_assigns_one_based_ranks() -> None:
    result = ProductSearchResult(
        query_id="q1",
        candidates=(candidate("p1", 0.7), candidate("p2", 0.9)),
    )

    ranked = ProductRanker().rank(result)

    assert [item.rank for item in ranked.candidates] == [1, 2]


def test_rank_tie_breaks_by_product_id() -> None:
    result = ProductSearchResult(
        query_id="q1",
        candidates=(candidate("p3", 0.8), candidate("p1", 0.8), candidate("p2", 0.8)),
    )

    ranked = ProductRanker().rank(result)

    assert [item.product_id for item in ranked.candidates] == ["p1", "p2", "p3"]


def test_rank_is_deterministic_independent_of_input_order() -> None:
    ranker = ProductRanker()
    first = ProductSearchResult(
        query_id="q1",
        candidates=(candidate("p3", 0.8), candidate("p1", 0.9), candidate("p2", 0.8)),
    )
    second = ProductSearchResult(
        query_id="q1",
        candidates=(candidate("p2", 0.8), candidate("p3", 0.8), candidate("p1", 0.9)),
    )

    ranked_first = ranker.rank(first)
    ranked_second = ranker.rank(second)

    assert ranked_first == ranked_second


def test_rank_preserves_metadata() -> None:
    original = ProductSearchCandidate(
        product_id="p1",
        category="Earrings",
        similarity=0.91,
        matched_image_ids=("a1", "a2"),
        matched_image_paths=("a1.jpg", "a2.jpg"),
    )
    result = ProductSearchResult(query_id="q1", candidates=(original,))

    ranked = ProductRanker().rank(result)
    item = ranked.candidates[0]

    assert item.product_id == "p1"
    assert item.category == "Earrings"
    assert item.similarity == pytest.approx(0.91)
    assert item.matched_image_ids == ("a1", "a2")
    assert item.matched_image_paths == ("a1.jpg", "a2.jpg")


def test_rank_does_not_mutate_input() -> None:
    result = ProductSearchResult(
        query_id="q1",
        candidates=(candidate("p2", 0.9), candidate("p1", 0.95)),
    )
    before = result

    ProductRanker().rank(result)

    assert result == before
    assert [item.product_id for item in result.candidates] == ["p2", "p1"]


def test_rank_accepts_empty_result() -> None:
    result = ProductSearchResult(query_id="q1", candidates=tuple())

    ranked = ProductRanker().rank(result)

    assert ranked.query_id == "q1"
    assert ranked.candidates == tuple()
    assert ranked.top_k == 0


def test_rank_rejects_duplicate_product_ids() -> None:
    result = ProductSearchResult(
        query_id="q1",
        candidates=(candidate("p1", 0.9), candidate("p1", 0.8)),
    )

    with pytest.raises(ValueError, match="duplicate product IDs"):
        ProductRanker().rank(result)


def test_rank_rejects_non_finite_similarity() -> None:
    result = ProductSearchResult(
        query_id="q1",
        candidates=(candidate("p1", float("nan")),),
    )

    with pytest.raises(ValueError, match="finite"):
        ProductRanker().rank(result)


def test_rank_rejects_wrong_result_type() -> None:
    with pytest.raises(TypeError, match="ProductSearchResult"):
        ProductRanker().rank(object())
