"""Tests for S3.10 Similarity Score conversion."""

from __future__ import annotations

import pytest

from src.retrieval.ranking import RankedProductCandidate, RankedProductSearchResult
from src.retrieval.scoring import (
    COSINE_DISTANCE,
    COSINE_SIMILARITY,
    EUCLIDEAN_DISTANCE,
    SimilarityScoreConverter,
)


def _ranked(*similarities: float) -> RankedProductSearchResult:
    return RankedProductSearchResult(
        query_id="query-1",
        candidates=tuple(
            RankedProductCandidate(
                rank=index,
                product_id=f"p{index}",
                category="Ring",
                similarity=similarity,
                matched_image_ids=(f"img{index}",),
                matched_image_paths=(None,),
            )
            for index, similarity in enumerate(similarities, start=1)
        ),
    )


def test_cosine_similarity_maps_endpoints_and_midpoint() -> None:
    converter = SimilarityScoreConverter()
    assert converter.convert(-1.0) == pytest.approx(0.0)
    assert converter.convert(0.0) == pytest.approx(50.0)
    assert converter.convert(1.0) == pytest.approx(100.0)


def test_cosine_distance_maps_endpoints() -> None:
    converter = SimilarityScoreConverter(COSINE_DISTANCE)
    assert converter.convert(0.0) == pytest.approx(100.0)
    assert converter.convert(1.0) == pytest.approx(50.0)
    assert converter.convert(2.0) == pytest.approx(0.0)


def test_euclidean_distance_maps_normalized_embedding_range() -> None:
    converter = SimilarityScoreConverter(EUCLIDEAN_DISTANCE)
    assert converter.convert(0.0) == pytest.approx(100.0)
    assert converter.convert(1.0) == pytest.approx(50.0)
    assert converter.convert(2.0) == pytest.approx(0.0)


def test_score_preserves_rank_order_and_raw_similarity() -> None:
    result = SimilarityScoreConverter().score(_ranked(0.9, 0.5, 0.1))
    assert [candidate.rank for candidate in result.candidates] == [1, 2, 3]
    assert [candidate.similarity for candidate in result.candidates] == [0.9, 0.5, 0.1]
    assert [candidate.similarity_score for candidate in result.candidates] == pytest.approx(
        [95.0, 75.0, 55.0]
    )


def test_score_returns_new_immutable_result_and_preserves_metadata() -> None:
    ranked = _ranked(0.8)
    scored = SimilarityScoreConverter().score(ranked)
    assert scored is not ranked
    assert scored.query_id == ranked.query_id
    assert scored.candidates[0].product_id == ranked.candidates[0].product_id
    assert scored.candidates[0].matched_image_ids == ranked.candidates[0].matched_image_ids
    assert scored.candidates[0].matched_image_paths == ranked.candidates[0].matched_image_paths


def test_empty_ranked_result_is_supported() -> None:
    scored = SimilarityScoreConverter().score(_ranked())
    assert scored.top_k == 0


def test_unsupported_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported score metric"):
        SimilarityScoreConverter("unknown")


@pytest.mark.parametrize(
    "metric, value",
    [
        (COSINE_SIMILARITY, float("nan")),
        (COSINE_SIMILARITY, float("inf")),
        (COSINE_SIMILARITY, -1.1),
        (COSINE_SIMILARITY, 1.1),
        (COSINE_DISTANCE, -0.1),
        (COSINE_DISTANCE, 2.1),
        (EUCLIDEAN_DISTANCE, -0.1),
        (EUCLIDEAN_DISTANCE, 2.1),
    ],
)
def test_invalid_metric_values_are_rejected(metric: str, value: float) -> None:
    with pytest.raises(ValueError):
        SimilarityScoreConverter(metric).convert(value)


def test_score_rejects_wrong_result_type() -> None:
    with pytest.raises(TypeError, match="RankedProductSearchResult"):
        SimilarityScoreConverter().score(object())  # type: ignore[arg-type]


def test_score_rejects_non_finite_raw_similarity() -> None:
    result = _ranked(0.8)
    candidate = result.candidates[0]
    invalid = RankedProductSearchResult(
        query_id=result.query_id,
        candidates=(
            RankedProductCandidate(
                rank=candidate.rank,
                product_id=candidate.product_id,
                category=candidate.category,
                similarity=float("nan"),
                matched_image_ids=candidate.matched_image_ids,
                matched_image_paths=candidate.matched_image_paths,
            ),
        ),
    )
    with pytest.raises(ValueError, match="Similarity must be finite"):
        SimilarityScoreConverter().score(invalid)
