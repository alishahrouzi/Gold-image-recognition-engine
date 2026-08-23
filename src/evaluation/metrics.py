"""Pure metric calculations for product-level image retrieval evaluation.

Every function here operates on a ranked sequence of booleans
(`is_positive_ranked`) describing, for a single query, whether each gallery
item in descending-similarity order is a Positive match. Functions have no
knowledge of embeddings, similarity, datasets, or file paths, which makes
them trivially unit-testable with small synthetic data and reusable by any
future evaluator (e.g. the Retrieval module).
"""

from __future__ import annotations

from typing import Optional, Sequence


def _validate_ranking(is_positive_ranked: Sequence[bool]) -> None:
    """Raise if `is_positive_ranked` is not a usable sequence."""
    if is_positive_ranked is None:
        raise ValueError("is_positive_ranked must not be None.")


def _validate_k(k: int) -> None:
    """Raise if `k` is not a positive integer."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}.")


def first_positive_rank(is_positive_ranked: Sequence[bool]) -> Optional[int]:
    """Return the 1-indexed rank of the first Positive, or None if absent.

    Args:
        is_positive_ranked: Gallery items in descending-similarity order,
            each flagged True if it is a Positive for the query.

    Returns:
        The 1-indexed rank of the first Positive, or None if no Positive
        is present in the ranking.
    """
    _validate_ranking(is_positive_ranked)
    for rank, is_positive in enumerate(is_positive_ranked, start=1):
        if is_positive:
            return rank
    return None


def reciprocal_rank(is_positive_ranked: Sequence[bool]) -> float:
    """Return 1 / rank_of_first_positive, or 0.0 if no Positive is present."""
    rank = first_positive_rank(is_positive_ranked)
    return 0.0 if rank is None else 1.0 / rank


def mean_reciprocal_rank(reciprocal_ranks: Sequence[float]) -> float:
    """Return the mean of a collection of per-query reciprocal ranks.

    Returns 0.0 for an empty collection rather than raising, since an
    empty query set has no meaningful MRR but callers should not need to
    special-case it.
    """
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def hit_at_k(is_positive_ranked: Sequence[bool], k: int) -> int:
    """Return 1 if at least one Positive appears within the top-K, else 0."""
    _validate_ranking(is_positive_ranked)
    _validate_k(k)
    return 1 if any(is_positive_ranked[:k]) else 0


def precision_at_k(is_positive_ranked: Sequence[bool], k: int) -> float:
    """Return (# Positives in top-K) / K.

    Following the evaluation protocol's definition literally, the
    denominator is always K, even if the gallery contains fewer than K
    items (this can only happen for pathologically small galleries).
    """
    _validate_ranking(is_positive_ranked)
    _validate_k(k)
    top_k = is_positive_ranked[:k]
    return sum(1 for is_positive in top_k if is_positive) / k


def recall_at_k(
    is_positive_ranked: Sequence[bool], k: int, total_positives: int
) -> float:
    """Return (# Positives retrieved in top-K) / total_positives.

    Args:
        is_positive_ranked: Gallery items in descending-similarity order.
        k: Cutoff rank.
        total_positives: Total number of Positive items available in the
            full Gallery for this query. Must be > 0 -- queries with no
            Positive in the Gallery must be excluded upstream (see
            evaluator.py) and never passed here.

    Raises:
        ValueError: If total_positives <= 0.
    """
    _validate_ranking(is_positive_ranked)
    _validate_k(k)
    if total_positives <= 0:
        raise ValueError(
            "total_positives must be > 0; exclude queries with no positive "
            "gallery item before calling recall_at_k."
        )
    top_k = is_positive_ranked[:k]
    return sum(1 for is_positive in top_k if is_positive) / total_positives
