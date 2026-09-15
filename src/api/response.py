"""S4.5 stable public response schema for the search API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from retrieval.scoring import ScoredProductSearchResult


class SearchResultItem(BaseModel):
    """One public product result returned by the MVP search API."""

    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1)
    similarity: float = Field(ge=0.0, le=100.0)


class SearchResponse(BaseModel):
    """Stable public response contract for one image-search request.

    ``category`` is the category of the highest-ranked returned product. The
    current retrieval pipeline does not run a separate query classifier, so
    this field represents the best-match category rather than an independent
    classifier prediction.
    """

    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1)
    results: list[SearchResultItem]

    @classmethod
    def from_result(cls, result: ScoredProductSearchResult) -> "SearchResponse":
        """Convert the internal scored result into the public API contract."""
        if not isinstance(result, ScoredProductSearchResult):
            raise TypeError("result must be a ScoredProductSearchResult.")
        if not result.candidates:
            raise ValueError("Cannot build a search response from an empty result.")

        return cls(
            category=result.candidates[0].category.strip().lower(),
            results=[
                SearchResultItem(
                    product_id=candidate.product_id,
                    similarity=float(candidate.similarity_score),
                )
                for candidate in result.candidates
            ],
        )


class ErrorDetail(BaseModel):
    """Stable public error payload."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ErrorResponse(BaseModel):
    """Stable public error response contract."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail
