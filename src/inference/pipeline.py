"""S4.1 inference orchestration for the functional MVP.

This module composes already-implemented S2/S3 components into the runtime
inference path. It deliberately does not own raw-upload/query validation,
gallery loading, HTTP/API concerns, response serialization, or exception
translation; those responsibilities belong to S4.2-S4.6.

Pipeline:

    preprocessed image tensor
        -> embedding extraction
        -> product similarity search
        -> product ranking
        -> display Similarity Score
        -> scored product-search result
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from src.retrieval.ranking import ProductRanker
from src.retrieval.scoring import ScoredProductSearchResult, SimilarityScoreConverter
from src.retrieval.search import SimilaritySearchEngine


class QueryEmbedder(Protocol):
    """Minimal embedding-extraction contract required by S4.1."""

    def extract_one(self, image: Tensor) -> Tensor:
        """Return one normalized query embedding from a preprocessed image."""
        ...


class InferencePipeline:
    """Compose embedding, search, ranking, and scoring for one query.

    Dependencies are injected so S4.1 remains independent of checkpoint and
    gallery filesystem paths. S4.2-S4.3 can construct those dependencies and
    S4.4 can expose this pipeline through an API without changing its core.

    The input must already be the project's deterministic preprocessed tensor
    with shape ``[3, 224, 224]``. Query preprocessing belongs to S4.2.
    """

    def __init__(
        self,
        embedder: QueryEmbedder,
        search_engine: SimilaritySearchEngine,
        *,
        ranker: ProductRanker | None = None,
        score_converter: SimilarityScoreConverter | None = None,
    ) -> None:
        self.embedder = embedder
        self.search_engine = search_engine
        self.ranker = ranker or ProductRanker()
        self.score_converter = score_converter or SimilarityScoreConverter()

    def run(
        self,
        image: Tensor,
        *,
        query_id: str = "query",
        k: int = 5,
        exclude_image_id: str | None = None,
    ) -> ScoredProductSearchResult:
        """Run the complete S4.1 inference chain for one preprocessed image.

        No stage is duplicated here: embedding extraction is delegated to the
        existing ``EmbeddingExtractor`` contract, while S3.8, S3.9, and S3.10
        remain the owners of search, ranking, and display-score conversion.
        Exceptions are intentionally propagated unchanged for S4.6 to handle.
        """
        embedding = self.embedder.extract_one(image)
        search_result = self.search_engine.search(
            embedding,
            query_id=query_id,
            k=k,
            exclude_image_id=exclude_image_id,
        )
        ranked_result = self.ranker.rank(search_result)
        return self.score_converter.score(ranked_result)
