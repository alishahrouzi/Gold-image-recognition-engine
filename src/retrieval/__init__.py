"""Retrieval components for the baseline and S3 metric-learning pipeline."""

from .embedding import EmbeddingExtractor, load_baseline_embedding_model
from .gallery import Gallery, GalleryBuilder
from .pipeline import BaselineRetrievalPipeline, build_baseline_gallery
from .ranking import ProductRanker, RankedProductCandidate, RankedProductSearchResult
from .result import RetrievalCandidate, RetrievalResult
from .scoring import (
    COSINE_DISTANCE,
    COSINE_SIMILARITY,
    EUCLIDEAN_DISTANCE,
    ScoredProductCandidate,
    ScoredProductSearchResult,
    SimilarityScoreConverter,
)
from .search import SimilaritySearchEngine
from .search_result import ProductSearchCandidate, ProductSearchResult
from .similarity import cosine_similarity_matrix
from .topk import TopKRetriever

__all__ = [
    "BaselineRetrievalPipeline",
    "COSINE_DISTANCE",
    "COSINE_SIMILARITY",
    "EmbeddingExtractor",
    "EUCLIDEAN_DISTANCE",
    "Gallery",
    "GalleryBuilder",
    "ProductRanker",
    "ProductSearchCandidate",
    "ProductSearchResult",
    "RankedProductCandidate",
    "RankedProductSearchResult",
    "RetrievalCandidate",
    "RetrievalResult",
    "ScoredProductCandidate",
    "ScoredProductSearchResult",
    "SimilaritySearchEngine",
    "SimilarityScoreConverter",
    "TopKRetriever",
    "build_baseline_gallery",
    "cosine_similarity_matrix",
    "load_baseline_embedding_model",
]
