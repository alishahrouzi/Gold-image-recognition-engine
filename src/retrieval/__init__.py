"""Retrieval components for the baseline and S3 metric-learning pipeline."""

from .embedding import EmbeddingExtractor, load_baseline_embedding_model
from .gallery import Gallery, GalleryBuilder
from .pipeline import BaselineRetrievalPipeline, build_baseline_gallery
from .result import RetrievalCandidate, RetrievalResult
from .search import SimilaritySearchEngine
from .search_result import ProductSearchCandidate, ProductSearchResult
from .similarity import cosine_similarity_matrix
from .topk import TopKRetriever

__all__ = [
    "BaselineRetrievalPipeline",
    "EmbeddingExtractor",
    "Gallery",
    "GalleryBuilder",
    "ProductSearchCandidate",
    "ProductSearchResult",
    "RetrievalCandidate",
    "RetrievalResult",
    "SimilaritySearchEngine",
    "TopKRetriever",
    "build_baseline_gallery",
    "cosine_similarity_matrix",
    "load_baseline_embedding_model",
]
