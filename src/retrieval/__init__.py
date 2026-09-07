"""S2.7 baseline image retrieval components."""

from .embedding import EmbeddingExtractor, load_baseline_embedding_model
from .gallery import Gallery, GalleryBuilder
from .pipeline import BaselineRetrievalPipeline, build_baseline_gallery
from .result import RetrievalCandidate, RetrievalResult
from .similarity import cosine_similarity_matrix
from .topk import TopKRetriever

__all__ = [
    "BaselineRetrievalPipeline",
    "EmbeddingExtractor",
    "Gallery",
    "GalleryBuilder",
    "RetrievalCandidate",
    "RetrievalResult",
    "TopKRetriever",
    "build_baseline_gallery",
    "cosine_similarity_matrix",
    "load_baseline_embedding_model",
]
