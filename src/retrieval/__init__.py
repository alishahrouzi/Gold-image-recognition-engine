"""S2.7 baseline image retrieval components."""

from .embedding import EmbeddingExtractor, load_baseline_embedding_model
from .gallery import Gallery, GalleryBuilder
from .result import RetrievalCandidate, RetrievalResult
from .similarity import cosine_similarity_matrix
from .topk import TopKRetriever

__all__ = [
    "EmbeddingExtractor",
    "Gallery",
    "GalleryBuilder",
    "RetrievalCandidate",
    "RetrievalResult",
    "TopKRetriever",
    "cosine_similarity_matrix",
    "load_baseline_embedding_model",
]
