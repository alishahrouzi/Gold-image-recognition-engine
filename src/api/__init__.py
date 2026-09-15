"""HTTP API layer for the functional MVP."""

from .app import SearchService, create_app
from .errors import (
    APIError,
    FileTooLargeError,
    GalleryUnavailableError,
    InternalAPIError,
    InvalidImageError,
    MissingFileError,
    ModelUnavailableError,
    NoResultsError,
)
from .response import ErrorResponse, SearchResponse, SearchResultItem
from .ui import ProductImageResolver

__all__ = [
    "APIError", "ErrorResponse", "FileTooLargeError", "GalleryUnavailableError",
    "InternalAPIError", "InvalidImageError", "MissingFileError", "ModelUnavailableError",
    "NoResultsError", "ProductImageResolver", "SearchResponse", "SearchResultItem",
    "SearchService", "create_app",
]
