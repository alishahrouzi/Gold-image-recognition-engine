"""HTTP API layer for the functional MVP."""

from .app import SearchService, create_app
from .response import SearchResponse, SearchResultItem

__all__ = ["SearchResponse", "SearchResultItem", "SearchService", "create_app"]
