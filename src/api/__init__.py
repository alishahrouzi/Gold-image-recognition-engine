"""HTTP API layer for the functional MVP."""

from .app import SearchService, create_app

__all__ = ["SearchService", "create_app"]
