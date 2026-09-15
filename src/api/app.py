"""S4.4 HTTP search API for the functional MVP.

The API layer is intentionally thin. It accepts an uploaded image, delegates
validation/preprocessing to S4.2, delegates retrieval orchestration to S4.1,
and returns the existing scored-result contract. S4.5 will own the stable
public response schema and S4.6 will translate domain exceptions to API errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.params import Query
from fastapi.responses import JSONResponse

from inference.pipeline import InferencePipeline
from inference.query import QueryProcessor
from retrieval.scoring import ScoredProductSearchResult


@dataclass(frozen=True)
class SearchService:
    """Runtime dependencies required by the S4.4 search endpoint."""

    query_processor: QueryProcessor
    inference_pipeline: InferencePipeline

    def search(self, image_bytes: bytes, *, k: int) -> ScoredProductSearchResult:
        """Process one upload and execute the complete S4.1 inference path."""
        query = self.query_processor.process(image_bytes)
        return self.inference_pipeline.run(query.tensor, query_id="api-query", k=k)


def create_app(service: SearchService) -> FastAPI:
    """Create the FastAPI application around explicit runtime dependencies.

    Dependencies are injected instead of loaded at module import time. This
    keeps API tests independent of local model/gallery artifacts and prevents
    S4.4 from owning checkpoint or gallery lifecycle decisions.
    """
    app = FastAPI(
        title="Gold Visual Search API",
        version="0.1.0",
        description="Functional MVP image-based gold product retrieval API.",
    )
    app.state.search_service = service

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return a minimal liveness response for local/API smoke tests."""
        return {"status": "ok"}

    @app.post("/search")
    async def search(
        file: UploadFile = File(..., description="Query jewelry image"),
        k: int = Query(5, ge=1, le=50, description="Number of products to return"),
    ) -> Any:
        """Search the configured gallery for products similar to one image.

        S4.4 deliberately exposes only transport concerns. Validation and
        preprocessing remain in S4.2; embedding/search/ranking/scoring remain
        in S4.1/S3.8-S3.10. User-facing exception translation is deferred to
        S4.6 and the response shape is formalized in S4.5.
        """
        image_bytes = await file.read()
        result = app.state.search_service.search(image_bytes, k=k)
        return JSONResponse(content=jsonable_encoder(result))

    return app
