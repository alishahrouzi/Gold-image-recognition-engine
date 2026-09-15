"""S4.4 HTTP search API, S4.6 error boundary, and S4.7 MVP UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.params import Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from data.errors import (
    FileTooLargeError as DataFileTooLargeError,
    ImageTooSmallError as DataImageTooSmallError,
    InvalidImageError as DataInvalidImageError,
    PreprocessingError,
)
from inference.gallery import RuntimeGallery
from inference.pipeline import InferencePipeline
from inference.query import QueryProcessor
from retrieval.scoring import ScoredProductSearchResult

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
from .response import ErrorResponse, SearchResponse
from .ui import ProductImageResolver


STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass(frozen=True)
class SearchService:
    """Runtime dependencies required by the search endpoint."""

    query_processor: QueryProcessor
    inference_pipeline: InferencePipeline
    image_resolver: ProductImageResolver | None = None

    def search(self, image_bytes: bytes, *, k: int) -> ScoredProductSearchResult:
        """Process one upload and execute the complete S4.1 inference path."""
        try:
            query = self.query_processor.process(image_bytes)
        except DataFileTooLargeError as exc:
            raise FileTooLargeError() from exc
        except (DataInvalidImageError, DataImageTooSmallError, PreprocessingError) as exc:
            raise InvalidImageError() from exc
        return self.inference_pipeline.run(query.tensor, query_id="api-query", k=k)


def _error_response(error: APIError) -> JSONResponse:
    body = ErrorResponse(error={"code": error.code, "message": error.message}).model_dump()
    return JSONResponse(status_code=error.status_code, content=body)


def create_app(service: SearchService) -> FastAPI:
    """Create the FastAPI application around explicit runtime dependencies."""
    app = FastAPI(
        title="Gold Visual Search API",
        version="0.1.0",
        description="Functional MVP image-based gold product retrieval API.",
    )
    app.state.search_service = service
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(APIError)
    async def api_error_handler(_request: Request, exc: APIError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        missing_file = any(
            error.get("loc", ())[-1:] == ("file",)
            and error.get("type") in {"missing", "value_error.missing"}
            for error in errors
        )
        if missing_file:
            return _error_response(MissingFileError())
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(NoResultsError)
    async def no_results_handler(_request: Request, exc: NoResultsError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable_handler(_request: Request, exc: ModelUnavailableError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(GalleryUnavailableError)
    async def gallery_unavailable_handler(_request: Request, exc: GalleryUnavailableError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, _exc: Exception) -> JSONResponse:
        return _error_response(InternalAPIError())

    @app.get("/")
    def ui() -> FileResponse:
        """Serve the dependency-free MVP web interface."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/result-image/{product_id:path}")
    def result_image(product_id: str) -> FileResponse:
        """Serve a representative image from trusted runtime gallery metadata."""
        resolver = app.state.search_service.image_resolver
        if resolver is None:
            raise GalleryUnavailableError("Product image gallery is not configured.")
        return resolver.response(product_id)

    @app.post("/search", response_model=SearchResponse)
    async def search(
        file: UploadFile = File(..., description="Query jewelry image"),
        k: int = Query(5, ge=1, le=50, description="Number of products to return"),
    ) -> SearchResponse:
        image_bytes = await file.read()
        result = app.state.search_service.search(image_bytes, k=k)
        try:
            return SearchResponse.from_result(result)
        except ValueError as exc:
            raise NoResultsError() from exc

    return app
