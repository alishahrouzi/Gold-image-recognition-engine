"""S4.6 API error-handling contract tests."""

from __future__ import annotations

from io import BytesIO

import torch
from fastapi.testclient import TestClient
from PIL import Image

from api.app import SearchService, create_app
from api.errors import GalleryUnavailableError, ModelUnavailableError
from inference.query import QueryProcessor
from retrieval.scoring import ScoredProductCandidate, ScoredProductSearchResult


class FakePipeline:
    def __init__(self, error: Exception | None = None, *, empty: bool = False) -> None:
        self.error = error
        self.empty = empty

    def run(self, image: torch.Tensor, *, query_id: str, k: int):
        if self.error is not None:
            raise self.error
        if self.empty:
            return ScoredProductSearchResult(query_id=query_id, candidates=())
        candidate = ScoredProductCandidate(
            rank=1,
            product_id="p-ring",
            category="Ring",
            similarity=0.9,
            similarity_score=95.0,
            matched_image_ids=("p-ring-1",),
            matched_image_paths=("p-ring-1.jpg",),
        )
        return ScoredProductSearchResult(query_id=query_id, candidates=(candidate,))


def _image_bytes(size: tuple[int, int] = (128, 96)) -> bytes:
    image = Image.new("RGB", size)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _client(
    *,
    pipeline: FakePipeline | None = None,
    processor: QueryProcessor | None = None,
) -> TestClient:
    service = SearchService(processor or QueryProcessor(), pipeline or FakePipeline())  # type: ignore[arg-type]
    return TestClient(create_app(service))


def test_non_image_file_returns_invalid_image() -> None:
    response = _client().post(
        "/search",
        files={"file": ("note.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "INVALID_IMAGE",
            "message": "The uploaded file is not a supported image.",
        }
    }


def test_corrupt_image_returns_invalid_image() -> None:
    response = _client().post(
        "/search",
        files={"file": ("broken.png", b"\x89PNG\r\nnot-a-real-png", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_empty_upload_returns_invalid_image() -> None:
    response = _client().post(
        "/search",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_very_large_upload_returns_file_too_large() -> None:
    processor = QueryProcessor(max_bytes=64)
    response = _client(processor=processor).post(
        "/search",
        files={"file": ("large.bin", b"x" * 65, "application/octet-stream")},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_empty_request_returns_missing_file() -> None:
    response = _client().post("/search")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MISSING_FILE"
    assert response.json()["error"]["message"] == "An image file is required."


def test_model_unavailable_returns_503() -> None:
    response = _client(pipeline=FakePipeline(ModelUnavailableError())).post(
        "/search",
        files={"file": ("query.png", _image_bytes(), "image/png")},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_gallery_unavailable_returns_503() -> None:
    response = _client(pipeline=FakePipeline(GalleryUnavailableError())).post(
        "/search",
        files={"file": ("query.png", _image_bytes(), "image/png")},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "GALLERY_UNAVAILABLE"


def test_unexpected_error_returns_safe_500() -> None:
    response = _client(pipeline=FakePipeline(RuntimeError("secret internal detail"))).post(
        "/search",
        files={"file": ("query.png", _image_bytes(), "image/png")},
    )
    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred while processing the request.",
        }
    }


def test_no_results_returns_404() -> None:
    response = _client(pipeline=FakePipeline(empty=True)).post(
        "/search",
        files={"file": ("query.png", _image_bytes(), "image/png")},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NO_RESULTS"
