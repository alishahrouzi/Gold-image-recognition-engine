"""Tests for the S4.4 HTTP search API boundary."""

from __future__ import annotations

from io import BytesIO

import torch
from fastapi.testclient import TestClient
from PIL import Image

from api.app import SearchService, create_app
from inference.query import QueryProcessor
from retrieval.scoring import ScoredProductCandidate, ScoredProductSearchResult


class FakePipeline:
    """Small S4.1-compatible test double that records API inputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, str, int]] = []

    def run(self, image: torch.Tensor, *, query_id: str, k: int):
        self.calls.append((image, query_id, k))
        candidate = ScoredProductCandidate(
            rank=1,
            product_id="p-ring",
            category="Ring",
            similarity=0.9,
            similarity_score=95.0,
            matched_image_ids=("ring-1",),
            matched_image_paths=("ring-1.jpg",),
        )
        return ScoredProductSearchResult(query_id=query_id, candidates=(candidate,))


def _image_bytes() -> bytes:
    image = Image.new("RGB", (128, 96))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _client() -> tuple[TestClient, FakePipeline]:
    fake_pipeline = FakePipeline()
    service = SearchService(QueryProcessor(), fake_pipeline)  # type: ignore[arg-type]
    return TestClient(create_app(service)), fake_pipeline


def test_health_endpoint_is_available() -> None:
    client, _ = _client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_accepts_image_upload_and_forwards_processed_tensor() -> None:
    client, pipeline = _client()

    response = client.post(
        "/search",
        files={"file": ("query.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query_id"] == "api-query"
    assert len(body["candidates"]) == 1
    assert body["candidates"][0]["product_id"] == "p-ring"
    assert body["candidates"][0]["similarity_score"] == 95.0
    assert len(pipeline.calls) == 1
    image_tensor, query_id, k = pipeline.calls[0]
    assert image_tensor.shape == (3, 224, 224)
    assert image_tensor.dtype == torch.float32
    assert query_id == "api-query"
    assert k == 5


def test_search_forwards_custom_k() -> None:
    client, pipeline = _client()

    response = client.post(
        "/search?k=10",
        files={"file": ("query.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert pipeline.calls[0][2] == 10


def test_search_rejects_invalid_k_at_http_boundary() -> None:
    client, _ = _client()

    response = client.post(
        "/search?k=0",
        files={"file": ("query.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 422
