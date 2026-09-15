"""Tests for the S4.4/S4.5 HTTP search API boundary."""

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
        candidates = tuple(
            ScoredProductCandidate(
                rank=rank,
                product_id=product_id,
                category=category,
                similarity=raw_similarity,
                similarity_score=display_score,
                matched_image_ids=(f"{product_id}-1",),
                matched_image_paths=(f"{product_id}-1.jpg",),
            )
            for rank, product_id, category, raw_similarity, display_score in (
                (1, "p-ring", "Ring", 0.9, 95.0),
                (2, "p-necklace", "Necklace", 0.7, 85.0),
            )
        )
        return ScoredProductSearchResult(query_id=query_id, candidates=candidates)


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


def test_search_returns_stable_public_response_schema() -> None:
    client, pipeline = _client()

    response = client.post(
        "/search",
        files={"file": ("query.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"category", "results"}
    assert body["category"] == "ring"
    assert body["results"] == [
        {"product_id": "p-ring", "similarity": 95.0},
        {"product_id": "p-necklace", "similarity": 85.0},
    ]
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
