"""S4.7 MVP UI/API integration tests."""

from pathlib import Path

import torch
from fastapi.testclient import TestClient

from api.app import SearchService, create_app
from api.ui import ProductImageResolver
from inference.query import QueryProcessor
from retrieval.scoring import ScoredProductCandidate, ScoredProductSearchResult


class _QueryProcessor:
    def process(self, image_bytes: bytes):
        class Query:
            tensor = torch.zeros(3, 224, 224)
        return Query()


class _Pipeline:
    def run(self, tensor, *, query_id: str, k: int):
        candidates = tuple(
            ScoredProductCandidate(
                rank=index + 1,
                product_id=product_id,
                category="Ring" if index == 0 else "Earrings",
                similarity=0.9 - index * 0.1,
                similarity_score=95.0 - index * 5.0,
                matched_image_ids=(f"image-{index}",),
                matched_image_paths=(None,),
            )
            for index, product_id in enumerate(("P001", "P002"))
        )
        return ScoredProductSearchResult(query_id=query_id, candidates=candidates[:k])


def _client(tmp_path: Path, *, processor=None) -> TestClient:
    image_path = tmp_path / "P001.jpg"
    image_path.write_bytes(b"fake-image")
    from types import SimpleNamespace

    gallery = SimpleNamespace(
        gallery=SimpleNamespace(
            metadata=(
                {"product_id": "P001", "category": "Ring", "image": str(image_path), "image_id": "image-0"},
            )
        )
    )
    service = SearchService(
        processor or _QueryProcessor(),
        _Pipeline(),
        ProductImageResolver(gallery),
    )
    return TestClient(create_app(service), raise_server_exceptions=False)


def test_ui_page_is_served(tmp_path: Path) -> None:
    response = _client(tmp_path).get("/")
    assert response.status_code == 200
    assert "Gold Visual Search" in response.text
    assert "/static/app.js" in response.text


def test_search_api_contract_is_renderable_by_ui(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/search?k=2",
        files={"file": ("query.jpg", b"image-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "category": "ring",
        "results": [
            {"product_id": "P001", "similarity": 95.0},
            {"product_id": "P002", "similarity": 90.0},
        ],
    }


def test_ui_receives_public_error_envelope(tmp_path: Path) -> None:
    response = _client(tmp_path, processor=QueryProcessor()).post(
        "/search",
        files={"file": ("bad.txt", b"not-image", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "INVALID_IMAGE",
            "message": "The uploaded file is not a supported image.",
        }
    }


def test_result_image_endpoint_serves_gallery_image(tmp_path: Path) -> None:
    response = _client(tmp_path).get("/result-image/P001")
    assert response.status_code == 200
    assert response.content == b"fake-image"


def test_result_image_endpoint_returns_404_for_unknown_product(tmp_path: Path) -> None:
    response = _client(tmp_path).get("/result-image/UNKNOWN")
    assert response.status_code == 404


def test_static_assets_are_served(tmp_path: Path) -> None:
    response = _client(tmp_path).get("/static/app.js")
    assert response.status_code == 200
    assert "fetch(`/search" in response.text
