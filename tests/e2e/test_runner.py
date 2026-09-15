"""Unit tests for S4.8 E2E scenario isolation and failure reporting."""

from pathlib import Path

from scripts.run_e2e_tests import _run_scenario


class _Response:
    def __init__(self, body: dict):
        self._body = body
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


class _Client:
    def __init__(self, body: dict):
        self.body = body

    def post(self, *args, **kwargs) -> _Response:
        return _Response(self.body)

    def get(self, *args, **kwargs) -> _Response:
        return _Response({"status": "ok"})


def test_failed_category_is_recorded_without_raising(tmp_path: Path) -> None:
    image_path = tmp_path / "ring.jpg"
    image_path.write_bytes(b"image")
    client = _Client(
        {
            "category": "pendant",
            "results": [
                {"product_id": "P001", "similarity": 91.0},
                {"product_id": "P002", "similarity": 88.0},
            ],
        }
    )

    result = _run_scenario(
        client,
        "http://127.0.0.1:8000",
        "Test 3 — different angle",
        image_path,
        2,
        expected_category="ring",
    )

    assert result["status"] == "failed"
    assert result["expected_category"] == "ring"
    assert result["category"] == "pendant"
    assert result["results"] == [
        {"product_id": "P001", "similarity": 91.0},
        {"product_id": "P002", "similarity": 88.0},
    ]
    assert "expected category 'ring', got 'pendant'" in result["error"]


def test_request_failure_is_recorded_without_raising(tmp_path: Path) -> None:
    image_path = tmp_path / "ring.jpg"
    image_path.write_bytes(b"image")

    class _FailingClient:
        def post(self, *args, **kwargs):
            raise OSError("connection refused")

    result = _run_scenario(
        _FailingClient(),
        "http://127.0.0.1:8000",
        "Test 4 — different lighting",
        image_path,
        5,
        expected_category="ring",
    )

    assert result["status"] == "failed"
    assert result["error"] == "connection refused"
    assert result["expected_category"] == "ring"
