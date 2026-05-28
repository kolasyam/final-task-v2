"""Tests for the FastAPI routes.

Tests all API endpoints with mocked backends, covering:
- Happy paths for each endpoint
- Error handling and HTTP status code mapping
- Authentication when API key is configured
- Request validation
"""

import os
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.constants import SUPPORTED_CATEGORIES
from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """Build headers with API key if one is set."""
    api_key: str = os.getenv("API_KEY", "")
    if api_key:
        return {"X-API-Key": api_key}
    return {}


def _setup_mock_predictor(**overrides: object) -> MagicMock:
    """Helper to create a mock predictor with default return values.

    Args:
        **overrides: Keyword arguments to override default return values.

    Returns:
        Tuple of (mock_instance, MockPredictor_class) for patching.
    """
    defaults: Dict[str, object] = {
        "issue_category": "supply_chain_delay",
        "confidence": 0.95,
        "method": "ollama_base",
        "latency_seconds": "1.23",
        "reasoning": "Test prediction",
    }
    defaults.update(overrides)

    mock_instance: MagicMock = MagicMock()
    mock_instance.predict.return_value = defaults
    return mock_instance


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        data: Dict = response.json()
        assert data["status"] in ("healthy", "degraded")

    def test_health_returns_model_info(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        data: Dict = response.json()
        assert "model_name" in data

    def test_health_returns_qlora_available(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        data: Dict = response.json()
        assert "qlora_available" in data
        assert "sklearn_available" in data

    def test_health_returns_supported_categories(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        data: Dict = response.json()
        assert "supported_categories" in data
        assert len(data["supported_categories"]) == 5

    def test_health_is_exempt_from_auth(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200


class TestCategoriesEndpoint:
    """Tests for the categories endpoint."""

    def test_categories_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/categories")
        assert response.status_code == 200

    def test_categories_returns_list(self, client: TestClient) -> None:
        response = client.get("/api/v1/categories")
        data: Dict = response.json()
        assert "categories" in data
        assert len(data["categories"]) == 5

    def test_categories_returns_display_names(self, client: TestClient) -> None:
        response = client.get("/api/v1/categories")
        data: Dict = response.json()
        assert "display_names" in data

    def test_categories_match_supported(self, client: TestClient) -> None:
        response = client.get("/api/v1/categories")
        data: Dict = response.json()
        for cat in SUPPORTED_CATEGORIES:
            assert cat in data["categories"]


class TestPredictEndpoint:
    """Tests for the prediction endpoint."""

    def test_predict_returns_200(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        mock_instance = _setup_mock_predictor()

        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": "Retailers complaining about delayed replenishment"},
                headers=auth_headers,
            )
            assert response.status_code == 200

    def test_predict_returns_category(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        mock_instance = _setup_mock_predictor()

        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": "test note"},
                headers=auth_headers,
            )
            data: Dict = response.json()
            assert "issue_category" in data

    def test_predict_returns_confidence(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        mock_instance = _setup_mock_predictor()

        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": "test note"},
                headers=auth_headers,
            )
            data: Dict = response.json()
            assert "confidence" in data
            assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_returns_method(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        mock_instance = _setup_mock_predictor()

        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": "test note"},
                headers=auth_headers,
            )
            data: Dict = response.json()
            assert "method" in data

    def test_predict_returns_latency(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        mock_instance = _setup_mock_predictor()

        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": "test note"},
                headers=auth_headers,
            )
            data: Dict = response.json()
            assert "latency_seconds" in data

    def test_predict_returns_reasoning(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        mock_instance = _setup_mock_predictor()

        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": "test note"},
                headers=auth_headers,
            )
            data: Dict = response.json()
            assert "reasoning" in data

    def test_predict_empty_note_returns_422(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        mock_instance = _setup_mock_predictor()
        mock_instance.predict.side_effect = ValueError("Note cannot be empty")

        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": ""},
                headers=auth_headers,
            )
            assert response.status_code == 422

    def test_predict_runtime_error_returns_503(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        mock_instance = _setup_mock_predictor()
        mock_instance.predict.side_effect = RuntimeError("Model inference failed")

        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": "test"},
                headers=auth_headers,
            )
            assert response.status_code == 503

    def test_predict_missing_field_returns_422(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        response = client.post(
            "/api/v1/predict",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_predict_note_too_long_returns_422(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        response = client.post(
            "/api/v1/predict",
            json={"rep_note": "x" * 1001},
            headers=auth_headers,
        )
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for the batch prediction endpoint."""

    def test_batch_predict_returns_200(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        mock_instance = _setup_mock_predictor()

        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict/batch",
                json={"notes": ["note one", "note two"]},
                headers=auth_headers,
            )
            assert response.status_code == 200

    def test_batch_predict_returns_all_results(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        mock_instance = _setup_mock_predictor()

        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict/batch",
                json={"notes": ["note one", "note two"]},
                headers=auth_headers,
            )
            data: Dict = response.json()
            assert data["total_notes"] == 2
            assert len(data["predictions"]) == 2

    def test_batch_predict_single_note(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        mock_instance = _setup_mock_predictor()

        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict/batch",
                json={"notes": ["single note"]},
                headers=auth_headers,
            )
            data: Dict = response.json()
            assert data["total_notes"] == 1

    def test_batch_predict_empty_list_returns_422(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        response = client.post(
            "/api/v1/predict/batch",
            json={"notes": []},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_batch_predict_too_many_notes_returns_422(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        response = client.post(
            "/api/v1/predict/batch",
            json={"notes": ["note"] * 51},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_batch_predict_empty_note_in_list_returns_422(
        self, client: TestClient, auth_headers: Dict[str, str],
    ) -> None:
        response = client.post(
            "/api/v1/predict/batch",
            json={"notes": ["valid note", ""]},
            headers=auth_headers,
        )
        assert response.status_code == 422


class TestStatusEndpoint:
    """Tests for the full status endpoint."""

    def test_status_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/status")
        assert response.status_code == 200

    def test_status_returns_inference_info(self, client: TestClient) -> None:
        response = client.get("/api/v1/status")
        data: Dict = response.json()
        assert "inference" in data
        assert "status" in data


class TestOpenApiDocs:
    """Tests for OpenAPI documentation endpoints."""

    def test_docs_accessible(self, client: TestClient) -> None:
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json_accessible(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_redoc_accessible(self, client: TestClient) -> None:
        response = client.get("/redoc")
        assert response.status_code == 200
