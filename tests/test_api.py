"""Tests for the FastAPI routes."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def api_key() -> str:
    """Get the API key from environment for authenticated requests."""
    return os.getenv("API_KEY", "")


@pytest.fixture
def auth_headers(api_key: str) -> dict:
    """Build headers with API key if one is set."""
    if api_key:
        return {"X-API-Key": api_key}
    return {}


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        data: dict = response.json()
        assert data["status"] in ("healthy", "degraded")

    def test_health_returns_ollama_model(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        data: dict = response.json()
        assert "model_name" in data
        assert "ollama_available" in data
        assert "sklearn_available" in data


class TestCategoriesEndpoint:
    """Tests for the categories endpoint."""

    def test_categories_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/categories")
        assert response.status_code == 200

    def test_categories_returns_list(self, client: TestClient) -> None:
        response = client.get("/api/v1/categories")
        data: dict = response.json()
        assert "categories" in data
        assert len(data["categories"]) == 5

    def test_categories_returns_display_names(self, client: TestClient) -> None:
        response = client.get("/api/v1/categories")
        data: dict = response.json()
        assert "display_names" in data


class TestPredictEndpoint:
    """Tests for the prediction endpoint."""

    def test_predict_returns_200(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = {
                "issue_category": "supply_chain_delay",
                "confidence": 0.95,
                "method": "ollama_base",
                "latency_seconds": "1.23",
                "reasoning": "Test prediction",
            }
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": "Retailers complaining about delayed replenishment"},
                headers=auth_headers,
            )
            assert response.status_code == 200

    def test_predict_returns_category(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = {
                "issue_category": "supply_chain_delay",
                "confidence": 0.95,
                "method": "ollama_base",
                "latency_seconds": "1.23",
                "reasoning": "Test prediction",
            }
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": "test note"},
                headers=auth_headers,
            )
            data: dict = response.json()
            assert "issue_category" in data
            assert data["issue_category"] == "supply_chain_delay"

    def test_predict_returns_confidence(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = {
                "issue_category": "supply_chain_delay",
                "confidence": 0.95,
                "method": "ollama_base",
                "latency_seconds": "1.23",
                "reasoning": "Test prediction",
            }
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": "test note"},
                headers=auth_headers,
            )
            data: dict = response.json()
            assert "confidence" in data
            assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_returns_method(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = {
                "issue_category": "supply_chain_delay",
                "confidence": 0.95,
                "method": "ollama_base",
                "latency_seconds": "1.23",
                "reasoning": "Test prediction",
            }
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": "test note"},
                headers=auth_headers,
            )
            data: dict = response.json()
            assert "method" in data

    def test_predict_empty_note_returns_422(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            mock_instance = MagicMock()
            mock_instance.predict.side_effect = ValueError("Note cannot be empty")
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": ""},
                headers=auth_headers,
            )
            assert response.status_code == 422

    def test_predict_runtime_error_returns_503(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            mock_instance = MagicMock()
            mock_instance.predict.side_effect = RuntimeError("Model inference failed")
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": "test"},
                headers=auth_headers,
            )
            assert response.status_code == 503

    def test_predict_missing_field_returns_422(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        response = client.post(
            "/api/v1/predict",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for the batch prediction endpoint."""

    def test_batch_predict_returns_200(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = {
                "issue_category": "supply_chain_delay",
                "confidence": 0.95,
                "method": "ollama_base",
                "latency_seconds": "1.23",
                "reasoning": "Test prediction",
            }
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict/batch",
                json={"notes": ["note one", "note two"]},
                headers=auth_headers,
            )
            assert response.status_code == 200

    def test_batch_predict_returns_all_results(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = {
                "issue_category": "supply_chain_delay",
                "confidence": 0.95,
                "method": "ollama_base",
                "latency_seconds": "1.23",
                "reasoning": "Test prediction",
            }
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict/batch",
                json={"notes": ["note one", "note two"]},
                headers=auth_headers,
            )
            data: dict = response.json()
            assert data["total_notes"] == 2
            assert len(data["predictions"]) == 2
