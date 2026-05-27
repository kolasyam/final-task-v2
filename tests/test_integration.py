"""Integration tests for the Sales Intelligence pipeline.

Tests the end-to-end flow from dataset preparation through training,
prediction, and evaluation without requiring a real Ollama server or GPU.

Uses the shared fixtures from tests/conftest.py.
"""

import os
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.constants import SUPPORTED_CATEGORIES
from app.core.exceptions import CircuitBreakerOpenError
from app.main import app
from app.services.circuit_breaker import CircuitBreaker


# --- API Integration Tests ---


class TestApiIntegration:
    """Integration tests for the API layer."""

    def test_health_endpoint_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_categories_endpoint_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/categories")
        assert response.status_code == 200
        data: Dict = response.json()
        assert len(data["categories"]) == 5

    def test_correlation_id_header_present(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert "X-Correlation-ID" in response.headers

    def test_custom_correlation_id_preserved(self, client: TestClient) -> None:
        custom_id: str = "test-correlation-123"
        response = client.get(
            "/api/v1/health",
            headers={"X-Correlation-ID": custom_id},
        )
        assert response.headers["X-Correlation-ID"] == custom_id

    def test_predict_with_mock_returns_200(self, client: TestClient) -> None:
        auth_headers: Dict[str, str] = {}
        api_key: str = os.getenv("API_KEY", "")
        if api_key:
            auth_headers["X-API-Key"] = api_key

        with patch("app.api.routes._predictor", new=None), \
             patch("app.api.routes.SalesNotePredictor") as MockPredictor:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = {
                "issue_category": "supply_chain_delay",
                "confidence": 0.95,
                "method": "ollama_base",
                "latency_seconds": "1.23",
                "reasoning": "Test",
            }
            MockPredictor.return_value = mock_instance

            response = client.post(
                "/api/v1/predict",
                json={"rep_note": "Test note"},
                headers=auth_headers,
            )
            assert response.status_code == 200

    def test_openapi_docs_accessible(self, client: TestClient) -> None:
        response = client.get("/docs")
        assert response.status_code == 200

    def test_full_status_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data: Dict = response.json()
        assert "inference" in data
        assert "status" in data


# --- Circuit Breaker Integration Tests ---


class TestCircuitBreakerIntegration:
    """Integration tests for the circuit breaker pattern."""

    def test_starts_closed(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3)
        assert breaker.state.value == "closed"

    def test_opens_after_threshold(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2)

        def fail() -> None:
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            breaker.call(fail)
        assert breaker.state.value == "closed"

        with pytest.raises(RuntimeError):
            breaker.call(fail)
        assert breaker.state.value == "open"

    def test_raises_breaker_error_when_open(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60)

        def fail() -> None:
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            breaker.call(fail)

        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(fail)

    def test_closes_after_recovery(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0)

        def fail() -> None:
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            breaker.call(fail)

        # With recovery_timeout=0, should transition to half_open immediately
        breaker._last_failure_time = 0  # Force expiry

        call_count: int = 0

        def succeed() -> int:
            nonlocal call_count
            call_count += 1
            return 1

        result = breaker.call(succeed)
        assert result == 1
        assert breaker.state.value == "closed"

    def test_status_report(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3)
        status: Dict = breaker.get_status()
        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status


# --- Dataset Preparation Tests ---


class TestDatasetPreparation:
    """Tests for the dataset preparation pipeline."""

    def test_load_dataset(self, sample_dataset: str) -> None:
        from training.prepare_dataset import prepare_dataset

        train_df, test_df = prepare_dataset(
            dataset_path=sample_dataset,
            test_size=0.2,
            random_seed=42,
        )
        assert len(train_df) > 0
        assert len(test_df) > 0

    def test_dataset_has_required_columns(self, sample_dataset: str) -> None:
        from training.prepare_dataset import prepare_dataset

        train_df, test_df = prepare_dataset(
            dataset_path=sample_dataset,
            test_size=0.2,
            random_seed=42,
        )
        assert "input" in train_df.columns
        assert "output" in train_df.columns
        assert "input" in test_df.columns
        assert "output" in test_df.columns

    def test_all_categories_present(self, sample_dataset: str) -> None:
        from training.prepare_dataset import prepare_dataset

        train_df, _ = prepare_dataset(
            dataset_path=sample_dataset,
            test_size=0.2,
            random_seed=42,
        )
        unique_cats = set(train_df["output"].unique())
        for cat in SUPPORTED_CATEGORIES:
            assert cat in unique_cats, f"Category {cat} missing from training data"

    def test_missing_file_raises_error(self) -> None:
        from training.prepare_dataset import prepare_dataset

        with pytest.raises(FileNotFoundError):
            prepare_dataset(dataset_path="/nonexistent/path.xlsx")

    def test_stratified_split(self, sample_dataset: str) -> None:
        from training.prepare_dataset import prepare_dataset

        train_df, test_df = prepare_dataset(
            dataset_path=sample_dataset,
            test_size=0.2,
            random_seed=42,
        )
        # Each category should be present in both splits
        train_cats = set(train_df["output"].unique())
        test_cats = set(test_df["output"].unique())
        assert len(train_cats) == 5
        assert len(test_cats) == 5


# --- Training Pipeline Tests ---


class TestTrainingPipeline:
    """Tests for the training pipeline."""

    def test_training_produces_metrics(self, sample_dataset: str, tmp_path) -> None:
        from training.prepare_dataset import prepare_dataset
        from training.train import train

        train_df, test_df = prepare_dataset(
            dataset_path=sample_dataset,
            test_size=0.2,
            random_seed=42,
        )

        model_dir: str = str(tmp_path / "model")
        metrics: Dict = train(train_df, test_df, output_dir=model_dir)

        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_training_saves_artifacts(self, sample_dataset: str, tmp_path) -> None:
        from training.prepare_dataset import prepare_dataset
        from training.train import train

        train_df, test_df = prepare_dataset(
            dataset_path=sample_dataset,
            test_size=0.2,
            random_seed=42,
        )

        model_dir: str = str(tmp_path / "model")
        train(train_df, test_df, output_dir=model_dir)

        assert os.path.exists(os.path.join(model_dir, "vectorizer.joblib"))
        assert os.path.exists(os.path.join(model_dir, "classifier.joblib"))
        assert os.path.exists(os.path.join(model_dir, "label_encoder.joblib"))

    def test_model_can_predict(self, sample_dataset: str, tmp_path) -> None:
        import joblib

        from training.prepare_dataset import prepare_dataset
        from training.train import train

        train_df, test_df = prepare_dataset(
            dataset_path=sample_dataset,
            test_size=0.2,
            random_seed=42,
        )

        model_dir: str = str(tmp_path / "model")
        train(train_df, test_df, output_dir=model_dir)

        # Load saved artifacts and predict
        vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
        classifier = joblib.load(os.path.join(model_dir, "classifier.joblib"))
        label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

        features = vectorizer.transform(["Stock running out at store"])
        prediction = classifier.predict(features)
        category: str = label_encoder.inverse_transform(prediction)[0]

        assert category in SUPPORTED_CATEGORIES


# --- Prediction Storage Tests ---


class TestPredictionStorage:
    """Tests for the prediction storage service."""

    def test_save_and_retrieve(self, temp_storage: PredictionStorage) -> None:
        temp_storage.save_prediction("test note", "supply_chain_delay")

        history = temp_storage.get_prediction_history()
        assert len(history) == 1
        assert history[0]["input_note"] == "test note"
        assert history[0]["issue_category"] == "supply_chain_delay"

    def test_category_counts(self, temp_storage: PredictionStorage) -> None:
        temp_storage.save_prediction("note 1", "supply_chain_delay")
        temp_storage.save_prediction("note 2", "supply_chain_delay")
        temp_storage.save_prediction("note 3", "pricing_conflict")

        counts: Dict[str, int] = temp_storage.get_category_counts()
        assert counts.get("supply_chain_delay") == 2
        assert counts.get("pricing_conflict") == 1

    def test_empty_history(self, temp_storage: PredictionStorage) -> None:
        history = temp_storage.get_prediction_history()
        assert len(history) == 0
