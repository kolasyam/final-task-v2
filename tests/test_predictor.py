"""Comprehensive tests for the prediction service.

Tests the SalesNotePredictor orchestration logic with QLoRA as primary
backend and sklearn as fallback. Also tests the category extraction
and confidence estimation logic.

Uses the shared fixtures from tests/conftest.py.
"""

from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from app.core.constants import InferenceMethod, SUPPORTED_CATEGORIES
from app.core.exceptions import (
    EmptyInputError,
    NoInferenceBackendError,
    QLoRAInferenceError,
)
from app.services.predictor import SalesNotePredictor
from app.services.qlora_predictor import QLoraPredictor


class TestSalesNotePredictorQLoRA:
    """Tests for predictor using QLoRA backend (primary path)."""

    def test_predict_returns_valid_category(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_qlora_predictor.predict("Stock running out")
        assert result["issue_category"] in SUPPORTED_CATEGORIES

    def test_predict_returns_qlora_method(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_qlora_predictor.predict("Test note")
        assert result["method"] == InferenceMethod.QLORA_DIRECT.value

    def test_predict_returns_confidence(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_qlora_predictor.predict("Test note")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_returns_latency(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_qlora_predictor.predict("Test note")
        assert "latency_seconds" in result

    def test_predict_returns_reasoning(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_qlora_predictor.predict("Test note")
        assert "reasoning" in result
        assert len(result["reasoning"]) > 0

    def test_predict_reasoning_mentions_qlora(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_qlora_predictor.predict("Test note")
        assert "QLoRA" in result["reasoning"] or "qlora" in result["reasoning"].lower()

    def test_predict_empty_string_raises_error(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        with pytest.raises(EmptyInputError):
            mock_qlora_predictor.predict("")

    def test_predict_whitespace_raises_error(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        with pytest.raises(EmptyInputError):
            mock_qlora_predictor.predict("   ")

    def test_predict_none_raises_error(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        with pytest.raises(AttributeError):
            mock_qlora_predictor.predict(None)  # type: ignore[arg-type]

    def test_predict_saves_to_storage(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        mock_qlora_predictor.predict("Test note")
        mock_qlora_predictor.storage.save_prediction.assert_called_once()

    def test_predict_calls_preprocess(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        with patch.object(
            mock_qlora_predictor.preprocessor, "preprocess",
            wraps=mock_qlora_predictor.preprocessor.preprocess,
        ) as mock_preprocess:
            mock_qlora_predictor.predict("Test note")
            mock_preprocess.assert_called_once_with("Test note")

    def test_predict_calls_qlora_classify(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        mock_qlora_predictor.predict("Stock shortage reported")
        mock_qlora_predictor.qlora_predictor.classify.assert_called_once()


class TestSalesNotePredictorSklearnFallback:
    """Tests for predictor falling back to sklearn when QLoRA fails."""

    def test_predict_returns_sklearn_category(
        self, mock_sklearn_only_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_sklearn_only_predictor.predict("Price too high")
        assert result["issue_category"] in SUPPORTED_CATEGORIES
        assert result["method"] == InferenceMethod.SKLEARN_TFIDF.value

    def test_predict_sklearn_returns_confidence(
        self, mock_sklearn_only_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_sklearn_only_predictor.predict("Test note")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_sklearn_returns_reasoning(
        self, mock_sklearn_only_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_sklearn_only_predictor.predict("Test note")
        assert "reasoning" in result

    def test_predict_sklearn_mentions_tfidf(
        self, mock_sklearn_only_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_sklearn_only_predictor.predict("Test note")
        assert "TF-IDF" in result["reasoning"] or "RandomForest" in result["reasoning"]


class TestSalesNotePredictorFallbackChain:
    """Tests for QLoRA → sklearn fallback behavior."""

    def test_falls_back_to_sklearn_when_qlora_fails(
        self, mock_qlora_failure_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_qlora_failure_predictor.predict("Test note")
        assert result["method"] == InferenceMethod.SKLEARN_TFIDF.value
        assert result["issue_category"] == "demand_spike"

    def test_qlora_available_flagged_false_after_failure(
        self, mock_qlora_failure_predictor: SalesNotePredictor,
    ) -> None:
        assert mock_qlora_failure_predictor.qlora_available is True
        mock_qlora_failure_predictor.predict("Test note")
        assert mock_qlora_failure_predictor.qlora_available is False

    def test_subsequent_calls_use_sklearn_after_qlora_failure(
        self, mock_qlora_failure_predictor: SalesNotePredictor,
    ) -> None:
        # First call triggers fallback
        mock_qlora_failure_predictor.predict("First call")
        # Second call should go directly to sklearn
        result = mock_qlora_failure_predictor.predict("Second call")
        assert result["method"] == InferenceMethod.SKLEARN_TFIDF.value


class TestSalesNotePredictorNoBackend:
    """Tests for predictor with no available backend."""

    def test_predict_raises_no_backend_error(
        self, mock_no_backend_predictor: SalesNotePredictor,
    ) -> None:
        with pytest.raises(NoInferenceBackendError):
            mock_no_backend_predictor.predict("Test note")


class TestSupportedCategories:
    """Tests for the supported categories configuration."""

    def test_five_categories(self) -> None:
        assert len(SUPPORTED_CATEGORIES) == 5

    def test_all_expected_present(self) -> None:
        expected = [
            "supply_chain_delay",
            "retailer_dissatisfaction",
            "pricing_conflict",
            "competitor_pressure",
            "demand_spike",
        ]
        for cat in expected:
            assert cat in SUPPORTED_CATEGORIES


class TestPredictorStatus:
    """Tests for the get_status method."""

    def test_status_structure(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        status: Dict = mock_qlora_predictor.get_status()
        assert "qlora_available" in status
        assert "sklearn_available" in status
        assert "supported_categories" in status
        assert "base_model" in status
        assert "adapter" in status

    def test_status_qlora_available(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        status: Dict = mock_qlora_predictor.get_status()
        assert status["qlora_available"] is True

    def test_status_sklearn_available(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        status: Dict = mock_qlora_predictor.get_status()
        assert status["sklearn_available"] is True

    def test_status_qlora_details(
        self, mock_qlora_predictor: SalesNotePredictor,
    ) -> None:
        status: Dict = mock_qlora_predictor.get_status()
        assert "qlora_details" in status
        qlora_status = status["qlora_details"]
        assert "loaded" in qlora_status
        assert "device" in qlora_status
        assert "is_cuda" in qlora_status
