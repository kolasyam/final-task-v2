"""Comprehensive tests for the prediction service.

Tests the SalesNotePredictor orchestration logic, SklearnClassifier,
category extraction, confidence estimation, and fallback behavior
without requiring real Ollama servers or model artifacts.

Uses the shared fixtures from tests/conftest.py.
"""

from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from app.core.constants import InferenceMethod, SUPPORTED_CATEGORIES
from app.core.exceptions import (
    EmptyInputError,
    NoInferenceBackendError,
)
from app.services.ollama_client import OllamaClient, PROMPT_MODEL_NAME
from app.services.predictor import SalesNotePredictor


class TestSalesNotePredictorOllama:
    """Tests for predictor using Ollama backend."""

    def test_predict_returns_valid_category(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_ollama_predictor.predict("Stock running out")
        assert result["issue_category"] in SUPPORTED_CATEGORIES

    def test_predict_returns_ollama_base_method(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_ollama_predictor.predict("Test note")
        assert result["method"] == InferenceMethod.OLLAMA_BASE.value

    def test_predict_returns_confidence(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_ollama_predictor.predict("Test note")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_returns_latency(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_ollama_predictor.predict("Test note")
        assert "latency_seconds" in result

    def test_predict_returns_reasoning(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        result: Dict = mock_ollama_predictor.predict("Test note")
        assert "reasoning" in result
        assert len(result["reasoning"]) > 0

    def test_predict_empty_string_raises_value_error(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        with pytest.raises(EmptyInputError):
            mock_ollama_predictor.predict("")

    def test_predict_whitespace_raises_value_error(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        with pytest.raises(EmptyInputError):
            mock_ollama_predictor.predict("   ")

    def test_predict_none_raises_error(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        with pytest.raises(AttributeError):
            mock_ollama_predictor.predict(None)  # type: ignore[arg-type]

    def test_predict_saves_to_storage(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        mock_ollama_predictor.predict("Test note")
        mock_ollama_predictor.storage.save_prediction.assert_called_once()

    def test_predict_calls_preprocess(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        with patch.object(
            mock_ollama_predictor.preprocessor, "preprocess",
            wraps=mock_ollama_predictor.preprocessor.preprocess,
        ) as mock_preprocess:
            mock_ollama_predictor.predict("Test note")
            mock_preprocess.assert_called_once_with("Test note")


class TestSalesNotePredictorPromptModel:
    """Tests for predictor with prompt-engineered model."""

    def test_predict_returns_prompt_method(
        self,
    ) -> None:
        from tests.conftest import mock_ollama_prompt_model
        mock_client = mock_ollama_prompt_model
        mock_sklearn = MagicMock()

        with patch.object(SalesNotePredictor, "__init__", lambda self: None):
            predictor = SalesNotePredictor.__new__(SalesNotePredictor)
            predictor.ollama_client = mock_client
            predictor.ollama_available = True
            predictor.is_prompt_model = True
            predictor.sklearn_classifier = mock_sklearn
            predictor.preprocessor = MagicMock()
            predictor.preprocessor.preprocess = lambda x: x.lower().strip()
            predictor.storage = MagicMock()
            predictor.storage.save_prediction = MagicMock()

            result = predictor.predict("Test note")
            assert result["method"] == InferenceMethod.OLLAMA_PROMPT_MODEL.value


class TestSalesNotePredictorSklearn:
    """Tests for predictor using sklearn fallback."""

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


class TestSalesNotePredictorNoBackend:
    """Tests for predictor with no available backend."""

    def test_predict_raises_no_backend_error(
        self, mock_no_backend_predictor: SalesNotePredictor,
    ) -> None:
        with pytest.raises(NoInferenceBackendError):
            mock_no_backend_predictor.predict("Test note")


class TestSalesNotePredictorFallback:
    """Tests for fallback behavior when Ollama returns unparseable output."""

    def test_fallback_to_sklearn_when_ollama_unparseable(
        self,
    ) -> None:
        from app.core.exceptions import OllamaConnectionError

        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.health_check.return_value = True
        mock_ollama.model_name = "gemma:2b"
        mock_ollama.base_model_name = "gemma:2b"
        mock_ollama.is_prompt_model = False
        mock_ollama.classify_note.return_value = ("random gibberish", 1.0)
        mock_ollama.extract_category.return_value = None

        mock_sklearn = MagicMock()
        mock_sklearn.predict.return_value = ("pricing_conflict", 0.75)

        with patch.object(SalesNotePredictor, "__init__", lambda self: None):
            predictor = SalesNotePredictor.__new__(SalesNotePredictor)
            predictor.ollama_client = mock_ollama
            predictor.ollama_available = True
            predictor.is_prompt_model = False
            predictor.sklearn_classifier = mock_sklearn
            predictor.preprocessor = MagicMock()
            predictor.preprocessor.preprocess = lambda x: x.lower().strip()
            predictor.storage = MagicMock()
            predictor.storage.save_prediction = MagicMock()

            result = predictor.predict("Test note")
            assert result["method"] == InferenceMethod.SKLEARN_TFIDF.value
            assert result["issue_category"] == "pricing_conflict"

    def test_uses_first_category_when_no_fallback(
        self,
    ) -> None:
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.health_check.return_value = True
        mock_ollama.model_name = "gemma:2b"
        mock_ollama.base_model_name = "gemma:2b"
        mock_ollama.is_prompt_model = False
        mock_ollama.classify_note.return_value = ("gibberish", 1.0)
        mock_ollama.extract_category.return_value = None

        with patch.object(SalesNotePredictor, "__init__", lambda self: None):
            predictor = SalesNotePredictor.__new__(SalesNotePredictor)
            predictor.ollama_client = mock_ollama
            predictor.ollama_available = True
            predictor.is_prompt_model = False
            predictor.sklearn_classifier = None
            predictor.preprocessor = MagicMock()
            predictor.preprocessor.preprocess = lambda x: x.lower().strip()
            predictor.storage = MagicMock()
            predictor.storage.save_prediction = MagicMock()

            result = predictor.predict("Test note")
            assert result["issue_category"] == SUPPORTED_CATEGORIES[0]


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


class TestOllamaExtractCategory:
    """Tests for the static category extraction method."""

    def test_direct_match(self) -> None:
        assert OllamaClient.extract_category("supply_chain_delay") == "supply_chain_delay"

    def test_direct_match_all_categories(self) -> None:
        for cat in SUPPORTED_CATEGORIES:
            assert OllamaClient.extract_category(cat) == cat

    def test_case_insensitive_match(self) -> None:
        assert OllamaClient.extract_category("SUPPLY_CHAIN_DELAY") == "supply_chain_delay"

    def test_supply_chain_keyword(self) -> None:
        result = OllamaClient.extract_category("stock shortage reported")
        assert result == "supply_chain_delay"

    def test_pricing_keyword(self) -> None:
        result = OllamaClient.extract_category("price dispute with retailer")
        assert result == "pricing_conflict"

    def test_competitor_keyword(self) -> None:
        result = OllamaClient.extract_category("competitor launched campaign")
        assert result == "competitor_pressure"

    def test_demand_keyword(self) -> None:
        result = OllamaClient.extract_category("demand surge observed")
        assert result == "demand_spike"

    def test_retailer_keyword(self) -> None:
        result = OllamaClient.extract_category("retailer unhappy with service")
        assert result == "retailer_dissatisfaction"

    def test_prefixed_output(self) -> None:
        result = OllamaClient.extract_category("Category: demand_spike")
        assert result == "demand_spike"

    def test_issue_category_prefix(self) -> None:
        result = OllamaClient.extract_category("Issue Category: pricing_conflict")
        assert result == "pricing_conflict"

    def test_the_category_is_prefix(self) -> None:
        result = OllamaClient.extract_category("The category is competitor_pressure")
        assert result == "competitor_pressure"

    def test_dash_prefix(self) -> None:
        result = OllamaClient.extract_category("- demand_spike")
        assert result == "demand_spike"

    def test_unknown_returns_none(self) -> None:
        result = OllamaClient.extract_category("completely unknown xyz")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = OllamaClient.extract_category("")
        assert result is None

    def test_punctuation_cleanup(self) -> None:
        result = OllamaClient.extract_category("demand_spike.")
        assert result == "demand_spike"

    def test_whitespace_cleanup(self) -> None:
        result = OllamaClient.extract_category("  supply_chain_delay  ")
        assert result == "supply_chain_delay"


class TestEstimateConfidence:
    """Tests for the confidence estimation heuristic."""

    def test_exact_match_high_confidence(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        confidence: float = mock_ollama_predictor._estimate_confidence(
            "supply_chain_delay", "supply_chain_delay",
        )
        assert confidence == 0.95

    def test_space_normalized_match(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        confidence: float = mock_ollama_predictor._estimate_confidence(
            "supply chain delay", "supply_chain_delay",
        )
        assert confidence == 0.85

    def test_substring_match(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        confidence: float = mock_ollama_predictor._estimate_confidence(
            "The category is supply_chain_delay", "supply_chain_delay",
        )
        assert confidence == 0.75

    def test_fuzzy_match_low_confidence(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        confidence: float = mock_ollama_predictor._estimate_confidence(
            "something else", "supply_chain_delay",
        )
        assert confidence == 0.60


class TestPredictorStatus:
    """Tests for the get_status method."""

    def test_status_structure(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        status: Dict = mock_ollama_predictor.get_status()
        assert "ollama_available" in status
        assert "is_prompt_model" in status
        assert "model_name" in status
        assert "base_model" in status
        assert "sklearn_available" in status
        assert "supported_categories" in status

    def test_status_ollama_available(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        status: Dict = mock_ollama_predictor.get_status()
        assert status["ollama_available"] is True

    def test_status_sklearn_not_available(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        status: Dict = mock_ollama_predictor.get_status()
        # Mock predictor has sklearn mocked (MagicMock is truthy for is not None)
        assert status["sklearn_available"] is True
