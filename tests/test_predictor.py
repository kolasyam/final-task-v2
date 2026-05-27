"""Tests for the prediction service using mocked backends.

Verifies the SalesNotePredictor orchestration logic, category
extraction, and the dual-mode (Ollama + sklearn) inference flow
without requiring a real model or Ollama server.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.predictor import SalesNotePredictor, SUPPORTED_CATEGORIES
from app.services.ollama_client import OllamaClient


@pytest.fixture
def mock_ollama_predictor() -> SalesNotePredictor:
    """Create a predictor with mocked Ollama client and sklearn.

    Returns:
        SalesNotePredictor instance backed by mocks.
    """
    mock_ollama = MagicMock()
    mock_ollama.health_check.return_value = True
    mock_ollama.model_name = "gemma:2b"
    mock_ollama.classify_note.return_value = ("supply_chain_delay", 1.5)
    mock_ollama.extract_category.return_value = "supply_chain_delay"

    mock_sklearn = MagicMock()

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

        yield predictor


@pytest.fixture
def mock_sklearn_only_predictor() -> SalesNotePredictor:
    """Create a predictor with only sklearn backend (no Ollama).

    Returns:
        SalesNotePredictor instance with sklearn-only fallback.
    """
    mock_sklearn = MagicMock()
    mock_sklearn.predict.return_value = ("pricing_conflict", 0.87)

    with patch.object(SalesNotePredictor, "__init__", lambda self: None):
        predictor = SalesNotePredictor.__new__(SalesNotePredictor)
        predictor.ollama_client = MagicMock()
        predictor.ollama_available = False
        predictor.is_prompt_model = False
        predictor.sklearn_classifier = mock_sklearn
        predictor.preprocessor = MagicMock()
        predictor.preprocessor.preprocess = lambda x: x.lower().strip()
        predictor.storage = MagicMock()
        predictor.storage.save_prediction = MagicMock()

        yield predictor


class TestSalesNotePredictorOllama:
    """Tests for predictor using Ollama backend."""

    def test_predict_returns_ollama_category(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        result = mock_ollama_predictor.predict("Stock running out at store")
        assert result["issue_category"] in SUPPORTED_CATEGORIES
        assert result["method"] in ("ollama_base", "ollama_finetuned")

    def test_predict_returns_confidence(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        result = mock_ollama_predictor.predict("Test note")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_returns_latency(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        result = mock_ollama_predictor.predict("Test note")
        assert "latency_seconds" in result

    def test_predict_empty_raises_value_error(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            mock_ollama_predictor.predict("")

    def test_predict_whitespace_raises_value_error(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            mock_ollama_predictor.predict("   ")

    def test_predict_saves_to_storage(
        self, mock_ollama_predictor: SalesNotePredictor,
    ) -> None:
        mock_ollama_predictor.predict("Test note")
        mock_ollama_predictor.storage.save_prediction.assert_called_once()


class TestSalesNotePredictorSklearn:
    """Tests for predictor using sklearn fallback."""

    def test_predict_returns_sklearn_category(
        self, mock_sklearn_only_predictor: SalesNotePredictor,
    ) -> None:
        result = mock_sklearn_only_predictor.predict("Price too high")
        assert result["issue_category"] in SUPPORTED_CATEGORIES
        assert result["method"] == "sklearn_tfidf"

    def test_predict_sklearn_returns_confidence(
        self, mock_sklearn_only_predictor: SalesNotePredictor,
    ) -> None:
        result = mock_sklearn_only_predictor.predict("Test note")
        assert 0.0 <= result["confidence"] <= 1.0


class TestSupportedCategories:
    """Tests for the supported categories list."""

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

    def test_unknown_returns_none(self) -> None:
        result = OllamaClient.extract_category("completely unknown xyz")
        assert result is None
