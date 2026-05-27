"""Tests for Pydantic schema validation.

Tests all request/response schemas for correct validation
behavior including edge cases.
"""

from typing import Dict

import pytest
from pydantic import ValidationError

from app.core.constants import (
    CATEGORY_DISPLAY_NAMES,
    MAX_BATCH_SIZE,
    MAX_NOTE_LENGTH,
    SUPPORTED_CATEGORIES,
)
from app.schemas.note_schema import (
    BatchPredictionRequest,
    CategoriesResponse,
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""

    def test_valid_request(self) -> None:
        req = PredictionRequest(rep_note="Test note")
        assert req.rep_note == "Test note"

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest(rep_note="")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest(rep_note="x" * (MAX_NOTE_LENGTH + 1))

    def test_accepts_max_length(self) -> None:
        req = PredictionRequest(rep_note="x" * MAX_NOTE_LENGTH)
        assert len(req.rep_note) == MAX_NOTE_LENGTH

    def test_rejects_missing_field(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest()  # type: ignore[call-arg]


class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest schema."""

    def test_valid_request(self) -> None:
        req = BatchPredictionRequest(notes=["note 1", "note 2"])
        assert len(req.notes) == 2

    def test_rejects_empty_list(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictionRequest(notes=[])

    def test_rejects_too_many_notes(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictionRequest(notes=["note"] * (MAX_BATCH_SIZE + 1))

    def test_rejects_empty_note_in_list(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictionRequest(notes=["valid", ""])

    def test_rejects_whitespace_note_in_list(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictionRequest(notes=["valid", "   "])

    def test_rejects_note_exceeding_max_length(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictionRequest(notes=["x" * (MAX_NOTE_LENGTH + 1)])


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    def test_valid_response(self) -> None:
        resp = PredictionResponse(
            issue_category="supply_chain_delay",
            confidence=0.95,
            method="ollama_base",
            latency_seconds="1.23",
            reasoning="Test reasoning",
        )
        assert resp.issue_category == "supply_chain_delay"
        assert resp.confidence == 0.95

    def test_rejects_confidence_above_1(self) -> None:
        with pytest.raises(ValidationError):
            PredictionResponse(
                issue_category="supply_chain_delay",
                confidence=1.5,
                method="ollama_base",
                latency_seconds="1.0",
                reasoning="test",
            )

    def test_rejects_confidence_below_0(self) -> None:
        with pytest.raises(ValidationError):
            PredictionResponse(
                issue_category="supply_chain_delay",
                confidence=-0.1,
                method="ollama_base",
                latency_seconds="1.0",
                reasoning="test",
            )

    def test_accepts_boundary_confidence(self) -> None:
        resp = PredictionResponse(
            issue_category="supply_chain_delay",
            confidence=0.0,
            method="ollama_base",
            latency_seconds="1.0",
            reasoning="test",
        )
        assert resp.confidence == 0.0

        resp2 = PredictionResponse(
            issue_category="supply_chain_delay",
            confidence=1.0,
            method="ollama_base",
            latency_seconds="1.0",
            reasoning="test",
        )
        assert resp2.confidence == 1.0


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_valid_response(self) -> None:
        resp = HealthResponse(
            status="healthy",
            ollama_available=True,
            sklearn_available=True,
            model_name="gemma:2b",
            supported_categories=SUPPORTED_CATEGORIES,
        )
        assert resp.status == "healthy"
        assert resp.ollama_available is True


class TestCategoriesResponse:
    """Tests for CategoriesResponse schema."""

    def test_valid_response(self) -> None:
        resp = CategoriesResponse(
            categories=SUPPORTED_CATEGORIES,
            display_names=CATEGORY_DISPLAY_NAMES,
        )
        assert len(resp.categories) == 5
        assert "supply_chain_delay" in resp.display_names


class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_valid_response(self) -> None:
        resp = ErrorResponse(
            error_code="TEST_ERROR",
            message="Something went wrong",
        )
        assert resp.error_code == "TEST_ERROR"
        assert resp.message == "Something went wrong"
        assert resp.details == {}

    def test_with_details(self) -> None:
        resp = ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Invalid input",
            details={"field": "rep_note", "max_length": 1000},
        )
        assert resp.details["field"] == "rep_note"
