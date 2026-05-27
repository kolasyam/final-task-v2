"""Tests for the centralized exception hierarchy.

Verifies that all exception types carry correct error codes,
HTTP status codes, and structured response data.
"""

from typing import Dict

import pytest

from app.core.exceptions import (
    AuthenticationError,
    BatchSizeExceededError,
    CategoryExtractionError,
    CircuitBreakerOpenError,
    DatasetError,
    DatasetNotFoundError,
    DatasetValidationError,
    EmptyInputError,
    InputTooLongError,
    ModelNotFoundError,
    NoInferenceBackendError,
    OllamaConnectionError,
    OllamaTimeoutError,
    PredictionError,
    RateLimitExceededError,
    SalesIntelligenceError,
    ServiceUnavailableError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Test that the exception hierarchy is correct."""

    def test_all_exceptions_inherit_from_base(self) -> None:
        exception_classes = [
            ValidationError,
            EmptyInputError,
            InputTooLongError,
            BatchSizeExceededError,
            ServiceUnavailableError,
            OllamaConnectionError,
            OllamaTimeoutError,
            NoInferenceBackendError,
            ModelNotFoundError,
            PredictionError,
            CategoryExtractionError,
            DatasetError,
            DatasetNotFoundError,
            DatasetValidationError,
            CircuitBreakerOpenError,
            RateLimitExceededError,
            AuthenticationError,
        ]
        for cls in exception_classes:
            assert issubclass(cls, SalesIntelligenceError), f"{cls.__name__} should inherit from SalesIntelligenceError"

    def test_base_exception_is_catchable(self) -> None:
        with pytest.raises(SalesIntelligenceError):
            raise SalesIntelligenceError("test")

    def test_validation_error_is_sales_intelligence_error(self) -> None:
        with pytest.raises(SalesIntelligenceError):
            raise ValidationError("test")

    def test_service_error_is_sales_intelligence_error(self) -> None:
        with pytest.raises(SalesIntelligenceError):
            raise ServiceUnavailableError("test")


class TestEmptyInputError:
    """Tests for EmptyInputError."""

    def test_default_message(self) -> None:
        error = EmptyInputError()
        assert "cannot be empty" in error.message

    def test_custom_field_name(self) -> None:
        error = EmptyInputError("Sales note")
        assert "Sales note" in error.message

    def test_http_status(self) -> None:
        error = EmptyInputError()
        assert error.HTTP_STATUS == 422

    def test_error_code(self) -> None:
        error = EmptyInputError()
        assert error.ERROR_CODE == "EMPTY_INPUT"

    def test_to_dict(self) -> None:
        error = EmptyInputError("Note")
        result = error.to_dict()
        assert result["error_code"] == "EMPTY_INPUT"
        assert "field" in result["details"]


class TestInputTooLongError:
    """Tests for InputTooLongError."""

    def test_message_includes_lengths(self) -> None:
        error = InputTooLongError("note", 100, 150)
        assert "100" in error.message
        assert "150" in error.message

    def test_http_status(self) -> None:
        error = InputTooLongError("note", 100, 150)
        assert error.HTTP_STATUS == 422


class TestBatchSizeExceededError:
    """Tests for BatchSizeExceededError."""

    def test_message(self) -> None:
        error = BatchSizeExceededError(50, 75)
        assert "75" in error.message
        assert "50" in error.message


class TestOllamaConnectionError:
    """Tests for OllamaConnectionError."""

    def test_inherits_from_service_unavailable(self) -> None:
        error = OllamaConnectionError("http://localhost:11434")
        assert isinstance(error, ServiceUnavailableError)

    def test_message_includes_url(self) -> None:
        error = OllamaConnectionError("http://localhost:11434")
        assert "localhost:11434" in error.message

    def test_message_includes_hint(self) -> None:
        error = OllamaConnectionError("http://localhost:11434")
        assert "ollama serve" in error.message


class TestOllamaTimeoutError:
    """Tests for OllamaTimeoutError."""

    def test_message_includes_timeout(self) -> None:
        error = OllamaTimeoutError(120)
        assert "120" in error.message

    def test_http_status(self) -> None:
        error = OllamaTimeoutError(120)
        assert error.HTTP_STATUS == 503


class TestNoInferenceBackendError:
    """Tests for NoInferenceBackendError."""

    def test_message(self) -> None:
        error = NoInferenceBackendError()
        assert "No inference backend" in error.message


class TestModelNotFoundError:
    """Tests for ModelNotFoundError."""

    def test_message_includes_path(self) -> None:
        error = ModelNotFoundError("/path/to/model.joblib")
        assert "/path/to/model.joblib" in error.message

    def test_custom_artifact_type(self) -> None:
        error = ModelNotFoundError("/path/to/vec.joblib", "vectorizer")
        assert "vectorizer" in error.message


class TestCircuitBreakerOpenError:
    """Tests for CircuitBreakerOpenError."""

    def test_inherits_from_service_unavailable(self) -> None:
        error = CircuitBreakerOpenError(retry_after=30.0)
        assert isinstance(error, ServiceUnavailableError)

    def test_retry_after_in_details(self) -> None:
        error = CircuitBreakerOpenError(retry_after=15.0)
        result = error.to_dict()
        assert "retry_after_seconds" in result["details"]


class TestRateLimitExceededError:
    """Tests for RateLimitExceededError."""

    def test_http_status_429(self) -> None:
        error = RateLimitExceededError(retry_after=60)
        assert error.HTTP_STATUS == 429

    def test_retry_after_in_details(self) -> None:
        error = RateLimitExceededError(retry_after=60)
        result = error.to_dict()
        assert result["details"]["retry_after_seconds"] == 60


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_http_status_401(self) -> None:
        error = AuthenticationError()
        assert error.HTTP_STATUS == 401

    def test_message(self) -> None:
        error = AuthenticationError()
        assert "Unauthorized" in error.message


class TestDatasetNotFoundError:
    """Tests for DatasetNotFoundError."""

    def test_message_includes_path(self) -> None:
        error = DatasetNotFoundError("/data/file.xlsx")
        assert "/data/file.xlsx" in error.message


class TestCategoryExtractionError:
    """Tests for CategoryExtractionError."""

    def test_message_includes_raw_output(self) -> None:
        error = CategoryExtractionError("gibberish")
        assert "gibberish" in error.message


class TestToDictFormat:
    """Test that all exceptions produce consistent to_dict output."""

    def test_all_exceptions_have_required_keys(self) -> None:
        exceptions = [
            EmptyInputError(),
            InputTooLongError("f", 1, 2),
            BatchSizeExceededError(1, 2),
            OllamaConnectionError("http://test"),
            OllamaTimeoutError(10),
            NoInferenceBackendError(),
            ModelNotFoundError("/path"),
            CircuitBreakerOpenError(5.0),
            RateLimitExceededError(10),
            AuthenticationError(),
            DatasetNotFoundError("/path"),
        ]
        for exc in exceptions:
            result = exc.to_dict()
            assert "error_code" in result
            assert "message" in result
            assert isinstance(result["error_code"], str)
            assert isinstance(result["message"], str)
