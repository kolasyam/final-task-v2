"""Centralized exception hierarchy for the Sales Intelligence System.

All custom exceptions inherit from SalesIntelligenceError, enabling
consistent error handling across the application and API layer.

Each exception maps to a specific HTTP status code for FastAPI responses.
"""

from typing import Any, Dict, List, Optional


class SalesIntelligenceError(Exception):
    """Base exception for all application errors.

    Attributes:
        message: Human-readable error message.
        error_code: Machine-readable error identifier.
        details: Additional context for debugging.
        http_status: HTTP status code for API responses.
    """

    ERROR_CODE: str = "UNKNOWN_ERROR"
    HTTP_STATUS: int = 500

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message: str = message
        self.error_code: str = error_code or self.ERROR_CODE
        self.details: Dict[str, Any] = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to a structured dictionary for API responses."""
        result: Dict[str, Any] = {
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


class ValidationError(SalesIntelligenceError):
    """Raised when input validation fails.

    Maps to HTTP 422 Unprocessable Entity.
    """

    ERROR_CODE = "VALIDATION_ERROR"
    HTTP_STATUS = 422


class EmptyInputError(ValidationError):
    """Raised when required input is empty or whitespace."""

    ERROR_CODE = "EMPTY_INPUT"
    HTTP_STATUS = 422

    def __init__(self, field_name: str = "input") -> None:
        super().__init__(
            message=f"{field_name} cannot be empty or whitespace",
            error_code=self.ERROR_CODE,
            details={"field": field_name},
        )


class InputTooLongError(ValidationError):
    """Raised when input exceeds maximum length."""

    ERROR_CODE = "INPUT_TOO_LONG"
    HTTP_STATUS = 422

    def __init__(self, field_name: str, max_length: int, actual_length: int) -> None:
        super().__init__(
            message=(
                f"{field_name} exceeds maximum length of {max_length} "
                f"characters (got {actual_length})"
            ),
            error_code=self.ERROR_CODE,
            details={
                "field": field_name,
                "max_length": max_length,
                "actual_length": actual_length,
            },
        )


class BatchSizeExceededError(ValidationError):
    """Raised when batch request exceeds maximum size."""

    ERROR_CODE = "BATCH_SIZE_EXCEEDED"
    HTTP_STATUS = 422

    def __init__(self, max_size: int, actual_size: int) -> None:
        super().__init__(
            message=(
                f"Batch size {actual_size} exceeds maximum of {max_size}"
            ),
            error_code=self.ERROR_CODE,
            details={"max_size": max_size, "actual_size": actual_size},
        )


class ServiceUnavailableError(SalesIntelligenceError):
    """Raised when an external service (Ollama) is unreachable.

    Maps to HTTP 503 Service Unavailable.
    """

    ERROR_CODE = "SERVICE_UNAVAILABLE"
    HTTP_STATUS = 503


class OllamaConnectionError(ServiceUnavailableError):
    """Raised when the Ollama server cannot be reached."""

    ERROR_CODE = "OLLAMA_CONNECTION_ERROR"
    HTTP_STATUS = 503

    def __init__(self, base_url: str) -> None:
        super().__init__(
            message=(
                f"Cannot connect to Ollama at {base_url}. "
                "Ensure Ollama is running: 'ollama serve'"
            ),
            error_code=self.ERROR_CODE,
            details={"ollama_url": base_url},
        )


class OllamaTimeoutError(ServiceUnavailableError):
    """Raised when an Ollama request times out."""

    ERROR_CODE = "OLLAMA_TIMEOUT"
    HTTP_STATUS = 503

    def __init__(self, timeout: int) -> None:
        super().__init__(
            message=(
                f"Ollama request timed out after {timeout}s. "
                "The model may still be loading."
            ),
            error_code=self.ERROR_CODE,
            details={"timeout_seconds": timeout},
        )


class NoInferenceBackendError(ServiceUnavailableError):
    """Raised when no inference backend (Ollama or sklearn) is available."""

    ERROR_CODE = "NO_INFERENCE_BACKEND"
    HTTP_STATUS = 503

    def __init__(self) -> None:
        super().__init__(
            message=(
                "No inference backend available. "
                "Start Ollama or run: python -m training.train"
            ),
            error_code=self.ERROR_CODE,
        )


class ModelNotFoundError(SalesIntelligenceError):
    """Raised when a required model artifact is not found on disk.

    Maps to HTTP 503 Service Unavailable.
    """

    ERROR_CODE = "MODEL_NOT_FOUND"
    HTTP_STATUS = 503

    def __init__(self, artifact_path: str, artifact_type: str = "model") -> None:
        super().__init__(
            message=(
                f"{artifact_type} artifact not found at '{artifact_path}'. "
                f"Run training: python -m training.train"
            ),
            error_code=self.ERROR_CODE,
            details={"artifact_path": artifact_path, "artifact_type": artifact_type},
        )


class PredictionError(SalesIntelligenceError):
    """Raised when prediction fails for a non-infrastructure reason.

    Maps to HTTP 500 Internal Server Error.
    """

    ERROR_CODE = "PREDICTION_ERROR"
    HTTP_STATUS = 500


class CategoryExtractionError(PredictionError):
    """Raised when a valid category cannot be extracted from model output."""

    ERROR_CODE = "CATEGORY_EXTRACTION_FAILED"
    HTTP_STATUS = 500

    def __init__(self, raw_output: str) -> None:
        super().__init__(
            message=f"Could not extract a valid category from model output: '{raw_output}'",
            error_code=self.ERROR_CODE,
            details={"raw_output": raw_output},
        )


class DatasetError(SalesIntelligenceError):
    """Raised when the source dataset cannot be loaded or is invalid.

    Maps to HTTP 500 Internal Server Error.
    """

    ERROR_CODE = "DATASET_ERROR"
    HTTP_STATUS = 500


class DatasetNotFoundError(DatasetError):
    """Raised when the dataset file does not exist."""

    ERROR_CODE = "DATASET_NOT_FOUND"
    HTTP_STATUS = 500

    def __init__(self, path: str) -> None:
        super().__init__(
            message=f"Dataset file not found: {path}",
            error_code=self.ERROR_CODE,
            details={"path": path},
        )


class DatasetValidationError(DatasetError):
    """Raised when the dataset fails validation checks."""

    ERROR_CODE = "DATASET_VALIDATION_ERROR"
    HTTP_STATUS = 500

    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=f"Dataset validation failed: {reason}",
            error_code=self.ERROR_CODE,
            details=details or {},
        )


class CircuitBreakerOpenError(ServiceUnavailableError):
    """Raised when the circuit breaker is open and blocking requests."""

    ERROR_CODE = "CIRCUIT_BREAKER_OPEN"
    HTTP_STATUS = 503

    def __init__(self, retry_after: float) -> None:
        super().__init__(
            message=(
                f"Circuit breaker is OPEN. Service unavailable. "
                f"Will retry in {retry_after:.0f}s"
            ),
            error_code=self.ERROR_CODE,
            details={"retry_after_seconds": round(retry_after, 1)},
        )


class RateLimitExceededError(SalesIntelligenceError):
    """Raised when a client exceeds the rate limit.

    Maps to HTTP 429 Too Many Requests.
    """

    ERROR_CODE = "RATE_LIMIT_EXCEEDED"
    HTTP_STATUS = 429

    def __init__(self, retry_after: int) -> None:
        super().__init__(
            message="Too many requests. Please slow down.",
            error_code=self.ERROR_CODE,
            details={"retry_after_seconds": retry_after},
        )


class AuthenticationError(SalesIntelligenceError):
    """Raised when API key authentication fails.

    Maps to HTTP 401 Unauthorized.
    """

    ERROR_CODE = "AUTHENTICATION_FAILED"
    HTTP_STATUS = 401

    def __init__(self) -> None:
        super().__init__(
            message="Unauthorized. Provide valid X-API-Key header.",
            error_code=self.ERROR_CODE,
        )
