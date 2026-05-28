"""FastAPI routes for the Sales Intelligence prediction API.

Provides endpoints for predicting issue categories from sales
representative notes using a tiered inference strategy:
  1. QLoRA direct inference (Transformers + PEFT on GPU) — best accuracy
  2. scikit-learn TF-IDF + RandomForest — fast fallback

All endpoints include structured error handling that maps internal
exceptions to appropriate HTTP status codes with error response schemas.
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from app.core.constants import (
    CATEGORY_DISPLAY_NAMES,
    SUPPORTED_CATEGORIES,
    SUPPORTED_CATEGORIES_SET,
)
from app.core.exceptions import (
    EmptyInputError,
    NoInferenceBackendError,
    SalesIntelligenceError,
    ServiceUnavailableError,
    ValidationError,
)
from app.schemas.note_schema import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CategoriesResponse,
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.predictor import SalesNotePredictor

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()
_predictor: Optional[SalesNotePredictor] = None


def get_predictor() -> SalesNotePredictor:
    """Get or create the singleton predictor instance.

    The predictor is lazily initialized on first call and reused
    for all subsequent requests. The QLoRA model itself is further
    lazily loaded on first inference call.

    Returns:
        SalesNotePredictor instance.
    """
    global _predictor
    if _predictor is None:
        _predictor = SalesNotePredictor()
    return _predictor


def _map_exception_to_http(exc: Exception) -> HTTPException:
    """Map application exceptions to HTTP exceptions.

    Provides a centralized mapping from internal exception types
    to HTTP status codes and structured error responses.

    Args:
        exc: The caught exception.

    Returns:
        HTTPException with appropriate status code and detail.
    """
    if isinstance(exc, SalesIntelligenceError):
        return HTTPException(
            status_code=exc.HTTP_STATUS,
            detail=exc.to_dict(),
        )
    # Fallback for unexpected exceptions
    return HTTPException(
        status_code=500,
        detail=ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An unexpected internal error occurred",
        ).model_dump(),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def predict_category(request: PredictionRequest) -> PredictionResponse:
    """Predict the issue category from a sales representative note.

    Uses the QLoRA fine-tuned Gemma-2B model (direct Transformers + PEFT
    inference on GPU). Falls back to sklearn TF-IDF + RandomForest if
    QLoRA inference fails.

    Args:
        request: Prediction request containing the sales note text.

    Returns:
        Prediction response with category, confidence, and metadata.

    Raises:
        HTTPException: 422 for validation errors, 503 for model errors.
    """
    try:
        predictor: SalesNotePredictor = get_predictor()
        result: Dict[str, Any] = predictor.predict(request.rep_note)
        return PredictionResponse(**result)
    except SalesIntelligenceError as exc:
        logger.error("Prediction error [%s]: %s", exc.error_code, exc.message)
        raise _map_exception_to_http(exc) from exc
    except Exception as exc:
        logger.exception("Unexpected error during prediction: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message="Internal server error",
            ).model_dump(),
        ) from exc


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def predict_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Predict issue categories for multiple notes in a single request.

    Args:
        request: Batch prediction request containing a list of notes.

    Returns:
        Batch prediction response with individual results.

    Raises:
        HTTPException: 422 for validation errors, 503 for model errors.
    """
    try:
        predictor: SalesNotePredictor = get_predictor()
        return _execute_batch_prediction(predictor, request.notes)
    except SalesIntelligenceError as exc:
        logger.error("Batch prediction error [%s]: %s", exc.error_code, exc.message)
        raise _map_exception_to_http(exc) from exc
    except Exception as exc:
        logger.exception("Unexpected batch error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message="Internal server error",
            ).model_dump(),
        ) from exc


def _execute_batch_prediction(
    predictor: SalesNotePredictor,
    notes: list,
) -> BatchPredictionResponse:
    """Execute batch prediction and build response.

    Args:
        predictor: Initialized predictor instance.
        notes: List of note strings to classify.

    Returns:
        Batch prediction response with all results.
    """
    predictions: list = []
    total_start: float = time.time()

    for note in notes:
        result: Dict[str, Any] = predictor.predict(note)
        predictions.append(PredictionResponse(**result))

    total_elapsed: float = time.time() - total_start

    return BatchPredictionResponse(
        predictions=predictions,
        total_notes=len(predictions),
        total_latency_seconds=f"{total_elapsed:.2f}",
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Reports the status of all inference backends including whether
    the QLoRA model is loaded and ready for inference.

    Returns:
        HealthResponse with service status information.
    """
    predictor: SalesNotePredictor = get_predictor()
    status_info: Dict[str, Any] = predictor.get_status()

    is_healthy: bool = (
        status_info["qlora_available"] or status_info["sklearn_available"]
    )

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        qlora_available=status_info["qlora_available"],
        sklearn_available=status_info["sklearn_available"],
        model_name=status_info.get("base_model", "N/A"),
        supported_categories=SUPPORTED_CATEGORIES,
    )


@router.get("/categories", response_model=CategoriesResponse)
async def get_categories() -> CategoriesResponse:
    """Get the list of supported issue categories.

    Returns:
        CategoriesResponse with category names and display labels.
    """
    return CategoriesResponse(
        categories=SUPPORTED_CATEGORIES,
        display_names=CATEGORY_DISPLAY_NAMES,
    )


@router.get("/status")
async def get_full_status() -> dict:
    """Get detailed status of all system components.

    Returns:
        Dictionary with full system status including model info.
    """
    predictor: SalesNotePredictor = get_predictor()
    status_info: Dict[str, Any] = predictor.get_status()

    return {
        "status": (
            "healthy"
            if (status_info["qlora_available"] or status_info["sklearn_available"])
            else "degraded"
        ),
        "inference": status_info,
    }
