"""FastAPI routes for the Sales Intelligence prediction API.

Provides endpoints for predicting issue categories from sales
representative notes using a tiered inference strategy:
  1. Prompt-engineered Ollama model (gemma-sales-intel) — best accuracy
  2. Base Ollama model (gemma:2b) — general purpose
  3. scikit-learn TF-IDF + RandomForest — fast fallback
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.schemas.note_schema import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CategoriesResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.predictor import SalesNotePredictor, SUPPORTED_CATEGORIES

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()
_predictor: Optional[SalesNotePredictor] = None


def get_predictor() -> SalesNotePredictor:
    """Get or create the singleton predictor instance.

    The predictor is lazily initialized on first call and reused
    for all subsequent requests.

    Returns:
        SalesNotePredictor instance.
    """
    global _predictor
    if _predictor is None:
        _predictor = SalesNotePredictor()
    return _predictor


@router.post("/predict", response_model=PredictionResponse)
async def predict_category(request: PredictionRequest) -> PredictionResponse:
    """Predict the issue category from a sales representative note.

    Uses the prompt-engineered gemma-sales-intel model if available,
    otherwise falls back to base gemma:2b or sklearn.

    Args:
        request: Prediction request containing the sales note text.

    Returns:
        Prediction response with category, confidence, and metadata.

    Raises:
        HTTPException: 422 for validation errors, 503 for model errors,
                      500 for unexpected errors.
    """
    try:
        predictor: SalesNotePredictor = get_predictor()
        result = predictor.predict(request.rep_note)
        return PredictionResponse(**result)
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected error during prediction: %s", exc)
        raise HTTPException(
            status_code=500, detail="Internal server error",
        ) from exc


@router.post("/predict/batch", response_model=BatchPredictionResponse)
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
        predictor = get_predictor()
        predictions = []
        total_start = time.time()

        for note in request.notes:
            result = predictor.predict(note)
            predictions.append(PredictionResponse(**result))

        total_elapsed = time.time() - total_start

        return BatchPredictionResponse(
            predictions=predictions,
            total_notes=len(predictions),
            total_latency_seconds=f"{total_elapsed:.2f}",
        )
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected batch error: %s", exc)
        raise HTTPException(
            status_code=500, detail="Internal server error",
        ) from exc


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Reports the status of all inference backends including whether
    the prompt-engineered model is active.

    Returns:
        HealthResponse with service status information.
    """
    predictor = get_predictor()
    status_info = predictor.get_status()

    is_healthy = status_info["ollama_available"] or status_info["sklearn_available"]

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        ollama_available=status_info["ollama_available"],
        sklearn_available=status_info["sklearn_available"],
        model_name=status_info["model_name"],
        supported_categories=SUPPORTED_CATEGORIES,
    )


@router.get("/categories", response_model=CategoriesResponse)
async def get_categories() -> CategoriesResponse:
    """Get the list of supported issue categories.

    Returns:
        CategoriesResponse with category names and display labels.
    """
    from app.config import CATEGORY_DISPLAY_NAMES

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
    predictor = get_predictor()
    return {
        "status": "healthy" if (predictor.ollama_available or predictor.sklearn_classifier) else "degraded",
        "inference": predictor.get_status(),
    }
