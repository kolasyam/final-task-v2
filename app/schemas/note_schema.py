"""Pydantic schemas for request/response validation.

All schemas enforce strict input validation and provide clear
documentation for the OpenAPI/Swagger interface.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Schema for single prediction request.

    Attributes:
        rep_note: Sales representative note text to classify.
    """

    rep_note: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Sales representative note to classify",
        examples=[
            "Retailer reported stock running out of fast movers, "
            "customers already asking questions about 1L Juice Bottle.",
        ],
    )


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request.

    Attributes:
        notes: List of sales representative notes to classify.
    """

    notes: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of sales representative notes to classify",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "notes": [
                    "Retailer reported stock running out of fast movers",
                    "Competitor launched aggressive pricing campaign",
                ],
            },
        },
    )

    @classmethod
    def validate_notes(cls, v: List[str]) -> List[str]:
        """Validate individual notes in the batch."""
        for i, note in enumerate(v):
            if len(note) > 1000:
                raise ValueError(f"Note at index {i} exceeds max length of 1000 characters")
            if not note.strip():
                raise ValueError(f"Note at index {i} is empty or whitespace")
        return v


class PredictionResponse(BaseModel):
    """Schema for prediction response.

    Attributes:
        issue_category: The predicted issue category name.
        confidence: Prediction confidence score (0.0 to 1.0).
        method: Inference method used (ollama_gemma2b or sklearn_tfidf).
        latency_seconds: Time taken for inference.
        reasoning: Human-readable explanation of the prediction.
    """

    issue_category: str = Field(
        ...,
        description="Predicted issue category",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score",
    )
    method: str = Field(
        ...,
        description="Inference method used",
    )
    latency_seconds: str = Field(
        ...,
        description="Inference latency in seconds",
    )
    reasoning: str = Field(
        ...,
        description="Human-readable reasoning for the prediction",
    )


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response.

    Attributes:
        predictions: List of individual prediction results.
        total_notes: Total number of notes processed.
        total_latency_seconds: Total processing time.
    """

    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of prediction results",
    )
    total_notes: int = Field(
        ...,
        description="Total number of notes processed",
    )
    total_latency_seconds: str = Field(
        ...,
        description="Total processing time in seconds",
    )


class HealthResponse(BaseModel):
    """Schema for health check response.

    Attributes:
        status: Service health status.
        ollama_available: Whether Ollama is reachable.
        sklearn_available: Whether sklearn fallback is loaded.
        model_name: Name of the Ollama model.
        supported_categories: List of supported issue categories.
    """

    status: str = Field(..., description="Service health status")
    ollama_available: bool = Field(
        ..., description="Whether Ollama is reachable",
    )
    sklearn_available: bool = Field(
        ..., description="Whether sklearn fallback is loaded",
    )
    model_name: str = Field(..., description="Ollama model name")
    supported_categories: List[str] = Field(
        ..., description="Supported issue categories",
    )


class CategoriesResponse(BaseModel):
    """Schema for categories response.

    Attributes:
        categories: List of supported category names.
        display_names: Mapping of category names to display labels.
    """

    categories: List[str] = Field(
        ..., description="Supported category names",
    )
    display_names: dict = Field(
        ..., description="Category display name mapping",
    )
