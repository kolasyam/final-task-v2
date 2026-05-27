"""Prediction service with tiered inference orchestration.

Model priority:
  1. Prompt-engineered Ollama model (gemma-sales-intel via Modelfile)
  2. Base Ollama model (gemma:2b) — zero-shot
  3. scikit-learn TF-IDF + RandomForest — fast local fallback
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.config import config
from app.core.constants import InferenceMethod, SUPPORTED_CATEGORIES
from app.core.exceptions import (
    EmptyInputError,
    ModelNotFoundError,
    NoInferenceBackendError,
    PredictionError,
)
from app.services.ollama_client import PROMPT_MODEL_NAME, OllamaClient
from app.services.preprocessing import TextPreprocessor
from app.services.storage import PredictionStorage

logger = logging.getLogger(__name__)

# Confidence thresholds based on output clarity
_EXACT_MATCH_CONFIDENCE: float = 0.95
_SPACE_NORMALIZED_CONFIDENCE: float = 0.85
_SUBSTRING_MATCH_CONFIDENCE: float = 0.75
_FUZZY_MATCH_CONFIDENCE: float = 0.60


class SklearnClassifier:
    """Wrapper for scikit-learn TF-IDF + RandomForest classification.

    Provides fast local classification as the final fallback when
    no Ollama backend is available.

    Attributes:
        _vectorizer: Fitted TF-IDF vectorizer.
        _classifier: Trained RandomForest classifier.
        _label_encoder: Fitted label encoder.
    """

    def __init__(self) -> None:
        """Initialize the sklearn classifier, loading persisted artifacts."""
        self._vectorizer: Any = self._load_artifact(config.vectorizer_path, "vectorizer")
        self._classifier: Any = self._load_artifact(config.classifier_path, "classifier")
        self._label_encoder: Any = self._load_artifact(config.label_encoder_path, "label encoder")
        logger.info("SklearnClassifier initialized (artifacts loaded)")

    @staticmethod
    def _load_artifact(path: str, artifact_name: str = "artifact") -> Any:
        """Load a persisted artifact from disk.

        Args:
            path: Filesystem path to the joblib artifact.
            artifact_name: Human-readable name for error messages.

        Returns:
            The deserialized artifact.

        Raises:
            ModelNotFoundError: If the artifact file does not exist.
        """
        import joblib
        from pathlib import Path

        if not Path(path).exists():
            raise ModelNotFoundError(path, artifact_name)
        artifact: Any = joblib.load(path)
        logger.info("Loaded %s from %s", artifact_name, path)
        return artifact

    def predict(self, text: str) -> Tuple[str, float]:
        """Classify text using the sklearn pipeline.

        Args:
            text: Preprocessed text to classify.

        Returns:
            Tuple of (predicted_category, confidence_score).
        """
        features: Any = self._vectorizer.transform([text])
        prediction: Any = self._classifier.predict(features)
        category: str = self._label_encoder.inverse_transform(prediction)[0]
        proba: np.ndarray = self._classifier.predict_proba(features)[0]
        confidence: float = float(np.max(proba))

        logger.info("Sklearn prediction: '%s' (confidence=%.3f)", category, confidence)
        return category, confidence


class SalesNotePredictor:
    """Orchestrates sales note classification with tiered inference.

    Inference priority:
        1. Prompt-engineered Ollama model (gemma-sales-intel via Modelfile)
        2. Base Ollama model (gemma:2b) — zero-shot
        3. scikit-learn TF-IDF + RandomForest — fast fallback
    """

    def __init__(self) -> None:
        """Initialize the predictor, probing Ollama availability."""
        self.preprocessor: TextPreprocessor = TextPreprocessor()
        self.storage: PredictionStorage = PredictionStorage()
        self.ollama_client: OllamaClient = OllamaClient()
        self.sklearn_classifier: Optional[SklearnClassifier] = None

        self.ollama_available: bool = self.ollama_client.health_check()
        self.is_prompt_model: bool = self.ollama_client.is_prompt_model

        self._log_initialization_status()
        self._load_sklearn_fallback()

    def _log_initialization_status(self) -> None:
        """Log the initialization status of inference backends."""
        if self.ollama_available:
            if self.is_prompt_model:
                logger.info("Using prompt-engineered model: gemma-sales-intel")
            else:
                logger.info("Using base model: gemma:2b")
        else:
            logger.warning("Ollama not available — will use sklearn fallback if available")

    def _load_sklearn_fallback(self) -> None:
        """Attempt to load the sklearn fallback classifier."""
        try:
            self.sklearn_classifier = SklearnClassifier()
            logger.info("Sklearn fallback classifier loaded")
        except ModelNotFoundError as exc:
            logger.warning("Sklearn fallback not available: %s", exc)
            self.sklearn_classifier = None
        except Exception as exc:
            logger.warning("Unexpected error loading sklearn fallback: %s", exc)
            self.sklearn_classifier = None

    def predict(self, note: str) -> Dict[str, Any]:
        """Classify a sales representative note into an issue category.

        Args:
            note: Raw sales representative note text.

        Returns:
            Dictionary with keys:
                - issue_category: Predicted category string.
                - confidence: Prediction confidence (0.0 to 1.0).
                - method: Inference method identifier.
                - latency_seconds: Inference time as formatted string.
                - reasoning: Human-readable explanation.

        Raises:
            EmptyInputError: If the note is empty or whitespace.
            NoInferenceBackendError: If no backend is available.
        """
        self._validate_input(note)

        cleaned_note: str = self.preprocessor.preprocess(note)
        result: Dict[str, Any] = self._execute_inference(cleaned_note)

        self.storage.save_prediction(
            input_note=note,
            issue_category=result["issue_category"],
        )
        return result

    @staticmethod
    def _validate_input(note: str) -> None:
        """Validate the input note.

        Args:
            note: Input text to validate.

        Raises:
            EmptyInputError: If the note is empty or whitespace.
        """
        if not note or not note.strip():
            raise EmptyInputError("Note")

    def _execute_inference(self, cleaned_note: str) -> Dict[str, Any]:
        """Execute inference using the best available backend.

        Args:
            cleaned_note: Preprocessed note text.

        Returns:
            Prediction result dictionary.

        Raises:
            NoInferenceBackendError: If no backend is available.
        """
        if self.ollama_available:
            return self._predict_ollama(cleaned_note)
        if self.sklearn_classifier is not None:
            return self._predict_sklearn(cleaned_note)
        raise NoInferenceBackendError()

    def _predict_ollama(self, cleaned_note: str) -> Dict[str, Any]:
        """Classify via Ollama (fine-tuned or base model).

        If Ollama returns an unparseable result, falls back to sklearn
        if available.

        Args:
            cleaned_note: Preprocessed sales note text.

        Returns:
            Prediction result dictionary.
        """
        start_time: float = time.time()
        raw_output: str
        ollama_elapsed: float
        raw_output, ollama_elapsed = self.ollama_client.classify_note(cleaned_note)
        category: Optional[str] = self.ollama_client.extract_category(raw_output)

        if category is None:
            logger.warning("Ollama returned unparseable: '%s'. Attempting fallback.", raw_output)
            if self.sklearn_classifier is not None:
                return self._predict_sklearn(cleaned_note)
            logger.warning("No fallback available. Using first supported category.")
            category = SUPPORTED_CATEGORIES[0]

        total_elapsed: float = time.time() - start_time
        confidence: float = self._estimate_confidence(raw_output, category)

        method: InferenceMethod = (
            InferenceMethod.OLLAMA_PROMPT_MODEL
            if self.is_prompt_model
            else InferenceMethod.OLLAMA_BASE
        )
        model_label: str = (
            PROMPT_MODEL_NAME
            if self.is_prompt_model
            else self.ollama_client.base_model_name
        )

        return {
            "issue_category": category,
            "confidence": confidence,
            "method": method.value,
            "latency_seconds": f"{total_elapsed:.2f}",
            "reasoning": (
                f"{model_label} classified the note as '{category}' "
                f"in {total_elapsed:.2f}s"
            ),
        }

    def _predict_sklearn(self, cleaned_note: str) -> Dict[str, Any]:
        """Classify via sklearn pipeline.

        Args:
            cleaned_note: Preprocessed note text.

        Returns:
            Prediction result dictionary.
        """
        start_time: float = time.time()
        category: str
        confidence: float
        category, confidence = self.sklearn_classifier.predict(cleaned_note)
        elapsed: float = time.time() - start_time

        return {
            "issue_category": category,
            "confidence": confidence,
            "method": InferenceMethod.SKLEARN_TFIDF.value,
            "latency_seconds": f"{elapsed:.4f}",
            "reasoning": (
                f"TF-IDF + RandomForest classified as '{category}' "
                f"({confidence:.1%} confidence)"
            ),
        }

    @staticmethod
    def _estimate_confidence(raw_output: str, category: str) -> float:
        """Estimate confidence from Ollama output clarity.

        Uses heuristics based on how cleanly the category was extracted:
          - Exact match: highest confidence
          - Space-normalized match (underscores replaced with spaces)
          - Substring match (category found within output)
          - Fuzzy match (keyword-based): lowest confidence

        Args:
            raw_output: Raw model output string.
            category: Extracted category name.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        cleaned: str = raw_output.lower().strip()

        if cleaned == category:
            return _EXACT_MATCH_CONFIDENCE
        if category.replace("_", " ") in cleaned:
            return _SPACE_NORMALIZED_CONFIDENCE
        if category in cleaned:
            return _SUBSTRING_MATCH_CONFIDENCE
        return _FUZZY_MATCH_CONFIDENCE

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of all inference backends.

        Returns:
            Status dictionary for health checks and dashboards.
        """
        return {
            "ollama_available": self.ollama_available,
            "is_prompt_model": self.is_prompt_model,
            "model_name": self.ollama_client.model_name,
            "base_model": self.ollama_client.base_model_name,
            "sklearn_available": self.sklearn_classifier is not None,
            "supported_categories": SUPPORTED_CATEGORIES,
        }
