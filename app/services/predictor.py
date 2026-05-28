"""Prediction service with tiered inference orchestration.

Model priority:
  1. QLoRA direct inference (Transformers + PEFT on GPU)
  2. scikit-learn TF-IDF + RandomForest — fast local fallback

No Ollama dependency — QLoRA loads the Gemma base model + LoRA adapter
directly via Transformers and PEFT, running fully offline on GPU.
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
    QLoRAInferenceError,
)
from app.services.preprocessing import TextPreprocessor
from app.services.qlora_predictor import QLoraPredictor
from app.services.storage import PredictionStorage

logger = logging.getLogger(__name__)


class SklearnClassifier:
    """Wrapper for scikit-learn TF-IDF + RandomForest classification.

    Provides fast local classification as the final fallback when
    the QLoRA GPU backend is unavailable.

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
        1. QLoRA direct inference (GPU, Transformers + PEFT)
        2. scikit-learn TF-IDF + RandomForest — fast fallback

    The QLoRA predictor is initialized lazily — the model loads on
    the first .classify() call. If loading fails, the predictor falls
    back to sklearn for all subsequent requests.
    """

    def __init__(self) -> None:
        """Initialize the predictor, setting up backends."""
        self.preprocessor: TextPreprocessor = TextPreprocessor()
        self.storage: PredictionStorage = PredictionStorage()

        # QLoRA is always attempted (lazy-loaded on first inference)
        self.qlora_predictor: QLoraPredictor = QLoraPredictor(
            base_model_path=config.base_model_path,
            adapter_path=config.qlora_adapter_path,
            max_new_tokens=config.qlora_max_new_tokens,
            temperature=config.qlora_temperature,
        )
        self.qlora_available: bool = True  # Will be set to False on hard failure
        self.sklearn_classifier: Optional[SklearnClassifier] = None

        self._load_sklearn_fallback()
        self._log_initialization_status()

    def _log_initialization_status(self) -> None:
        """Log the initialization status of inference backends."""
        logger.info(
            "SalesNotePredictor initialized (qlora=%s, sklearn=%s, base_model=%s)",
            self.qlora_available,
            self.sklearn_classifier is not None,
            config.base_model_path,
        )

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

        Priority:
          1. QLoRA direct (GPU) — loads model lazily on first call
          2. sklearn (CPU) — fast fallback

        Args:
            cleaned_note: Preprocessed note text.

        Returns:
            Prediction result dictionary.

        Raises:
            NoInferenceBackendError: If no backend is available.
        """
        if self.qlora_available:
            try:
                return self._predict_qlora(cleaned_note)
            except QLoRAInferenceError as exc:
                logger.warning("QLoRA inference failed: %s — attempting sklearn fallback", exc)
                self.qlora_available = False
            except Exception as exc:
                logger.warning("Unexpected QLoRA error: %s — attempting sklearn fallback", exc)
                self.qlora_available = False

        if self.sklearn_classifier is not None:
            return self._predict_sklearn(cleaned_note)

        raise NoInferenceBackendError()

    def _predict_qlora(self, cleaned_note: str) -> Dict[str, Any]:
        """Classify via QLoRA direct inference (Transformers + PEFT).

        The QLoraPredictor loads the model lazily on first call.

        Args:
            cleaned_note: Preprocessed sales note text.

        Returns:
            Prediction result dictionary.

        Raises:
            QLoRAInferenceError: If inference fails.
        """
        start_time: float = time.time()

        try:
            category: str
            confidence: float
            qlora_latency: float
            category, confidence, qlora_latency = self.qlora_predictor.classify(cleaned_note)
        except Exception as exc:
            raise QLoRAInferenceError(
                f"QLoRA classify failed: {exc}"
            ) from exc

        total_elapsed: float = time.time() - start_time

        return {
            "issue_category": category,
            "confidence": confidence,
            "method": InferenceMethod.QLORA_DIRECT.value,
            "latency_seconds": f"{total_elapsed:.2f}",
            "reasoning": (
                f"QLoRA fine-tuned Gemma-2B classified the note as '{category}' "
                f"in {total_elapsed:.2f}s (confidence: {confidence:.0%})"
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

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of all inference backends.

        Returns:
            Status dictionary for health checks and dashboards.
        """
        status: Dict[str, Any] = {
            "qlora_available": self.qlora_available,
            "sklearn_available": self.sklearn_classifier is not None,
            "supported_categories": SUPPORTED_CATEGORIES,
            "base_model": config.base_model_path,
            "adapter": config.qlora_adapter_path,
        }

        # Include QLoRA detailed status if the predictor exists
        try:
            status["qlora_details"] = self.qlora_predictor.get_status()
        except Exception:
            pass

        return status
