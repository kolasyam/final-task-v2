"""Prediction service with tiered inference.

Model priority:
  1. Prompt-engineered Ollama model (gemma-sales-intel via Modelfile) — best accuracy
  2. Base Ollama model (gemma:2b) — good zero-shot accuracy
  3. scikit-learn TF-IDF + RandomForest — fast local fallback

Note: The prompt-engineered model uses an Ollama Modelfile with embedded
few-shot examples. For true parameter fine-tuning, use QLoRA training.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.config import SUPPORTED_CATEGORIES, config
from app.services.ollama_client import (
    PROMPT_MODEL_NAME,
    OllamaClient,
)
from app.services.preprocessing import TextPreprocessor
from app.services.storage import PredictionStorage

logger = logging.getLogger(__name__)


class SklearnClassifier:
    """Wrapper for scikit-learn TF-IDF + RandomForest classification.

    Provides fast local classification as the final fallback when
    no Ollama backend is available.
    """

    def __init__(self) -> None:
        """Initialize the sklearn classifier, loading persisted artifacts."""
        self._vectorizer = self._load_artifact(config.vectorizer_path)
        self._classifier = self._load_artifact(config.classifier_path)
        self._label_encoder = self._load_artifact(config.label_encoder_path)
        logger.info("SklearnClassifier initialized (artifacts loaded)")

    @staticmethod
    def _load_artifact(path: str) -> Any:
        """Load a persisted artifact from disk.

        Args:
            path: Filesystem path to the joblib artifact.

        Returns:
            The deserialized artifact.

        Raises:
            FileNotFoundError: If the artifact does not exist.
        """
        import joblib
        from pathlib import Path

        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{path}'. "
                f"Run training: python -m training.train",
            )
        artifact = joblib.load(path)
        logger.info("Loaded artifact from %s", path)
        return artifact

    def predict(self, text: str) -> Tuple[str, float]:
        """Classify text using the sklearn pipeline.

        Args:
            text: Preprocessed text to classify.

        Returns:
            Tuple of (predicted_category, confidence_score).
        """
        features = self._vectorizer.transform([text])
        prediction = self._classifier.predict(features)
        category = self._label_encoder.inverse_transform(prediction)[0]
        proba = self._classifier.predict_proba(features)[0]
        confidence = float(np.max(proba))

        logger.info("Sklearn prediction: '%s' (confidence=%.3f)", category, confidence)
        return category, confidence


class SalesNotePredictor:
    """Orchestrates sales note classification with tiered inference.

    Inference priority:
        1. Prompt-engineered Ollama model (gemma-sales-intel via Modelfile)
        2. Base Ollama model (gemma:2b) — zero-shot
        3. scikit-learn TF-IDF + RandomForest — fast fallback

    For true parameter-efficient fine-tuning, use QLoRA training
    (training/finetune_qlora.py) instead of the Modelfile approach.

    Attributes:
        ollama_client: Client for Ollama LLM inference.
        ollama_available: Whether any Ollama backend is reachable.
        is_prompt_model: Whether the prompt-engineered model is being used.
        sklearn_classifier: Fallback sklearn classifier.
        preprocessing: Text preprocessing pipeline.
        storage: Prediction logging service.
    """

    def __init__(self) -> None:
        """Initialize the predictor, probing Ollama availability."""
        self.preprocessor = TextPreprocessor()
        self.storage = PredictionStorage()
        self.ollama_client = OllamaClient()
        self.sklearn_classifier: Optional[SklearnClassifier] = None

        self.ollama_available: bool = self.ollama_client.health_check()
        self.is_prompt_model: bool = self.ollama_client.is_prompt_model

        if self.ollama_available:
            if self.is_prompt_model:
                logger.info("✅ Using prompt-engineered model: gemma-sales-intel")
            else:
                logger.info("ℹ️  Using base model: gemma:2b (fine-tune for better accuracy)")
        else:
            logger.warning("Ollama not available — using sklearn fallback")

        self._load_sklearn_fallback()

    def _load_sklearn_fallback(self) -> None:
        """Attempt to load the sklearn fallback classifier."""
        try:
            self.sklearn_classifier = SklearnClassifier()
            logger.info("Sklearn fallback classifier loaded")
        except FileNotFoundError as exc:
            logger.warning("Sklearn fallback not available: %s", exc)
            self.sklearn_classifier = None

    def predict(self, note: str) -> Dict[str, Any]:
        """Classify a sales representative note into an issue category.

        Args:
            note: Raw sales representative note text.

        Returns:
            Dictionary with keys:
                - issue_category (str): Predicted category.
                - confidence (float): Prediction confidence (0.0 to 1.0).
                - method (str): 'ollama_finetuned', 'ollama_base', or 'sklearn_tfidf'.
                - latency_seconds (str): Inference time.
                - reasoning (str): Human-readable reasoning.

        Raises:
            ValueError: If the note is empty or whitespace.
            RuntimeError: If no inference backend is available.
        """
        if not note or not note.strip():
            raise ValueError("Note cannot be empty")

        cleaned_note: str = self.preprocessor.preprocess(note)

        if self.ollama_available:
            result = self._predict_ollama(cleaned_note)
        elif self.sklearn_classifier is not None:
            result = self._predict_sklearn(cleaned_note)
        else:
            raise RuntimeError(
                "No inference backend available. "
                f"Start Ollama or run: python -m training.train"
            )

        self.storage.save_prediction(
            input_note=note,
            issue_category=result["issue_category"],
        )
        return result

    def _predict_ollama(self, cleaned_note: str) -> Dict[str, Any]:
        """Classify via Ollama (fine-tuned or base model).

        Args:
            cleaned_note: Preprocessed sales note text.

        Returns:
            Prediction result dictionary.
        """
        start_time = time.time()
        raw_output, ollama_elapsed = self.ollama_client.classify_note(cleaned_note)
        category = self.ollama_client.extract_category(raw_output)

        if category is None:
            logger.warning("Ollama returned unparseable: '%s'. Falling back.", raw_output)
            if self.sklearn_classifier is not None:
                return self._predict_sklearn(cleaned_note)
            category = SUPPORTED_CATEGORIES[0]

        total_elapsed = time.time() - start_time
        confidence = self._estimate_confidence(raw_output, category)

        method = "ollama_prompt_model" if self.is_prompt_model else "ollama_base"
        model_label = PROMPT_MODEL_NAME if self.is_prompt_model else self.ollama_client.base_model_name

        return {
            "issue_category": category,
            "confidence": confidence,
            "method": method,
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
        start_time = time.time()
        category, confidence = self.sklearn_classifier.predict(cleaned_note)
        elapsed = time.time() - start_time

        return {
            "issue_category": category,
            "confidence": confidence,
            "method": "sklearn_tfidf",
            "latency_seconds": f"{elapsed:.4f}",
            "reasoning": (
                f"TF-IDF + RandomForest classified as '{category}' "
                f"({confidence:.1%} confidence)"
            ),
        }

    @staticmethod
    def _estimate_confidence(raw_output: str, category: str) -> float:
        """Estimate confidence from Ollama output clarity.

        Args:
            raw_output: Raw model output string.
            category: Extracted category name.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        cleaned = raw_output.lower().strip()
        if cleaned == category:
            return 0.95
        if category.replace("_", " ") in cleaned:
            return 0.85
        if category in cleaned:
            return 0.75
        return 0.60

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
