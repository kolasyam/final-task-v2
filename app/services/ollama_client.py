"""Ollama client for local LLM inference via gemma:2b.

Communicates with a local Ollama server through its REST API.
All inference runs locally — no internet access required.

Supports three model tiers:
  1. QLoRA-finetuned model (real parameter fine-tuning on GPU)
  2. Prompt-engineered model (gemma-sales-intel via Ollama Modelfile)
  3. Base gemma:2b (zero-shot)
"""

import logging
import time
from typing import Any, Dict, FrozenSet, Optional, Tuple

import requests

from app.config import config
from app.core.constants import (
    CATEGORY_KEYWORDS,
    CATEGORY_PREFIXES_TO_STRIP,
    CLASSIFICATION_PROMPT,
    OLLAMA_MAX_TOKENS,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
    PROMPT_MODEL_NAME,
    SUPPORTED_CATEGORIES,
    SUPPORTED_CATEGORIES_SET,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
    DEFAULT_RETRY_BACKOFF_FACTOR,
)
from app.core.exceptions import (
    CategoryExtractionError,
    OllamaConnectionError,
    OllamaTimeoutError,
)

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        max_delay: float = DEFAULT_RETRY_MAX_DELAY,
        backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR,
    ) -> None:
        self.max_retries: int = max_retries
        self.base_delay: float = base_delay
        self.max_delay: float = max_delay
        self.backoff_factor: float = backoff_factor


class OllamaClient:
    """Client for communicating with a local Ollama server.

    Sends prompts to the Ollama REST API and returns generated text.
    Automatically detects and uses the fine-tuned 'gemma-sales-intel'
    model if available, otherwise falls back to base 'gemma:2b'.

    Includes retry logic with exponential backoff for transient failures.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        """Initialize the Ollama client.

        Args:
            base_url: Ollama server URL. Defaults to config or localhost:11434.
            model_name: Model to use. Defaults to config or 'gemma:2b'.
            timeout: Request timeout in seconds. Defaults to config or 120.
            retry_config: Retry configuration. Uses defaults if not specified.
        """
        self.base_url: str = (base_url or config.ollama_base_url).rstrip("/")
        self.base_model_name: str = model_name or config.model_name
        self.timeout: int = timeout or config.ollama_timeout
        self.retry_config: RetryConfig = retry_config or RetryConfig()
        self._generate_url: str = f"{self.base_url}/api/generate"

        # Detect which model to use: prompt-engineered or base
        self.model_name: str = self._detect_best_model()
        self.is_prompt_model: bool = self.model_name == PROMPT_MODEL_NAME

        logger.info(
            "OllamaClient initialized (url=%s, model=%s, prompt_model=%s, timeout=%ds)",
            self.base_url,
            self.model_name,
            self.is_prompt_model,
            self.timeout,
        )

    def _detect_best_model(self) -> str:
        """Detect the best available model.

        Prefers the prompt-engineered model if available, otherwise
        falls back to the base model.

        Returns:
            Model name string to use for inference.
        """
        available: list = self._list_models()

        if PROMPT_MODEL_NAME in available:
            logger.info("Prompt-engineered model '%s' detected — using it", PROMPT_MODEL_NAME)
            return PROMPT_MODEL_NAME

        if self.base_model_name in available:
            logger.info("Base model '%s' detected — using it", self.base_model_name)
            return self.base_model_name

        logger.warning(
            "Neither prompt-engineered nor base model found. "
            "Available: %s. Defaulting to '%s'.",
            available,
            self.base_model_name,
        )
        return self.base_model_name

    def _list_models(self) -> list:
        """List all available Ollama models.

        Returns:
            List of model name strings.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=5,
            )
            if response.status_code == 200:
                return [m.get("name", "") for m in response.json().get("models", [])]
        except requests.exceptions.RequestException as exc:
            logger.debug("Could not list models: %s", exc)
        return []

    def health_check(self) -> bool:
        """Check if the Ollama server is reachable and the model is loaded.

        Returns:
            True if the server responds and the model is available.
        """
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code != 200:
                logger.error("Ollama server returned status %d", response.status_code)
                return False

            available: list = self._list_models()
            if self.model_name in available:
                logger.info(
                    "Ollama health check passed (model=%s, finetuned=%s)",
                    self.model_name,
                    self.is_prompt_model,
                )
                return True

            logger.warning(
                "Model '%s' not found. Available: %s",
                self.model_name,
                available,
            )
            return False
        except requests.exceptions.RequestException as exc:
            logger.error("Ollama health check failed: %s", exc)
            return False

    def generate(
        self, prompt: str, max_tokens: int = OLLAMA_MAX_TOKENS,
    ) -> Tuple[str, float]:
        """Send a prompt to Ollama with retry logic and exponential backoff.

        Args:
            prompt: The full prompt to send.
            max_tokens: Maximum tokens to generate.

        Returns:
            Tuple of (generated_text, elapsed_seconds).

        Raises:
            OllamaConnectionError: If the server is unreachable after retries.
            OllamaTimeoutError: If the request times out after retries.
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": OLLAMA_TEMPERATURE,
                "top_p": OLLAMA_TOP_P,
                "num_predict": max_tokens,
            },
        }

        last_exception: Optional[Exception] = None
        retry_config: RetryConfig = self.retry_config

        for attempt in range(retry_config.max_retries + 1):
            if attempt > 0:
                delay: float = min(
                    retry_config.base_delay * (retry_config.backoff_factor ** (attempt - 1)),
                    retry_config.max_delay,
                )
                logger.info("Retry attempt %d/%d after %.1fs delay", attempt, retry_config.max_retries, delay)
                time.sleep(delay)

            try:
                return self._execute_generate(payload, prompt)
            except (OllamaConnectionError, OllamaTimeoutError) as exc:
                last_exception = exc
                logger.warning(
                    "Ollama request failed (attempt %d/%d): %s",
                    attempt + 1,
                    retry_config.max_retries + 1,
                    exc,
                )
            except RuntimeError:
                raise

        # All retries exhausted
        logger.error("All %d retry attempts exhausted for Ollama request", retry_config.max_retries + 1)
        raise last_exception  # type: ignore[misc]

    def _execute_generate(
        self, payload: Dict[str, Any], prompt: str,
    ) -> Tuple[str, float]:
        """Execute a single generate request to Ollama.

        Args:
            payload: JSON payload for the request.
            prompt: The original prompt (for logging).

        Returns:
            Tuple of (generated_text, elapsed_seconds).

        Raises:
            OllamaConnectionError: On connection failure.
            OllamaTimeoutError: On timeout.
            RuntimeError: On other request failures.
        """
        logger.debug("Sending prompt to Ollama (model=%s)", self.model_name)
        start_time: float = time.time()

        try:
            response = requests.post(
                self._generate_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            elapsed: float = time.time() - start_time
            result: Dict[str, Any] = response.json()
            generated_text: str = result.get("response", "").strip()
            logger.debug("Ollama response in %.2fs: '%s'", elapsed, generated_text)
            return generated_text, elapsed
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(self.base_url) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaTimeoutError(self.timeout) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    def classify_note(self, note: str) -> Tuple[str, float]:
        """Classify a sales note using the Ollama model.

        For the prompt-engineered model, sends just the note (system prompt
        already has the instructions). For the base model, uses the
        full classification prompt.

        Args:
            note: Preprocessed sales note text.

        Returns:
            Tuple of (raw_category_string, elapsed_seconds).
        """
        prompt: str = self._build_classification_prompt(note)
        return self.generate(prompt, max_tokens=OLLAMA_MAX_TOKENS)

    def _build_classification_prompt(self, note: str) -> str:
        """Build the classification prompt based on model type.

        Args:
            note: Preprocessed sales note text.

        Returns:
            Formatted prompt string.
        """
        if self.is_prompt_model:
            return f"Classify this sales note: {note}\nCategory:"
        return CLASSIFICATION_PROMPT.format(note=note)

    @staticmethod
    def extract_category(raw_output: str) -> Optional[str]:
        """Extract a valid category from raw model output.

        Uses a three-tier matching strategy:
          1. Direct match against supported categories
          2. Substring match (handling formatting variations)
          3. Keyword-based fuzzy match using CATEGORY_KEYWORDS

        Args:
            raw_output: Raw string output from the model.

        Returns:
            A valid category string, or None if no match found.
        """
        cleaned: str = OllamaClient._clean_raw_output(raw_output)

        # Direct match
        if cleaned in SUPPORTED_CATEGORIES_SET:
            return cleaned

        # Substring match
        category: Optional[str] = OllamaClient._substring_match(cleaned)
        if category is not None:
            return category

        # Keyword-based fuzzy match
        return OllamaClient._keyword_match(cleaned)

    @staticmethod
    def _clean_raw_output(raw_output: str) -> str:
        """Clean raw model output by lowercasing and stripping prefixes.

        Args:
            raw_output: Raw model output.

        Returns:
            Cleaned and normalized string.
        """
        cleaned: str = raw_output.lower().strip().rstrip(".!?\n")
        for prefix in CATEGORY_PREFIXES_TO_STRIP:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        return cleaned

    @staticmethod
    def _substring_match(cleaned: str) -> Optional[str]:
        """Attempt substring matching against supported categories.

        Args:
            cleaned: Pre-cleaned output string.

        Returns:
            Matching category or None.
        """
        for category in SUPPORTED_CATEGORIES:
            if category in cleaned:
                return category
        return None

    @staticmethod
    def _keyword_match(cleaned: str) -> Optional[str]:
        """Attempt keyword-based fuzzy matching.

        Uses the CATEGORY_KEYWORDS lookup table from constants.
        Scores each category by keyword overlap with the cleaned output.

        Args:
            cleaned: Pre-cleaned output string.

        Returns:
            Best-matching category or None if no keywords match.
        """
        best_category: Optional[str] = None
        best_score: int = 0
        cleaned_words: FrozenSet[str] = frozenset(cleaned.split())

        for category, keywords in CATEGORY_KEYWORDS.items():
            score: int = len(cleaned_words & keywords)
            if score > best_score:
                best_score = score
                best_category = category

        if best_category is not None and best_score > 0:
            logger.warning(
                "Fuzzy matched '%s' to '%s' (score=%d)",
                cleaned,
                best_category,
                best_score,
            )
            return best_category

        return None
