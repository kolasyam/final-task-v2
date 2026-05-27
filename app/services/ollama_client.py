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
from typing import Any, Dict, Optional, Tuple

import requests

from app.config import SUPPORTED_CATEGORIES, config

logger = logging.getLogger(__name__)

# Name of the fine-tuned model created by training.generate_modelfile
PROMPT_MODEL_NAME: str = "gemma-sales-intel"

# Classification prompt — used with the base gemma:2b model.
# The fine-tuned model has this baked into its system prompt.
CLASSIFICATION_PROMPT: str = (
    "You are a sales intelligence analyst. Classify the following "
    "sales representative field note into exactly ONE of these categories:\n\n"
    "1. supply_chain_delay — stock shortages, delivery delays, replenishment issues\n"
    "2. retailer_dissatisfaction — complaints, unhappy retailers, service issues\n"
    "3. pricing_conflict — price disputes, margin concerns, discount conflicts\n"
    "4. competitor_pressure — competitor actions, market share threats, rival offers\n"
    "5. demand_spike — unexpected demand surges, stockout from high volume\n\n"
    "Sales Note: {note}\n\n"
    "Respond with ONLY the category name (e.g., supply_chain_delay). "
    "No explanation, no extra text.\n"
    "Category:"
)


class OllamaClient:
    """Client for communicating with a local Ollama server.

    Sends prompts to the Ollama REST API and returns generated text.
    Automatically detects and uses the fine-tuned 'gemma-sales-intel'
    model if available, otherwise falls back to base 'gemma:2b'.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Initialize the Ollama client.

        Args:
            base_url: Ollama server URL. Defaults to config or localhost:11434.
            model_name: Model to use. Defaults to config or 'gemma:2b'.
            timeout: Request timeout in seconds. Defaults to config or 120.
        """
        self.base_url = (base_url or config.ollama_base_url).rstrip("/")
        self.base_model_name = model_name or config.model_name
        self.timeout = timeout or config.ollama_timeout
        self._generate_url = f"{self.base_url}/api/generate"

        # Detect which model to use: prompt-engineered or base
        self.model_name = self._detect_best_model()
        self.is_prompt_model = self.model_name == PROMPT_MODEL_NAME

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
        available = self._list_models()

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
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
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

            available = self._list_models()
            if self.model_name in available:
                logger.info(
                    "Ollama health check passed (model=%s, finetuned=%s)",
                    self.model_name, self.is_prompt_model,
                )
                return True

            logger.warning(
                "Model '%s' not found. Available: %s",
                self.model_name, available,
            )
            return False
        except requests.exceptions.RequestException as exc:
            logger.error("Ollama health check failed: %s", exc)
            return False

    def generate(
        self, prompt: str, max_tokens: int = 30,
    ) -> Tuple[str, float]:
        """Send a prompt to Ollama and return the generated text.

        Args:
            prompt: The full prompt to send.
            max_tokens: Maximum tokens to generate.

        Returns:
            Tuple of (generated_text, elapsed_seconds).

        Raises:
            RuntimeError: If the Ollama server is unreachable or returns an error.
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": max_tokens,
            },
        }

        logger.debug("Sending prompt to Ollama (model=%s)", self.model_name)
        start_time = time.time()

        try:
            response = requests.post(
                self._generate_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            elapsed = time.time() - start_time
            result = response.json()
            generated_text: str = result.get("response", "").strip()
            logger.debug("Ollama response in %.2fs: '%s'", elapsed, generated_text)
            return generated_text, elapsed
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running: 'ollama serve'",
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s. "
                f"The model may still be loading.",
            ) from exc
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
        if self.is_prompt_model:
            # Prompt-engineered model already has the system prompt
            prompt = f"Classify this sales note: {note}\nCategory:"
        else:
            # Base model needs the full prompt
            prompt = CLASSIFICATION_PROMPT.format(note=note)

        return self.generate(prompt, max_tokens=30)

    @staticmethod
    def extract_category(raw_output: str) -> Optional[str]:
        """Extract a valid category from raw model output.

        Attempts direct matching, substring matching, and keyword-based
        fallback in that order.

        Args:
            raw_output: Raw string output from the model.

        Returns:
            A valid category string, or None if no match found.
        """
        cleaned: str = raw_output.lower().strip().rstrip(".!?\n")

        # Remove common prefixes
        for prefix in [
            "issue category:",
            "category:",
            "the issue category is",
            "the category is",
            "-",
        ]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        # Direct match
        if cleaned in SUPPORTED_CATEGORIES:
            return cleaned

        # Substring match
        for category in SUPPORTED_CATEGORIES:
            if category in cleaned:
                return category

        # Keyword-based fallback
        keyword_map: Dict[str, str] = {
            "supply_chain_delay": (
                "supply chain delay supply chain stock shortage delivery "
                "delay replenish shipment backlog warehouse inventory mismatch "
                "running out fast movers stock movement"
            ),
            "retailer_dissatisfaction": (
                "retailer dissatisfaction unhappy complain angry "
                "frustrated poor service bad experience relationship issue "
                "dissatisfied"
            ),
            "pricing_conflict": (
                "pricing conflict price dispute margin discount "
                "expensive cheap billing charge rate cost conflict"
            ),
            "competitor_pressure": (
                "competitor pressure competition rival alternative "
                "switching market share competitor launched campaign"
            ),
            "demand_spike": (
                "demand spike surge overflow high volume rush "
                "stockout demand surge unexpected demand high demand"
            ),
        }

        best_category: Optional[str] = None
        best_score: int = 0
        for category, keywords in keyword_map.items():
            score: int = sum(1 for kw in keywords.split() if kw in cleaned)
            if score > best_score:
                best_score = score
                best_category = category

        if best_category and best_score > 0:
            logger.warning(
                "Fuzzy matched '%s' to '%s' (score=%d)",
                raw_output, best_category, best_score,
            )
            return best_category

        logger.warning("Could not extract category from: '%s'", raw_output)
        return None
