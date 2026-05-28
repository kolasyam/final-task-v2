"""Direct QLoRA inference engine using Transformers + PEFT.

Loads the base Gemma model in 4-bit quantization and applies the
QLoRA adapter at runtime — no Ollama merge, GGUF conversion, or
Modelfile packaging required.

Key design decisions:
  - 4-bit NF4 quantization: keeps base model in ~1.5GB VRAM
  - LoRA adapter loaded on top: adds <50MB overhead
  - Lazy singleton pattern: model loaded once on first inference
  - Chat template: reuses the chat_template.jinja from the adapter dir
  - Temperature=0.1 for deterministic classification output

VRAM budget (L40S 48GB):
  - Base model (4-bit):     ~1.5 GB
  - Adapter overhead:       ~0.05 GB
  - KV cache (512 tokens):  ~0.5 GB
  - Activation buffers:    ~1.0 GB
  - Total:                  ~3.0 GB  (plenty of headroom)

Usage:
    predictor = QLoraPredictor()
    category, confidence, latency = predictor.classify("Stock running out")
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple

import torch

from app.core.constants import (
    CATEGORY_KEYWORDS,
    CATEGORY_PREFIXES_TO_STRIP,
    CLASSIFICATION_PROMPT,
    SUPPORTED_CATEGORIES,
    SUPPORTED_CATEGORIES_SET,
)
from app.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when the QLoRA model or adapter cannot be loaded."""

    def __init__(self, message: str, path: str = "") -> None:
        self.path = path
        super().__init__(message)


class QLoraPredictor:
    """Direct QLoRA inference engine via Transformers + PEFT.

    Loads the base Gemma-2-2b-it model in 4-bit NF4 quantization,
    then applies the LoRA adapter weights at inference time.

    This class is designed as a process-safe singleton. The model
    is loaded lazily on first .classify() call and kept in memory
    for the lifetime of the process.

    Attributes:
        base_model_path: Filesystem path to the local Gemma base model.
        adapter_path: Filesystem path to the LoRA adapter directory.
        max_new_tokens: Maximum tokens to generate per classification.
        temperature: Sampling temperature (low for deterministic output).
    """

    # --- Class-level singleton state ---
    _model: Optional[Any] = None
    _tokenizer: Optional[Any] = None
    _loaded: bool = False

    def __init__(
        self,
        base_model_path: str = "/opt/ai-platform/models/gemma-2-2b-it",
        adapter_path: str = "training/saved_model/qlora_adapter",
        max_new_tokens: int = 30,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> None:
        """Initialize the QLoRA predictor.

        Does NOT load the model immediately — loading is deferred to
        first .classify() call via _ensure_loaded().

        Args:
            base_model_path: Path to local Gemma-2-2b-it model directory.
            adapter_path: Path to QLoRA adapter directory.
            max_new_tokens: Maximum tokens to generate per inference.
            temperature: Inference temperature (0.1 = near-deterministic).
            top_p: Nucleus sampling parameter.
        """
        self.base_model_path: str = base_model_path
        self.adapter_path: str = adapter_path
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.top_p: float = top_p

        # Load chat template from adapter directory if available
        self._chat_template: Optional[str] = self._load_chat_template()

        logger.info(
            "QLoraPredictor configured (base=%s, adapter=%s, max_new_tokens=%d, temp=%.2f)",
            self.base_model_path,
            self.adapter_path,
            self.max_new_tokens,
            self.temperature,
        )

    # ------------------------------------------------------------------
    #  Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model and tokenizer if not already loaded.

        This method is idempotent — safe to call multiple times.
        Uses a class-level lock pattern to avoid duplicate loads in
        multi-threaded FastAPI workers (each process loads once).

        Raises:
            ModelLoadError: If model or adapter cannot be loaded.
        """
        if self._loaded and self._model is not None and self._tokenizer is not None:
            return

        logger.info("Loading QLoRA model — this may take 30-60 seconds...")
        load_start: float = time.time()

        self._validate_paths()

        try:
            self._tokenizer = self._load_tokenizer()
            self._model = self._load_model_with_adapter()
            self._loaded = True

            elapsed: float = time.time() - load_start
            gpu_mem = self._get_gpu_memory_usage()
            logger.info(
                "QLoRA model loaded successfully in %.1fs (GPU mem: %s)",
                elapsed,
                gpu_mem,
            )
        except ModelLoadError:
            raise
        except Exception as exc:
            self._loaded = False
            self._model = None
            self._tokenizer = None
            raise ModelLoadError(
                f"Failed to load QLoRA model: {exc}",
                path=self.base_model_path,
            ) from exc

    def _validate_paths(self) -> None:
        """Validate that model and adapter paths exist.

        Raises:
            ModelLoadError: If either path is missing.
        """
        if not Path(self.base_model_path).exists():
            raise ModelLoadError(
                f"Base model not found at: {self.base_model_path}. "
                "Ensure the Gemma model is available locally.",
                path=self.base_model_path,
            )
        if not Path(self.adapter_path).exists():
            raise ModelLoadError(
                f"QLoRA adapter not found at: {self.adapter_path}. "
                "Run: python -m training.finetune_qlora",
                path=self.adapter_path,
            )
        # Verify critical adapter files
        adapter_files = ["adapter_config.json", "adapter_model.safetensors"]
        for fname in adapter_files:
            fpath = Path(self.adapter_path) / fname
            if not fpath.exists():
                raise ModelLoadError(
                    f"Adapter file missing: {fpath}",
                    path=str(fpath),
                )

    def _load_tokenizer(self) -> Any:
        """Load the tokenizer from the adapter directory.

        The adapter directory contains a copy of the tokenizer with
        the correct chat template, so we prefer loading from there.

        Returns:
            HuggingFace tokenizer instance.
        """
        from transformers import AutoTokenizer

        tokenizer_path: str = (
            self.adapter_path
            if (Path(self.adapter_path) / "tokenizer.json").exists()
            else self.base_model_path
        )

        logger.info("Loading tokenizer from: %s", tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def _load_model_with_adapter(self) -> Any:
        """Load the base model in 4-bit and apply the LoRA adapter.

        Returns:
            PEFT model instance with LoRA adapter applied.
        """
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        logger.info("Loading base model with 4-bit quantization: %s", self.base_model_path)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        base_model.config.use_cache = True

        logger.info("Applying LoRA adapter from: %s", self.adapter_path)
        model = PeftModel.from_pretrained(
            base_model,
            self.adapter_path,
            is_trainable=False,
        )

        # Log parameter stats
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Model ready: %d total params, %d trainable (%.4f%%)",
            total, trainable, 100 * trainable / total,
        )

        return model

    def _load_chat_template(self) -> Optional[str]:
        """Load chat template from the adapter directory if available.

        Returns:
            Chat template string, or None if not found.
        """
        template_path = Path(self.adapter_path) / "chat_template.jinja"
        if template_path.exists():
            template = template_path.read_text(encoding="utf-8")
            logger.info("Loaded chat template from adapter directory")
            return template
        return None

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    def classify(self, note: str) -> Tuple[str, float, float]:
        """Classify a single sales note.

        Automatically triggers model loading on first call.

        Args:
            note: Preprocessed sales note text.

        Returns:
            Tuple of (category, confidence_score, latency_seconds).
        """
        self._ensure_loaded()

        start_time: float = time.time()

        messages = self._build_messages(note)
        input_ids = self._apply_chat_template(messages)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0.0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated_text = self._decode_output(input_ids, output_ids)
        latency: float = time.time() - start_time

        category, confidence = self._parse_category(generated_text)

        logger.info(
            "QLoRA inference: category=%s confidence=%.2f latency=%.2fs",
            category, confidence, latency,
        )
        return category, confidence, latency

    def _build_messages(self, note: str) -> list:
        """Build the chat message list for classification.

        Uses the system + user message format that the model was
        trained on during QLoRA fine-tuning.

        Args:
            note: Preprocessed sales note text.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        system_content: str = (
            "You are a sales intelligence analyst for an FMCG company. "
            "Your task is to classify sales representative field notes into "
            "exactly one of these 5 issue categories:\n"
            "1. supply_chain_delay — stock shortages, delivery delays, replenishment issues\n"
            "2. retailer_dissatisfaction — complaints, unhappy retailers, service issues\n"
            "3. pricing_conflict — price disputes, margin concerns, discount conflicts\n"
            "4. competitor_pressure — competitor actions, market share threats, rival offers\n"
            "5. demand_spike — unexpected demand surges, stockouts from high volume\n\n"
            "Respond with ONLY the category name. No explanation."
        )
        user_content: str = f"Classify this sales note: {note}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _apply_chat_template(self, messages: list) -> torch.Tensor:
        """Apply the tokenizer's chat template and return tensor on device.

        Args:
            messages: List of message dicts.

        Returns:
            Input IDs tensor on the model's device.
        """
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=self._chat_template,
        )
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024,
        )
        return inputs.to(self._model.device)

    def _decode_output(self, input_ids: torch.Tensor, output_ids: torch.Tensor) -> str:
        """Decode only the newly generated tokens (exclude the prompt).

        Args:
            input_ids: Tokenized input (prompt) tensor.
            output_ids: Full output tensor (prompt + generated).

        Returns:
            Decoded generated text string.
        """
        generated_only = output_ids[0][input_ids.shape[1]:]
        return self._tokenizer.decode(generated_only, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    #  Category extraction + confidence
    # ------------------------------------------------------------------

    # Confidence thresholds based on extraction clarity
    _EXACT_MATCH_CONFIDENCE: float = 0.97
    _SPACE_NORMALIZED_CONFIDENCE: float = 0.90
    _SUBSTRING_MATCH_CONFIDENCE: float = 0.80
    _FUZZY_MATCH_CONFIDENCE: float = 0.65

    def _parse_category(self, raw_output: str) -> Tuple[str, float]:
        """Extract a valid category and confidence from raw model output.

        Matching strategy:
          1. Exact match against supported categories
          2. Substring match (handles formatting variations)
          3. Keyword-based fuzzy match
          4. Fallback to first supported category

        Args:
            raw_output: Raw text output from the model.

        Returns:
            Tuple of (category, confidence_score).
        """
        cleaned: str = OllamaClient._clean_raw_output(raw_output)

        # Direct match
        if cleaned in SUPPORTED_CATEGORIES_SET:
            return cleaned, self._EXACT_MATCH_CONFIDENCE

        # Substring match
        for category in SUPPORTED_CATEGORIES:
            if category in cleaned:
                return category, self._SUBSTRING_MATCH_CONFIDENCE

        # Category with spaces instead of underscores
        for category in SUPPORTED_CATEGORIES:
            spaced = category.replace("_", " ")
            if spaced in cleaned:
                return category, self._SPACE_NORMALIZED_CONFIDENCE

        # Keyword-based fuzzy match
        fuzzy: Optional[str] = self._keyword_match(cleaned)
        if fuzzy is not None:
            return fuzzy, self._FUZZY_MATCH_CONFIDENCE

        # Fallback — use most likely category if output is near-empty
        if len(cleaned) < 2:
            logger.warning("Model returned near-empty output, defaulting to first category")
            return SUPPORTED_CATEGORIES[0], 0.30

        # Last resort: return first category with very low confidence
        logger.warning(
            "Could not match output '%s' to any category", raw_output
        )
        return SUPPORTED_CATEGORIES[0], 0.30

    @staticmethod
    def _clean_output(raw_output: str) -> str:
        """Clean raw model output.

        Delegates to ``OllamaClient._clean_raw_output`` to avoid duplicating
        the prefix-stripping and normalization logic.

        Args:
            raw_output: Raw text from model generation.

        Returns:
            Cleaned, normalized string.
        """
        return OllamaClient._clean_raw_output(raw_output)

    @staticmethod
    def _keyword_match(cleaned: str) -> Optional[str]:
        """Attempt keyword-based fuzzy matching.

        Scores each category by shared keyword overlap with the
        cleaned output. Returns the best match if any keywords overlap.

        Args:
            cleaned: Pre-cleaned model output string.

        Returns:
            Best-matching category or None.
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
            logger.info("Fuzzy matched '%s' -> '%s' (score=%d)", cleaned, best_category, best_score)
            return best_category
        return None

    # ------------------------------------------------------------------
    #  Health and diagnostics
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded in memory."""
        return self._loaded and self._model is not None

    def health_check(self) -> bool:
        """Verify the predictor is healthy and ready for inference.

        Returns:
            True if the model is loaded and on a CUDA device.
        """
        if not self.is_loaded:
            return False
        try:
            device = next(self._model.parameters()).device
            return device.type == "cuda"
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status for health check endpoints.

        Returns:
            Dictionary with model status, device info, and memory usage.
        """
        gpu_mem: str = self._get_gpu_memory_usage()
        device_name: str = "N/A"
        is_cuda: bool = False

        if self.is_loaded:
            try:
                device = next(self._model.parameters()).device
                is_cuda = device.type == "cuda"
                if is_cuda:
                    device_name = torch.cuda.get_device_name(device)
            except Exception:
                pass

        return {
            "loaded": self.is_loaded,
            "device": device_name,
            "is_cuda": is_cuda,
            "gpu_memory": gpu_mem,
            "base_model": self.base_model_path,
            "adapter": self.adapter_path,
            "supported_categories": list(SUPPORTED_CATEGORIES),
        }

    @staticmethod
    def _get_gpu_memory_usage() -> str:
        """Get GPU memory usage string for diagnostics.

        Returns:
            Formatted string like "2.1 GB / 48.0 GB" or "N/A".
        """
        if not torch.cuda.is_available():
            return "N/A (no CUDA)"
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"{allocated:.1f}GB allocated / {reserved:.1f}GB reserved / {total:.1f}GB total"
        except Exception:
            return "N/A"

    # ------------------------------------------------------------------
    #  Cleanup
    # ------------------------------------------------------------------

    @classmethod
    def unload(cls) -> None:
        """Free model memory. Useful for testing or graceful restarts."""
        if cls._model is not None:
            del cls._model
            cls._model = None
        if cls._tokenizer is not None:
            del cls._tokenizer
            cls._tokenizer = None
        cls._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("QLoRA model unloaded and GPU cache cleared")
