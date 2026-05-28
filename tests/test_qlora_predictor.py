"""Tests for the QLoRA direct inference engine (QLoraPredictor).

These tests use fully mocked Transformers/PEFT imports to avoid
requiring GPU or model files. The QLoraPredictor unit logic —
path validation, category parsing, message building — is tested
independently of the actual model loading.

Usage:
    pytest tests/test_qlora_predictor.py -v
"""

import json
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from app.core.constants import SUPPORTED_CATEGORIES
from app.services.qlora_predictor import ModelLoadError, QLoraPredictor


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_adapter_dir(tmp_path: Path) -> str:
    """Create a minimal valid adapter directory structure."""
    adapter_dir = tmp_path / "qlora_adapter"
    adapter_dir.mkdir()

    config_data: Dict = {
        "base_model_name_or_path": "/opt/ai-platform/models/gemma-2-2b-it",
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config_data))
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 100)
    (adapter_dir / "tokenizer.json").write_text("{}")
    (adapter_dir / "tokenizer_config.json").write_text("{}")

    base_dir = tmp_path / "gemma-2-2b-it"
    base_dir.mkdir()
    (base_dir / "config.json").write_text(json.dumps({
        "model_type": "gemma2",
        "vocab_size": 256000,
    }))

    return str(adapter_dir)


@pytest.fixture
def predictor(valid_adapter_dir: str, tmp_path: Path) -> QLoraPredictor:
    """Create a QLoraPredictor with mocked model loading (no GPU)."""
    base_dir = str(tmp_path / "gemma-2-2b-it")
    with patch("app.services.qlora_predictor.QLoraPredictor._load_model_with_adapter"):
        with patch("app.services.qlora_predictor.QLoraPredictor._load_tokenizer"):
            pred = QLoraPredictor(
                base_model_path=base_dir,
                adapter_path=valid_adapter_dir,
            )
            pred._loaded = True
            pred._model = MagicMock()
            pred._model.device = MagicMock()
            pred._model.device.type = "cuda"
            pred._model.device.index = 0
            pred._tokenizer = MagicMock()
            pred._tokenizer.pad_token_id = 0
            pred._tokenizer.eos_token_id = 1
            return pred


# =============================================================================
# Path Validation Tests
# =============================================================================


class TestPathValidation:
    """Tests for model and adapter path validation."""

    def test_missing_base_model_raises_error(self, valid_adapter_dir: str) -> None:
        pred = QLoraPredictor(
            base_model_path="/nonexistent/path/to/model",
            adapter_path=valid_adapter_dir,
        )
        with pytest.raises(ModelLoadError, match="Base model not found"):
            pred._validate_paths()

    def test_missing_adapter_raises_error(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "gemma-2-2b-it"
        base_dir.mkdir()
        pred = QLoraPredictor(
            base_model_path=str(base_dir),
            adapter_path="/nonexistent/adapter",
        )
        with pytest.raises(ModelLoadError, match="adapter not found"):
            pred._validate_paths()

    def test_missing_adapter_config_raises_error(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "gemma-2-2b-it"
        base_dir.mkdir()
        adapter_dir = tmp_path / "broken_adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00")

        pred = QLoraPredictor(
            base_model_path=str(base_dir),
            adapter_path=str(adapter_dir),
        )
        with pytest.raises(ModelLoadError, match="adapter_config.json"):
            pred._validate_paths()

    def test_missing_safetensors_raises_error(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "gemma-2-2b-it"
        base_dir.mkdir()
        adapter_dir = tmp_path / "broken_adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")

        pred = QLoraPredictor(
            base_model_path=str(base_dir),
            adapter_path=str(adapter_dir),
        )
        with pytest.raises(ModelLoadError, match="adapter_model.safetensors"):
            pred._validate_paths()

    def test_valid_paths_pass(self, valid_adapter_dir: str, tmp_path: Path) -> None:
        base_dir = str(tmp_path / "gemma-2-2b-it")
        pred = QLoraPredictor(
            base_model_path=base_dir,
            adapter_path=valid_adapter_dir,
        )
        pred._validate_paths()


# =============================================================================
# Category Parsing Tests
# =============================================================================


class TestCategoryParsing:
    """Tests for the _parse_category method."""

    def _make_pred(self, tmp_path: Path) -> QLoraPredictor:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "test", "peft_type": "LORA",
            "r": 16, "lora_alpha": 32, "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
        }))
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00")
        (adapter_dir / "tokenizer.json").write_text("{}")
        (adapter_dir / "tokenizer_config.json").write_text("{}")
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        with patch("app.services.qlora_predictor.QLoraPredictor._load_model_with_adapter"):
            with patch("app.services.qlora_predictor.QLoraPredictor._load_tokenizer"):
                return QLoraPredictor(str(base_dir), str(adapter_dir))

    def test_exact_match(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        category, confidence = pred._parse_category("supply_chain_delay")
        assert category == "supply_chain_delay"
        assert confidence == 0.97

    def test_exact_match_all_categories(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        for cat in SUPPORTED_CATEGORIES:
            result_cat, result_conf = pred._parse_category(cat)
            assert result_cat == cat
            assert result_conf == 0.97

    def test_substring_match(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        category, confidence = pred._parse_category("The category is pricing_conflict")
        assert category == "pricing_conflict"
        assert confidence == 0.80

    def test_case_insensitive(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        category, _ = pred._parse_category("SUPPLY_CHAIN_DELAY")
        assert category == "supply_chain_delay"

    def test_space_normalized(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        category, confidence = pred._parse_category("supply chain delay")
        assert category == "supply_chain_delay"
        assert confidence == 0.90

    def test_prefix_stripping_category_colon(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        category, _ = pred._parse_category("category: demand_spike")
        assert category == "demand_spike"

    def test_prefix_stripping_issue_category(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        category, _ = pred._parse_category("issue category: competitor_pressure")
        assert category == "competitor_pressure"

    def test_prefix_stripping_dash(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        category, _ = pred._parse_category("- demand_spike")
        assert category == "demand_spike"

    def test_keyword_fuzzy_match_supply(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        category, confidence = pred._parse_category("stock shortage delivery delay")
        assert category == "supply_chain_delay"
        assert confidence == 0.65

    def test_keyword_fuzzy_match_pricing(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        category, _ = pred._parse_category("price dispute margin concern")
        assert category == "pricing_conflict"

    def test_keyword_fuzzy_match_competitor(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        category, _ = pred._parse_category("competing company launched campaign")
        assert category == "competitor_pressure"

    def test_unknown_output_fallback(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        category, confidence = pred._parse_category("xyz gibberish unknown")
        assert category == SUPPORTED_CATEGORIES[0]
        assert confidence == 0.30

    def test_near_empty_output_fallback(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        category, confidence = pred._parse_category("x")
        assert category == SUPPORTED_CATEGORIES[0]
        assert confidence == 0.30


# =============================================================================
# Health and Status Tests
# =============================================================================


class TestHealthAndStatus:
    """Tests for health_check and get_status methods."""

    def test_is_loaded_false_when_not_loaded(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "test", "peft_type": "LORA",
            "r": 16, "lora_alpha": 32, "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
        }))
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00")
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        pred = QLoraPredictor(str(base_dir), str(adapter_dir))
        assert pred.is_loaded is False

    def test_health_check_false_when_not_loaded(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "test", "peft_type": "LORA",
            "r": 16, "lora_alpha": 32, "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
        }))
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00")
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        pred = QLoraPredictor(str(base_dir), str(adapter_dir))
        assert pred.health_check() is False

    def test_get_status_structure(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "test", "peft_type": "LORA",
            "r": 16, "lora_alpha": 32, "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
        }))
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00")
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        pred = QLoraPredictor(str(base_dir), str(adapter_dir))
        status = pred.get_status()
        assert "loaded" in status
        assert "device" in status
        assert "is_cuda" in status
        assert "gpu_memory" in status
        assert "base_model" in status
        assert "adapter" in status
        assert "supported_categories" in status


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for QLoraPredictor configuration defaults."""

    def _make_pred(self, tmp_path: Path, **kwargs) -> QLoraPredictor:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "test", "peft_type": "LORA",
            "r": 16, "lora_alpha": 32, "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
        }))
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00")
        (adapter_dir / "tokenizer.json").write_text("{}")
        (adapter_dir / "tokenizer_config.json").write_text("{}")
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        return QLoraPredictor(str(base_dir), str(adapter_dir), **kwargs)

    def test_default_max_new_tokens(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        assert pred.max_new_tokens == 30

    def test_custom_max_new_tokens(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path, max_new_tokens=64)
        assert pred.max_new_tokens == 64

    def test_default_temperature(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        assert pred.temperature == 0.1

    def test_custom_temperature(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path, temperature=0.5)
        assert pred.temperature == 0.5


# =============================================================================
# Message Building Tests
# =============================================================================


class TestMessageBuilding:
    """Tests for _build_messages method."""

    def _make_pred(self, tmp_path: Path) -> QLoraPredictor:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "test", "peft_type": "LORA",
            "r": 16, "lora_alpha": 32, "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
        }))
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00")
        (adapter_dir / "tokenizer.json").write_text("{}")
        (adapter_dir / "tokenizer_config.json").write_text("{}")
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        return QLoraPredictor(str(base_dir), str(adapter_dir))

    def test_message_structure(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        messages = pred._build_messages("Stock running out")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Stock running out" in messages[1]["content"]

    def test_system_prompt_contains_categories(self, tmp_path: Path) -> None:
        pred = self._make_pred(tmp_path)
        messages = pred._build_messages("test")
        system_content = messages[0]["content"]

        for cat in SUPPORTED_CATEGORIES:
            assert cat in system_content
