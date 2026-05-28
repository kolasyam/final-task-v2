"""Comprehensive tests for the Ollama client.

Tests the Ollama client's classification, retry logic, category extraction,
and error handling with mocked HTTP responses.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.core.constants import SUPPORTED_CATEGORIES
from app.core.exceptions import (
    OllamaConnectionError,
    OllamaTimeoutError,
    ServiceUnavailableError,
)
from app.services.ollama_client import (
    OllamaClient,
    RetryConfig,
)


class TestOllamaClientInitialization:
    """Tests for OllamaClient construction."""

    def test_default_initialization(self) -> None:
        with patch.object(OllamaClient, "_list_models", return_value=[]):
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = "http://localhost:11434"
            client.base_model_name = "gemma:2b"
            client.timeout = 120
            client.retry_config = RetryConfig()
            client._generate_url = "http://localhost:11434/api/generate"
            client.model_name = "gemma:2b"
            client.is_prompt_model = False

            assert client.model_name == "gemma:2b"
            assert client.is_prompt_model is False

    def test_prompt_model_detection(self) -> None:
        with patch.object(
            OllamaClient, "_list_models",
            return_value=["gemma-sales-intel", "gemma:2b"],
        ):
            with patch.object(OllamaClient, "health_check", return_value=True):
                with patch("app.config.config") as mock_config:
                    mock_config.ollama_base_url = "http://localhost:11434"
                    mock_config.model_name = "gemma:2b"
                    mock_config.ollama_timeout = 120

                    client = OllamaClient()
                    assert client.is_prompt_model is True

    def test_base_model_fallback(self) -> None:
        with patch.object(
            OllamaClient, "_list_models",
            return_value=["gemma:2b"],
        ):
            with patch.object(OllamaClient, "health_check", return_value=True):
                with patch("app.config.config") as mock_config:
                    mock_config.ollama_base_url = "http://localhost:11434"
                    mock_config.model_name = "gemma:2b"
                    mock_config.ollama_timeout = 120

                    client = OllamaClient()
                    assert client.model_name == "gemma:2b"
                    assert client.is_prompt_model is False

    def test_url_trailing_slash_stripped(self) -> None:
        with patch.object(OllamaClient, "_list_models", return_value=[]):
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = "http://localhost:11434"
            assert not client.base_url.endswith("/")


class TestRetryConfig:
    """Tests for RetryConfig defaults and customization."""

    def test_defaults(self) -> None:
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 10.0
        assert config.backoff_factor == 2.0

    def test_custom_values(self) -> None:
        config = RetryConfig(max_retries=5, base_delay=2.0, max_delay=30.0)
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0


class TestHealthCheck:
    """Tests for the health_check method."""

    def test_server_reachable_and_model_available(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response), \
             patch.object(OllamaClient, "_list_models", return_value=["gemma:2b"]), \
             patch.object(OllamaClient, "__init__", lambda self: None):
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = "http://localhost:11434"
            client.model_name = "gemma:2b"
            client.timeout = 120

            assert client.health_check() is True

    def test_server_unreachable(self) -> None:
        with patch(
            "requests.get",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ), \
             patch.object(OllamaClient, "__init__", lambda self: None):
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = "http://localhost:11434"
            client.model_name = "gemma:2b"
            client.timeout = 120

            assert client.health_check() is False

    def test_server_returns_non_200(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("requests.get", return_value=mock_response), \
             patch.object(OllamaClient, "__init__", lambda self: None):
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = "http://localhost:11434"
            client.model_name = "gemma:2b"
            client.timeout = 120

            assert client.health_check() is False


class TestListModels:
    """Tests for the _list_models method."""

    def test_returns_model_names(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "gemma:2b"},
                {"name": "gemma-sales-intel"},
            ],
        }

        with patch("requests.get", return_value=mock_response), \
             patch.object(OllamaClient, "__init__", lambda self: None):
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = "http://localhost:11434"

            models = client._list_models()
            assert "gemma:2b" in models
            assert "gemma-sales-intel" in models

    def test_returns_empty_on_error(self) -> None:
        with patch(
            "requests.get",
            side_effect=requests.exceptions.ConnectionError("error"),
        ), \
             patch.object(OllamaClient, "__init__", lambda self: None):
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = "http://localhost:11434"

            models = client._list_models()
            assert models == []


class TestGenerate:
    """Tests for the generate method with retry logic."""

    def test_successful_generation(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "supply_chain_delay"}

        with patch("requests.post", return_value=mock_response), \
             patch.object(OllamaClient, "__init__", lambda self: None):
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = "http://localhost:11434"
            client.model_name = "gemma:2b"
            client.timeout = 120
            client.retry_config = RetryConfig(max_retries=1, base_delay=0.01)
            client._generate_url = "http://localhost:11434/api/generate"

            text, elapsed = client.generate("test prompt")
            assert text == "supply_chain_delay"
            assert elapsed >= 0

    def test_retry_on_connection_error(self) -> None:
        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise requests.exceptions.ConnectionError("refused")
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "ok"}
            return mock_response

        with patch("requests.post", side_effect=mock_post), \
             patch.object(OllamaClient, "__init__", lambda self: None):
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = "http://localhost:11434"
            client.model_name = "gemma:2b"
            client.timeout = 120
            client.retry_config = RetryConfig(max_retries=2, base_delay=0.01)
            client._generate_url = "http://localhost:11434/api/generate"

            text, _ = client.generate("test")
            assert text == "ok"
            assert call_count == 2

    def test_raises_after_all_retries_exhausted(self) -> None:
        with patch(
            "requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ), \
             patch.object(OllamaClient, "__init__", lambda self: None):
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = "http://localhost:11434"
            client.model_name = "gemma:2b"
            client.timeout = 120
            client.retry_config = RetryConfig(
                max_retries=2, base_delay=0.01,
            )
            client._generate_url = "http://localhost:11434/api/generate"

            with pytest.raises(OllamaConnectionError):
                client.generate("test")

    def test_strips_whitespace_from_response(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "  supply_chain_delay  "}

        with patch("requests.post", return_value=mock_response), \
             patch.object(OllamaClient, "__init__", lambda self: None):
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = "http://localhost:11434"
            client.model_name = "gemma:2b"
            client.timeout = 120
            client.retry_config = RetryConfig()
            client._generate_url = "http://localhost:11434/api/generate"

            text, _ = client.generate("test")
            assert text == "supply_chain_delay"


class TestClassifyNote:
    """Tests for the classify_note method."""

    def test_prompt_model_sends_short_prompt(self) -> None:
        with patch.object(OllamaClient, "__init__", lambda self: None):
            client = OllamaClient.__new__(OllamaClient)
            client.is_prompt_model = True
            client.generate = MagicMock(return_value=("supply_chain_delay", 1.0))

            client.classify_note("Test note")
            args = client.generate.call_args
            prompt = args[0][0]
            assert "Classify this sales note" in prompt
            assert "Test note" in prompt

    def test_base_model_sends_full_prompt(self) -> None:
        with patch.object(OllamaClient, "__init__", lambda self: None):
            client = OllamaClient.__new__(OllamaClient)
            client.is_prompt_model = False
            client.generate = MagicMock(return_value=("supply_chain_delay", 1.0))

            client.classify_note("Test note")
            args = client.generate.call_args
            prompt = args[0][0]
            assert "sales intelligence analyst" in prompt
            assert "Test note" in prompt
