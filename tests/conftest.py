"""Shared test fixtures for the Sales Intelligence test suite.

Provides reusable fixtures for mocking QLoRA predictor, sklearn,
and setting up predictor instances with controlled backends.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.circuit_breaker import CircuitBreaker
from app.services.predictor import SalesNotePredictor
from app.services.preprocessing import TextPreprocessor
from app.services.qlora_predictor import QLoraPredictor
from app.services.storage import PredictionStorage


# =============================================================================
# QLoRA Predictor Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_qlora_predictor() -> Generator:
    """Create a QLoRA-backed SalesNotePredictor.

    The QLoRA predictor is mocked to return controlled responses
    without requiring GPU or model files.

    Yields:
        SalesNotePredictor instance with mocked QLoRA backend.
    """
    mock_qlora = MagicMock(spec=QLoraPredictor)
    mock_qlora.classify.return_value = ("supply_chain_delay", 0.97, 1.5)
    mock_qlora.health_check.return_value = True
    mock_qlora.is_loaded = False
    mock_qlora.get_status.return_value = {
        "loaded": False,
        "device": "N/A",
        "is_cuda": False,
        "gpu_memory": "N/A (no CUDA)",
        "base_model": "/opt/ai-platform/models/gemma-2-2b-it",
        "adapter": "training/saved_model/qlora_adapter",
        "supported_categories": [
            "supply_chain_delay",
            "retailer_dissatisfaction",
            "pricing_conflict",
            "competitor_pressure",
            "demand_spike",
        ],
    }

    mock_sklearn = MagicMock()
    mock_sklearn.predict.return_value = ("supply_chain_delay", 0.92)

    with patch.object(SalesNotePredictor, "__init__", lambda self: None):
        predictor = SalesNotePredictor.__new__(SalesNotePredictor)
        predictor.qlora_predictor = mock_qlora
        predictor.qlora_available = True
        predictor.sklearn_classifier = mock_sklearn
        predictor.preprocessor = TextPreprocessor()
        predictor.storage = MagicMock()
        predictor.storage.save_prediction = MagicMock()
        yield predictor


@pytest.fixture
def mock_sklearn_only_predictor() -> Generator:
    """Create a SalesNotePredictor with only sklearn backend (no QLoRA).

    Yields:
        SalesNotePredictor instance with sklearn-only fallback.
    """
    mock_sklearn = MagicMock()
    mock_sklearn.predict.return_value = ("pricing_conflict", 0.87)

    with patch.object(SalesNotePredictor, "__init__", lambda self: None):
        predictor = SalesNotePredictor.__new__(SalesNotePredictor)
        predictor.qlora_predictor = MagicMock()
        predictor.qlora_available = False
        predictor.sklearn_classifier = mock_sklearn
        predictor.preprocessor = TextPreprocessor()
        predictor.storage = MagicMock()
        predictor.storage.save_prediction = MagicMock()
        yield predictor


@pytest.fixture
def mock_no_backend_predictor() -> Generator:
    """Create a SalesNotePredictor with no inference backend.

    Yields:
        SalesNotePredictor with neither QLoRA nor sklearn.
    """
    with patch.object(SalesNotePredictor, "__init__", lambda self: None):
        predictor = SalesNotePredictor.__new__(SalesNotePredictor)
        predictor.qlora_predictor = MagicMock()
        predictor.qlora_available = False
        predictor.sklearn_classifier = None
        predictor.preprocessor = TextPreprocessor()
        predictor.storage = MagicMock()
        predictor.storage.save_prediction = MagicMock()
        yield predictor


@pytest.fixture
def mock_qlora_failure_predictor() -> Generator:
    """Create a predictor where QLoRA fails and falls back to sklearn.

    Yields:
        SalesNotePredictor where QLoRA raises, forcing sklearn fallback.
    """
    from app.core.exceptions import QLoRAInferenceError

    mock_qlora = MagicMock(spec=QLoraPredictor)
    mock_qlora.classify.side_effect = QLoRAInferenceError("GPU not available")

    mock_sklearn = MagicMock()
    mock_sklearn.predict.return_value = ("demand_spike", 0.80)

    with patch.object(SalesNotePredictor, "__init__", lambda self: None):
        predictor = SalesNotePredictor.__new__(SalesNotePredictor)
        predictor.qlora_predictor = mock_qlora
        predictor.qlora_available = True
        predictor.sklearn_classifier = mock_sklearn
        predictor.preprocessor = TextPreprocessor()
        predictor.storage = MagicMock()
        predictor.storage.save_prediction = MagicMock()
        yield predictor


# =============================================================================
# API Client Fixture
# =============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client.

    Returns:
        TestClient instance for the app.
    """
    return TestClient(app)


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """Build authentication headers based on environment.

    Returns:
        Dictionary with API key header if configured, empty dict otherwise.
    """
    api_key: str = os.getenv("API_KEY", "")
    if api_key:
        return {"X-API-Key": api_key}
    return {}


# =============================================================================
# Circuit Breaker Fixtures
# =============================================================================


@pytest.fixture
def circuit_breaker() -> CircuitBreaker:
    """Create a CircuitBreaker with default settings.

    Returns:
        CircuitBreaker instance.
    """
    return CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)


@pytest.fixture
def fast_circuit_breaker() -> CircuitBreaker:
    """Create a CircuitBreaker with short recovery for testing.

    Returns:
        CircuitBreaker instance with 0s recovery timeout.
    """
    return CircuitBreaker(failure_threshold=2, recovery_timeout=0.0)


# =============================================================================
# Dataset Fixtures
# =============================================================================


@pytest.fixture
def sample_dataset(tmp_path: Path) -> str:
    """Create a minimal sample dataset for testing.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the created Excel file.
    """
    categories: List[tuple] = [
        ("SUPPLY_CHAIN_ISSUE", "supply_chain_delay"),
        ("RETAILER_RELATIONSHIP_ISSUE", "retailer_dissatisfaction"),
        ("PRICING_AND_MARGIN_CONFLICT", "pricing_conflict"),
        ("COMPETITOR_MARKET_PRESSURE", "competitor_pressure"),
        ("DEMAND_SURGE", "demand_spike"),
    ]

    records: List[Dict[str, str]] = []
    for i in range(50):
        cat_raw, cat_norm = categories[i % 5]
        records.append({
            "rep_note": (
                f"Sample sales note number {i} for {cat_norm}."
                f" Additional unique text: {'x' * (i * 3)}"
            ),
            "issue_category": cat_raw,
        })

    df = pd.DataFrame(records)
    path: str = str(tmp_path / "test_dataset.xlsx")
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def empty_dataset(tmp_path: Path) -> str:
    """Create an empty dataset with correct headers but no data rows.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the created Excel file.
    """
    df = pd.DataFrame(columns=["rep_note", "issue_category"])
    path: str = str(tmp_path / "empty_dataset.xlsx")
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def malformed_dataset(tmp_path: Path) -> str:
    """Create a dataset with missing required columns.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the created Excel file.
    """
    df = pd.DataFrame({"wrong_column": [1, 2, 3], "another_wrong": ["a", "b", "c"]})
    path: str = str(tmp_path / "malformed_dataset.xlsx")
    df.to_excel(path, index=False)
    return path


# =============================================================================
# Storage Fixtures
# =============================================================================


@pytest.fixture
def temp_storage(tmp_path: Path) -> PredictionStorage:
    """Create a PredictionStorage instance with temporary file paths.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        PredictionStorage instance using temp directory.
    """
    csv_path: str = str(tmp_path / "test_predictions.csv")
    jsonl_path: str = str(tmp_path / "test_predictions.jsonl")
    return PredictionStorage(csv_path=csv_path, jsonl_path=jsonl_path)


# =============================================================================
# Preprocessing Fixtures
# =============================================================================


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    """Create a TextPreprocessor instance.

    Returns:
        TextPreprocessor instance.
    """
    return TextPreprocessor()


# =============================================================================
# Trained Model Fixtures
# =============================================================================


@pytest.fixture
def trained_model_dir(sample_dataset: str, tmp_path: Path) -> str:
    """Train a model on the sample dataset and return the model dir.

    Args:
        sample_dataset: Path to sample dataset.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the directory containing trained model artifacts.
    """
    from training.prepare_dataset import prepare_dataset
    from training.train import train

    train_df, test_df = prepare_dataset(
        dataset_path=sample_dataset,
        test_size=0.2,
        random_seed=42,
    )

    model_dir: str = str(tmp_path / "saved_model")
    train(train_df, test_df, output_dir=model_dir)
    return model_dir
