"""Centralized configuration management for the Sales Intelligence System.

Loads settings from environment variables with sensible defaults.
All configuration is centralized here to avoid magic values scattered
throughout the codebase (SonarQube maintainability compliance).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category mapping: raw dataset values -> normalized output labels
# ---------------------------------------------------------------------------
CATEGORY_MAP: Dict[str, str] = {
    "SUPPLY_CHAIN_ISSUE": "supply_chain_delay",
    "RETAILER_RELATIONSHIP_ISSUE": "retailer_dissatisfaction",
    "PRICING_AND_MARGIN_CONFLICT": "pricing_conflict",
    "COMPETITOR_MARKET_PRESSURE": "competitor_pressure",
    "DEMAND_SURGE": "demand_spike",
}

SUPPORTED_CATEGORIES: List[str] = list(CATEGORY_MAP.values())

# Reverse mapping: normalized label -> display name
CATEGORY_DISPLAY_NAMES: Dict[str, str] = {
    "supply_chain_delay": "Supply Chain Delay",
    "retailer_dissatisfaction": "Retailer Dissatisfaction",
    "pricing_conflict": "Pricing Conflict",
    "competitor_pressure": "Competitor Pressure",
    "demand_spike": "Demand Spike",
}


@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded from environment variables.

    Attributes:
        dataset_path: Absolute path to the source Excel dataset.
        ollama_base_url: Ollama server URL.
        model_name: Ollama model name (e.g., 'gemma:2b').
        ollama_timeout: Timeout in seconds for Ollama API calls.
        api_host: FastAPI bind address.
        api_port: FastAPI listen port.
        api_base_url: Full base URL for API.
        streamlit_port: Streamlit dashboard port.
        mlflow_tracking_uri: MLflow tracking backend URI.
        mlflow_experiment_name: MLflow experiment name.
        model_dir: Directory for saved model artifacts.
        vectorizer_path: Path to saved TF-IDF vectorizer.
        classifier_path: Path to saved scikit-learn classifier.
        label_encoder_path: Path to saved label encoder.
        test_size: Fraction of data reserved for testing.
        random_seed: Random seed for reproducibility.
        n_estimators: Number of trees in the ensemble classifier.
        max_features: Maximum number of TF-IDF features.
        log_level: Python logging level string.
    """

    # --- Dataset ---
    dataset_path: str = field(default_factory=lambda: os.getenv(
        "DATASET_PATH",
        "data/final_standardized_sales_dataset.xlsx",
    ))

    # --- Ollama ---
    ollama_base_url: str = field(default_factory=lambda: os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434",
    ))
    model_name: str = field(default_factory=lambda: os.getenv(
        "MODEL_NAME", "gemma:2b",
    ))
    ollama_timeout: int = field(default_factory=lambda: int(os.getenv(
        "OLLAMA_TIMEOUT", "120",
    )))

    # --- API ---
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_base_url: str = field(default_factory=lambda: os.getenv(
        "API_BASE_URL", "http://localhost:8000",
    ))

    # --- Streamlit ---
    streamlit_port: int = field(default_factory=lambda: int(os.getenv(
        "STREAMLIT_PORT", "8501",
    )))

    # --- MLflow ---
    mlflow_tracking_uri: str = field(default_factory=lambda: os.getenv(
        "MLFLOW_TRACKING_URI", "file:./mlruns",
    ))
    mlflow_experiment_name: str = field(default_factory=lambda: os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "sales-intelligence",
    ))

    # --- Model Paths ---
    model_dir: str = field(default_factory=lambda: os.getenv(
        "MODEL_DIR", "training/saved_model",
    ))
    vectorizer_path: str = field(default_factory=lambda: os.getenv(
        "VECTORIZER_PATH", "training/saved_model/vectorizer.joblib",
    ))
    classifier_path: str = field(default_factory=lambda: os.getenv(
        "CLASSIFIER_PATH", "training/saved_model/classifier.joblib",
    ))
    label_encoder_path: str = field(default_factory=lambda: os.getenv(
        "LABEL_ENCODER_PATH", "training/saved_model/label_encoder.joblib",
    ))

    # --- Training ---
    test_size: float = field(default_factory=lambda: float(os.getenv(
        "TEST_SIZE", "0.2",
    )))
    random_seed: int = field(default_factory=lambda: int(os.getenv(
        "RANDOM_SEED", "42",
    )))
    n_estimators: int = field(default_factory=lambda: int(os.getenv(
        "N_ESTIMATORS", "200",
    )))
    max_features: int = field(default_factory=lambda: int(os.getenv(
        "MAX_FEATURES", "5000",
    )))

    # --- Logging ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # --- Security ---
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))

    # --- QLoRA Finetuning ---
    qlora_adapter_path: str = field(default_factory=lambda: os.getenv(
        "QLORA_ADAPTER_PATH", "training/saved_model/qlora_adapter",
    ))
    qlora_epochs: int = field(default_factory=lambda: int(os.getenv(
        "QLORA_EPOCHS", "3",
    )))
    qlora_learning_rate: float = field(default_factory=lambda: float(os.getenv(
        "QLORA_LEARNING_RATE", "2e-4",
    )))
    qlora_batch_size: int = field(default_factory=lambda: int(os.getenv(
        "QLORA_BATCH_SIZE", "4",
    )))


# Singleton config instance imported throughout the application
config = AppConfig()
