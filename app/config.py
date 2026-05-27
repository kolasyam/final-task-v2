"""Centralized configuration management for the Sales Intelligence System.

Loads settings from environment variables with sensible defaults
defined in app.core.constants. All configuration is centralized here
to avoid magic values scattered throughout the codebase.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from app.core.constants import (
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_CORS_ORIGINS,
    DEFAULT_MAX_FEATURES,
    DEFAULT_MODEL_NAME,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_TIMEOUT,
    DEFAULT_RATE_LIMIT_RPM,
    DEFAULT_RANDOM_SEED,
    DEFAULT_STREAMLIT_PORT,
    DEFAULT_TEST_SIZE,
)

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded from environment variables.

    All defaults are sourced from app.core.constants to maintain
    a single source of truth for default values.
    """

    # --- Dataset ---
    dataset_path: str = field(default_factory=lambda: os.getenv(
        "DATASET_PATH",
        "data/final_standardized_sales_dataset.xlsx",
    ))

    # --- Ollama ---
    ollama_base_url: str = field(default_factory=lambda: os.getenv(
        "OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL,
    ))
    model_name: str = field(default_factory=lambda: os.getenv(
        "MODEL_NAME", DEFAULT_MODEL_NAME,
    ))
    ollama_timeout: int = field(default_factory=lambda: int(os.getenv(
        "OLLAMA_TIMEOUT", str(DEFAULT_OLLAMA_TIMEOUT),
    )))

    # --- API ---
    api_host: str = field(default_factory=lambda: os.getenv(
        "API_HOST", DEFAULT_API_HOST,
    ))
    api_port: int = field(default_factory=lambda: int(os.getenv(
        "API_PORT", str(DEFAULT_API_PORT),
    )))
    api_base_url: str = field(default_factory=lambda: os.getenv(
        "API_BASE_URL", f"http://localhost:{DEFAULT_API_PORT}",
    ))

    # --- Rate Limiting ---
    rate_limit_rpm: int = field(default_factory=lambda: int(os.getenv(
        "RATE_LIMIT_RPM", str(DEFAULT_RATE_LIMIT_RPM),
    )))

    # --- CORS ---
    cors_origins: str = field(default_factory=lambda: os.getenv(
        "CORS_ORIGINS", DEFAULT_CORS_ORIGINS,
    ))

    # --- Streamlit ---
    streamlit_port: int = field(default_factory=lambda: int(os.getenv(
        "STREAMLIT_PORT", str(DEFAULT_STREAMLIT_PORT),
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
        "TEST_SIZE", str(DEFAULT_TEST_SIZE),
    )))
    random_seed: int = field(default_factory=lambda: int(os.getenv(
        "RANDOM_SEED", str(DEFAULT_RANDOM_SEED),
    )))
    n_estimators: int = field(default_factory=lambda: int(os.getenv(
        "N_ESTIMATORS", str(DEFAULT_N_ESTIMATORS),
    )))
    max_features: int = field(default_factory=lambda: int(os.getenv(
        "MAX_FEATURES", str(DEFAULT_MAX_FEATURES),
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

    @property
    def cors_origin_list(self) -> list:
        """Parse CORS origins string into a list of stripped values."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

# --- Category Normalization Map ---
CATEGORY_MAP = {
    "SUPPLY_CHAIN_ISSUE": "supply_chain_delay",
    "RETAILER_RELATIONSHIP_ISSUE": "retailer_dissatisfaction",
    "PRICING_AND_MARGIN_CONFLICT": "pricing_conflict",
    "COMPETITOR_MARKET_PRESSURE": "competitor_pressure",
    "DEMAND_SURGE": "demand_spike",
}
SUPPORTED_CATEGORIES = [
    "supply_chain_delay",
    "retailer_dissatisfaction",
    "pricing_conflict",
    "competitor_pressure",
    "demand_spike",
]
# Singleton config instance imported throughout the application
config = AppConfig()
