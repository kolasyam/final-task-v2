"""FastAPI application entry point for the Sales Intelligence API.

Initializes the FastAPI application with CORS middleware, API routes,
and startup/shutdown event handlers. The predictor is lazily loaded
on first prediction request.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.middleware.auth import ApiKeyMiddleware
from app.middleware.correlation import CorrelationIdMiddleware
from app.middleware.rate_limit import RateLimitMiddleware

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    logger.info("Sales Intelligence API starting up")
    logger.info("API_HOST: %s", os.getenv("API_HOST", "0.0.0.0"))
    logger.info("API_PORT: %s", os.getenv("API_PORT", "8000"))
    logger.info("OLLAMA_BASE_URL: %s", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    logger.info("MODEL_NAME: %s", os.getenv("MODEL_NAME", "gemma:2b"))
    logger.info("BACKEND: Ollama (gemma:2b + prompt-engineered) + scikit-learn fallback")
    yield
    logger.info("Sales Intelligence API shutting down")


app: FastAPI = FastAPI(
    title="Sales Intelligence API",
    description=(
        "AI-Powered Sales Representative Intelligence Extractor. "
        "Classifies sales notes into issue categories using "
        "Ollama gemma:2b (primary) or scikit-learn TF-IDF + "
        "RandomForest (fallback)."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# --- CORS (env-configurable) ---
cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000")
origin_list: list = [o.strip() for o in cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rate limiting ---
rpm: int = int(os.getenv("RATE_LIMIT_RPM", "60"))
app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm)

# --- Correlation IDs ---
app.add_middleware(CorrelationIdMiddleware)

# --- API Key Auth ---
app.add_middleware(ApiKeyMiddleware)

app.include_router(router, prefix="/api/v1", tags=["prediction"])


if __name__ == "__main__":
    import uvicorn

    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
