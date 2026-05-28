"""FastAPI application entry point for the Sales Intelligence API.

Initializes the FastAPI application with CORS middleware, API routes,
middleware layers, and startup/shutdown event handlers.

The predictor is lazily loaded on first prediction request.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import config
from app.core.constants import (
    APP_DESCRIPTION,
    APP_NAME,
    APP_VERSION,
)
from app.core.logging_config import setup_logging
from app.middleware.auth import ApiKeyMiddleware
from app.middleware.correlation import CorrelationIdMiddleware
from app.middleware.rate_limit import RateLimitMiddleware

load_dotenv()

# Configure logging before anything else
setup_logging(log_level=os.getenv("LOG_LEVEL", config.log_level))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events.

    Logs key configuration values on startup for operational visibility.
    """
    logger.info("Sales Intelligence API starting up")
    logger.info("API_HOST: %s", config.api_host)
    logger.info("API_PORT: %s", config.api_port)
    logger.info("OLLAMA_BASE_URL: %s", config.ollama_base_url)
    logger.info("MODEL_NAME: %s", config.model_name)
    logger.info("BACKEND: Ollama (gemma:2b + prompt-engineered) + scikit-learn fallback")
    yield
    logger.info("Sales Intelligence API shutting down")


app: FastAPI = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

# --- CORS (env-configurable) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rate limiting ---
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=config.rate_limit_rpm,
)

# --- Correlation IDs ---
app.add_middleware(CorrelationIdMiddleware)

# --- API Key Auth ---
# app.add_middleware(ApiKeyMiddleware)

# --- API Routes ---
app.include_router(
    router,
    prefix="/api/v1",
    tags=["prediction"],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
    )
