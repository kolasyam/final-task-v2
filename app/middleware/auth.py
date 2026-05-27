"""API key authentication middleware for FastAPI.

Protects API endpoints with a simple API key passed in the X-API-Key header.

Usage:
    Set API_KEY in .env, then include X-API-Key header in requests.

    curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/predict ...
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY_HEADER: str = "X-API-Key"
API_KEY: str = os.getenv("API_KEY", "")

# Paths that don't require authentication
EXEMPT_PATHS: list = [
    "/api/v1/health",
    "/api/v1/categories",
    "/docs",
    "/openapi.json",
    "/redoc",
]


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Validates API key on protected endpoints.

    If API_KEY is not set in environment, authentication is disabled.
    Health check and docs endpoints are always exempt.
    """

    async def dispatch(self, request: Request, call_next) -> None:
        """Check API key before processing request.

        Args:
            request: Incoming request.
            call_next: Next handler.

        Returns:
            Response or 401 Unauthorized.
        """
        # If no API key configured, skip auth
        if not API_KEY:
            return await call_next(request)

        # Exempt health check and docs
        path: str = request.url.path
        if path in EXEMPT_PATHS or any(path.startswith(p) for p in EXEMPT_PATHS):
            return await call_next(request)

        # Validate API key
        provided_key: Optional[str] = request.headers.get(API_KEY_HEADER)
        if not provided_key or provided_key != API_KEY:
            client_ip: str = request.client.host if request.client else "unknown"
            logger.warning(
                "Unauthorized request from %s to %s", client_ip, path,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Unauthorized. Provide valid X-API-Key header.",
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return await call_next(request)
