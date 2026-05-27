"""API key authentication middleware for FastAPI.

Protects API endpoints with a simple API key passed in the X-API-Key header.

Usage:
    Set API_KEY in .env, then include X-API-Key header in requests.

    curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/predict ...
"""

import logging
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import config
from app.core.constants import API_KEY_HEADER, AUTH_EXEMPT_PATHS
from app.core.exceptions import AuthenticationError

logger = logging.getLogger(__name__)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Validates API key on protected endpoints.

    If API_KEY is not set in environment, authentication is disabled.
    Health check and docs endpoints are always exempt.
    """

    async def dispatch(self, request: Request, call_next) -> JSONResponse:
        """Check API key before processing request.

        Args:
            request: Incoming request.
            call_next: Next handler.

        Returns:
            Response or 401 Unauthorized.
        """
        if not config.api_key:
            return await call_next(request)

        if self._is_exempt(request.url.path):
            return await call_next(request)

        error: Optional[AuthenticationError] = self._validate_api_key(request)
        if error is not None:
            return JSONResponse(
                status_code=error.HTTP_STATUS,
                content=error.to_dict(),
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return await call_next(request)

    @staticmethod
    def _is_exempt(path: str) -> bool:
        """Check if the request path is exempt from authentication.

        Args:
            path: Request URL path.

        Returns:
            True if the path is exempt.
        """
        return path in AUTH_EXEMPT_PATHS or any(
            path.startswith(exempt) for exempt in AUTH_EXEMPT_PATHS
        )

    @staticmethod
    def _validate_api_key(request: Request) -> Optional[AuthenticationError]:
        """Validate the API key from the request header.

        Args:
            request: Incoming request.

        Returns:
            AuthenticationError if validation fails, None if valid.
        """
        provided_key: Optional[str] = request.headers.get(API_KEY_HEADER)
        if not provided_key or provided_key != config.api_key:
            client_ip: str = request.client.host if request.client else "unknown"
            logger.warning(
                "Unauthorized request from %s to %s",
                client_ip,
                request.url.path,
            )
            return AuthenticationError()
        return None
