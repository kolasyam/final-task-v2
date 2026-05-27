"""Rate limiting middleware for FastAPI.

Uses a simple in-memory token bucket algorithm.
For production deployments, replace with Redis-based rate limiting.
"""

import logging
import time
from collections import defaultdict
from typing import Dict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import config
from app.core.constants import DEFAULT_RATE_LIMIT_WINDOW_SECONDS
from app.core.exceptions import RateLimitExceededError

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter.

    Limits requests per client IP within a configurable time window.

    Attributes:
        requests_per_minute: Maximum requests per window.
        window_seconds: Time window in seconds.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = config.rate_limit_rpm,
        window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        super().__init__(app)
        self.requests_per_minute: int = requests_per_minute
        self.window_seconds: int = window_seconds
        self._clients: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"tokens": float(requests_per_minute), "last_refill": time.time()},
        )

    async def dispatch(self, request: Request, call_next) -> JSONResponse:
        """Process request through rate limiter.

        Args:
            request: Incoming request.
            call_next: Next handler.

        Returns:
            Response or 429 Too Many Requests.
        """
        client_ip: str = request.client.host if request.client else "unknown"

        if not self._allow_request(client_ip):
            logger.warning("Rate limit exceeded for %s", client_ip)
            error = RateLimitExceededError(retry_after=self.window_seconds)
            return JSONResponse(
                status_code=error.HTTP_STATUS,
                content=error.to_dict(),
                headers={"Retry-After": str(self.window_seconds)},
            )

        return await call_next(request)

    def _allow_request(self, client_ip: str) -> bool:
        """Check if a request should be allowed under the current rate limit.

        Args:
            client_ip: Client IP address.

        Returns:
            True if the request is allowed.
        """
        now: float = time.time()
        client: Dict[str, float] = self._clients[client_ip]

        self._refill_tokens(client, now)

        if client["tokens"] >= 1:
            client["tokens"] -= 1
            return True
        return False

    def _refill_tokens(self, client: Dict[str, float], now: float) -> None:
        """Refill tokens based on elapsed time.

        Args:
            client: Client state dictionary.
            now: Current timestamp.
        """
        elapsed: float = now - client["last_refill"]
        refill_rate: float = self.requests_per_minute / self.window_seconds
        client["tokens"] = min(
            self.requests_per_minute,
            client["tokens"] + elapsed * refill_rate,
        )
        client["last_refill"] = now
