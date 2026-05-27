"""Rate limiting middleware for FastAPI.

Uses a simple in-memory token bucket algorithm.
For production, use Redis-based rate limiting.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter.

    Limits requests per client IP within a time window.

    Attributes:
        requests_per_minute: Maximum requests per window.
        window_seconds: Time window in seconds.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self._clients: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"tokens": float(requests_per_minute), "last_refill": time.time()},
        )

    async def dispatch(self, request: Request, call_next) -> None:
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
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. Please slow down.",
                    "retry_after": self.window_seconds,
                },
                headers={"Retry-After": str(self.window_seconds)},
            )

        response = await call_next(request)
        return response

    def _allow_request(self, client_ip: str) -> bool:
        """Check if a request should be allowed.

        Args:
            client_ip: Client IP address.

        Returns:
            True if the request is allowed.
        """
        now = time.time()
        client = self._clients[client_ip]

        # Refill tokens
        elapsed = now - client["last_refill"]
        refill_rate = self.requests_per_minute / self.window_seconds
        client["tokens"] = min(
            self.requests_per_minute,
            client["tokens"] + elapsed * refill_rate,
        )
        client["last_refill"] = now

        if client["tokens"] >= 1:
            client["tokens"] -= 1
            return True
        return False
