"""Request correlation ID middleware.

Adds a unique correlation ID to every request for distributed tracing.
"""

import logging
import uuid
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

CORRELATION_ID_HEADER: str = "X-Correlation-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Adds correlation ID to requests and logs.

    If the request includes an X-Correlation-ID header, it is reused.
    Otherwise, a new UUID is generated.
    """

    async def dispatch(self, request: Request, call_next) -> None:
        """Process request with correlation ID.

        Args:
            request: Incoming request.
            call_next: Next handler.

        Returns:
            Response with correlation ID header.
        """
        correlation_id: str = request.headers.get(
            CORRELATION_ID_HEADER,
            str(uuid.uuid4()),
        )

        # Store in request state for access in route handlers
        request.state.correlation_id = correlation_id

        logger.debug("[%s] %s %s", correlation_id, request.method, request.url.path)

        response = await call_next(request)
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response


def get_correlation_id(request: Request) -> Optional[str]:
    """Get the correlation ID from a request.

    Args:
        request: FastAPI request.

    Returns:
        Correlation ID string or None.
    """
    return getattr(request.state, "correlation_id", None)
