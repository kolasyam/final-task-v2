"""Circuit breaker pattern for external service calls.

Prevents cascading failures when Ollama or other external services
are unavailable. States:
  - CLOSED: Normal operation, requests pass through
  - OPEN: Service is down, requests fail fast
  - HALF_OPEN: Testing if service has recovered

Usage:
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
    result = breaker.call(some_function, arg1, arg2)
"""

import logging
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

from app.core.constants import (
    DEFAULT_CIRCUIT_BREAKER_RECOVERY_SECONDS,
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
)
from app.core.exceptions import CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker state enumeration."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    Attributes:
        failure_threshold: Number of failures before opening the circuit.
        recovery_timeout: Seconds to wait before trying half-open.
    """

    def __init__(
        self,
        failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        recovery_timeout: float = DEFAULT_CIRCUIT_BREAKER_RECOVERY_SECONDS,
        expected_exception: type = Exception,
    ) -> None:
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening.
            recovery_timeout: Seconds before transitioning to half-open.
            expected_exception: Exception type that counts as a failure.
        """
        self.failure_threshold: int = failure_threshold
        self.recovery_timeout: float = recovery_timeout
        self.expected_exception: type = expected_exception

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._success_count: int = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery timeout."""
        if self._state == CircuitState.OPEN:
            elapsed: float = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if the circuit is currently open."""
        return self.state == CircuitState.OPEN

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call a function through the circuit breaker.

        Args:
            func: Function to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.

        Raises:
            CircuitBreakerOpenError: If the circuit is open.
            Exception: The original exception if the function raises.
        """
        if self.is_open:
            retry_after: float = self._calculate_retry_after()
            raise CircuitBreakerOpenError(retry_after=retry_after)

        try:
            result: Any = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    def _calculate_retry_after(self) -> float:
        """Calculate remaining time until the circuit may transition."""
        remaining: float = self.recovery_timeout - (time.time() - self._last_failure_time)
        return max(0.0, remaining)

    def _on_success(self) -> None:
        """Handle a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker CLOSED (service recovered)")
            self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count += 1

    def _on_failure(self) -> None:
        """Handle a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            logger.warning(
                "Circuit breaker OPEN after %d consecutive failures",
                self._failure_count,
            )
            self._state = CircuitState.OPEN

    def get_status(self) -> dict:
        """Get circuit breaker status as a dictionary.

        Returns:
            Status dictionary with state, counts, and timing.
        """
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
        }

    def reset(self) -> None:
        """Reset the circuit breaker to closed state.

        Useful for testing or manual recovery scenarios.
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._success_count = 0
        logger.info("Circuit breaker reset to CLOSED")


def circuit_breaker(
    failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    recovery_timeout: float = DEFAULT_CIRCUIT_BREAKER_RECOVERY_SECONDS,
    expected_exception: type = Exception,
) -> Callable:
    """Decorator for applying circuit breaker to functions.

    Args:
        failure_threshold: Failures before opening.
        recovery_timeout: Seconds before half-open.
        expected_exception: Exception type to catch.

    Returns:
        Decorated function.
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return breaker.call(func, *args, **kwargs)
        wrapper.circuit_breaker = breaker  # type: ignore[attr-defined]
        return wrapper
    return decorator
