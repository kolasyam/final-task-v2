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

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when the circuit breaker is open."""

    def __init__(self, message: str = "Circuit breaker is open") -> None:
        self.message = message
        super().__init__(self.message)


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    Attributes:
        failure_threshold: Number of failures before opening the circuit.
        recovery_timeout: Seconds to wait before trying half-open.
        state: Current circuit state.
        failure_count: Consecutive failure count.
        last_failure_time: Timestamp of the last failure.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: type = Exception,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._success_count: int = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery timeout."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
        return self._state

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call a function through the circuit breaker.

        Args:
            func: Function to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.

        Raises:
            CircuitBreakerError: If the circuit is open.
            Exception: If the function raises an exception.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerError(
                f"Circuit breaker is OPEN. Service unavailable. "
                f"Will retry in {self.recovery_timeout - (time.time() - self._last_failure_time):.0f}s"
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as exc:
            self._on_failure()
            raise

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
        """Get circuit breaker status.

        Returns:
            Status dictionary.
        """
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
        }


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
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
