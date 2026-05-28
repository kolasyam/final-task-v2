"""Comprehensive tests for the circuit breaker pattern.

Tests all circuit breaker states, transitions, and edge cases.
"""

import time
from typing import NoReturn

import pytest

from app.core.exceptions import CircuitBreakerOpenError
from app.services.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    circuit_breaker,
)


class TestCircuitBreakerInitialization:
    """Tests for circuit breaker initialization."""

    def test_starts_closed(self, circuit_breaker: CircuitBreaker) -> None:
        assert circuit_breaker.state == CircuitState.CLOSED

    def test_default_failure_count_zero(self, circuit_breaker: CircuitBreaker) -> None:
        assert circuit_breaker._failure_count == 0

    def test_default_success_count_zero(self, circuit_breaker: CircuitBreaker) -> None:
        assert circuit_breaker._success_count == 0

    def test_custom_threshold(self) -> None:
        breaker = CircuitBreaker(failure_threshold=10)
        assert breaker.failure_threshold == 10

    def test_custom_recovery_timeout(self) -> None:
        breaker = CircuitBreaker(recovery_timeout=60.0)
        assert breaker.recovery_timeout == 60.0


class TestCircuitBreakerClosedState:
    """Tests while the circuit is in CLOSED state."""

    def test_successful_call(self, circuit_breaker: CircuitBreaker) -> None:
        result = circuit_breaker.call(lambda: "success")
        assert result == "success"

    def test_success_resets_failure_count(self, circuit_breaker: CircuitBreaker) -> None:
        # Cause one failure
        with pytest.raises(RuntimeError):
            circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        # Success should reset
        circuit_breaker.call(lambda: "ok")
        assert circuit_breaker._failure_count == 0

    def test_success_increments_success_count(self, circuit_breaker: CircuitBreaker) -> None:
        circuit_breaker.call(lambda: "ok")
        assert circuit_breaker._success_count == 1

    def test_failure_below_threshold_stays_closed(self, circuit_breaker: CircuitBreaker) -> None:
        with pytest.raises(RuntimeError):
            circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert circuit_breaker.state == CircuitState.CLOSED

    def test_multiple_failures_below_threshold(self, circuit_breaker: CircuitBreaker) -> None:
        for _ in range(2):
            with pytest.raises(RuntimeError):
                circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert circuit_breaker.state == CircuitState.CLOSED


class TestCircuitBreakerOpenState:
    """Tests for circuit breaker transitioning to OPEN state."""

    def test_opens_after_threshold(self, circuit_breaker: CircuitBreaker) -> None:
        for _ in range(3):
            with pytest.raises(RuntimeError):
                circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert circuit_breaker.state == CircuitState.OPEN

    def test_raises_breaker_error_when_open(self, circuit_breaker: CircuitBreaker) -> None:
        for _ in range(3):
            with pytest.raises(RuntimeError):
                circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        with pytest.raises(CircuitBreakerOpenError):
            circuit_breaker.call(lambda: "should not run")

    def test_is_open_property(self, circuit_breaker: CircuitBreaker) -> None:
        assert circuit_breaker.is_open is False
        for _ in range(3):
            with pytest.raises(RuntimeError):
                circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert circuit_breaker.is_open is True


class TestCircuitBreakerHalfOpenState:
    """Tests for HALF_OPEN state and recovery."""

    def test_transitions_to_half_open_after_timeout(
        self, fast_circuit_breaker: CircuitBreaker,
    ) -> None:
        for _ in range(2):
            with pytest.raises(RuntimeError):
                fast_circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        # With recovery_timeout=0, should transition to half_open immediately
        assert fast_circuit_breaker.state == CircuitState.HALF_OPEN

    def test_success_in_half_open_closes_circuit(
        self, fast_circuit_breaker: CircuitBreaker,
    ) -> None:
        for _ in range(2):
            with pytest.raises(RuntimeError):
                fast_circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        result = fast_circuit_breaker.call(lambda: "recovered")
        assert result == "recovered"
        assert fast_circuit_breaker.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens_circuit(
        self, fast_circuit_breaker: CircuitBreaker,
    ) -> None:
        for _ in range(2):
            with pytest.raises(RuntimeError):
                fast_circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        with pytest.raises(RuntimeError):
            fast_circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail again")))

        assert fast_circuit_breaker.state == CircuitState.OPEN


class TestCircuitBreakerReset:
    """Tests for manual reset functionality."""

    def test_reset_clears_all_state(self, circuit_breaker: CircuitBreaker) -> None:
        for _ in range(3):
            with pytest.raises(RuntimeError):
                circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        circuit_breaker.reset()
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failure_count == 0
        assert circuit_breaker._success_count == 0

    def test_reset_allows_calls_again(self, circuit_breaker: CircuitBreaker) -> None:
        for _ in range(3):
            with pytest.raises(RuntimeError):
                circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        circuit_breaker.reset()
        result = circuit_breaker.call(lambda: "works")
        assert result == "works"


class TestCircuitBreakerStatus:
    """Tests for the status reporting method."""

    def test_status_structure(self, circuit_breaker: CircuitBreaker) -> None:
        status = circuit_breaker.get_status()
        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status
        assert "last_failure_time" in status

    def test_status_initial_values(self, circuit_breaker: CircuitBreaker) -> None:
        status = circuit_breaker.get_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["success_count"] == 0

    def test_status_after_failures(self, circuit_breaker: CircuitBreaker) -> None:
        with pytest.raises(RuntimeError):
            circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        status = circuit_breaker.get_status()
        assert status["failure_count"] == 1


class TestCircuitBreakerExpectedException:
    """Tests for expected_exception filtering."""

    def test_only_catches_expected_exception(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, expected_exception=ValueError)

        with pytest.raises(TypeError):
            breaker.call(lambda: (_ for _ in ()).throw(TypeError("wrong type")))

        # TypeError should not count as a failure
        assert breaker._failure_count == 0

    def test_expected_exception_counts(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, expected_exception=ValueError)

        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("expected")))

        assert breaker._failure_count == 1


class TestCircuitBreakerDecorator:
    """Tests for the circuit_breaker decorator."""

    def test_decorator_applies_breaker(self) -> None:
        call_count: int = 0

        @circuit_breaker(failure_threshold=2, recovery_timeout=0)
        def flaky_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("fail")
            return "success"

        with pytest.raises(RuntimeError):
            flaky_function()
        with pytest.raises(RuntimeError):
            flaky_function()

        with pytest.raises(CircuitBreakerOpenError):
            flaky_function()

    def test_decorator_exposes_breaker(self) -> None:
        @circuit_breaker(failure_threshold=5)
        def my_func() -> str:
            return "ok"

        assert hasattr(my_func, "circuit_breaker")
        assert isinstance(my_func.circuit_breaker, CircuitBreaker)


class TestCircuitBreakerOpenError:
    """Tests for the CircuitBreakerOpenError exception."""

    def test_default_message(self) -> None:
        error = CircuitBreakerOpenError(retry_after=30.0)
        assert "Circuit breaker is OPEN" in error.message

    def test_retry_after_in_message(self) -> None:
        error = CircuitBreakerOpenError(retry_after=15.0)
        assert "15" in error.message

    def test_to_dict(self) -> None:
        error = CircuitBreakerOpenError(retry_after=10.0)
        result = error.to_dict()
        assert "error_code" in result
        assert "message" in result
        assert "details" in result
