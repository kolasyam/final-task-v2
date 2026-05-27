# ADR-002: Circuit Breaker for Ollama Calls

## Status: Accepted

## Context

The prediction service depends on an Ollama server which may become unavailable. Without protection, failed requests would pile up, causing cascading failures.

## Decision

Implement a circuit breaker pattern with 3 states:
- **CLOSED**: Normal operation (default)
- **OPEN**: After 5 consecutive failures, block requests for 30 seconds
- **HALF_OPEN**: After cooldown, allow one test request

## Rationale

- Prevents cascading failures when Ollama is down
- Provides graceful degradation to sklearn fallback
- Fast failure (no waiting for timeouts)
- Configurable thresholds

## Implementation

File: `app/services/circuit_breaker.py`
- Wraps Ollama client calls
- Configurable failure threshold and recovery timeout
- Exposes status for monitoring

## Consequences

- **Positive**: System resilience significantly improved
- **Positive**: Clear failure mode visible in logs
- **Negative**: Adds complexity to the codebase
- **Negative**: In-memory state (not shared across workers)
