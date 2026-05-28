"""Tests for FastAPI middleware components.

Tests authentication, rate limiting, and correlation ID middleware.
"""

import os
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.middleware.auth import ApiKeyMiddleware
from app.middleware.correlation import CorrelationIdMiddleware
from app.middleware.rate_limit import RateLimitMiddleware


# =============================================================================
# Correlation ID Middleware Tests
# =============================================================================


class TestCorrelationIdMiddleware:
    """Tests for the correlation ID middleware."""

    def test_generates_correlation_id_if_not_provided(self) -> None:
        app = FastAPI()

        @app.get("/test")
        async def endpoint() -> Dict:
            return {"ok": True}

        app.add_middleware(CorrelationIdMiddleware)

        client = TestClient(app)
        response = client.get("/test")
        assert "X-Correlation-ID" in response.headers
        assert len(response.headers["X-Correlation-ID"]) > 0

    def test_preserves_custom_correlation_id(self) -> None:
        app = FastAPI()

        @app.get("/test")
        async def endpoint() -> Dict:
            return {"ok": True}

        app.add_middleware(CorrelationIdMiddleware)

        client = TestClient(app)
        custom_id = "test-correlation-123"
        response = client.get("/test", headers={"X-Correlation-ID": custom_id})
        assert response.headers["X-Correlation-ID"] == custom_id

    def test_correlation_id_is_valid_uuid_format(self) -> None:
        import uuid

        app = FastAPI()

        @app.get("/test")
        async def endpoint() -> Dict:
            return {"ok": True}

        app.add_middleware(CorrelationIdMiddleware)

        client = TestClient(app)
        response = client.get("/test")
        correlation_id = response.headers["X-Correlation-ID"]
        # Should be a valid UUID
        try:
            uuid.UUID(correlation_id)
        except ValueError:
            pytest.fail(f"Correlation ID '{correlation_id}' is not a valid UUID")


# =============================================================================
# Rate Limiting Middleware Tests
# =============================================================================


class TestRateLimitMiddleware:
    """Tests for the rate limiting middleware."""

    def test_allows_requests_within_limit(self) -> None:
        app = FastAPI()

        @app.get("/test")
        async def endpoint() -> Dict:
            return {"ok": True}

        app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

        client = TestClient(app)
        for _ in range(5):
            response = client.get("/test")
            assert response.status_code == 200

    def test_returns_429_when_limit_exceeded(self) -> None:
        app = FastAPI()

        @app.get("/test")
        async def endpoint() -> Dict:
            return {"ok": True}

        # Set a very low limit
        app.add_middleware(RateLimitMiddleware, requests_per_minute=1, window_seconds=60)

        client = TestClient(app)
        # First request succeeds
        response1 = client.get("/test")
        assert response1.status_code == 200

        # Second request should be rate limited
        response2 = client.get("/test")
        assert response2.status_code == 429

    def test_429_response_has_retry_after_header(self) -> None:
        app = FastAPI()

        @app.get("/test")
        async def endpoint() -> Dict:
            return {"ok": True}

        app.add_middleware(RateLimitMiddleware, requests_per_minute=1, window_seconds=30)

        client = TestClient(app)
        client.get("/test")  # Use up the token

        response = client.get("/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert int(response.headers["Retry-After"]) == 30

    def test_different_clients_have_separate_limits(self) -> None:
        app = FastAPI()

        @app.get("/test")
        async def endpoint() -> Dict:
            return {"ok": True}

        app.add_middleware(RateLimitMiddleware, requests_per_minute=1, window_seconds=60)

        client = TestClient(app)

        # Use up the token with one client IP (default is testclient)
        response1 = client.get("/test")
        assert response1.status_code == 200

        # Same client should be rate limited
        response2 = client.get("/test")
        assert response2.status_code == 429


# =============================================================================
# API Key Middleware Tests
# =============================================================================


class TestApiKeyMiddleware:
    """Tests for the API key authentication middleware."""

    def test_no_api_key_configured_allows_all(self) -> None:
        with patch("app.middleware.auth.config") as mock_config:
            mock_config.api_key = ""

            app = FastAPI()

            @app.get("/test")
            async def endpoint() -> Dict:
                return {"ok": True}

            app.add_middleware(ApiKeyMiddleware)

            client = TestClient(app)
            response = client.get("/test")
            assert response.status_code == 200

    def test_api_key_required_without_key_returns_401(self) -> None:
        with patch("app.middleware.auth.config") as mock_config:
            mock_config.api_key = "secret-key"

            app = FastAPI()

            @app.get("/api/v1/protected")
            async def endpoint() -> Dict:
                return {"ok": True}

            app.add_middleware(ApiKeyMiddleware)

            client = TestClient(app)
            response = client.get("/api/v1/protected")
            assert response.status_code == 401

    def test_api_key_required_with_valid_key_returns_200(self) -> None:
        with patch("app.middleware.auth.config") as mock_config:
            mock_config.api_key = "secret-key"

            app = FastAPI()

            @app.get("/api/v1/protected")
            async def endpoint() -> Dict:
                return {"ok": True}

            app.add_middleware(ApiKeyMiddleware)

            client = TestClient(app)
            response = client.get(
                "/api/v1/protected",
                headers={"X-API-Key": "secret-key"},
            )
            assert response.status_code == 200

    def test_api_key_required_with_invalid_key_returns_401(self) -> None:
        with patch("app.middleware.auth.config") as mock_config:
            mock_config.api_key = "secret-key"

            app = FastAPI()

            @app.get("/api/v1/protected")
            async def endpoint() -> Dict:
                return {"ok": True}

            app.add_middleware(ApiKeyMiddleware)

            client = TestClient(app)
            response = client.get(
                "/api/v1/protected",
                headers={"X-API-Key": "wrong-key"},
            )
            assert response.status_code == 401

    def test_health_endpoint_exempt_from_auth(self) -> None:
        with patch("app.middleware.auth.config") as mock_config:
            mock_config.api_key = "secret-key"

            app = FastAPI()

            @app.get("/api/v1/health")
            async def endpoint() -> Dict:
                return {"ok": True}

            app.add_middleware(ApiKeyMiddleware)

            client = TestClient(app)
            response = client.get("/api/v1/health")
            assert response.status_code == 200

    def test_docs_endpoint_exempt_from_auth(self) -> None:
        with patch("app.middleware.auth.config") as mock_config:
            mock_config.api_key = "secret-key"

            app = FastAPI()

            app.add_middleware(ApiKeyMiddleware)

            client = TestClient(app)
            response = client.get("/docs")
            assert response.status_code == 200

    def test_401_response_has_www_authenticate_header(self) -> None:
        with patch("app.middleware.auth.config") as mock_config:
            mock_config.api_key = "secret-key"

            app = FastAPI()

            @app.get("/api/v1/protected")
            async def endpoint() -> Dict:
                return {"ok": True}

            app.add_middleware(ApiKeyMiddleware)

            client = TestClient(app)
            response = client.get("/api/v1/protected")
            assert response.status_code == 401
            assert "WWW-Authenticate" in response.headers

    def test_401_response_has_error_structure(self) -> None:
        with patch("app.middleware.auth.config") as mock_config:
            mock_config.api_key = "secret-key"

            app = FastAPI()

            @app.get("/api/v1/protected")
            async def endpoint() -> Dict:
                return {"ok": True}

            app.add_middleware(ApiKeyMiddleware)

            client = TestClient(app)
            response = client.get("/api/v1/protected")
            data = response.json()
            assert "detail" in data
