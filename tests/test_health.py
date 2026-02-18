"""Tests for /health endpoints."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from llmgateway.main import app


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    async def test_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200

    async def test_response_structure(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        body = response.json()
        assert body["status"] == "healthy"
        assert "timestamp" in body
        assert "version" in body

    async def test_version_matches_settings(self, client: AsyncClient) -> None:
        from llmgateway.config import settings

        response = await client.get("/health")
        assert response.json()["version"] == settings.app_version


# ---------------------------------------------------------------------------
# /health/live
# ---------------------------------------------------------------------------
class TestLivenessEndpoint:
    async def test_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/health/live")
        assert response.status_code == 200

    async def test_response_body(self, client: AsyncClient) -> None:
        response = await client.get("/health/live")
        assert response.json()["status"] == "alive"


# ---------------------------------------------------------------------------
# /health/ready
# ---------------------------------------------------------------------------
class TestReadinessEndpoint:
    async def test_all_dependencies_healthy(self, client: AsyncClient) -> None:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.aclose = AsyncMock()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)
        mock_engine.dispose = AsyncMock()

        with (
            patch("llmgateway.api.health.Redis") as mock_redis_patch,
            patch("llmgateway.api.health.create_async_engine", return_value=mock_engine),
        ):
            mock_redis_patch.from_url.return_value = mock_redis
            response = await client.get("/health/ready")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["checks"]["redis"] == "ok"
        assert body["checks"]["postgres"] == "ok"

    async def test_redis_down_returns_503(self, client: AsyncClient) -> None:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("Redis unreachable"))
        mock_redis.aclose = AsyncMock()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)
        mock_engine.dispose = AsyncMock()

        with (
            patch("llmgateway.api.health.Redis") as mock_redis_patch,
            patch("llmgateway.api.health.create_async_engine", return_value=mock_engine),
        ):
            mock_redis_patch.from_url.return_value = mock_redis
            response = await client.get("/health/ready")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unavailable"
        assert "redis" in body["errors"]

    async def test_postgres_down_returns_503(self, client: AsyncClient) -> None:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.aclose = AsyncMock()

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(side_effect=ConnectionError("Postgres unreachable"))
        mock_engine.dispose = AsyncMock()

        with (
            patch("llmgateway.api.health.Redis") as mock_redis_patch,
            patch("llmgateway.api.health.create_async_engine", return_value=mock_engine),
        ):
            mock_redis_patch.from_url.return_value = mock_redis
            response = await client.get("/health/ready")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unavailable"
        assert "postgres" in body["errors"]

    async def test_both_dependencies_down_returns_503(self, client: AsyncClient) -> None:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("Redis unreachable"))
        mock_redis.aclose = AsyncMock()

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(side_effect=ConnectionError("Postgres unreachable"))
        mock_engine.dispose = AsyncMock()

        with (
            patch("llmgateway.api.health.Redis") as mock_redis_patch,
            patch("llmgateway.api.health.create_async_engine", return_value=mock_engine),
        ):
            mock_redis_patch.from_url.return_value = mock_redis
            response = await client.get("/health/ready")

        assert response.status_code == 503
        body = response.json()
        assert "redis" in body["errors"]
        assert "postgres" in body["errors"]
