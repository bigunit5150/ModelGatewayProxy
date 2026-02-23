"""Tests for GET /admin/costs/summary endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from llmgateway.cost import CostTracker
from llmgateway.main import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUMMARY = {
    "total_cost_usd": 0.05,
    "request_count": 3,
    "cost_by_model": {"gpt-4o": 0.05},
}


def _mock_tracker(summary: dict | None = None) -> CostTracker:
    tracker = AsyncMock(spec=CostTracker)
    tracker.get_summary = AsyncMock(return_value=summary or _SUMMARY)
    return tracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_state():
    """Remove app.state.cost_tracker after every test."""
    yield
    if hasattr(app.state, "cost_tracker"):
        delattr(app.state, "cost_tracker")


@pytest.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Authentication / configuration guard
# ---------------------------------------------------------------------------


class TestAdminAuth:
    async def test_returns_503_when_admin_key_not_configured(self, client: AsyncClient) -> None:
        app.state.cost_tracker = _mock_tracker()
        with patch("llmgateway.api.admin.settings") as mock_settings:
            mock_settings.admin_api_key = None
            response = await client.get("/admin/costs/summary")
        assert response.status_code == 503

    async def test_returns_401_with_wrong_key(self, client: AsyncClient) -> None:
        app.state.cost_tracker = _mock_tracker()
        with patch("llmgateway.api.admin.settings") as mock_settings:
            mock_settings.admin_api_key = "secret123"
            response = await client.get("/admin/costs/summary", headers={"X-Admin-Key": "wrongkey"})
        assert response.status_code == 401

    async def test_returns_401_when_key_missing(self, client: AsyncClient) -> None:
        app.state.cost_tracker = _mock_tracker()
        with patch("llmgateway.api.admin.settings") as mock_settings:
            mock_settings.admin_api_key = "secret123"
            response = await client.get("/admin/costs/summary")
        assert response.status_code == 401

    async def test_returns_200_with_correct_key(self, client: AsyncClient) -> None:
        app.state.cost_tracker = _mock_tracker()
        with patch("llmgateway.api.admin.settings") as mock_settings:
            mock_settings.admin_api_key = "secret123"
            response = await client.get(
                "/admin/costs/summary", headers={"X-Admin-Key": "secret123"}
            )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Cost tracker availability
# ---------------------------------------------------------------------------


class TestCostTrackerDependency:
    async def test_returns_503_when_no_tracker(self, client: AsyncClient) -> None:
        with patch("llmgateway.api.admin.settings") as mock_settings:
            mock_settings.admin_api_key = "secret"
            response = await client.get("/admin/costs/summary", headers={"X-Admin-Key": "secret"})
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


class TestCostsSummaryResponse:
    async def _get(self, client: AsyncClient, **params) -> dict:
        app.state.cost_tracker = _mock_tracker()
        with patch("llmgateway.api.admin.settings") as mock_settings:
            mock_settings.admin_api_key = "key"
            response = await client.get(
                "/admin/costs/summary",
                headers={"X-Admin-Key": "key"},
                params=params,
            )
        return response

    async def test_response_has_expected_keys(self, client: AsyncClient) -> None:
        r = await self._get(client)
        body = r.json()
        assert "total_cost_usd" in body
        assert "request_count" in body
        assert "cost_by_model" in body

    async def test_total_cost_matches_mock(self, client: AsyncClient) -> None:
        r = await self._get(client)
        assert r.json()["total_cost_usd"] == 0.05

    async def test_request_count_matches_mock(self, client: AsyncClient) -> None:
        r = await self._get(client)
        assert r.json()["request_count"] == 3


# ---------------------------------------------------------------------------
# Query parameter forwarding
# ---------------------------------------------------------------------------


class TestQueryParameters:
    async def test_user_id_forwarded_to_tracker(self, client: AsyncClient) -> None:
        tracker = _mock_tracker()
        app.state.cost_tracker = tracker

        with patch("llmgateway.api.admin.settings") as mock_settings:
            mock_settings.admin_api_key = "key"
            await client.get(
                "/admin/costs/summary",
                headers={"X-Admin-Key": "key"},
                params={"user_id": "alice"},
            )

        tracker.get_summary.assert_awaited_once()
        _, kwargs = tracker.get_summary.call_args
        assert kwargs.get("user_id") == "alice"

    async def test_invalid_start_date_returns_400(self, client: AsyncClient) -> None:
        app.state.cost_tracker = _mock_tracker()
        with patch("llmgateway.api.admin.settings") as mock_settings:
            mock_settings.admin_api_key = "key"
            response = await client.get(
                "/admin/costs/summary",
                headers={"X-Admin-Key": "key"},
                params={"start_date": "not-a-date"},
            )
        assert response.status_code == 400

    async def test_invalid_end_date_returns_400(self, client: AsyncClient) -> None:
        app.state.cost_tracker = _mock_tracker()
        with patch("llmgateway.api.admin.settings") as mock_settings:
            mock_settings.admin_api_key = "key"
            response = await client.get(
                "/admin/costs/summary",
                headers={"X-Admin-Key": "key"},
                params={"end_date": "bad-date"},
            )
        assert response.status_code == 400

    async def test_valid_date_range_forwarded(self, client: AsyncClient) -> None:
        tracker = _mock_tracker()
        app.state.cost_tracker = tracker

        with patch("llmgateway.api.admin.settings") as mock_settings:
            mock_settings.admin_api_key = "key"
            await client.get(
                "/admin/costs/summary",
                headers={"X-Admin-Key": "key"},
                params={
                    "start_date": "2025-01-01T00:00:00",
                    "end_date": "2025-01-31T23:59:59",
                },
            )

        tracker.get_summary.assert_awaited_once()
        _, kwargs = tracker.get_summary.call_args
        assert kwargs.get("start") is not None
        assert kwargs.get("end") is not None
