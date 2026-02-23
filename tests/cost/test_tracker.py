"""Tests for CostTracker (cost/tracker.py) using mocked SQLAlchemy sessions."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmgateway.cost.tracker import CostTracker, UsageRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker() -> CostTracker:
    """Return a CostTracker with a mocked async engine (no real DB needed)."""
    with patch("llmgateway.cost.tracker.create_async_engine") as mock_engine:
        mock_engine.return_value = MagicMock()
        tracker = CostTracker(database_url="postgresql+asyncpg://test/test")
    return tracker


def _mock_session_ctx(rows=None, scalar=None):
    """Return a context manager mock that yields a mocked AsyncSession."""
    session = AsyncMock()
    # session.add is synchronous — override so it doesn't produce an unawaited coroutine.
    session.add = MagicMock()

    execute_result = MagicMock()
    execute_result.all.return_value = rows if rows is not None else []
    execute_result.scalar.return_value = scalar  # May be None — that's valid
    session.execute = AsyncMock(return_value=execute_result)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, session


# ---------------------------------------------------------------------------
# UsageRecord ORM model
# ---------------------------------------------------------------------------


class TestUsageRecord:
    def test_tablename(self) -> None:
        assert UsageRecord.__tablename__ == "usage_records"

    def test_columns_exist(self) -> None:
        cols = {c.name for c in UsageRecord.__table__.columns}
        expected = {
            "id",
            "timestamp",
            "user_id",
            "model",
            "input_tokens",
            "output_tokens",
            "cost_usd",
            "cached",
            "cache_type",
        }
        assert expected <= cols

    def test_timestamp_indexed(self) -> None:
        col = UsageRecord.__table__.columns["timestamp"]
        assert col.index is True

    def test_user_id_indexed(self) -> None:
        col = UsageRecord.__table__.columns["user_id"]
        assert col.index is True


# ---------------------------------------------------------------------------
# CostTracker.record_usage
# ---------------------------------------------------------------------------


class TestRecordUsage:
    async def test_inserts_record_and_commits(self) -> None:
        tracker = _make_tracker()
        ctx, session = _mock_session_ctx()

        with patch("llmgateway.cost.tracker.AsyncSession", return_value=ctx):
            await tracker.record_usage(
                user_id="alice",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )

        session.add.assert_called_once()
        session.commit.assert_awaited_once()

    async def test_record_has_correct_fields(self) -> None:
        tracker = _make_tracker()
        ctx, session = _mock_session_ctx()

        with patch("llmgateway.cost.tracker.AsyncSession", return_value=ctx):
            await tracker.record_usage(
                user_id="bob",
                model="claude-sonnet-4-6",
                input_tokens=200,
                output_tokens=80,
                cost_usd=0.002,
                cached=True,
                cache_type="EXACT",
            )

        added: UsageRecord = session.add.call_args[0][0]
        assert added.user_id == "bob"
        assert added.model == "claude-sonnet-4-6"
        assert added.input_tokens == 200
        assert added.output_tokens == 80
        assert added.cached is True
        assert added.cache_type == "EXACT"

    async def test_swallows_db_errors(self) -> None:
        tracker = _make_tracker()
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(side_effect=Exception("connection refused"))
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("llmgateway.cost.tracker.AsyncSession", return_value=ctx):
            # Must not raise
            await tracker.record_usage(
                user_id="user",
                model="gpt-4o",
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
            )

    async def test_defaults_cached_to_false(self) -> None:
        tracker = _make_tracker()
        ctx, session = _mock_session_ctx()

        with patch("llmgateway.cost.tracker.AsyncSession", return_value=ctx):
            await tracker.record_usage(
                user_id="u",
                model="gpt-4o",
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
            )

        added: UsageRecord = session.add.call_args[0][0]
        assert added.cached is False
        assert added.cache_type is None


# ---------------------------------------------------------------------------
# CostTracker.get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    def _row(self, model: str, total_cost: float, count: int):
        r = MagicMock()
        r.model = model
        r.total_cost = Decimal(str(total_cost))
        r.request_count = count
        return r

    async def test_returns_expected_keys(self) -> None:
        tracker = _make_tracker()
        ctx, _ = _mock_session_ctx(rows=[self._row("gpt-4o", 0.05, 3)])

        with patch("llmgateway.cost.tracker.AsyncSession", return_value=ctx):
            result = await tracker.get_summary()

        assert "total_cost_usd" in result
        assert "request_count" in result
        assert "cost_by_model" in result

    async def test_aggregates_multiple_models(self) -> None:
        tracker = _make_tracker()
        rows = [self._row("gpt-4o", 0.10, 2), self._row("claude-sonnet-4-6", 0.06, 1)]
        ctx, _ = _mock_session_ctx(rows=rows)

        with patch("llmgateway.cost.tracker.AsyncSession", return_value=ctx):
            result = await tracker.get_summary()

        assert result["total_cost_usd"] == pytest.approx(0.16, rel=1e-6)
        assert result["request_count"] == 3
        assert "gpt-4o" in result["cost_by_model"]
        assert "claude-sonnet-4-6" in result["cost_by_model"]

    async def test_empty_result(self) -> None:
        tracker = _make_tracker()
        ctx, _ = _mock_session_ctx(rows=[])

        with patch("llmgateway.cost.tracker.AsyncSession", return_value=ctx):
            result = await tracker.get_summary()

        assert result["total_cost_usd"] == 0.0
        assert result["request_count"] == 0
        assert result["cost_by_model"] == {}

    async def test_none_total_cost_treated_as_zero(self) -> None:
        tracker = _make_tracker()
        row = MagicMock()
        row.model = "gpt-4o"
        row.total_cost = None
        row.request_count = 1
        ctx, _ = _mock_session_ctx(rows=[row])

        with patch("llmgateway.cost.tracker.AsyncSession", return_value=ctx):
            result = await tracker.get_summary()

        assert result["total_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# CostTracker.get_daily_cost
# ---------------------------------------------------------------------------


class TestGetDailyCost:
    async def test_returns_float(self) -> None:
        tracker = _make_tracker()
        ctx, _ = _mock_session_ctx(scalar=Decimal("1.234567"))

        with patch("llmgateway.cost.tracker.AsyncSession", return_value=ctx):
            cost = await tracker.get_daily_cost()

        assert isinstance(cost, float)
        assert cost == pytest.approx(1.234567, rel=1e-5)

    async def test_none_scalar_returns_zero(self) -> None:
        tracker = _make_tracker()
        ctx, _ = _mock_session_ctx(scalar=None)

        with patch("llmgateway.cost.tracker.AsyncSession", return_value=ctx):
            cost = await tracker.get_daily_cost()

        assert cost == 0.0

    async def test_defaults_to_today_utc(self) -> None:
        """Passing day=None should not raise and should query today."""
        tracker = _make_tracker()
        ctx, _ = _mock_session_ctx(scalar=Decimal("0.5"))

        with patch("llmgateway.cost.tracker.AsyncSession", return_value=ctx):
            cost = await tracker.get_daily_cost(day=None)

        assert cost == pytest.approx(0.5, rel=1e-6)

    async def test_explicit_day(self) -> None:
        tracker = _make_tracker()
        ctx, _ = _mock_session_ctx(scalar=Decimal("2.0"))

        with patch("llmgateway.cost.tracker.AsyncSession", return_value=ctx):
            cost = await tracker.get_daily_cost(day=date(2025, 1, 15))

        assert cost == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# CostTracker.close
# ---------------------------------------------------------------------------


class TestClose:
    async def test_disposes_engine(self) -> None:
        tracker = _make_tracker()
        tracker._engine = AsyncMock()
        await tracker.close()
        tracker._engine.dispose.assert_awaited_once()
