"""Async cost tracker — records per-request usage to PostgreSQL.

Schema
------
A single ``usage_records`` table stores one row per request with columns for
user, model, token counts, computed cost, and cache metadata.  Indices on
``timestamp`` and ``user_id`` support the common query patterns (daily totals
and per-user summaries).

Fail-open design
----------------
:meth:`CostTracker.record_usage` swallows all exceptions so that a database
outage never causes a request failure.  The error is logged at WARNING level.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import structlog
from opentelemetry import trace
from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase

_log = structlog.get_logger(__name__)
_tracer = trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class UsageRecord(_Base):
    """One row per LLM request in the ``usage_records`` table."""

    __tablename__ = "usage_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    model = Column(String, nullable=False)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Numeric(10, 6), nullable=False, default=0.0)
    cached = Column(Boolean, nullable=False, default=False)
    cache_type = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Records per-request usage to PostgreSQL and exposes query helpers.

    Args:
        database_url: Async-compatible SQLAlchemy URL
                      (e.g. ``"postgresql+asyncpg://..."``).
    """

    def __init__(self, database_url: str) -> None:
        self._engine = create_async_engine(database_url, pool_pre_ping=True)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    async def record_usage(
        self,
        *,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cached: bool = False,
        cache_type: str | None = None,
    ) -> None:
        """Insert one usage record.  Fails silently to avoid blocking the response.

        Args:
            user_id:       Identifier for the requesting user.
            model:         LiteLLM model string.
            input_tokens:  Prompt token count.
            output_tokens: Completion token count.
            cost_usd:      Estimated cost in USD (use ``0.0`` for cached hits).
            cached:        ``True`` when the response came from cache.
            cache_type:    ``"EXACT"``, ``"SEMANTIC"``, or ``None``.
        """
        with _tracer.start_as_current_span("cost.record_usage") as span:
            span.set_attribute("cost.user_id", user_id)
            span.set_attribute("cost.model", model)
            span.set_attribute("cost.usd", cost_usd)
            try:
                async with AsyncSession(self._engine) as session:
                    record = UsageRecord(
                        timestamp=datetime.now(UTC),
                        user_id=user_id,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost_usd,
                        cached=cached,
                        cache_type=cache_type,
                    )
                    session.add(record)
                    await session.commit()
            except Exception as exc:
                _log.warning("cost.record_failed", error=str(exc), user_id=user_id)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    async def get_summary(
        self,
        user_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        """Return aggregated cost summary, optionally filtered by user and date range.

        Returns:
            Dict with keys ``total_cost_usd``, ``request_count``, ``cost_by_model``.
        """
        with _tracer.start_as_current_span("cost.get_summary"):
            async with AsyncSession(self._engine) as session:
                stmt = select(
                    func.sum(UsageRecord.cost_usd).label("total_cost"),
                    func.count(UsageRecord.id).label("request_count"),
                    UsageRecord.model,
                ).group_by(UsageRecord.model)

                if user_id is not None:
                    stmt = stmt.where(UsageRecord.user_id == user_id)
                if start is not None:
                    stmt = stmt.where(UsageRecord.timestamp >= start)
                if end is not None:
                    stmt = stmt.where(UsageRecord.timestamp <= end)

                rows = (await session.execute(stmt)).all()

                total_cost = sum(float(r.total_cost or 0) for r in rows)
                total_requests = sum(int(r.request_count or 0) for r in rows)
                cost_by_model = {r.model: round(float(r.total_cost or 0), 8) for r in rows}

                return {
                    "total_cost_usd": round(total_cost, 8),
                    "request_count": total_requests,
                    "cost_by_model": cost_by_model,
                }

    async def get_daily_cost(self, day: date | None = None) -> float:
        """Return the total cost across all users for *day* (defaults to today UTC).

        Args:
            day: The calendar date to query.  Defaults to today in UTC.

        Returns:
            Total cost in USD for that day.
        """
        if day is None:
            day = datetime.now(UTC).date()
        day_start = datetime(day.year, day.month, day.day, tzinfo=UTC)
        day_end = datetime(day.year, day.month, day.day, 23, 59, 59, 999999, tzinfo=UTC)

        with _tracer.start_as_current_span("cost.get_daily_cost"):
            async with AsyncSession(self._engine) as session:
                stmt = select(func.sum(UsageRecord.cost_usd)).where(
                    UsageRecord.timestamp >= day_start,
                    UsageRecord.timestamp <= day_end,
                )
                result = (await session.execute(stmt)).scalar()
                return float(result or 0.0)

    async def close(self) -> None:
        """Dispose the database engine connection pool."""
        await self._engine.dispose()
