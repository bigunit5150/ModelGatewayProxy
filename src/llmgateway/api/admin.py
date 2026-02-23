"""Admin endpoints for cost and usage reporting.

All routes require an ``X-Admin-Key`` header whose value must match the
``ADMIN_API_KEY`` environment variable (or the ``admin_api_key`` config field).
When no admin key is configured the endpoints return ``503 Service Unavailable``.
"""

from __future__ import annotations

from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from llmgateway.config import settings
from llmgateway.cost import CostTracker

router = APIRouter(prefix="/admin", tags=["admin"])
_log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _require_admin_key(request: Request) -> None:
    """Raise 401/503 unless a valid admin key is present in the request."""
    if settings.admin_api_key is None:
        raise HTTPException(status_code=503, detail="Admin API not configured")
    provided = request.headers.get("X-Admin-Key")
    if not provided or provided != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Key")


def _get_cost_tracker(request: Request) -> CostTracker:
    """Return the shared :class:`CostTracker` or raise 503."""
    tracker: CostTracker | None = getattr(request.app.state, "cost_tracker", None)
    if tracker is None:
        raise HTTPException(status_code=503, detail="Cost tracker not available")
    return tracker


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/costs/summary")
async def costs_summary(
    request: Request,
    user_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    _auth: None = Depends(_require_admin_key),
    cost_tracker: CostTracker = Depends(_get_cost_tracker),
) -> JSONResponse:
    """Return an aggregated cost summary.

    Query parameters:
    - ``user_id``    Filter to a specific user.
    - ``start_date`` ISO-8601 datetime (inclusive lower bound).
    - ``end_date``   ISO-8601 datetime (inclusive upper bound).

    Returns a JSON object with ``total_cost_usd``, ``request_count``,
    and ``cost_by_model``.
    """
    start: datetime | None = None
    end: datetime | None = None

    try:
        if start_date is not None:
            start = datetime.fromisoformat(start_date)
        if end_date is not None:
            end = datetime.fromisoformat(end_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {exc}") from exc

    summary = await cost_tracker.get_summary(user_id=user_id, start=start, end=end)
    _log.info("admin.costs_summary", user_id=user_id, start=start_date, end=end_date)
    return JSONResponse(content=summary)
