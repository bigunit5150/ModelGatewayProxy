import asyncio
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from opentelemetry import trace
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import text

from llmgateway.config import settings

router = APIRouter(prefix="/health", tags=["health"])
log = structlog.get_logger()
tracer = trace.get_tracer(__name__)


@router.get("")
async def health() -> JSONResponse:
    return JSONResponse(
        content={
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "version": settings.app_version,
        }
    )


@router.get("/live")
async def liveness() -> JSONResponse:
    """Kubernetes liveness probe — always returns 200 if the process is running."""
    return JSONResponse(content={"status": "alive"})


@router.get("/ready")
async def readiness() -> JSONResponse:
    """Kubernetes readiness probe — checks connectivity to all downstream dependencies."""
    checks: dict[str, str] = {}
    errors: dict[str, str] = {}

    with tracer.start_as_current_span("health.readiness"):
        # ------------------------------------------------------------------
        # Redis
        # ------------------------------------------------------------------
        with tracer.start_as_current_span("health.check.redis"):
            redis_client: Redis = Redis.from_url(settings.redis_url, socket_timeout=5)
            try:
                await asyncio.wait_for(redis_client.ping(), timeout=5.0)
                checks["redis"] = "ok"
                log.debug("Redis ping succeeded")
            except Exception as exc:
                errors["redis"] = str(exc)
                log.warning("Redis ping failed", error=str(exc))
            finally:
                await redis_client.aclose()

        # ------------------------------------------------------------------
        # Postgres
        # ------------------------------------------------------------------
        with tracer.start_as_current_span("health.check.postgres"):
            engine = create_async_engine(settings.database_url, pool_pre_ping=False)
            try:
                async with engine.connect() as conn:
                    await asyncio.wait_for(conn.execute(text("SELECT 1")), timeout=5.0)
                checks["postgres"] = "ok"
                log.debug("Postgres check succeeded")
            except Exception as exc:
                errors["postgres"] = str(exc)
                log.warning("Postgres check failed", error=str(exc))
            finally:
                await engine.dispose()

    if errors:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "checks": checks, "errors": errors},
        )

    return JSONResponse(content={"status": "ready", "checks": checks})
