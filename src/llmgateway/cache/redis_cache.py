"""Redis-backed async cache implementing :class:`~llmgateway.cache.base.CacheBackend`.

Uses ``redis.asyncio`` (bundled with ``redis[hiredis]`` ≥ 4.2) for non-blocking
I/O.  Every operation is wrapped in an OpenTelemetry span and logged with
structlog.  All exceptions are caught and logged so that callers experience
fail-open behaviour (errors are treated as cache misses).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from llmgateway.cache.base import CacheEntry

if TYPE_CHECKING:
    from redis.asyncio import Redis

_log = structlog.get_logger(__name__)
_tracer = trace.get_tracer(__name__)

# Namespace prefix applied to every key stored in Redis so cache entries
# never collide with other data (e.g. rate-limit counters).
_KEY_PREFIX = "llmgw:cache:"


class RedisCache:
    """Async Redis cache backend.

    Args:
        redis: An already-connected :class:`redis.asyncio.Redis` client.
               Create one with ``redis.asyncio.from_url(url, decode_responses=True)``.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    # ------------------------------------------------------------------
    # CacheBackend implementation
    # ------------------------------------------------------------------

    async def get(self, key: str) -> CacheEntry | None:
        """Fetch the entry for *key* from Redis.

        Returns ``None`` on a miss or on any Redis / deserialisation error.
        """
        redis_key = _KEY_PREFIX + key

        with _tracer.start_as_current_span("cache.get") as span:
            span.set_attribute("db.system", "redis")
            span.set_attribute("cache.key", key)

            try:
                raw: str | bytes | None = await self._redis.get(redis_key)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                _log.warning("cache.get_error", key=key, error=str(exc))
                return None

            if raw is None:
                span.set_attribute("cache.hit", False)
                return None

            try:
                data: dict[str, Any] = json.loads(raw)
                entry = CacheEntry(
                    key=data["key"],
                    value=data["value"],
                    created_at=float(data["created_at"]),
                    ttl=int(data["ttl"]),
                    metadata=data.get("metadata", {}),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                _log.warning("cache.deserialise_error", key=key, error=str(exc))
                return None

            span.set_attribute("cache.hit", True)
            span.set_attribute("cache.ttl", entry.ttl)
            return entry

    async def set(self, entry: CacheEntry) -> None:
        """Persist *entry* in Redis with its configured TTL.

        Silently swallows errors so a Redis outage never breaks a request.
        """
        redis_key = _KEY_PREFIX + entry.key

        with _tracer.start_as_current_span("cache.set") as span:
            span.set_attribute("db.system", "redis")
            span.set_attribute("cache.key", entry.key)
            span.set_attribute("cache.ttl", entry.ttl)

            try:
                payload = json.dumps(
                    {
                        "key": entry.key,
                        "value": entry.value,
                        "created_at": entry.created_at,
                        "ttl": entry.ttl,
                        "metadata": entry.metadata,
                    }
                )
                await self._redis.set(redis_key, payload, ex=entry.ttl)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                _log.warning("cache.set_error", key=entry.key, error=str(exc))

    async def delete(self, key: str) -> None:
        """Remove the entry for *key* from Redis (no-op if absent)."""
        redis_key = _KEY_PREFIX + key

        with _tracer.start_as_current_span("cache.delete") as span:
            span.set_attribute("db.system", "redis")
            span.set_attribute("cache.key", key)

            try:
                await self._redis.delete(redis_key)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                _log.warning("cache.delete_error", key=key, error=str(exc))

    async def exists(self, key: str) -> bool:
        """Return ``True`` if a non-expired entry exists for *key*."""
        redis_key = _KEY_PREFIX + key

        with _tracer.start_as_current_span("cache.exists") as span:
            span.set_attribute("db.system", "redis")
            span.set_attribute("cache.key", key)

            try:
                result: int = await self._redis.exists(redis_key)
                found = bool(result)
                span.set_attribute("cache.hit", found)
                return found
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                _log.warning("cache.exists_error", key=key, error=str(exc))
                return False
