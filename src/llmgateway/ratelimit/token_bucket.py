"""Distributed token bucket rate limiter backed by a Redis Lua script.

The Lua script executes atomically inside Redis so that multiple gateway
replicas share a single consistent per-user bucket with no race conditions.

Redis key layout
----------------
``llmgw:rl:{user_id}``  — Hash with ``tokens`` and ``last_refill`` fields.
                           Auto-expires after 3 600 s of inactivity (set by
                           the Lua script's EXPIRE call).

Lua return convention
---------------------
The script returns a 3-element list ``{allowed, remaining, retry_after}``
where each element is a Redis integer (Lua numbers are truncated when
returned via RESP2).  Float precision is therefore limited to ±1 unit, which
is acceptable for token-bucket rate limiting.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from opentelemetry import trace
from prometheus_client import Counter, Histogram

if TYPE_CHECKING:
    from redis.asyncio import Redis

_log = structlog.get_logger(__name__)
_tracer = trace.get_tracer(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_RL_CHECKS = Counter(
    "llm_rate_limit_checks_total",
    "Total number of rate limit checks performed",
    ["result"],  # "allowed" | "denied"
)

_RL_DURATION = Histogram(
    "llm_rate_limit_check_duration_seconds",
    "Duration of a single rate limit check (Redis Lua round-trip)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
)

_RL_EXCEEDED = Counter(
    "llm_rate_limit_exceeded_total",
    "Total number of rate limit exceeded events per user",
    ["user_id"],
)

# ---------------------------------------------------------------------------
# Lua script
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent / "scripts" / "token_bucket.lua"
_LUA_SCRIPT: str = _SCRIPT_PATH.read_text()

_KEY_PREFIX = "llmgw:rl:"


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------


class TokenBucket:
    """Distributed token bucket backed by Redis and an atomic Lua script.

    Each user has a dedicated hash key in Redis.  The Lua script refills
    tokens based on elapsed time, then either deducts the requested amount
    (allowed) or reports the wait time (denied).  Because the script runs
    inside Redis, the check-and-deduct is atomic even across many gateway
    replicas.

    The Lua script SHA is loaded once on the first call and reused for the
    lifetime of the instance.  If Redis flushes its script cache (e.g.
    after a restart), the SHA is invalidated and reloaded transparently.

    Args:
        redis_client: Async Redis client.
        capacity:     Maximum token count (burst ceiling).
        rate:         Refill rate in tokens **per second**.
        key_prefix:   Redis key namespace prefix.
    """

    def __init__(
        self,
        redis_client: Redis,
        capacity: float,
        rate: float,
        key_prefix: str = _KEY_PREFIX,
    ) -> None:
        self._redis = redis_client
        self._capacity = capacity
        self._rate = rate
        self._key_prefix = key_prefix
        self._script_sha: str | None = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_sha(self) -> str:
        """Upload the Lua script to Redis on first call; cache the SHA."""
        if self._script_sha is None:
            self._script_sha = await self._redis.script_load(_LUA_SCRIPT)
        return self._script_sha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def consume(
        self,
        user_id: str,
        tokens: float = 1.0,
    ) -> tuple[bool, float, float]:
        """Attempt to consume *tokens* from *user_id*'s bucket.

        Returns:
            ``(allowed, remaining, retry_after)``

            * *allowed*     — ``True`` when the request is within budget.
            * *remaining*   — Tokens left after this call (integer precision).
            * *retry_after* — Seconds to wait before the next request will
                              succeed (``0`` when *allowed* is ``True``).

        Fails open: any Redis or Lua error returns ``(True, capacity, 0.0)``
        so a Redis outage never blocks legitimate traffic.
        """
        key = self._key_prefix + user_id
        now = time.time()

        start = time.monotonic()

        with _tracer.start_as_current_span("rate_limit.consume") as span:
            span.set_attribute("rate_limit.user_id", user_id)
            span.set_attribute("rate_limit.tokens_requested", tokens)
            span.set_attribute("rate_limit.capacity", self._capacity)
            span.set_attribute("rate_limit.rate_per_second", self._rate)

            try:
                sha = await self._get_sha()
                result = await self._redis.evalsha(
                    sha,
                    1,  # numkeys
                    key,  # KEYS[1]
                    self._capacity,  # ARGV[1]
                    self._rate,  # ARGV[2]
                    tokens,  # ARGV[3]
                    now,  # ARGV[4]
                )
                allowed = bool(int(result[0]))
                remaining = float(result[1])
                retry_after = float(result[2])

                span.set_attribute("rate_limit.allowed", allowed)
                span.set_attribute("rate_limit.remaining", remaining)

            except Exception as exc:
                _log.warning(
                    "rate_limit.redis_error",
                    user_id=user_id,
                    error=str(exc),
                )
                # Fail open: permit the request, report full capacity.
                return True, self._capacity, 0.0

        duration = time.monotonic() - start
        _RL_DURATION.observe(duration)
        label = "allowed" if allowed else "denied"
        _RL_CHECKS.labels(result=label).inc()

        if not allowed:
            _RL_EXCEEDED.labels(user_id=user_id).inc()

        _log.debug(
            "rate_limit.check",
            user_id=user_id,
            allowed=allowed,
            remaining=round(remaining, 2),
            retry_after=round(retry_after, 2),
        )

        return allowed, remaining, retry_after
