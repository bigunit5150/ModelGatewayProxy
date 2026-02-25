"""High-level rate limiter coordinating per-user token buckets.

Provides:

* :class:`RateLimitResult` — Structured result of a single rate limit check.
* :class:`RateLimiter`     — Routes each request to the correct
  :class:`~llmgateway.ratelimit.token_bucket.TokenBucket` based on the
  user's tier and exposes HTTP-layer helpers.

Tier system
-----------
Three built-in tiers control bucket capacity (burst limit) and steady-state
refill rate.  Pass a ``get_tier`` callable to map user IDs to tier names;
users that don't match any tier fall back to the default capacity/rate from
configuration.

=========== ============ ====================
Tier        Capacity     Refill rate
=========== ============ ====================
free        20           10 req / min
pro         100          60 req / min
enterprise  1 000        600 req / min
=========== ============ ====================
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from llmgateway.ratelimit.token_bucket import TokenBucket

if TYPE_CHECKING:
    from redis.asyncio import Redis

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Built-in tier definitions: (capacity, tokens_per_second)
# ---------------------------------------------------------------------------

TIER_CONFIGS: dict[str, tuple[float, float]] = {
    "free": (20.0, 10 / 60),  # 10 req/min, burst of 20
    "pro": (100.0, 60 / 60),  # 60 req/min, burst of 100
    "enterprise": (1000.0, 600 / 60),  # 600 req/min, burst of 1 000
}


# ---------------------------------------------------------------------------
# RateLimitResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimitResult:
    """Outcome of a single :meth:`RateLimiter.check_rate_limit` call.

    Attributes:
        allowed:     ``True`` when the request is within the rate limit.
        retry_after: Seconds the client should wait before retrying.
                     Always ``0.0`` when *allowed* is ``True``.
        remaining:   Token count remaining in the bucket after this call.
        reset_time:  Unix timestamp (float) at which the bucket will be full.
        limit:       Bucket capacity — the maximum burst the user can send.
    """

    allowed: bool
    retry_after: float
    remaining: float
    reset_time: float
    limit: float


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Per-user distributed rate limiter built on :class:`TokenBucket`.

    Creates one :class:`TokenBucket` instance per unique ``(capacity, rate)``
    pair (lazy, on first use) and routes each :meth:`check_rate_limit` call
    to the correct bucket based on the resolved tier for that user.

    Args:
        redis_client:     Async Redis client shared with the rest of the app.
        default_capacity: Bucket capacity for users not matched by a tier.
        default_rate:     Refill rate (tokens / second) for untiered users.
        enabled:          When ``False`` every check returns ``allowed=True``
                          without touching Redis — useful in local dev / tests.
        tier_configs:     Mapping of tier name → ``(capacity, rate/s)``.
                          Defaults to the built-in :data:`TIER_CONFIGS`.
        get_tier:         Callable ``(user_id: str) → str | None`` that maps
                          a user ID to a tier name, or ``None`` to use the
                          default bucket.  When omitted all users share the
                          default tier limits.
    """

    def __init__(
        self,
        redis_client: Redis,
        default_capacity: float,
        default_rate: float,
        *,
        enabled: bool = True,
        tier_configs: dict[str, tuple[float, float]] | None = None,
        get_tier: Callable[[str], str | None] | None = None,
    ) -> None:
        self._redis = redis_client
        self._default_capacity = default_capacity
        self._default_rate = default_rate
        self._enabled = enabled
        self._tier_configs = tier_configs if tier_configs is not None else TIER_CONFIGS
        self._get_tier = get_tier

        # Lazily-created TokenBucket instances, keyed by (capacity, rate).
        self._buckets: dict[tuple[float, float], TokenBucket] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bucket_for(self, capacity: float, rate: float) -> TokenBucket:
        """Return (or create) the :class:`TokenBucket` for the given params."""
        key = (capacity, rate)
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                redis_client=self._redis,
                capacity=capacity,
                rate=rate,
            )
        return self._buckets[key]

    def _params_for_user(self, user_id: str) -> tuple[float, float]:
        """Resolve ``(capacity, rate)`` for *user_id* via tier lookup."""
        if self._get_tier is not None:
            tier = self._get_tier(user_id)
            if tier is not None and tier in self._tier_configs:
                return self._tier_configs[tier]
        return self._default_capacity, self._default_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_rate_limit(
        self,
        user_id: str,
        cost: float = 1.0,
    ) -> RateLimitResult:
        """Check and consume *cost* tokens from *user_id*'s bucket.

        When rate limiting is disabled the call is a no-op and always returns
        an *allowed* result without touching Redis.

        Args:
            user_id: Identifier for the requesting user.
            cost:    Tokens to consume (default ``1.0`` per request).

        Returns:
            :class:`RateLimitResult` describing the outcome.
        """
        if not self._enabled:
            return RateLimitResult(
                allowed=True,
                retry_after=0.0,
                remaining=self._default_capacity,
                reset_time=time.time(),
                limit=self._default_capacity,
            )

        capacity, rate = self._params_for_user(user_id)
        bucket = self._bucket_for(capacity, rate)
        allowed, remaining, retry_after = await bucket.consume(user_id, cost)

        # Estimate when the bucket will be fully refilled.
        now = time.time()
        tokens_needed = capacity - remaining
        reset_time = now + (tokens_needed / rate) if rate > 0 else now

        result = RateLimitResult(
            allowed=allowed,
            retry_after=retry_after,
            remaining=remaining,
            reset_time=reset_time,
            limit=capacity,
        )

        _log.info(
            "rate_limit.result",
            user_id=user_id,
            allowed=allowed,
            remaining=round(remaining, 2),
            retry_after=round(retry_after, 2),
            reset_time=round(reset_time, 2),
            limit=capacity,
        )

        return result

    async def get_rate_limit_info(self, user_id: str) -> dict[str, Any]:
        """Return the current bucket state for *user_id* without consuming tokens.

        Useful for admin / debug endpoints.  Internally calls the Lua script
        with ``requested=0``, which refills the bucket based on elapsed time
        and returns the current state without deducting anything.

        Returns:
            Dict with keys ``user_id``, ``tier``, ``limit``,
            ``remaining``, ``rate_per_second``, ``reset_time``, ``enabled``.
        """
        capacity, rate = self._params_for_user(user_id)
        tier = "default"
        if self._get_tier is not None:
            t = self._get_tier(user_id)
            if t is not None:
                tier = t

        if not self._enabled:
            return {
                "user_id": user_id,
                "tier": tier,
                "limit": capacity,
                "remaining": capacity,
                "rate_per_second": rate,
                "reset_time": time.time(),
                "enabled": False,
            }

        bucket = self._bucket_for(capacity, rate)
        # consume(tokens=0) reads state without modifying the bucket.
        _, remaining, _ = await bucket.consume(user_id, tokens=0.0)

        now = time.time()
        tokens_needed = capacity - remaining
        reset_time = now + (tokens_needed / rate) if rate > 0 else now

        return {
            "user_id": user_id,
            "tier": tier,
            "limit": capacity,
            "remaining": round(remaining, 2),
            "rate_per_second": round(rate, 6),
            "reset_time": round(reset_time, 2),
            "enabled": True,
        }
