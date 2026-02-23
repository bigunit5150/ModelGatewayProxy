"""Tests for RateLimiter and RateLimitResult (ratelimit/limiter.py)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from llmgateway.ratelimit.limiter import TIER_CONFIGS, RateLimiter, RateLimitResult

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _redis(*, evalsha_return: list | None = None) -> AsyncMock:
    mock = AsyncMock()
    mock.script_load.return_value = "sha"
    mock.evalsha.return_value = evalsha_return if evalsha_return is not None else [1, 19, 0]
    return mock


def _limiter(
    redis_client: AsyncMock | None = None,
    *,
    capacity: float = 20.0,
    rate: float = 10 / 60,
    enabled: bool = True,
    get_tier=None,
) -> RateLimiter:
    return RateLimiter(
        redis_client=redis_client or _redis(),
        default_capacity=capacity,
        default_rate=rate,
        enabled=enabled,
        get_tier=get_tier,
    )


# ---------------------------------------------------------------------------
# RateLimitResult
# ---------------------------------------------------------------------------


class TestRateLimitResult:
    def test_frozen_raises_on_mutation(self) -> None:
        r = RateLimitResult(
            allowed=True, retry_after=0.0, remaining=19.0, reset_time=time.time(), limit=20.0
        )
        with pytest.raises((AttributeError, TypeError)):
            r.allowed = False  # type: ignore[misc]

    def test_field_values_accessible(self) -> None:
        r = RateLimitResult(
            allowed=False, retry_after=6.0, remaining=0.0, reset_time=1000.0, limit=20.0
        )
        assert r.allowed is False
        assert r.retry_after == 6.0
        assert r.remaining == 0.0
        assert r.reset_time == 1000.0
        assert r.limit == 20.0


# ---------------------------------------------------------------------------
# RateLimiter — disabled mode
# ---------------------------------------------------------------------------


class TestDisabledMode:
    async def test_always_returns_allowed(self) -> None:
        limiter = _limiter(enabled=False)
        result = await limiter.check_rate_limit("user-1")
        assert result.allowed is True

    async def test_retry_after_is_zero(self) -> None:
        limiter = _limiter(enabled=False)
        result = await limiter.check_rate_limit("user-1")
        assert result.retry_after == 0.0

    async def test_remaining_equals_capacity(self) -> None:
        limiter = _limiter(capacity=42.0, enabled=False)
        result = await limiter.check_rate_limit("user-1")
        assert result.remaining == 42.0

    async def test_no_redis_calls_when_disabled(self) -> None:
        r = _redis()
        limiter = _limiter(r, enabled=False)
        await limiter.check_rate_limit("user-1")
        r.script_load.assert_not_called()
        r.evalsha.assert_not_called()


# ---------------------------------------------------------------------------
# RateLimiter — check_rate_limit
# ---------------------------------------------------------------------------


class TestCheckRateLimit:
    async def test_allowed_when_bucket_permits(self) -> None:
        limiter = _limiter(_redis(evalsha_return=[1, 19, 0]))
        result = await limiter.check_rate_limit("user-1")
        assert result.allowed is True
        assert result.remaining == 19.0

    async def test_denied_when_bucket_exhausted(self) -> None:
        limiter = _limiter(_redis(evalsha_return=[0, 0, 6]))
        result = await limiter.check_rate_limit("user-1")
        assert result.allowed is False
        assert result.retry_after == 6.0

    async def test_limit_reflects_capacity(self) -> None:
        limiter = _limiter(capacity=50.0)
        result = await limiter.check_rate_limit("user-1")
        assert result.limit == 50.0

    async def test_reset_time_is_in_the_future_when_tokens_consumed(self) -> None:
        # 19 remaining out of 20 → 1 token needs to refill → reset slightly in future
        limiter = _limiter(_redis(evalsha_return=[1, 19, 0]))
        before = time.time()
        result = await limiter.check_rate_limit("user-1")
        assert result.reset_time >= before

    async def test_custom_cost_forwarded_to_bucket(self) -> None:
        r = _redis()
        limiter = _limiter(r)
        await limiter.check_rate_limit("user-1", cost=3.0)
        # The 6th positional arg to evalsha is the tokens (cost)
        tokens_arg = r.evalsha.call_args[0][5]
        assert tokens_arg == 3.0


# ---------------------------------------------------------------------------
# RateLimiter — tier routing
# ---------------------------------------------------------------------------


class TestTierRouting:
    async def test_uses_default_when_get_tier_returns_none(self) -> None:
        limiter = _limiter(capacity=42.0, get_tier=lambda uid: None)
        result = await limiter.check_rate_limit("user-1")
        assert result.limit == 42.0

    async def test_uses_tier_capacity_when_tier_matched(self) -> None:
        r = _redis(evalsha_return=[1, 99, 0])

        def get_tier(uid: str) -> str | None:
            return "pro" if uid.startswith("pro_") else None

        limiter = _limiter(r, get_tier=get_tier)
        result = await limiter.check_rate_limit("pro_alice")
        assert result.limit == TIER_CONFIGS["pro"][0]  # 100.0

    async def test_unknown_tier_falls_back_to_default(self) -> None:
        limiter = _limiter(
            capacity=7.0,
            get_tier=lambda uid: "nonexistent_tier",
        )
        result = await limiter.check_rate_limit("user-1")
        assert result.limit == 7.0

    async def test_same_tier_reuses_bucket_instance(self) -> None:
        limiter = _limiter()
        await limiter.check_rate_limit("user-a")
        await limiter.check_rate_limit("user-b")
        # Both resolve to the same (capacity, rate) → single bucket in cache
        assert len(limiter._buckets) == 1

    async def test_different_tiers_get_separate_buckets(self) -> None:
        r = _redis()

        def get_tier(uid: str) -> str | None:
            return "free" if "free" in uid else "pro"

        limiter = _limiter(r, get_tier=get_tier)
        await limiter.check_rate_limit("free_alice")
        await limiter.check_rate_limit("pro_bob")
        assert len(limiter._buckets) == 2


# ---------------------------------------------------------------------------
# RateLimiter — get_rate_limit_info
# ---------------------------------------------------------------------------


class TestGetRateLimitInfo:
    async def test_disabled_returns_enabled_false(self) -> None:
        limiter = _limiter(enabled=False)
        info = await limiter.get_rate_limit_info("user-1")
        assert info["enabled"] is False

    async def test_disabled_remaining_equals_capacity(self) -> None:
        limiter = _limiter(capacity=20.0, enabled=False)
        info = await limiter.get_rate_limit_info("user-1")
        assert info["remaining"] == 20.0

    async def test_enabled_returns_expected_keys(self) -> None:
        limiter = _limiter()
        info = await limiter.get_rate_limit_info("user-1")
        for key in ("user_id", "tier", "limit", "remaining", "rate_per_second", "reset_time"):
            assert key in info

    async def test_user_id_matches(self) -> None:
        limiter = _limiter()
        info = await limiter.get_rate_limit_info("alice")
        assert info["user_id"] == "alice"

    async def test_enabled_shows_true(self) -> None:
        limiter = _limiter()
        info = await limiter.get_rate_limit_info("user-1")
        assert info["enabled"] is True

    async def test_tier_label_included(self) -> None:
        def get_tier(uid: str) -> str | None:
            return "pro"

        limiter = _limiter(get_tier=get_tier)
        info = await limiter.get_rate_limit_info("pro_user")
        assert info["tier"] == "pro"

    async def test_consumes_zero_tokens(self) -> None:
        r = _redis()
        limiter = _limiter(r)
        await limiter.get_rate_limit_info("user-1")
        # The tokens arg (index 5) passed to evalsha should be 0.0
        tokens_arg = r.evalsha.call_args[0][5]
        assert tokens_arg == 0.0
