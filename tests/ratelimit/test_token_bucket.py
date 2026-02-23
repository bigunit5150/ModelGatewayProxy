"""Tests for TokenBucket (ratelimit/token_bucket.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock

from llmgateway.ratelimit.token_bucket import _KEY_PREFIX, TokenBucket

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _redis(
    *,
    evalsha_return: list | None = None,
    script_load_return: str = "abc123sha",
) -> AsyncMock:
    """Return a mock async Redis client pre-configured for happy-path tests."""
    mock = AsyncMock()
    mock.script_load.return_value = script_load_return
    mock.evalsha.return_value = evalsha_return if evalsha_return is not None else [1, 19, 0]
    return mock


def _bucket(
    redis_client: AsyncMock | None = None,
    *,
    capacity: float = 20.0,
    rate: float = 10 / 60,
) -> TokenBucket:
    return TokenBucket(
        redis_client=redis_client or _redis(),
        capacity=capacity,
        rate=rate,
    )


# ---------------------------------------------------------------------------
# consume — allowed path
# ---------------------------------------------------------------------------


class TestConsumeAllowed:
    async def test_returns_true_when_bucket_allows(self) -> None:
        bucket = _bucket(_redis(evalsha_return=[1, 19, 0]))
        allowed, remaining, retry_after = await bucket.consume("user-1")
        assert allowed is True

    async def test_remaining_reflects_lua_result(self) -> None:
        bucket = _bucket(_redis(evalsha_return=[1, 17, 0]))
        _, remaining, _ = await bucket.consume("user-1")
        assert remaining == 17.0

    async def test_retry_after_is_zero_when_allowed(self) -> None:
        bucket = _bucket(_redis(evalsha_return=[1, 19, 0]))
        _, _, retry_after = await bucket.consume("user-1")
        assert retry_after == 0.0


# ---------------------------------------------------------------------------
# consume — denied path
# ---------------------------------------------------------------------------


class TestConsumeDenied:
    async def test_returns_false_when_bucket_denies(self) -> None:
        bucket = _bucket(_redis(evalsha_return=[0, 0, 6]))
        allowed, _, _ = await bucket.consume("user-1")
        assert allowed is False

    async def test_remaining_is_zero_when_exhausted(self) -> None:
        bucket = _bucket(_redis(evalsha_return=[0, 0, 6]))
        _, remaining, _ = await bucket.consume("user-1")
        assert remaining == 0.0

    async def test_retry_after_reflects_lua_result(self) -> None:
        bucket = _bucket(_redis(evalsha_return=[0, 0, 30]))
        _, _, retry_after = await bucket.consume("user-1")
        assert retry_after == 30.0


# ---------------------------------------------------------------------------
# Redis key / argument passing
# ---------------------------------------------------------------------------


class TestScriptArguments:
    async def test_key_uses_default_prefix_and_user_id(self) -> None:
        r = _redis()
        bucket = TokenBucket(r, capacity=20, rate=0.5)
        await bucket.consume("alice")
        # evalsha(sha, numkeys, key, capacity, rate, tokens, now)
        key_arg = r.evalsha.call_args[0][2]
        assert key_arg == _KEY_PREFIX + "alice"

    async def test_custom_prefix_applied(self) -> None:
        r = _redis()
        bucket = TokenBucket(r, capacity=20, rate=0.5, key_prefix="custom:")
        await bucket.consume("bob")
        key_arg = r.evalsha.call_args[0][2]
        assert key_arg == "custom:bob"

    async def test_capacity_passed_to_script(self) -> None:
        r = _redis()
        bucket = TokenBucket(r, capacity=42.0, rate=0.5)
        await bucket.consume("user-1")
        assert r.evalsha.call_args[0][3] == 42.0

    async def test_rate_passed_to_script(self) -> None:
        r = _redis()
        bucket = TokenBucket(r, capacity=20, rate=1.5)
        await bucket.consume("user-1")
        assert r.evalsha.call_args[0][4] == 1.5

    async def test_custom_tokens_passed_to_script(self) -> None:
        r = _redis()
        bucket = TokenBucket(r, capacity=20, rate=0.5)
        await bucket.consume("user-1", tokens=3.0)
        assert r.evalsha.call_args[0][5] == 3.0


# ---------------------------------------------------------------------------
# SHA caching
# ---------------------------------------------------------------------------


class TestShaCaching:
    async def test_script_loaded_once_across_multiple_calls(self) -> None:
        r = _redis()
        bucket = _bucket(r)
        await bucket.consume("user-1")
        await bucket.consume("user-2")
        await bucket.consume("user-3")
        r.script_load.assert_called_once()

    async def test_sha_stored_after_first_call(self) -> None:
        r = _redis(script_load_return="deadbeef")
        bucket = _bucket(r)
        assert bucket._script_sha is None
        await bucket.consume("user-1")
        assert bucket._script_sha == "deadbeef"


# ---------------------------------------------------------------------------
# Fail-open behaviour
# ---------------------------------------------------------------------------


class TestFailOpen:
    async def test_evalsha_error_returns_allowed(self) -> None:
        r = AsyncMock()
        r.script_load.return_value = "sha"
        r.evalsha.side_effect = RuntimeError("connection refused")
        bucket = _bucket(r)
        allowed, remaining, retry_after = await bucket.consume("user-1")
        assert allowed is True
        assert retry_after == 0.0

    async def test_evalsha_error_remaining_equals_capacity(self) -> None:
        r = AsyncMock()
        r.script_load.return_value = "sha"
        r.evalsha.side_effect = RuntimeError("timeout")
        bucket = TokenBucket(r, capacity=50.0, rate=1.0)
        _, remaining, _ = await bucket.consume("user-1")
        assert remaining == 50.0

    async def test_script_load_error_returns_allowed(self) -> None:
        r = AsyncMock()
        r.script_load.side_effect = ConnectionError("redis down")
        bucket = _bucket(r)
        allowed, remaining, retry_after = await bucket.consume("user-1")
        assert allowed is True
        assert remaining == 20.0
        assert retry_after == 0.0
