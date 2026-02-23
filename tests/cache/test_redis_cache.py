"""Tests for RedisCache backend (cache/redis_cache.py)."""

import json
import time
from unittest.mock import AsyncMock

from llmgateway.cache.base import CacheEntry
from llmgateway.cache.redis_cache import RedisCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_redis() -> AsyncMock:
    return AsyncMock()


def _make_entry(key: str = "test-key", ttl: int = 3600) -> CacheEntry:
    return CacheEntry(
        key=key,
        value='{"answer": 42}',
        created_at=time.time(),
        ttl=ttl,
        metadata={"model": "gpt-4o"},
    )


def _serialise(entry: CacheEntry) -> str:
    return json.dumps(
        {
            "key": entry.key,
            "value": entry.value,
            "created_at": entry.created_at,
            "ttl": entry.ttl,
            "metadata": entry.metadata,
        }
    )


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestRedisCacheGet:
    async def test_returns_none_on_miss(self) -> None:
        redis = _mock_redis()
        redis.get.return_value = None
        result = await RedisCache(redis).get("test-key")
        assert result is None

    async def test_uses_key_prefix(self) -> None:
        redis = _mock_redis()
        redis.get.return_value = None
        await RedisCache(redis).get("mykey")
        redis.get.assert_called_once_with("llmgw:cache:mykey")

    async def test_returns_entry_on_hit(self) -> None:
        entry = _make_entry()
        redis = _mock_redis()
        redis.get.return_value = _serialise(entry)
        result = await RedisCache(redis).get("test-key")
        assert result is not None
        assert result.key == "test-key"
        assert result.ttl == 3600

    async def test_returns_none_on_redis_error(self) -> None:
        redis = _mock_redis()
        redis.get.side_effect = ConnectionError("Redis down")
        result = await RedisCache(redis).get("test-key")
        assert result is None

    async def test_returns_none_on_corrupt_json(self) -> None:
        redis = _mock_redis()
        redis.get.return_value = "not-valid-json}}}"
        result = await RedisCache(redis).get("test-key")
        assert result is None

    async def test_returns_none_on_missing_fields(self) -> None:
        redis = _mock_redis()
        redis.get.return_value = json.dumps({"key": "k"})  # missing value/ttl etc.
        result = await RedisCache(redis).get("test-key")
        assert result is None


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------


class TestRedisCacheSet:
    async def test_stores_entry_with_prefixed_key(self) -> None:
        redis = _mock_redis()
        entry = _make_entry(key="mykey")
        await RedisCache(redis).set(entry)
        call_key = redis.set.call_args.args[0]
        assert call_key == "llmgw:cache:mykey"

    async def test_stores_entry_with_ttl(self) -> None:
        redis = _mock_redis()
        entry = _make_entry(ttl=120)
        await RedisCache(redis).set(entry)
        assert redis.set.call_args.kwargs.get("ex") == 120

    async def test_swallows_redis_error(self) -> None:
        redis = _mock_redis()
        redis.set.side_effect = ConnectionError("Redis down")
        await RedisCache(redis).set(_make_entry())  # should not raise


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestRedisCacheDelete:
    async def test_deletes_with_prefixed_key(self) -> None:
        redis = _mock_redis()
        await RedisCache(redis).delete("mykey")
        redis.delete.assert_called_once_with("llmgw:cache:mykey")

    async def test_swallows_redis_error(self) -> None:
        redis = _mock_redis()
        redis.delete.side_effect = ConnectionError()
        await RedisCache(redis).delete("mykey")  # should not raise


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestRedisCacheExists:
    async def test_returns_true_when_key_exists(self) -> None:
        redis = _mock_redis()
        redis.exists.return_value = 1
        assert await RedisCache(redis).exists("test-key") is True

    async def test_returns_false_when_key_absent(self) -> None:
        redis = _mock_redis()
        redis.exists.return_value = 0
        assert await RedisCache(redis).exists("test-key") is False

    async def test_returns_false_on_redis_error(self) -> None:
        redis = _mock_redis()
        redis.exists.side_effect = ConnectionError()
        assert await RedisCache(redis).exists("test-key") is False
