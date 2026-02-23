#!/usr/bin/env python3
"""Manual test script for the Redis exact-match caching layer.

Exercises two distinct layers:

1. **Direct component tests** — import and exercise CacheManager / RedisCache /
   CacheEntry without starting the full gateway server.  These run against a
   live Redis instance (REDIS_URL from .env or environment).

2. **HTTP endpoint tests** — send real HTTP requests to a running gateway and
   inspect the ``X-Cache-Status`` response header.  These are skipped
   automatically when the gateway is not reachable.

Usage
-----
    # From the repo root (gateway *not* required for component tests):
    python scripts/test_cache.py

    # With a running gateway on a custom port:
    GATEWAY_URL=http://localhost:9000 python scripts/test_cache.py
"""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv

# Resolve .env relative to the repo root so the script works from any cwd.
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ---------------------------------------------------------------------------
# Import gateway internals *after* loading .env so settings picks them up.
# ---------------------------------------------------------------------------
from llmgateway.cache.base import CacheEntry  # noqa: E402
from llmgateway.cache.cache_manager import CacheManager  # noqa: E402
from llmgateway.cache.redis_cache import RedisCache  # noqa: E402
from llmgateway.config import settings  # noqa: E402
from llmgateway.providers.models import CompletionRequest  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
MODEL = os.getenv("TEST_MODEL", "claude-haiku-4-5-20251001")
REDIS_URL = settings.redis_url

SECTION = "=" * 60
PASS = "✅"
FAIL = "❌"
SKIP = "⏭️ "

_passed = 0
_failed = 0
_skipped = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(label: str, detail: str = "") -> None:
    global _passed
    _passed += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  {PASS} {label}{suffix}")


def _fail(label: str, detail: str = "") -> None:
    global _failed
    _failed += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  {FAIL} {label}{suffix}")


def _skip(label: str, reason: str = "") -> None:
    global _skipped
    _skipped += 1
    suffix = f"  ({reason})" if reason else ""
    print(f"  {SKIP} {label}{suffix}")


def _section(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)


def _make_request(temperature: float = 0.0, content: str = "What is 2+2?") -> CompletionRequest:
    return CompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        stream=False,
    )


def _dummy_response(content: str = "4") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }


# ---------------------------------------------------------------------------
# Component tests (no gateway server required)
# ---------------------------------------------------------------------------


async def test_cache_key_stability(manager: CacheManager) -> None:
    """The same logical request must always produce the same key."""
    req = _make_request()
    key1 = manager.generate_cache_key(req.model, req.messages, req.temperature, req.max_tokens)
    key2 = manager.generate_cache_key(req.model, req.messages, req.temperature, req.max_tokens)

    if key1 == key2 and len(key1) == 64:
        _ok("cache key is stable across calls", f"sha256={key1[:16]}…")
    else:
        _fail("cache key is not stable", f"key1={key1!r}, key2={key2!r}")


async def test_cache_key_uniqueness(manager: CacheManager) -> None:
    """Different requests must produce different keys."""
    req_a = _make_request(content="What is 2+2?")
    req_b = _make_request(content="What is 3+3?")
    req_c = _make_request(temperature=0.0)
    req_d = _make_request(temperature=0.7)  # different temperature

    key_a = manager.generate_cache_key(
        req_a.model, req_a.messages, req_a.temperature, req_a.max_tokens
    )
    key_b = manager.generate_cache_key(
        req_b.model, req_b.messages, req_b.temperature, req_b.max_tokens
    )
    key_c = manager.generate_cache_key(
        req_c.model, req_c.messages, req_c.temperature, req_c.max_tokens
    )
    key_d = manager.generate_cache_key(
        req_d.model, req_d.messages, req_d.temperature, req_d.max_tokens
    )

    if key_a != key_b:
        _ok("different messages → different keys")
    else:
        _fail("different messages produced the same key")

    if key_c != key_d:
        _ok("different temperatures → different keys")
    else:
        _fail("different temperatures produced the same key")


async def test_temperature_gating(manager: CacheManager) -> None:
    """Requests with temperature != 0 must be skipped (return None)."""
    for temp in (0.1, 0.5, 0.7, 1.0, 2.0):
        req = _make_request(temperature=temp)
        result = await manager.get_cached_response(req)
        if result is None:
            _ok(f"temperature={temp} correctly skipped (not cached)")
        else:
            _fail(f"temperature={temp} should not have been cached", str(result))


async def test_cache_miss(manager: CacheManager) -> None:
    """A request not in cache returns None."""
    req = _make_request(content=f"unique-miss-{time.time()}")
    result = await manager.get_cached_response(req)
    if result is None:
        _ok("cache miss returns None")
    else:
        _fail("expected None on cache miss", str(result))


async def test_cache_store_and_hit(manager: CacheManager) -> None:
    """Storing a response then fetching it returns the original data."""
    content_text = f"test-content-{time.time()}"
    req = _make_request(content=content_text)
    resp = _dummy_response(content="4")

    # Should be a miss initially
    before = await manager.get_cached_response(req)
    if before is not None:
        _fail("expected miss before storing", str(before))
        return

    # Store
    await manager.cache_response(req, resp)

    # Should now be a hit
    after = await manager.get_cached_response(req)
    if after is not None and after.get("id") == resp["id"]:
        _ok("store then get returns original response")
    else:
        _fail("cache hit returned unexpected value", str(after))


async def test_cache_ttl(manager: CacheManager) -> None:
    """Responses stored with TTL=1 expire after one second."""
    content_text = f"ttl-test-{time.time()}"
    req = _make_request(content=content_text)
    resp = _dummy_response()

    await manager.cache_response(req, resp, ttl=1)

    hit = await manager.get_cached_response(req)
    if hit is not None:
        _ok("entry present immediately after store")
    else:
        _fail("entry missing immediately after store")
        return

    print("    (waiting 2 s for TTL expiry…)")
    await asyncio.sleep(2)

    expired = await manager.get_cached_response(req)
    if expired is None:
        _ok("entry expired after TTL elapsed")
    else:
        _fail("entry still present after TTL should have expired")


async def test_redis_entry_serialisation(redis_cache: RedisCache) -> None:
    """CacheEntry round-trips cleanly through Redis."""
    entry = CacheEntry(
        key=f"test-serialise-{time.time()}",
        value=json.dumps({"hello": "world", "n": 42}),
        ttl=60,
        metadata={"model": MODEL},
    )

    await redis_cache.set(entry)
    retrieved = await redis_cache.get(entry.key)

    if retrieved is None:
        _fail("entry not found after set")
        return

    payload = json.loads(retrieved.value)
    if payload == {"hello": "world", "n": 42} and retrieved.metadata == {"model": MODEL}:
        _ok("CacheEntry serialises and deserialises correctly")
    else:
        _fail("CacheEntry round-trip mismatch", f"got={payload}")

    await redis_cache.delete(entry.key)


async def test_redis_exists(redis_cache: RedisCache) -> None:
    """exists() returns True after set and False after delete."""
    key = f"test-exists-{time.time()}"
    entry = CacheEntry(key=key, value="{}", ttl=60)

    before = await redis_cache.exists(key)
    await redis_cache.set(entry)
    after_set = await redis_cache.exists(key)
    await redis_cache.delete(key)
    after_delete = await redis_cache.exists(key)

    if not before and after_set and not after_delete:
        _ok("exists() transitions correctly: False → True → False")
    else:
        _fail(
            "exists() gave unexpected results",
            f"before={before}, after_set={after_set}, after_delete={after_delete}",
        )


# ---------------------------------------------------------------------------
# HTTP endpoint tests (require a running gateway)
# ---------------------------------------------------------------------------


async def _check_gateway_available(client: httpx.AsyncClient) -> bool:
    try:
        resp = await client.get(f"{GATEWAY_URL}/health/live", timeout=2.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


async def test_http_cache_miss_then_hit(client: httpx.AsyncClient) -> None:
    """First request → MISS; identical request → HIT."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"http-cache-test-{time.time()}"}],
        "temperature": 0.0,
        "stream": False,
    }

    try:
        r1 = await client.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload, timeout=30.0)
    except Exception as exc:
        _fail("first request failed", str(exc))
        return

    if r1.status_code != 200:
        _fail(f"first request returned {r1.status_code}", r1.text[:120])
        return

    status1 = r1.headers.get("X-Cache-Status", "(missing)")
    if status1 == "MISS":
        _ok("first request → X-Cache-Status: MISS")
    else:
        _fail("expected MISS on first request", f"got={status1!r}")

    # Second identical request — should be served from cache
    try:
        r2 = await client.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload, timeout=10.0)
    except Exception as exc:
        _fail("second request failed", str(exc))
        return

    if r2.status_code != 200:
        _fail(f"second request returned {r2.status_code}", r2.text[:120])
        return

    status2 = r2.headers.get("X-Cache-Status", "(missing)")
    if status2 == "HIT":
        _ok("second request → X-Cache-Status: HIT")
    else:
        _fail("expected HIT on second request", f"got={status2!r}")

    # Bodies should be identical
    if r1.json()["choices"] == r2.json()["choices"]:
        _ok("cached response body matches original")
    else:
        _fail("cached body differs from original")


async def test_http_nonzero_temp_always_miss(client: httpx.AsyncClient) -> None:
    """Requests with temperature > 0 are never cached."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "temp-test"}],
        "temperature": 0.7,
        "stream": False,
    }

    for attempt in range(1, 3):
        try:
            r = await client.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload, timeout=30.0)
        except Exception as exc:
            _fail(f"request {attempt} failed", str(exc))
            return

        if r.status_code != 200:
            _fail(f"request {attempt} returned {r.status_code}")
            return

        status = r.headers.get("X-Cache-Status", "(missing)")
        if status == "MISS":
            _ok(f"temperature=0.7 request {attempt} → X-Cache-Status: MISS")
        else:
            _fail(f"temperature=0.7 request {attempt} should never be cached", f"got={status!r}")


async def test_http_streaming_always_miss(client: httpx.AsyncClient) -> None:
    """Streaming requests always return X-Cache-Status: MISS."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hi"}],
        "temperature": 0.0,
        "stream": True,
    }

    try:
        async with client.stream(
            "POST", f"{GATEWAY_URL}/v1/chat/completions", json=payload, timeout=30.0
        ) as r:
            status = r.headers.get("X-Cache-Status", "(missing)")
            # Drain the stream
            async for _ in r.aiter_lines():
                pass

        if status == "MISS":
            _ok("streaming request → X-Cache-Status: MISS")
        else:
            _fail("expected MISS for streaming request", f"got={status!r}")
    except Exception as exc:
        _fail("streaming request failed", str(exc))


async def test_http_response_header_present(client: httpx.AsyncClient) -> None:
    """X-Cache-Status header is always present in responses."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "header-check"}],
        "temperature": 0.0,
        "stream": False,
    }
    try:
        r = await client.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload, timeout=30.0)
        if "X-Cache-Status" in r.headers:
            _ok("X-Cache-Status header present", f"value={r.headers['X-Cache-Status']!r}")
        else:
            _fail("X-Cache-Status header missing from response")
    except Exception as exc:
        _fail("request failed", str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    print(SECTION)
    print("  LLM Gateway — Cache Manual Test Suite")
    print(SECTION)
    print(f"  Redis URL : {REDIS_URL}")
    print(f"  Gateway   : {GATEWAY_URL}")
    print(f"  Model     : {MODEL}")

    # ------------------------------------------------------------------
    # Build shared Redis client / cache objects
    # ------------------------------------------------------------------
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
    redis_connected = False
    try:
        await redis_client.ping()
        redis_connected = True
    except Exception as exc:
        print(f"\n  ⚠️  Redis not reachable ({exc}); component tests will be skipped.")

    redis_cache = RedisCache(redis_client)
    manager = CacheManager(backend=redis_cache, default_ttl=settings.cache_ttl)

    # ------------------------------------------------------------------
    # 1. Cache key tests (no Redis needed)
    # ------------------------------------------------------------------
    _section("1 · Cache Key Generation")
    await test_cache_key_stability(manager)
    await test_cache_key_uniqueness(manager)

    # ------------------------------------------------------------------
    # 2. Temperature gating (no Redis needed)
    # ------------------------------------------------------------------
    _section("2 · Temperature Gating (never cache temp != 0)")
    await test_temperature_gating(manager)

    # ------------------------------------------------------------------
    # 3. RedisCache + CacheManager (Redis required)
    # ------------------------------------------------------------------
    _section("3 · RedisCache Serialisation")
    if redis_connected:
        await test_redis_entry_serialisation(redis_cache)
        await test_redis_exists(redis_cache)
    else:
        _skip("redis_entry_serialisation", "Redis not reachable")
        _skip("redis_exists", "Redis not reachable")

    _section("4 · CacheManager Miss → Store → Hit")
    if redis_connected:
        await test_cache_miss(manager)
        await test_cache_store_and_hit(manager)
    else:
        _skip("cache_miss", "Redis not reachable")
        _skip("cache_store_and_hit", "Redis not reachable")

    _section("5 · TTL Expiry (takes ~2 s)")
    if redis_connected:
        await test_cache_ttl(manager)
    else:
        _skip("cache_ttl", "Redis not reachable")

    # ------------------------------------------------------------------
    # 4. HTTP endpoint tests (gateway required)
    # ------------------------------------------------------------------
    _section("6 · HTTP Endpoint — X-Cache-Status Header")

    async with httpx.AsyncClient() as http_client:
        gateway_up = await _check_gateway_available(http_client)
        if not gateway_up:
            print(f"\n  ⚠️  Gateway not reachable at {GATEWAY_URL}; HTTP tests skipped.")
            print("      Start the server with:  make dev")
            _skip("http_cache_miss_then_hit", "gateway not running")
            _skip("http_nonzero_temp_always_miss", "gateway not running")
            _skip("http_streaming_always_miss", "gateway not running")
            _skip("http_response_header_present", "gateway not running")
        else:
            print("  Gateway reachable — running live HTTP tests…")
            await test_http_response_header_present(http_client)
            await test_http_cache_miss_then_hit(http_client)
            await test_http_nonzero_temp_always_miss(http_client)
            await test_http_streaming_always_miss(http_client)

    await redis_client.aclose()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = _passed + _failed + _skipped
    print(f"\n{SECTION}")
    print(f"  Results: {_passed}/{total} passed, {_failed} failed, {_skipped} skipped")
    if _failed == 0:
        print(f"  {PASS} All executed tests passed!")
    else:
        print(f"  {FAIL} {_failed} test(s) failed — see output above.")
    print(SECTION)

    if _failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
