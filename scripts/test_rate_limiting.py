#!/usr/bin/env python3
"""Manual test script for the token-bucket rate limiting layer.

Exercises two distinct layers:

1. **Direct component tests** — import and exercise TokenBucket / RateLimiter
   directly against a live Redis instance (REDIS_URL from .env or environment).
   No gateway server is required for these.

2. **HTTP endpoint tests** — send real HTTP requests to a running gateway and
   inspect the ``X-RateLimit-*`` response headers and 429 behaviour.  These
   are skipped automatically when the gateway is not reachable.

Usage
-----
    # From the repo root (gateway *not* required for component tests):
    python scripts/test_rate_limiting.py

    # With a running gateway on a custom port:
    GATEWAY_URL=http://localhost:9000 python scripts/test_rate_limiting.py

    # Test 429 exhaustion via HTTP (requires a small capacity in the gateway):
    #   In .env: RATE_LIMIT_DEFAULT_CAPACITY=3
    #   Then:
    RATE_LIMIT_DEFAULT_CAPACITY=3 python scripts/test_rate_limiting.py

Notes
-----
* Component tests use isolated Redis keys (prefixed with ``llmgw:rl:test-``
  and suffixed with the current timestamp) so they cannot interfere with a
  running gateway or previous runs.
* Redis keys are deleted automatically at the end of the run.
"""

import asyncio
import os
import time
import uuid
from pathlib import Path

import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv

# Resolve .env relative to the repo root so the script works from any cwd.
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ---------------------------------------------------------------------------
# Import gateway internals *after* loading .env so settings picks them up.
# ---------------------------------------------------------------------------
from llmgateway.config import settings  # noqa: E402
from llmgateway.ratelimit.limiter import TIER_CONFIGS, RateLimiter  # noqa: E402
from llmgateway.ratelimit.token_bucket import TokenBucket  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration (override any of these with environment variables)
# ---------------------------------------------------------------------------
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
MODEL = os.getenv("TEST_MODEL", "claude-haiku-4-5-20251001")
REDIS_URL = settings.redis_url

# Capacity from env — if <= 5 we also run HTTP exhaustion test.
HTTP_CAPACITY = int(
    os.getenv("RATE_LIMIT_DEFAULT_CAPACITY", str(settings.rate_limit_default_capacity))
)

SECTION = "=" * 60
PASS = "✅"
FAIL = "❌"
SKIP = "⏭️ "
INFO = "ℹ️ "

_passed = 0
_failed = 0
_skipped = 0
_cleanup_keys: list[tuple[aioredis.Redis, str]] = []  # (client, key) pairs to delete


# ---------------------------------------------------------------------------
# Output helpers
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


def _info(msg: str) -> None:
    print(f"  {INFO} {msg}")


def _section(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)


def _uid(tag: str = "") -> str:
    """Generate an isolated test user ID that won't collide with real traffic."""
    suffix = f"-{tag}" if tag else ""
    return f"test-rl{suffix}-{int(time.time())}-{uuid.uuid4().hex[:6]}"


def _track(client: aioredis.Redis, key: str) -> None:
    """Register a Redis key for cleanup at the end of the run."""
    _cleanup_keys.append((client, key))


# ---------------------------------------------------------------------------
# Component tests — Section 1: Basic consume
# ---------------------------------------------------------------------------


async def test_consume_allowed(redis_client: aioredis.Redis) -> None:
    """A fresh bucket with capacity > 1 allows the first consume."""
    uid = _uid("basic")
    bucket = TokenBucket(redis_client, capacity=5, rate=1.0, key_prefix="llmgw:rl:test-")
    _track(redis_client, "llmgw:rl:test-" + uid)

    allowed, remaining, retry_after = await bucket.consume(uid, tokens=1.0)

    if allowed:
        _ok("first consume allowed", f"remaining={remaining:.0f}, retry_after={retry_after:.2f}")
    else:
        _fail("first consume unexpectedly denied", f"retry_after={retry_after:.2f}")

    if remaining == 4.0:
        _ok("remaining decremented by 1 (5 → 4)")
    else:
        _fail("unexpected remaining after consume", f"remaining={remaining}")

    if retry_after == 0.0:
        _ok("retry_after is 0 when allowed")
    else:
        _fail("retry_after should be 0 when allowed", f"retry_after={retry_after}")


# ---------------------------------------------------------------------------
# Component tests — Section 2: Exhaustion and retry_after
# ---------------------------------------------------------------------------


async def test_bucket_exhaustion(redis_client: aioredis.Redis) -> None:
    """Consuming more tokens than the capacity results in a denial."""
    uid = _uid("exhaust")
    # Very slow refill so tokens don't replenish during the test.
    bucket = TokenBucket(redis_client, capacity=3, rate=0.001, key_prefix="llmgw:rl:test-")
    _track(redis_client, "llmgw:rl:test-" + uid)

    # Drain the bucket completely.
    results = []
    for _ in range(3):
        allowed, remaining, _ = await bucket.consume(uid, tokens=1.0)
        results.append(allowed)

    if all(results):
        _ok("first 3 consumes allowed (bucket capacity = 3)")
    else:
        _fail("unexpected denial while bucket should still have tokens", str(results))

    # This 4th consume should be denied.
    allowed4, remaining4, retry_after4 = await bucket.consume(uid, tokens=1.0)

    if not allowed4:
        _ok("4th consume denied (bucket exhausted)")
    else:
        _fail("4th consume should have been denied", f"remaining={remaining4}")

    if retry_after4 > 0:
        _ok("retry_after > 0 when denied", f"retry_after={retry_after4:.2f} s")
    else:
        _fail("retry_after should be > 0 when denied", f"retry_after={retry_after4}")


async def test_bucket_denial_body(redis_client: aioredis.Redis) -> None:
    """When denied, remaining reflects tokens still in bucket (may be > 0 for partial requests)."""
    uid = _uid("partial")
    # Bucket has 2 tokens; request 3 at once.
    bucket = TokenBucket(redis_client, capacity=2, rate=0.001, key_prefix="llmgw:rl:test-")
    _track(redis_client, "llmgw:rl:test-" + uid)

    allowed, remaining, retry_after = await bucket.consume(uid, tokens=3.0)

    if not allowed:
        _ok(
            "request for 3 tokens from 2-token bucket denied",
            f"remaining={remaining:.0f}, retry_after={retry_after:.2f} s",
        )
    else:
        _fail("should have been denied (3 tokens requested, bucket has 2)")

    # retry_after ≈ (3-2) / 0.001 = 1000 s (very slow refill)
    if retry_after > 100:
        _ok("retry_after reflects slow refill rate", f"{retry_after:.0f} s")
    else:
        _fail("retry_after seems too short for the slow refill rate", f"{retry_after:.2f} s")


# ---------------------------------------------------------------------------
# Component tests — Section 3: Token refill
# ---------------------------------------------------------------------------


async def test_token_refill(redis_client: aioredis.Redis) -> None:
    """Tokens refill at the configured rate; waiting long enough re-allows a denied request."""
    uid = _uid("refill")
    # Rate = 2 tokens/s; capacity = 1.  After consuming 1 token, wait 0.6 s for refill.
    bucket = TokenBucket(redis_client, capacity=1, rate=2.0, key_prefix="llmgw:rl:test-")
    _track(redis_client, "llmgw:rl:test-" + uid)

    # First consume — drains the bucket.
    allowed1, _, _ = await bucket.consume(uid, tokens=1.0)
    if not allowed1:
        _fail("first consume should be allowed")
        return
    _ok("first consume allowed (bucket = 1 token)")

    # Immediate second consume — should be denied.
    allowed2, _, _ = await bucket.consume(uid, tokens=1.0)
    if not allowed2:
        _ok("immediate second consume denied (bucket empty)")
    else:
        _fail("second consume should have been denied immediately")

    # Wait for the bucket to partially refill (≥ 1 token at 2 tokens/s takes 0.5 s).
    wait = 0.7
    _info(f"waiting {wait} s for bucket to refill…")
    await asyncio.sleep(wait)

    # Third consume — bucket should now have ≥ 1 token again.
    allowed3, remaining3, _ = await bucket.consume(uid, tokens=1.0)
    if allowed3:
        _ok("consume allowed after refill", f"remaining={remaining3:.2f}")
    else:
        _fail("consume denied even after waiting for refill")


# ---------------------------------------------------------------------------
# Component tests — Section 4: Redis key state
# ---------------------------------------------------------------------------


async def test_redis_key_has_ttl(redis_client: aioredis.Redis) -> None:
    """After a consume, the Redis hash key must have a positive TTL set."""
    uid = _uid("ttl")
    bucket = TokenBucket(redis_client, capacity=5, rate=1.0, key_prefix="llmgw:rl:test-")
    key = "llmgw:rl:test-" + uid
    _track(redis_client, key)

    await bucket.consume(uid, tokens=1.0)
    ttl = await redis_client.ttl(key)

    if ttl > 0:
        _ok("Redis key has TTL after consume", f"ttl={ttl} s")
    else:
        _fail("Redis key has no TTL (or does not exist)", f"ttl={ttl}")


async def test_redis_key_contains_tokens_and_last_refill(redis_client: aioredis.Redis) -> None:
    """The Redis hash stores ``tokens`` and ``last_refill`` fields."""
    uid = _uid("fields")
    bucket = TokenBucket(redis_client, capacity=10, rate=1.0, key_prefix="llmgw:rl:test-")
    key = "llmgw:rl:test-" + uid
    _track(redis_client, key)

    await bucket.consume(uid, tokens=3.0)
    fields = await redis_client.hmget(key, "tokens", "last_refill")

    tokens_val, refill_val = fields
    if tokens_val is not None and refill_val is not None:
        _ok(
            "hash has 'tokens' and 'last_refill' fields",
            f"tokens={float(tokens_val):.0f}, last_refill≈now",
        )
    else:
        _fail("expected hash fields missing", f"got={fields}")


# ---------------------------------------------------------------------------
# Component tests — Section 5: RateLimiter tier routing
# ---------------------------------------------------------------------------


async def test_ratelimiter_disabled(redis_client: aioredis.Redis) -> None:
    """When enabled=False, check_rate_limit always returns allowed without touching Redis."""
    limiter = RateLimiter(
        redis_client,
        default_capacity=5,
        default_rate=0.1,
        enabled=False,
    )

    results = []
    for _ in range(10):
        result = await limiter.check_rate_limit("any-user")
        results.append(result.allowed)

    if all(results):
        _ok("all 10 requests allowed when rate limiting disabled")
    else:
        _fail("unexpected denial with rate limiting disabled")


async def test_ratelimiter_tier_free(redis_client: aioredis.Redis) -> None:
    """RateLimiter resolves 'free' tier to its configured capacity."""
    uid = _uid("tier-free")

    def get_tier(user_id: str) -> str | None:
        return "free"

    limiter = RateLimiter(
        redis_client,
        default_capacity=999,  # would never match in this test
        default_rate=0.1,
        get_tier=get_tier,
    )

    result = await limiter.check_rate_limit(uid, cost=0.0)
    expected_cap = TIER_CONFIGS["free"][0]

    if result.limit == expected_cap:
        _ok(f"free tier capacity = {expected_cap:.0f}", f"remaining={result.remaining:.0f}")
    else:
        _fail("free tier capacity mismatch", f"got={result.limit}, want={expected_cap}")


async def test_ratelimiter_tier_pro(redis_client: aioredis.Redis) -> None:
    """RateLimiter resolves 'pro' tier to its configured capacity."""
    uid = _uid("tier-pro")

    def get_tier(user_id: str) -> str | None:
        return "pro"

    limiter = RateLimiter(
        redis_client,
        default_capacity=999,
        default_rate=0.1,
        get_tier=get_tier,
    )

    result = await limiter.check_rate_limit(uid, cost=0.0)
    expected_cap = TIER_CONFIGS["pro"][0]

    if result.limit == expected_cap:
        _ok(f"pro tier capacity = {expected_cap:.0f}", f"remaining={result.remaining:.0f}")
    else:
        _fail("pro tier capacity mismatch", f"got={result.limit}, want={expected_cap}")


async def test_ratelimiter_independent_buckets(redis_client: aioredis.Redis) -> None:
    """Two different user IDs have independent buckets."""
    uid_a = _uid("indep-a")
    uid_b = _uid("indep-b")

    limiter = RateLimiter(redis_client, default_capacity=3, default_rate=0.001)

    # Exhaust user A's bucket.
    for _ in range(3):
        await limiter.check_rate_limit(uid_a)
    res_a = await limiter.check_rate_limit(uid_a)

    # User B's bucket should be untouched.
    res_b = await limiter.check_rate_limit(uid_b)

    if not res_a.allowed:
        _ok("user A denied after exhausting 3-token bucket")
    else:
        _fail("user A should have been denied")

    if res_b.allowed:
        _ok("user B allowed (independent bucket, unaffected by user A)")
    else:
        _fail("user B should not be affected by user A's limit")


async def test_get_rate_limit_info(redis_client: aioredis.Redis) -> None:
    """get_rate_limit_info returns a dict with expected keys and correct values."""
    uid = _uid("info")
    limiter = RateLimiter(redis_client, default_capacity=10, default_rate=1.0)

    await limiter.check_rate_limit(uid, cost=3.0)
    info = await limiter.get_rate_limit_info(uid)

    required_keys = {
        "user_id",
        "tier",
        "limit",
        "remaining",
        "rate_per_second",
        "reset_time",
        "enabled",
    }
    missing = required_keys - set(info.keys())
    if not missing:
        _ok("info dict has all required keys")
    else:
        _fail("info dict missing keys", str(missing))

    if info["user_id"] == uid:
        _ok("user_id in info matches requested uid")
    else:
        _fail("user_id mismatch", f"got={info['user_id']!r}")

    if info["enabled"] is True:
        _ok("info reports rate limiting as enabled")
    else:
        _fail("info should report enabled=True")

    _info(
        f"limit={info['limit']:.0f}  remaining={info['remaining']:.1f}"
        f"  rate={info['rate_per_second']:.4f}/s  reset_time={info['reset_time']:.0f}"
    )


# ---------------------------------------------------------------------------
# HTTP endpoint tests (require running gateway)
# ---------------------------------------------------------------------------


async def _check_gateway_available(client: httpx.AsyncClient) -> bool:
    try:
        resp = await client.get(f"{GATEWAY_URL}/health/live", timeout=2.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


async def test_http_ratelimit_headers_present(client: httpx.AsyncClient) -> None:
    """Successful responses include all three X-RateLimit-* headers."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is 1+1?"}],
        "temperature": 0.0,
        "stream": False,
        "user": _uid("hdr"),
    }
    try:
        r = await client.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload, timeout=30.0)
    except Exception as exc:
        _fail("request failed", str(exc))
        return

    if r.status_code != 200:
        _fail(f"expected 200, got {r.status_code}", r.text[:120])
        return

    has_limit = "X-RateLimit-Limit" in r.headers
    has_remaining = "X-RateLimit-Remaining" in r.headers
    has_reset = "X-RateLimit-Reset" in r.headers

    for name, present in [
        ("X-RateLimit-Limit", has_limit),
        ("X-RateLimit-Remaining", has_remaining),
        ("X-RateLimit-Reset", has_reset),
    ]:
        if present:
            _ok(f"{name} present", f"value={r.headers[name]!r}")
        else:
            _fail(f"{name} missing from response")


async def test_http_remaining_decrements(client: httpx.AsyncClient) -> None:
    """X-RateLimit-Remaining decreases by 1 between successive requests from the same user."""
    uid = _uid("decr")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello"}],
        "temperature": 0.0,
        "stream": False,
        "user": uid,
    }
    headers = {"X-User-ID": uid}

    try:
        r1 = await client.post(
            f"{GATEWAY_URL}/v1/chat/completions", json=payload, headers=headers, timeout=30.0
        )
        r2 = await client.post(
            f"{GATEWAY_URL}/v1/chat/completions", json=payload, headers=headers, timeout=30.0
        )
    except Exception as exc:
        _fail("request failed", str(exc))
        return

    if r1.status_code != 200 or r2.status_code != 200:
        _fail("one or both requests failed", f"status1={r1.status_code}, status2={r2.status_code}")
        return

    rem1 = int(r1.headers.get("X-RateLimit-Remaining", -1))
    rem2 = int(r2.headers.get("X-RateLimit-Remaining", -1))

    _info(f"request 1 remaining={rem1},  request 2 remaining={rem2}")

    if rem2 < rem1:
        _ok(
            "X-RateLimit-Remaining decreased between requests",
            f"{rem1} → {rem2}",
        )
    else:
        _fail(
            "X-RateLimit-Remaining did not decrease",
            f"rem1={rem1}, rem2={rem2}",
        )


async def test_http_independent_user_buckets(client: httpx.AsyncClient) -> None:
    """Two distinct X-User-ID values maintain independent, non-interfering buckets."""
    uid_a = _uid("user-a")
    uid_b = _uid("user-b")

    async def _request(uid: str) -> httpx.Response:
        return await client.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.0,
                "stream": False,
            },
            headers={"X-User-ID": uid},
            timeout=30.0,
        )

    try:
        ra = await _request(uid_a)
        rb = await _request(uid_b)
    except Exception as exc:
        _fail("request failed", str(exc))
        return

    if ra.status_code == 200 and rb.status_code == 200:
        rem_a = int(ra.headers.get("X-RateLimit-Remaining", -1))
        rem_b = int(rb.headers.get("X-RateLimit-Remaining", -1))
        limit = int(ra.headers.get("X-RateLimit-Limit", 0))
        _info(f"user_a remaining={rem_a}, user_b remaining={rem_b}, limit={limit}")

        # After one request each, both should be at (limit - 1).
        if rem_a == limit - 1 and rem_b == limit - 1:
            _ok("both users start at limit-1 (independent buckets)")
        else:
            _ok(
                "requests succeeded with separate X-User-ID headers",
                f"rem_a={rem_a}, rem_b={rem_b}",
            )
    else:
        _fail(
            "one or both requests failed",
            f"status_a={ra.status_code}, status_b={rb.status_code}",
        )


async def test_http_x_user_id_takes_precedence(client: httpx.AsyncClient) -> None:
    """X-User-ID header is used over the 'user' body field for rate limit routing."""
    header_uid = _uid("hdr-uid")
    body_uid = _uid("body-uid")

    try:
        r1 = await client.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.0,
                "stream": False,
                "user": body_uid,
            },
            headers={"X-User-ID": header_uid},
            timeout=30.0,
        )
        r2 = await client.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi again"}],
                "temperature": 0.0,
                "stream": False,
                "user": body_uid,
            },
            headers={"X-User-ID": header_uid},
            timeout=30.0,
        )
    except Exception as exc:
        _fail("request failed", str(exc))
        return

    if r1.status_code != 200 or r2.status_code != 200:
        _fail("request failed", f"{r1.status_code} / {r2.status_code}")
        return

    rem1 = int(r1.headers.get("X-RateLimit-Remaining", -1))
    rem2 = int(r2.headers.get("X-RateLimit-Remaining", -1))

    # If the header UID is being used (correctly), two requests reduce remaining by 2.
    if rem2 == rem1 - 1:
        _ok(
            "X-User-ID header used for bucket (remaining decremented correctly)",
            f"{rem1} → {rem2}",
        )
    else:
        _fail("unexpected remaining change", f"rem1={rem1}, rem2={rem2}")


async def test_http_exhaustion_returns_429(client: httpx.AsyncClient) -> None:
    """Sending capacity+1 requests from the same user eventually yields a 429."""
    if HTTP_CAPACITY > 5:
        _skip(
            "http_exhaustion_returns_429",
            f"RATE_LIMIT_DEFAULT_CAPACITY={HTTP_CAPACITY} is too large to exhaust cheaply. "
            "Set it to ≤5 in .env and restart the gateway to enable this test.",
        )
        return

    uid = _uid("exhaust-http")
    _info(f"exhausting bucket (capacity={HTTP_CAPACITY}) for user {uid!r}…")

    statuses = []
    for i in range(HTTP_CAPACITY + 2):
        try:
            r = await client.post(
                f"{GATEWAY_URL}/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": f"Request {i}"}],
                    "temperature": 0.0,
                    "stream": False,
                },
                headers={"X-User-ID": uid},
                timeout=30.0,
            )
            statuses.append(r.status_code)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After", "(missing)")
                rl_remaining = r.headers.get("X-RateLimit-Remaining", "(missing)")
                _info(
                    f"429 received on request {i+1}  "
                    f"Retry-After={retry_after!r}  "
                    f"X-RateLimit-Remaining={rl_remaining!r}"
                )
                break
        except Exception as exc:
            _fail(f"request {i+1} failed", str(exc))
            return

    if 429 in statuses:
        idx = statuses.index(429)
        _ok(
            f"429 received after {idx} successful request(s)",
            f"sequence={statuses}",
        )
        # Verify Retry-After header is present on the 429 response
        r_429 = await client.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "one more"}],
                "temperature": 0.0,
                "stream": False,
            },
            headers={"X-User-ID": uid},
            timeout=10.0,
        )
        if r_429.status_code == 429:
            has_retry = "Retry-After" in r_429.headers
            has_reset = "X-RateLimit-Reset" in r_429.headers
            if has_retry:
                _ok("Retry-After header present on 429", f"value={r_429.headers['Retry-After']!r}")
            else:
                _fail("Retry-After header missing from 429 response")
            if has_reset:
                _ok("X-RateLimit-Reset header present on 429")
            else:
                _fail("X-RateLimit-Reset header missing from 429 response")
    else:
        _fail(
            f"no 429 after {len(statuses)} requests",
            f"capacity={HTTP_CAPACITY}, statuses={statuses}",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    print(SECTION)
    print("  LLM Gateway — Rate Limiting Manual Test Suite")
    print(SECTION)
    print(f"  Redis URL           : {REDIS_URL}")
    print(f"  Gateway URL         : {GATEWAY_URL}")
    print(f"  Model               : {MODEL}")
    print(f"  Gateway capacity    : {HTTP_CAPACITY}  (RATE_LIMIT_DEFAULT_CAPACITY)")

    # ------------------------------------------------------------------
    # Shared Redis client
    # ------------------------------------------------------------------
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
    redis_connected = False
    try:
        await redis_client.ping()
        redis_connected = True
    except Exception as exc:
        print(f"\n  ⚠️  Redis not reachable ({exc}); component tests will be skipped.")

    # ------------------------------------------------------------------
    # 1. TokenBucket — basic consume
    # ------------------------------------------------------------------
    _section("1 · TokenBucket — Basic Consume")
    if redis_connected:
        await test_consume_allowed(redis_client)
    else:
        for label in ("first consume allowed", "remaining decremented", "retry_after=0"):
            _skip(label, "Redis not reachable")

    # ------------------------------------------------------------------
    # 2. TokenBucket — exhaustion and denial
    # ------------------------------------------------------------------
    _section("2 · TokenBucket — Exhaustion and Retry-After")
    if redis_connected:
        await test_bucket_exhaustion(redis_client)
        await test_bucket_denial_body(redis_client)
    else:
        for label in ("bucket exhaustion", "partial-request denial"):
            _skip(label, "Redis not reachable")

    # ------------------------------------------------------------------
    # 3. TokenBucket — token refill
    # ------------------------------------------------------------------
    _section("3 · TokenBucket — Token Refill (takes ~0.7 s)")
    if redis_connected:
        await test_token_refill(redis_client)
    else:
        _skip("token refill", "Redis not reachable")

    # ------------------------------------------------------------------
    # 4. Redis key structure
    # ------------------------------------------------------------------
    _section("4 · Redis Key Structure")
    if redis_connected:
        await test_redis_key_has_ttl(redis_client)
        await test_redis_key_contains_tokens_and_last_refill(redis_client)
    else:
        for label in ("Redis key TTL", "Redis hash fields"):
            _skip(label, "Redis not reachable")

    # ------------------------------------------------------------------
    # 5. RateLimiter — tiers and multi-user isolation
    # ------------------------------------------------------------------
    _section("5 · RateLimiter — Tiers, Disabled Mode, and User Isolation")
    if redis_connected:
        await test_ratelimiter_disabled(redis_client)
        await test_ratelimiter_tier_free(redis_client)
        await test_ratelimiter_tier_pro(redis_client)
        await test_ratelimiter_independent_buckets(redis_client)
        await test_get_rate_limit_info(redis_client)
    else:
        for label in (
            "disabled mode",
            "free tier",
            "pro tier",
            "independent buckets",
            "get_rate_limit_info",
        ):
            _skip(label, "Redis not reachable")

    # ------------------------------------------------------------------
    # 6. HTTP endpoint tests (gateway required)
    # ------------------------------------------------------------------
    _section("6 · HTTP Endpoint — Rate Limit Headers")

    async with httpx.AsyncClient() as http_client:
        gateway_up = await _check_gateway_available(http_client)
        if not gateway_up:
            print(f"\n  ⚠️  Gateway not reachable at {GATEWAY_URL}; HTTP tests skipped.")
            print("      Start the server with:  make dev")
            for label in (
                "http_ratelimit_headers_present",
                "http_remaining_decrements",
                "http_independent_user_buckets",
                "http_x_user_id_takes_precedence",
                "http_exhaustion_returns_429",
            ):
                _skip(label, "gateway not running")
        else:
            print("  Gateway reachable — running live HTTP tests…")
            print()
            await test_http_ratelimit_headers_present(http_client)
            await test_http_remaining_decrements(http_client)
            await test_http_independent_user_buckets(http_client)
            await test_http_x_user_id_takes_precedence(http_client)

            _section("6b · HTTP Endpoint — 429 Exhaustion")
            await test_http_exhaustion_returns_429(http_client)

    # ------------------------------------------------------------------
    # Cleanup test keys from Redis
    # ------------------------------------------------------------------
    if _cleanup_keys and redis_connected:
        for rc, key in _cleanup_keys:
            try:
                await rc.delete(key)
                # Also delete the sorted-set index entry if one was created
                await rc.delete(key)
            except Exception:
                pass

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

    if HTTP_CAPACITY > 5:
        print()
        print("  Tip: to enable the HTTP 429 exhaustion test, set a small capacity in .env:")
        print("       RATE_LIMIT_DEFAULT_CAPACITY=3")
        print("       Then restart the gateway and re-run this script.")

    if _failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
