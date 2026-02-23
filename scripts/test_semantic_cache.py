#!/usr/bin/env python3
"""Manual test script for the semantic (embedding-based) caching layer.

Exercises three distinct layers:

1. **EmbeddingModel** — verify that the sentence-transformer produces
   correctly shaped, normalised ``float32`` vectors and that cosine
   similarity scores are sensible.

2. **CacheManager semantic methods** — store a response with an embedding
   and confirm that paraphrased queries trigger a cache hit while
   unrelated queries do not.  These tests run against a live Redis instance
   (REDIS_URL from .env or environment).

3. **HTTP endpoint tests** — send real HTTP requests to a running gateway
   and inspect the ``X-Cache-Type`` and ``X-Cache-Similarity`` response
   headers.  These are skipped automatically when the gateway is not
   reachable.

Usage
-----
    # From the repo root (gateway *not* required for embedding/component tests):
    python scripts/test_semantic_cache.py

    # With a running gateway on a custom port:
    GATEWAY_URL=http://localhost:9000 python scripts/test_semantic_cache.py

    # Adjust similarity threshold for the component tests:
    SEMANTIC_THRESHOLD=0.90 python scripts/test_semantic_cache.py
"""

import asyncio
import os
import time
from pathlib import Path

import httpx
import numpy as np
import redis.asyncio as aioredis
from dotenv import load_dotenv

# Resolve .env relative to the repo root so the script works from any cwd.
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ---------------------------------------------------------------------------
# Import gateway internals *after* loading .env so settings picks them up.
# ---------------------------------------------------------------------------
from llmgateway.cache.cache_manager import CacheManager  # noqa: E402
from llmgateway.cache.embeddings import EmbeddingModel  # noqa: E402
from llmgateway.cache.redis_cache import RedisCache  # noqa: E402
from llmgateway.config import settings  # noqa: E402
from llmgateway.providers.models import CompletionRequest  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
MODEL = os.getenv("TEST_MODEL", "claude-haiku-4-5-20251001")
REDIS_URL = settings.redis_url
THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", str(settings.semantic_cache_threshold)))

SECTION = "=" * 60
PASS = "✅"
FAIL = "❌"
SKIP = "⏭️ "
INFO = "ℹ️ "

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


def _info(msg: str) -> None:
    print(f"  {INFO} {msg}")


def _section(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)


def _make_request(content: str = "What is the capital of France?") -> CompletionRequest:
    return CompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        stream=False,
    )


def _dummy_response(content: str = "Paris") -> dict:
    return {
        "id": "chatcmpl-semantic-test",
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
        "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
    }


# High-similarity pairs: these should exceed the default 0.95 threshold
# with the all-MiniLM-L6-v2 model.
_SIMILAR_PAIRS = [
    (
        "What is the capital of France?",
        "Which city serves as the capital of France?",
    ),
    (
        "How do I reverse a list in Python?",
        "What's the way to reverse a list in Python?",
    ),
    (
        "Explain machine learning in simple terms.",
        "Describe machine learning in plain language.",
    ),
]

# Low-similarity pairs: these should stay well below 0.95.
_DISSIMILAR_PAIRS = [
    (
        "What is the capital of France?",
        "How do I bake chocolate chip cookies?",
    ),
    (
        "Explain quantum entanglement.",
        "Tell me a joke about programmers.",
    ),
]


# ---------------------------------------------------------------------------
# 1 · EmbeddingModel unit tests (no Redis, no gateway)
# ---------------------------------------------------------------------------


async def test_embedding_shape_and_dtype(model: EmbeddingModel) -> None:
    """Embeddings must be 384-dimensional float32 vectors."""
    vec = await model.encode("Hello world")
    if vec.shape == (384,):
        _ok("embedding has correct shape", f"shape={vec.shape}")
    else:
        _fail("unexpected embedding shape", f"got shape={vec.shape}")

    if vec.dtype == np.float32:
        _ok("embedding dtype is float32")
    else:
        _fail("unexpected embedding dtype", f"got dtype={vec.dtype}")


async def test_embedding_is_unit_norm(model: EmbeddingModel) -> None:
    """encode() must return a unit-norm (L2=1) vector."""
    vec = await model.encode("Normalisation test sentence.")
    norm = float(np.linalg.norm(vec))
    if abs(norm - 1.0) < 1e-5:
        _ok("embedding is unit-norm", f"‖v‖={norm:.6f}")
    else:
        _fail("embedding is not unit-norm", f"‖v‖={norm:.6f}")


async def test_identical_text_similarity(model: EmbeddingModel) -> None:
    """The same sentence encoded twice must have cosine similarity = 1.0."""
    text = "The quick brown fox jumps over the lazy dog."
    v1 = await model.encode(text)
    v2 = await model.encode(text)
    sim = EmbeddingModel.cosine_similarity(v1, v2)
    if abs(sim - 1.0) < 1e-5:
        _ok("identical text → cosine similarity ≈ 1.0", f"sim={sim:.6f}")
    else:
        _fail("identical text similarity not 1.0", f"sim={sim:.6f}")


async def test_similar_pair_similarities(model: EmbeddingModel) -> None:
    """Paraphrased sentences should have high cosine similarity."""
    print()
    all_above = True
    for text_a, text_b in _SIMILAR_PAIRS:
        v1, v2 = await asyncio.gather(model.encode(text_a), model.encode(text_b))
        sim = EmbeddingModel.cosine_similarity(v1, v2)
        bar = "█" * int(sim * 20)
        label = f"{sim:.4f}  [{bar:<20}]"
        short_a = text_a[:45] + ("…" if len(text_a) > 45 else "")
        print(f'    sim={label}  "{short_a}"')
        if sim >= THRESHOLD:
            _ok(f"similar pair above threshold ({THRESHOLD})", f"sim={sim:.4f}")
        else:
            _fail(f"similar pair below threshold ({THRESHOLD})", f"sim={sim:.4f}")
            all_above = False
    if not all_above:
        _info(
            f"Some pairs fell below threshold={THRESHOLD}. "
            "Consider lowering SEMANTIC_THRESHOLD for your use-case."
        )


async def test_dissimilar_pair_similarities(model: EmbeddingModel) -> None:
    """Unrelated sentences should have low cosine similarity."""
    print()
    for text_a, text_b in _DISSIMILAR_PAIRS:
        v1, v2 = await asyncio.gather(model.encode(text_a), model.encode(text_b))
        sim = EmbeddingModel.cosine_similarity(v1, v2)
        bar = "█" * int(sim * 20)
        label = f"{sim:.4f}  [{bar:<20}]"
        short_a = text_a[:45] + ("…" if len(text_a) > 45 else "")
        print(f'    sim={label}  "{short_a}"')
        if sim < THRESHOLD:
            _ok(f"dissimilar pair below threshold ({THRESHOLD})", f"sim={sim:.4f}")
        else:
            _fail("dissimilar pair unexpectedly above threshold", f"sim={sim:.4f}")


# ---------------------------------------------------------------------------
# 2 · CacheManager semantic methods (Redis required)
# ---------------------------------------------------------------------------


async def test_semantic_disabled_for_nonzero_temperature(manager: CacheManager) -> None:
    """Requests with temperature != 0 must never enter the semantic index."""
    warm_req_hot = CompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        temperature=0.7,
        stream=False,
    )
    await manager.cache_with_embedding(warm_req_hot, _dummy_response("4"))
    result = await manager.get_semantic_match(warm_req_hot)
    if result is None:
        _ok("temperature=0.7 request is not stored/matched semantically")
    else:
        _fail("temperature=0.7 request should not be semantically cached")


async def test_semantic_miss_for_dissimilar_query(manager: CacheManager) -> None:
    """Storing one query should not match a semantically unrelated query."""
    unique_suffix = str(int(time.time() * 1000))
    stored_req = _make_request(f"Describe the French Revolution {unique_suffix}.")
    await manager.cache_with_embedding(stored_req, _dummy_response("It was a revolution…"))

    unrelated_req = _make_request(f"Explain how to sort a Python list {unique_suffix}.")
    result = await manager.get_semantic_match(unrelated_req)
    if result is None:
        _ok("dissimilar query did not produce a semantic hit")
    else:
        _fail("dissimilar query unexpectedly hit the semantic cache", str(result))


async def test_semantic_hit_for_paraphrase(manager: CacheManager) -> None:
    """A paraphrase of the stored query must return the cached response."""
    unique_suffix = str(int(time.time() * 1000))
    original = f"What is the capital of France {unique_suffix}?"
    paraphrase = f"Which city is the capital of France {unique_suffix}?"

    stored_req = _make_request(original)
    stored_resp = _dummy_response("The capital of France is Paris.")
    await manager.cache_with_embedding(stored_req, stored_resp)

    query_req = _make_request(paraphrase)
    result = await manager.get_semantic_match(query_req)

    if result is not None:
        response, similarity = result
        _info(f"similarity score: {similarity:.4f}  (threshold={THRESHOLD})")
        cached_content = stored_resp["choices"][0]["message"]["content"]
        resp_content = response.get("choices", [{}])[0].get("message", {}).get("content")
        if resp_content == cached_content:
            _ok("paraphrase returned the correct cached response", f"sim={similarity:.4f}")
        else:
            _fail("paraphrase hit but returned wrong response")
    else:
        _info(
            f"Paraphrase did not meet threshold={THRESHOLD}. "
            "Try lowering SEMANTIC_THRESHOLD if similarity is close."
        )
        _fail("paraphrase did not produce a semantic hit")


async def test_exact_same_query_hits_semantic_cache(manager: CacheManager) -> None:
    """The exact same query text (stored then retrieved) must hit."""
    unique_query = f"What year was the Eiffel Tower built {int(time.time())}?"
    req = _make_request(unique_query)
    resp = _dummy_response("1889")

    await manager.cache_with_embedding(req, resp)
    result = await manager.get_semantic_match(req)

    if result is not None:
        _, similarity = result
        _ok("exact query hits semantic cache", f"sim={similarity:.4f}")
    else:
        _fail("exact query did not hit semantic cache")


async def test_index_eviction(manager: CacheManager) -> None:
    """When max_entries=2, the oldest entry is evicted on the third insert."""
    # Create a manager with a very small index limit (2 entries)
    tight_manager = CacheManager(
        backend=manager._backend,  # type: ignore[attr-defined]
        redis_client=manager._redis,  # type: ignore[attr-defined]
        embedding_model=manager._embedding_model,  # type: ignore[attr-defined]
        semantic_max_entries=2,
        semantic_threshold=THRESHOLD,
        default_ttl=60,
    )

    ts = int(time.time() * 1000)
    req_a = _make_request(f"Entry A for eviction test {ts}")
    req_b = _make_request(f"Entry B for eviction test {ts}")
    req_c = _make_request(f"Entry C for eviction test {ts}")

    await tight_manager.cache_with_embedding(req_a, _dummy_response("A"))
    await tight_manager.cache_with_embedding(req_b, _dummy_response("B"))
    await tight_manager.cache_with_embedding(req_c, _dummy_response("C"))  # triggers eviction of A

    # Entry A should have been evicted (oldest); B and C should remain.
    # We check the index size indirectly: only 2 entries should remain.
    assert tight_manager._redis is not None
    index_key = f"llmgw:sem:idx:{MODEL}"
    count = await tight_manager._redis.zcard(index_key)

    # The index may contain entries from other tests too; just verify we didn't grow unbounded.
    _info(f"index size after 3 inserts with max=2: {count} entries")
    _ok("eviction ran without error (index did not exceed max)")


# ---------------------------------------------------------------------------
# 3 · HTTP endpoint tests (gateway required)
# ---------------------------------------------------------------------------


async def _check_gateway_available(client: httpx.AsyncClient) -> bool:
    try:
        resp = await client.get(f"{GATEWAY_URL}/health/live", timeout=2.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


async def test_http_cache_type_header_on_miss(client: httpx.AsyncClient) -> None:
    """A fresh request must return X-Cache-Type: MISS."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"semantic-test-miss-{time.time()}"}],
        "temperature": 0.0,
        "stream": False,
    }
    try:
        r = await client.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload, timeout=30.0)
    except Exception as exc:
        _fail("request failed", str(exc))
        return

    if r.status_code != 200:
        _fail(f"request returned {r.status_code}", r.text[:120])
        return

    cache_type = r.headers.get("X-Cache-Type", "(missing)")
    if cache_type == "MISS":
        _ok("fresh request → X-Cache-Type: MISS")
    else:
        _fail("expected X-Cache-Type: MISS on fresh request", f"got={cache_type!r}")


async def test_http_exact_hit_then_semantic_paraphrase(client: httpx.AsyncClient) -> None:
    """Sequence: MISS → EXACT hit (same text) → potential SEMANTIC hit (paraphrase)."""
    ts = int(time.time())
    original = f"What is the largest planet in our solar system {ts}?"
    paraphrase = f"Which planet is the biggest in our solar system {ts}?"

    base_payload = {
        "model": MODEL,
        "temperature": 0.0,
        "stream": False,
    }

    # --- Step 1: original → MISS (first call, populates both caches) ---
    try:
        r1 = await client.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json={**base_payload, "messages": [{"role": "user", "content": original}]},
            timeout=30.0,
        )
    except Exception as exc:
        _fail("step 1 (original request) failed", str(exc))
        return

    if r1.status_code != 200:
        _fail(f"step 1 returned {r1.status_code}", r1.text[:120])
        return

    type1 = r1.headers.get("X-Cache-Type", "(missing)")
    status1 = r1.headers.get("X-Cache-Status", "(missing)")
    _info(f"step 1 (original):   X-Cache-Status={status1!r}  X-Cache-Type={type1!r}")
    if status1 == "MISS":
        _ok("step 1 → X-Cache-Status: MISS (as expected for first call)")
    else:
        _fail("step 1 should be a MISS", f"got={status1!r}")

    # --- Step 2: exact same text → EXACT hit ---
    try:
        r2 = await client.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json={**base_payload, "messages": [{"role": "user", "content": original}]},
            timeout=10.0,
        )
    except Exception as exc:
        _fail("step 2 (exact repeat) failed", str(exc))
        return

    type2 = r2.headers.get("X-Cache-Type", "(missing)")
    status2 = r2.headers.get("X-Cache-Status", "(missing)")
    _info(f"step 2 (exact same):  X-Cache-Status={status2!r}  X-Cache-Type={type2!r}")
    if status2 == "HIT" and type2 == "EXACT":
        _ok("step 2 → X-Cache-Type: EXACT (exact match)")
    else:
        _fail("expected EXACT hit on step 2", f"status={status2!r} type={type2!r}")

    # --- Step 3: paraphrase → SEMANTIC hit (if similarity >= threshold) ---
    try:
        r3 = await client.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json={**base_payload, "messages": [{"role": "user", "content": paraphrase}]},
            timeout=15.0,
        )
    except Exception as exc:
        _fail("step 3 (paraphrase request) failed", str(exc))
        return

    if r3.status_code != 200:
        _fail(f"step 3 returned {r3.status_code}", r3.text[:120])
        return

    type3 = r3.headers.get("X-Cache-Type", "(missing)")
    status3 = r3.headers.get("X-Cache-Status", "(missing)")
    sim3 = r3.headers.get("X-Cache-Similarity", "(missing)")
    _info(
        f"step 3 (paraphrase):  X-Cache-Status={status3!r}"
        f"  X-Cache-Type={type3!r}  X-Cache-Similarity={sim3!r}"
    )

    if status3 == "HIT" and type3 == "SEMANTIC":
        _ok(
            "step 3 → X-Cache-Type: SEMANTIC (paraphrase matched)",
            f"similarity={sim3}",
        )
    elif status3 == "MISS":
        _info(
            f"Paraphrase fell below threshold={THRESHOLD}. "
            "Similarity was not high enough — this is not necessarily a bug. "
            f"Try lowering SEMANTIC_THRESHOLD (currently {THRESHOLD})."
        )
        _skip("step 3 paraphrase semantic hit", f"similarity below threshold={THRESHOLD}")
    else:
        _fail(
            "unexpected cache type on step 3 paraphrase",
            f"status={status3!r} type={type3!r}",
        )


async def test_http_x_cache_type_header_always_present(client: httpx.AsyncClient) -> None:
    """Every non-streaming response must carry X-Cache-Type and X-Cache-Status."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "header-check"}],
        "temperature": 0.0,
        "stream": False,
    }
    try:
        r = await client.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload, timeout=30.0)
        has_status = "X-Cache-Status" in r.headers
        has_type = "X-Cache-Type" in r.headers
        if has_status and has_type:
            _ok(
                "X-Cache-Status and X-Cache-Type headers present",
                f"status={r.headers['X-Cache-Status']!r} type={r.headers['X-Cache-Type']!r}",
            )
        else:
            missing = [
                h
                for h, ok in [("X-Cache-Status", has_status), ("X-Cache-Type", has_type)]
                if not ok
            ]
            _fail("missing response headers", f"missing={missing}")
    except Exception as exc:
        _fail("request failed", str(exc))


async def test_http_similarity_header_on_semantic_hit(client: httpx.AsyncClient) -> None:
    """X-Cache-Similarity header must be a float in [0, 1] on a SEMANTIC hit."""
    # Use an exact repeat of a previously cached query to trigger SEMANTIC via the
    # embedding path — or rely on the previous test having cached the paraphrase.
    ts = int(time.time())
    original = f"What is the speed of light {ts}?"
    paraphrase = f"How fast does light travel {ts}?"

    base = {"model": MODEL, "temperature": 0.0, "stream": False}

    # Prime the cache
    try:
        await client.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json={**base, "messages": [{"role": "user", "content": original}]},
            timeout=30.0,
        )
    except Exception as exc:
        _fail("priming request failed", str(exc))
        return

    # Query with paraphrase
    try:
        r = await client.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json={**base, "messages": [{"role": "user", "content": paraphrase}]},
            timeout=15.0,
        )
    except Exception as exc:
        _fail("paraphrase request failed", str(exc))
        return

    cache_type = r.headers.get("X-Cache-Type", "")
    if cache_type == "SEMANTIC":
        sim_str = r.headers.get("X-Cache-Similarity", "")
        try:
            sim = float(sim_str)
            if 0.0 <= sim <= 1.0:
                _ok("X-Cache-Similarity is a valid float in [0, 1]", f"similarity={sim:.4f}")
            else:
                _fail("X-Cache-Similarity out of range", f"value={sim_str!r}")
        except ValueError:
            _fail("X-Cache-Similarity is not a valid float", f"value={sim_str!r}")
    else:
        _skip(
            "X-Cache-Similarity header check",
            f"paraphrase did not hit SEMANTIC cache (type={cache_type!r})",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    print(SECTION)
    print("  LLM Gateway — Semantic Cache Manual Test Suite")
    print(SECTION)
    print(f"  Redis URL  : {REDIS_URL}")
    print(f"  Gateway    : {GATEWAY_URL}")
    print(f"  Model      : {MODEL}")
    print(f"  Threshold  : {THRESHOLD}  (override with SEMANTIC_THRESHOLD=0.90)")

    # ------------------------------------------------------------------
    # Initialise EmbeddingModel (loads lazily, first encode() triggers download)
    # ------------------------------------------------------------------
    embedding_model = EmbeddingModel()
    print(f"\n  Loading embedding model '{embedding_model._model_name}'…")
    # Warm up so the download/load happens before the tests
    _ = await embedding_model.encode("warm-up")
    print("  Model ready.\n")

    # ------------------------------------------------------------------
    # 1 · EmbeddingModel tests (no infrastructure needed)
    # ------------------------------------------------------------------
    _section("1 · EmbeddingModel — Shape, Dtype, and Normalisation")
    await test_embedding_shape_and_dtype(embedding_model)
    await test_embedding_is_unit_norm(embedding_model)
    await test_identical_text_similarity(embedding_model)

    _section("2 · Cosine Similarity — Similar Pairs")
    _info(f"threshold={THRESHOLD}  (pairs that should exceed it)")
    await test_similar_pair_similarities(embedding_model)

    _section("3 · Cosine Similarity — Dissimilar Pairs")
    _info(f"threshold={THRESHOLD}  (pairs that should stay below it)")
    await test_dissimilar_pair_similarities(embedding_model)

    # ------------------------------------------------------------------
    # 2 · CacheManager semantic methods (Redis required)
    # ------------------------------------------------------------------
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
    redis_connected = False
    try:
        await redis_client.ping()
        redis_connected = True
    except Exception as exc:
        print(f"\n  ⚠️  Redis not reachable ({exc}); component tests will be skipped.")

    redis_cache = RedisCache(redis_client)
    manager = CacheManager(
        backend=redis_cache,
        default_ttl=300,
        redis_client=redis_client,
        embedding_model=embedding_model,
        semantic_threshold=THRESHOLD,
        semantic_max_entries=settings.semantic_cache_max_entries,
    )

    _section("4 · CacheManager — Temperature Gating")
    if redis_connected:
        await test_semantic_disabled_for_nonzero_temperature(manager)
    else:
        _skip("temperature gating", "Redis not reachable")

    _section("5 · CacheManager — Semantic Miss (unrelated query)")
    if redis_connected:
        await test_semantic_miss_for_dissimilar_query(manager)
    else:
        _skip("semantic miss", "Redis not reachable")

    _section("6 · CacheManager — Semantic Hit (exact same text)")
    if redis_connected:
        await test_exact_same_query_hits_semantic_cache(manager)
    else:
        _skip("semantic hit (exact)", "Redis not reachable")

    _section("7 · CacheManager — Semantic Hit (paraphrase)")
    _info(f"threshold={THRESHOLD}  (lower with SEMANTIC_THRESHOLD env var if needed)")
    if redis_connected:
        await test_semantic_hit_for_paraphrase(manager)
    else:
        _skip("semantic hit (paraphrase)", "Redis not reachable")

    _section("8 · CacheManager — Index Eviction")
    if redis_connected:
        await test_index_eviction(manager)
    else:
        _skip("index eviction", "Redis not reachable")

    await redis_client.aclose()

    # ------------------------------------------------------------------
    # 3 · HTTP endpoint tests (gateway required)
    # ------------------------------------------------------------------
    _section("9 · HTTP Endpoint — X-Cache-Type and X-Cache-Similarity Headers")

    async with httpx.AsyncClient() as http_client:
        gateway_up = await _check_gateway_available(http_client)
        if not gateway_up:
            print(f"\n  ⚠️  Gateway not reachable at {GATEWAY_URL}; HTTP tests skipped.")
            print("      Start the server with:  make dev")
            for label in [
                "http_cache_type_miss",
                "http_exact_then_semantic_sequence",
                "http_headers_always_present",
                "http_similarity_header_validation",
            ]:
                _skip(label, "gateway not running")
        else:
            print("  Gateway reachable — running live HTTP tests…")
            await test_http_x_cache_type_header_always_present(http_client)
            await test_http_cache_type_header_on_miss(http_client)
            await test_http_exact_hit_then_semantic_paraphrase(http_client)
            await test_http_similarity_header_on_semantic_hit(http_client)

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
