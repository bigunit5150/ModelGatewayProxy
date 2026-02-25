"""CacheManager: exact-match and semantic caching for LLM completions.

Responsibilities
----------------
* Decide whether a request is cacheable (temperature == 0 only).
* Derive a deterministic SHA-256 cache key from the request parameters.
* Coordinate exact-match lookups and stores through a :class:`CacheBackend`.
* On exact misses, optionally search the semantic index in Redis for a
  similar query above a cosine-similarity threshold.
* Store new responses in both the exact-match cache and the semantic index.
* Export Prometheus metrics for hits, misses, and operation durations.
* Log every cache operation with structlog for observability.

Cache failures are always treated as misses (fail-open): the exact-match
backend swallows exceptions internally, and all semantic operations are
wrapped in try/except blocks that log warnings and return ``None``.

Redis key layout for the semantic index
-----------------------------------------
``llmgw:sem:emb:{uuid}``   — JSON list of floats (embedding vector), with TTL
``llmgw:sem:resp:{uuid}``  — JSON-encoded response dict, with TTL
``llmgw:sem:idx:{model}``  — Sorted set of UUIDs scored by creation timestamp
                             (used for FIFO eviction when the index exceeds
                             *semantic_max_entries*)
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import structlog
from prometheus_client import Counter, Histogram

from llmgateway.cache.base import CacheBackend, CacheEntry
from llmgateway.cache.embeddings import EmbeddingModel

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from llmgateway.providers import CompletionRequest

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Redis key prefixes
# ---------------------------------------------------------------------------

_SEM_EMB_PREFIX = "llmgw:sem:emb:"
_SEM_RESP_PREFIX = "llmgw:sem:resp:"
_SEM_IDX_PREFIX = "llmgw:sem:idx:"

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_CACHE_HITS = Counter(
    "llm_cache_hits_total",
    "Total number of cache hits for LLM responses",
    ["model"],
)

_CACHE_MISSES = Counter(
    "llm_cache_misses_total",
    "Total number of cache misses for LLM responses",
    ["model"],
)

_CACHE_LOOKUP_DURATION = Histogram(
    "llm_cache_lookup_duration_seconds",
    "End-to-end duration of a cache lookup operation (hit or miss)",
    ["result"],  # "hit" | "miss"
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

_SEMANTIC_CACHE_HITS = Counter(
    "llm_semantic_cache_hits_total",
    "Total number of semantic cache hits",
    ["model"],
)

_SEMANTIC_LOOKUP_DURATION = Histogram(
    "llm_semantic_cache_lookups_duration_seconds",
    "End-to-end duration of a semantic cache lookup (embedding + similarity scan)",
    ["result"],  # "hit" | "miss"
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_user_text(messages: list[dict[str, str]]) -> str:
    """Return the content of the last user-role message, or empty string."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------


class CacheManager:
    """Manages exact-match and semantic caching of non-streaming LLM responses.

    Only requests with ``temperature=0`` are cached because those are the
    only ones guaranteed to be deterministic.  All other requests pass through
    to the provider.

    **Exact-match cache** (always enabled when *backend* is provided):
    Uses a SHA-256 digest of ``(model, messages, temperature, max_tokens)`` as
    the cache key, stored via any :class:`~llmgateway.cache.base.CacheBackend`.

    **Semantic cache** (optional, requires *redis_client* and *embedding_model*):
    On an exact miss, encodes the last user message with a sentence-transformer
    model and scans a per-model Redis index for the most similar stored
    embedding.  Returns the associated response if the cosine similarity
    exceeds *semantic_threshold*.

    Args:
        backend:               Any object implementing :class:`CacheBackend`.
        default_ttl:           Seconds to keep a cached entry.  Defaults to
                               3 600 s (1 hour).
        redis_client:          ``redis.asyncio.Redis`` client used for the
                               semantic index.  Semantic caching is disabled
                               when ``None``.
        embedding_model:       :class:`EmbeddingModel` instance.  Semantic
                               caching is disabled when ``None``.
        semantic_threshold:    Minimum cosine similarity to count as a hit.
                               Defaults to ``0.95``.
        semantic_max_entries:  Maximum embeddings stored per model in the
                               index.  Oldest entries are evicted first.
                               Defaults to ``1 000``.
    """

    def __init__(
        self,
        backend: CacheBackend,
        default_ttl: int = 3600,
        redis_client: Redis | None = None,
        embedding_model: EmbeddingModel | None = None,
        semantic_threshold: float = 0.95,
        semantic_max_entries: int = 1000,
    ) -> None:
        self._backend = backend
        self._default_ttl = default_ttl
        self._redis = redis_client
        self._embedding_model = embedding_model
        self._semantic_threshold = semantic_threshold
        self._semantic_max_entries = semantic_max_entries

    @property
    def semantic_enabled(self) -> bool:
        """``True`` when both a Redis client and an embedding model are available."""
        return self._redis is not None and self._embedding_model is not None

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def generate_cache_key(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        """Return a stable SHA-256 hex digest for the given request parameters."""
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Exact-match cache
    # ------------------------------------------------------------------

    async def get_cached_response(self, request: CompletionRequest) -> dict[str, Any] | None:
        """Look up an exact-match cached response for *request*.

        Returns the deserialised response dict on a cache hit, or ``None``
        when the request is not cacheable, the entry is absent, or any error
        occurs (fail-open).
        """
        if request.temperature != 0.0:
            _log.debug("cache.skip", reason="temperature_nonzero", model=request.model)
            return None

        cache_key = self.generate_cache_key(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        start = time.monotonic()
        entry = await self._backend.get(cache_key)
        duration = time.monotonic() - start

        if entry is None:
            _CACHE_MISSES.labels(model=request.model).inc()
            _CACHE_LOOKUP_DURATION.labels(result="miss").observe(duration)
            _log.info("cache.miss", key=cache_key, model=request.model)
            return None

        _CACHE_HITS.labels(model=request.model).inc()
        _CACHE_LOOKUP_DURATION.labels(result="hit").observe(duration)
        _log.info(
            "cache.hit",
            key=cache_key,
            model=request.model,
            ttl=entry.ttl,
            age_s=round(time.time() - entry.created_at, 1),
        )

        try:
            return cast(dict[str, Any], json.loads(entry.value))
        except json.JSONDecodeError as exc:
            _log.warning("cache.decode_error", key=cache_key, error=str(exc))
            return None

    async def cache_response(
        self,
        request: CompletionRequest,
        response_data: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Store *response_data* in the exact-match cache for *request*.

        No-ops silently when the request is not cacheable (temperature != 0)
        or when the backend raises (fail-open).
        """
        if request.temperature != 0.0:
            return

        cache_key = self.generate_cache_key(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        effective_ttl = ttl if ttl is not None else self._default_ttl

        entry = CacheEntry(
            key=cache_key,
            value=json.dumps(response_data),
            created_at=time.time(),
            ttl=effective_ttl,
            metadata={"model": request.model},
        )

        await self._backend.set(entry)
        _log.info("cache.stored", key=cache_key, model=request.model, ttl=effective_ttl)

    # ------------------------------------------------------------------
    # Semantic cache
    # ------------------------------------------------------------------

    async def get_semantic_match(
        self, request: CompletionRequest
    ) -> tuple[dict[str, Any], float] | None:
        """Search the semantic index for a response similar to *request*.

        Extracts the last user message, encodes it, then scans all stored
        embeddings for the same model using cosine similarity.  Returns the
        best-matching response and its similarity score if the score meets
        :attr:`_semantic_threshold`, otherwise ``None``.

        Fails open: any Redis or embedding error returns ``None`` without
        raising.

        Args:
            request: The incoming completion request.

        Returns:
            ``(response_dict, similarity)`` on a semantic hit, or ``None``.
        """
        if not self.semantic_enabled or request.temperature != 0.0:
            return None

        user_text = _extract_user_text(request.messages)
        if not user_text:
            return None

        start = time.monotonic()

        # Generate query embedding (runs in thread pool).
        try:
            assert self._embedding_model is not None  # guarded by semantic_enabled
            query_vec = await self._embedding_model.encode(user_text)
        except Exception as exc:
            _log.warning("semantic.encode_error", error=str(exc))
            return None

        # Fetch the list of UUIDs for this model's index.
        index_key = _SEM_IDX_PREFIX + request.model
        try:
            assert self._redis is not None  # guarded by semantic_enabled
            entry_ids: list[str] = await self._redis.zrange(index_key, 0, -1)
        except Exception as exc:
            _log.warning("semantic.index_fetch_error", error=str(exc))
            return None

        if not entry_ids:
            _SEMANTIC_LOOKUP_DURATION.labels(result="miss").observe(time.monotonic() - start)
            return None

        # Batch-fetch all embedding vectors in a single MGET round trip.
        emb_keys = [_SEM_EMB_PREFIX + eid for eid in entry_ids]
        try:
            raw_embeddings: list[str | None] = await self._redis.mget(*emb_keys)
        except Exception as exc:
            _log.warning("semantic.mget_error", error=str(exc))
            return None

        # Find the most similar entry.
        best_similarity = -1.0
        best_entry_id: str | None = None

        for entry_id, raw_emb in zip(entry_ids, raw_embeddings):
            if raw_emb is None:
                # Entry expired or was evicted between the ZRANGE and MGET.
                continue
            try:
                stored_vec = np.array(json.loads(raw_emb), dtype=np.float32)
                similarity = EmbeddingModel.cosine_similarity(query_vec, stored_vec)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry_id = entry_id
            except Exception as exc:
                _log.warning("semantic.similarity_error", entry_id=entry_id, error=str(exc))
                continue

        duration = time.monotonic() - start

        if best_entry_id is None or best_similarity < self._semantic_threshold:
            _SEMANTIC_LOOKUP_DURATION.labels(result="miss").observe(duration)
            _log.info(
                "semantic.miss",
                model=request.model,
                best_similarity=round(best_similarity, 4),
                threshold=self._semantic_threshold,
                candidates=len(entry_ids),
            )
            return None

        # Fetch the stored response for the best-matching entry.
        resp_key = _SEM_RESP_PREFIX + best_entry_id
        try:
            raw_resp: str | None = await self._redis.get(resp_key)
            if raw_resp is None:
                # Response expired even though the embedding was still present.
                _SEMANTIC_LOOKUP_DURATION.labels(result="miss").observe(duration)
                return None
            response_data: dict[str, Any] = json.loads(raw_resp)
        except Exception as exc:
            _log.warning("semantic.resp_fetch_error", entry_id=best_entry_id, error=str(exc))
            _SEMANTIC_LOOKUP_DURATION.labels(result="miss").observe(duration)
            return None

        _SEMANTIC_CACHE_HITS.labels(model=request.model).inc()
        _SEMANTIC_LOOKUP_DURATION.labels(result="hit").observe(duration)
        _log.info(
            "semantic.hit",
            model=request.model,
            entry_id=best_entry_id,
            similarity=round(best_similarity, 4),
            threshold=self._semantic_threshold,
            duration_ms=round(duration * 1000, 2),
        )
        return response_data, best_similarity

    async def cache_with_embedding(
        self,
        request: CompletionRequest,
        response_data: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Store *response_data* in the semantic index for *request*.

        Generates an embedding for the last user message and stores it
        alongside the response in Redis.  Evicts the oldest entry from the
        per-model sorted-set index when it exceeds :attr:`_semantic_max_entries`.

        No-ops silently when semantic caching is disabled, the request is not
        cacheable, or any error occurs (fail-open).

        Args:
            request:       The completion request that produced *response_data*.
            response_data: The full OpenAI-format response dict to cache.
            ttl:           Override TTL in seconds; falls back to
                           :attr:`_default_ttl` when ``None``.
        """
        if not self.semantic_enabled or request.temperature != 0.0:
            return

        user_text = _extract_user_text(request.messages)
        if not user_text:
            return

        # Generate embedding for the user message.
        try:
            assert self._embedding_model is not None  # guarded by semantic_enabled
            vec = await self._embedding_model.encode(user_text)
        except Exception as exc:
            _log.warning("semantic.store_encode_error", error=str(exc))
            return

        effective_ttl = ttl if ttl is not None else self._default_ttl
        entry_id = str(uuid.uuid4())
        now = time.time()

        emb_key = _SEM_EMB_PREFIX + entry_id
        resp_key = _SEM_RESP_PREFIX + entry_id
        index_key = _SEM_IDX_PREFIX + request.model

        try:
            assert self._redis is not None  # guarded by semantic_enabled

            # Store embedding and response with TTL so Redis auto-cleans them.
            await self._redis.set(emb_key, json.dumps(vec.tolist()), ex=effective_ttl)
            await self._redis.set(resp_key, json.dumps(response_data), ex=effective_ttl)

            # Add to the per-model index (score = creation timestamp for FIFO eviction).
            await self._redis.zadd(index_key, {entry_id: now})

            # Evict the oldest entry if the index has grown beyond the limit.
            count: int = await self._redis.zcard(index_key)
            if count > self._semantic_max_entries:
                evicted: list[tuple[str, float]] = await self._redis.zpopmin(index_key)
                # zpopmin returns [(member, score), ...]; delete the evicted entry's keys.
                for evicted_id, _ in evicted:
                    await self._redis.delete(
                        _SEM_EMB_PREFIX + evicted_id,
                        _SEM_RESP_PREFIX + evicted_id,
                    )
                _log.debug(
                    "semantic.evicted",
                    model=request.model,
                    evicted_count=len(evicted),
                    index_size=count - len(evicted),
                )

        except Exception as exc:
            _log.warning("semantic.store_error", entry_id=entry_id, error=str(exc))
            return

        _log.info(
            "semantic.stored",
            model=request.model,
            entry_id=entry_id,
            text_length=len(user_text),
            ttl=effective_ttl,
        )
