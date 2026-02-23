"""CacheManager: high-level caching logic for LLM completions.

Responsibilities
----------------
* Decide whether a request is cacheable (temperature == 0 only).
* Derive a deterministic SHA-256 cache key from the request parameters.
* Coordinate cache lookups and stores through a :class:`CacheBackend`.
* Export Prometheus metrics for hits, misses, and lookup duration.
* Log every cache operation with structlog for observability.

Cache failures are always treated as misses (fail-open) because the backend
implementations (:class:`~llmgateway.cache.redis_cache.RedisCache`) already
swallow exceptions internally.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

import structlog
from prometheus_client import Counter, Histogram

from llmgateway.cache.base import CacheBackend, CacheEntry

if TYPE_CHECKING:
    from llmgateway.providers import CompletionRequest

_log = structlog.get_logger(__name__)

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


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------


class CacheManager:
    """Manages exact-match caching of non-streaming LLM responses.

    Only requests with ``temperature=0`` are cached because those are the
    only ones that are truly deterministic.  All other requests pass through
    directly to the provider.

    Args:
        backend:     Any object implementing :class:`~llmgateway.cache.base.CacheBackend`.
        default_ttl: Seconds to keep a cached entry (overridable per call).
                     Defaults to 3 600 s (1 hour).
    """

    def __init__(self, backend: CacheBackend, default_ttl: int = 3600) -> None:
        self._backend = backend
        self._default_ttl = default_ttl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_cache_key(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        """Return a stable SHA-256 hex digest for the given request parameters.

        The digest is computed over a canonical JSON representation so that
        the same logical request always maps to the same key regardless of
        dict insertion order.
        """
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

    async def get_cached_response(self, request: CompletionRequest) -> dict[str, Any] | None:
        """Look up a cached response for *request*.

        Returns the deserialised response dict on a cache hit, or ``None``
        when the request is not cacheable, the entry is absent, or any error
        occurs (fail-open).

        Args:
            request: The incoming completion request.

        Returns:
            Deserialised JSON response dict, or ``None``.
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
        # RedisCache already swallows exceptions and returns None on error.
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
            return json.loads(entry.value)
        except json.JSONDecodeError as exc:
            _log.warning("cache.decode_error", key=cache_key, error=str(exc))
            return None

    async def cache_response(
        self,
        request: CompletionRequest,
        response_data: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Store *response_data* in the cache for *request*.

        No-ops silently when the request is not cacheable (temperature != 0)
        or when the backend raises (fail-open).

        Args:
            request:       The completion request that produced *response_data*.
            response_data: The full OpenAI-format response dict to cache.
            ttl:           Override TTL in seconds; falls back to
                           :attr:`default_ttl` when ``None``.
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

        # RedisCache already swallows exceptions internally.
        await self._backend.set(entry)
        _log.info(
            "cache.stored",
            key=cache_key,
            model=request.model,
            ttl=effective_ttl,
        )
