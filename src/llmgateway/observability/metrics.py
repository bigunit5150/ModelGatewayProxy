"""Centralised Prometheus metrics for the LLM Gateway.

New metrics defined here (not duplicating those already defined elsewhere):
  llm_requests_total                — Request counter per model/provider/status
  llm_request_duration_seconds      — End-to-end gateway latency histogram
  llm_token_count                   — Input/output token count histogram
  llm_provider_api_duration_seconds — Provider-side API call latency histogram
  llm_active_requests               — In-flight request gauge

Pre-existing metrics (defined in their respective modules, not re-registered):
  llm_cost_usd_total                         — api/completions.py
  llm_cache_hits_total                       — cache/cache_manager.py
  llm_cache_misses_total                     — cache/cache_manager.py
  llm_cache_lookup_duration_seconds          — cache/cache_manager.py
  llm_semantic_cache_hits_total              — cache/cache_manager.py
  llm_semantic_cache_lookups_duration_seconds — cache/cache_manager.py
  llm_embedding_generation_duration_seconds  — cache/embeddings.py
  llm_rate_limit_checks_total                — ratelimit/token_bucket.py
  llm_rate_limit_exceeded_total              — ratelimit/token_bucket.py
  llm_rate_limit_check_duration_seconds      — ratelimit/token_bucket.py
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Request counters and end-to-end latency
# ---------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM completion requests processed by the gateway",
    ["model", "provider", "status"],  # status: "success" | "error"
)

REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "End-to-end gateway request duration from receipt to response",
    ["model", "provider", "cached"],  # cached: "true" | "false"
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Number of LLM completion requests currently being processed",
    ["provider"],
)

# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

TOKEN_COUNT = Histogram(
    "llm_token_count",
    "Token count per completion request (provider calls only, excludes cache hits)",
    ["model", "type"],  # type: "input" | "output"
    buckets=[50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000],
)

# ---------------------------------------------------------------------------
# Provider-side latency
# ---------------------------------------------------------------------------

PROVIDER_API_DURATION = Histogram(
    "llm_provider_api_duration_seconds",
    "LiteLLM provider API call duration (single attempt, excludes retry overhead)",
    ["model", "provider"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
