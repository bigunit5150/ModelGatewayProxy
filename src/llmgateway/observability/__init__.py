"""Observability package: centralised Prometheus metrics and OTel span helpers."""

from llmgateway.observability.metrics import (
    ACTIVE_REQUESTS,
    PROVIDER_API_DURATION,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    TOKEN_COUNT,
)
from llmgateway.observability.tracing import (
    cache_lookup_span,
    cost_record_span,
    gateway_request_span,
    provider_generate_span,
)

__all__ = [
    "ACTIVE_REQUESTS",
    "PROVIDER_API_DURATION",
    "REQUEST_DURATION",
    "REQUESTS_TOTAL",
    "TOKEN_COUNT",
    "cache_lookup_span",
    "cost_record_span",
    "gateway_request_span",
    "provider_generate_span",
]
