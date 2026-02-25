"""Custom OTel span helpers with a defined hierarchy for the LLM Gateway.

Span hierarchy
--------------
  llm_gateway.request           — root span per incoming HTTP request
    ├── cache.lookup             — exact-match and semantic cache checks
    ├── rate_limit.check         — token-bucket Redis Lua check
    └── provider.generate        — full provider round-trip
          ├── provider.api_call  — raw litellm.acompletion call
          ├── cost.calculate     — pricing table look-up
          └── cost.record        — async PostgreSQL write

Usage
-----
These helpers wrap :func:`opentelemetry.trace.get_tracer` to provide
a consistent span naming scheme.  They are context managers, so callers
use ``with`` syntax::

    with gateway_request_span(model="gpt-4o", provider="openai",
                               user_id="user123") as span:
        span.set_attribute("extra.key", "value")
        ...
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator

from opentelemetry import trace
from opentelemetry.trace import Span

_tracer = trace.get_tracer("llmgateway.observability")


@contextlib.contextmanager
def gateway_request_span(
    model: str,
    provider: str,
    user_id: str,
    stream: bool = False,
) -> Generator[Span, None, None]:
    """Root span covering the full lifecycle of a gateway HTTP request.

    Args:
        model:    LiteLLM model string (e.g. ``"gpt-4o"``).
        provider: Short provider name (e.g. ``"openai"``).
        user_id:  Resolved user identifier for the request.
        stream:   ``True`` for streaming responses.
    """
    with _tracer.start_as_current_span("llm_gateway.request") as span:
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.system", provider)
        span.set_attribute("enduser.id", user_id)
        span.set_attribute("llm.stream", stream)
        yield span


@contextlib.contextmanager
def cache_lookup_span(
    model: str,
    cache_type: str = "exact",
) -> Generator[Span, None, None]:
    """Child span for a single cache lookup operation.

    Args:
        model:      LiteLLM model string — used to label the span.
        cache_type: ``"exact"`` or ``"semantic"``.
    """
    with _tracer.start_as_current_span("cache.lookup") as span:
        span.set_attribute("cache.model", model)
        span.set_attribute("cache.type", cache_type)
        yield span


@contextlib.contextmanager
def provider_generate_span(
    model: str,
    provider: str,
) -> Generator[Span, None, None]:
    """Child span wrapping a full provider generation call (including retries).

    Args:
        model:    LiteLLM model string.
        provider: Short provider name.
    """
    with _tracer.start_as_current_span("provider.generate") as span:
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.system", provider)
        yield span


@contextlib.contextmanager
def cost_record_span(
    user_id: str,
    model: str,
    cost_usd: float,
) -> Generator[Span, None, None]:
    """Child span for an async PostgreSQL cost record write.

    Args:
        user_id:  User identifier for attribution.
        model:    LiteLLM model string.
        cost_usd: Estimated cost in USD for this request.
    """
    with _tracer.start_as_current_span("cost.record") as span:
        span.set_attribute("cost.user_id", user_id)
        span.set_attribute("cost.model", model)
        span.set_attribute("cost.usd", cost_usd)
        yield span
