# Provider Layer — Design Decisions

Architectural decisions are recorded here to explain *why* the system is built the way it is, not just *what* it does. Each entry follows the format: context → decision → rationale → consequences.

---

## ADR-001: Use LiteLLM as the Provider Abstraction Layer

**Date:** 2026-02
**Status:** Accepted

### Context

The gateway needs to call multiple LLM providers (Anthropic, OpenAI, Groq, Together AI, and others). Each provider has a different HTTP API, authentication scheme, request/response shape, streaming protocol, and error vocabulary. Without an abstraction, we would need to write and maintain a separate adapter for each.

### Decision

Use [LiteLLM](https://github.com/BerriAI/litellm) as the single translation layer between our code and every upstream provider API.

### Options Considered

**Option A: Custom per-provider adapters**
Write a `BaseProvider` interface with one implementation per provider (`AnthropicProvider`, `OpenAIProvider`, etc.).

*Rejected because:* High initial and ongoing maintenance cost. Every new provider requires a new adapter; every API change requires a patch. The streaming protocol, error mapping, and token counting logic would all need reimplementing per provider. This is commodity plumbing that does not differentiate the product.

**Option B: LiteLLM (chosen)**
Wrap a single `litellm.acompletion` call. LiteLLM handles translation, authentication, and streaming normalisation for 100+ providers.

*Accepted because:* Shifts the per-provider maintenance burden to the open-source community. Adding a new provider requires zero code changes — just a new API key and the right model string.

**Option C: OpenAI Python SDK only (with Azure/Anthropic compatibility modes)**
Several providers expose OpenAI-compatible endpoints. Route all traffic through the OpenAI SDK.

*Rejected because:* Only covers providers with an OpenAI-compatible API. Does not work for native Anthropic, Gemini, or Bedrock endpoints. Introduces vendor lock-in on SDK semantics.

### Rationale

LiteLLM has become the de-facto standard for multi-provider LLM routing. It is actively maintained, battle-tested in production at many companies, and covers the full surface area we need (streaming, function calling, token counting, embeddings). The abstraction cost — a thin wrapper class — is much lower than building equivalent breadth ourselves.

### Consequences

**Positive:**
- Supporting a new provider takes minutes, not days.
- Streaming, tool use, and embeddings are normalised automatically.
- Upstream bug fixes in LiteLLM flow to us for free.

**Negative:**
- We take a transitive dependency on LiteLLM's release cadence. Breaking changes in LiteLLM require gateway updates.
- LiteLLM's error types (`litellm.AuthenticationError`, etc.) must be mapped to our own hierarchy at the boundary to avoid leaking third-party types into application code.
- LiteLLM's own logging is verbose; we suppress it and emit our own structured logs instead.

---

## ADR-002: Custom Typed Exception Hierarchy

**Date:** 2026-02
**Status:** Accepted

### Context

LiteLLM raises its own exception types (which inherit from the OpenAI Python SDK exceptions). Application code that catches `litellm.RateLimitError` is now coupled to LiteLLM's type hierarchy. If we ever replace or wrap LiteLLM, every call site that catches LiteLLM exceptions breaks.

### Decision

Define a gateway-owned exception hierarchy in `llmgateway.providers.errors` and map all LiteLLM exceptions to gateway types at the `_call_litellm` boundary. No LiteLLM exception type ever crosses this boundary into application code.

```
ProviderError              (base)
├── RateLimitError         HTTP 429
├── AuthError              HTTP 401/403
├── TimeoutError           request timeout
├── InvalidRequestError    HTTP 400/422
└── ProviderUnavailableError  HTTP 5xx / network
```

### Rationale

**Stable API contract.** Application code catches `AuthError`, not `litellm.AuthenticationError`. If LiteLLM renames or restructures its exceptions, only `_map_error()` needs updating — zero changes to call sites.

**Retry policy as a type property.** The tenacity retry predicate can be expressed cleanly as `retry_if_exception_type((RateLimitError, TimeoutError, ProviderUnavailableError))`. This is readable and correct by construction: if `AuthError` is never in that tuple, auth failures are never retried — regardless of what LiteLLM does internally.

**Richer metadata.** Our exceptions carry `.provider`, `.retry_after`, and `.original_error` fields that LiteLLM exceptions do not always expose consistently.

### Consequences

**Positive:**
- Callers have a stable, documented contract for error handling.
- Retry logic is expressed as a simple type predicate.
- The original LiteLLM exception is preserved on `.original_error` for debugging without polluting the public interface.

**Negative:**
- Every new LiteLLM exception type requires a corresponding entry in `_map_error()`. Unknown exceptions fall through to the base `ProviderError`, which is intentionally conservative but may lose specificity.

---

## ADR-003: OpenTelemetry for Distributed Tracing

**Date:** 2026-02
**Status:** Accepted

### Context

A production LLM gateway is a distributed system component. Understanding request latency, failure modes, and provider behaviour requires observability beyond simple logs. We need to correlate a single user request through the gateway, the LiteLLM layer, and the upstream provider.

### Decision

Instrument every `generate()` call with OpenTelemetry spans following the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). Export traces to Jaeger via OTLP.

### What We Track

**Span: `llm.generate`** (outer)
- `gen_ai.system` — provider name (`anthropic`, `openai`, …)
- `gen_ai.request.model` — model identifier
- `gen_ai.request.temperature` — sampling temperature
- `gen_ai.request.max_tokens` — token limit (if set)
- `llm.stream` — streaming mode flag
- `enduser.id` — caller identifier (if provided)
- Exception details and `ERROR` status on failure

**Span: `llm.api_call`** (inner, one per LiteLLM call)
- `call_type` — `streaming` or `non_streaming`
- `gen_ai.usage.input_tokens` — prompt tokens consumed
- `gen_ai.usage.output_tokens` — completion tokens generated
- `gen_ai.response.finish_reasons` — stop reason (non-streaming)

### Why OTel Over a Custom Metrics Solution

OTel is the CNCF standard for observability instrumentation. Using it means:

1. **Backend-agnostic.** We export to Jaeger today; switching to Honeycomb, Datadog, or Tempo requires only a configuration change, not code changes.
2. **Composability.** FastAPI is already instrumented via `FastAPIInstrumentor`. The `llm.generate` span automatically becomes a child of the HTTP request span, giving full end-to-end trace context.
3. **Industry vocabulary.** The GenAI semantic conventions are becoming the standard way to describe LLM calls in telemetry. Adopting them now means dashboards and alerts built today will remain valid as tooling matures.

### Consequences

**Positive:**
- Full distributed traces from HTTP request through to LLM response in Jaeger UI.
- Per-request token usage visible in span attributes — useful for cost attribution.
- Error details (exception type, message) automatically recorded on failed spans.

**Negative:**
- OTel SDK adds ~50ms to cold-start time (mitigated by `cache_logger_on_first_use`).
- Span export is async and best-effort via `BatchSpanProcessor`. Traces may be lost if Jaeger is unreachable — this is acceptable for observability data but would not be acceptable for billing data.
- The Jaeger endpoint (`http://jaeger:4318`) is a Docker Compose hostname. Deployments outside Docker Compose must set `OTEL_EXPORTER_OTLP_ENDPOINT` explicitly.

---

## ADR-004: Tenacity for Retry Logic

**Date:** 2026-02
**Status:** Accepted

### Context

Transient failures (rate limits, timeouts, 5xx errors) are common when calling external APIs. Naive immediate retries cause thundering-herd problems; no retries surface transient failures unnecessarily to callers.

### Decision

Use [tenacity](https://github.com/jd/tenacity) with `AsyncRetrying` to implement exponential-backoff retry only for the subset of errors that are safe to retry.

```python
AsyncRetrying(
    stop=stop_after_attempt(max_retries),
    retry=retry_if_exception_type((RateLimitError, TimeoutError, ProviderUnavailableError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
```

### Rationale

**Retry predicate on types, not status codes.** The retry decision is made on our typed exception hierarchy (ADR-002), not on raw HTTP status codes. This keeps the retry policy readable and decoupled from provider-specific HTTP semantics.

**Never retry permanent errors.** `AuthError` and `InvalidRequestError` are never retried. Retrying an auth failure would waste quota; retrying a 400 Bad Request would always fail. The type predicate enforces this by construction.

**Cap at 30 seconds.** The `max=30` ceiling prevents a single slow request from monopolising a thread for too long.

**`reraise=True`.** When retries are exhausted, the last exception is re-raised directly (already mapped to a gateway type) rather than wrapped in a tenacity `RetryError`.

### Consequences

**Positive:**
- Transient rate limits and 5xx errors resolve automatically without caller involvement.
- Retry behaviour is configurable at provider construction time (`max_retries` param).
- Backoff delays are skipped in tests by patching `tenacity.asyncio._portable_async_sleep`.

**Negative:**
- A request that ultimately fails after 3 attempts takes `2 + 4 = 6` seconds longer than one that fails immediately. Callers must set their own timeout budgets accordingly.
- Mid-stream errors (errors that occur during token iteration *after* the initial connection succeeds) are not retried — resuming a partial stream is unsafe without server-side support.
