"""Integration tests for the complete LLM Gateway HTTP flow.

These tests make *real* API calls and require valid API keys in the environment.
All tests are marked ``integration`` and are excluded from the default ``pytest``
run.  Run them explicitly when you have keys and infrastructure available:

    # Run only integration tests
    pytest -m integration -v

    # Run with a specific provider key only
    ANTHROPIC_API_KEY=sk-ant-... pytest -m integration -v

Jaeger and Prometheus tests additionally require the Docker Compose stack to be
running (``docker compose up -d jaeger``).
"""

# Load .env before any app imports so API keys reach os.environ for LiteLLM.
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections.abc import AsyncGenerator  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from llmgateway.main import app, tracer_provider  # noqa: E402
from llmgateway.providers import LLMGatewayProvider  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level integration marker — applied to every test in this file
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Skip conditions evaluated at collection time
# ---------------------------------------------------------------------------
_HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
_HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))

needs_anthropic = pytest.mark.skipif(
    not _HAS_ANTHROPIC,
    reason="ANTHROPIC_API_KEY not set — skipping Anthropic integration test",
)
needs_openai = pytest.mark.skipif(
    not _HAS_OPENAI,
    reason="OPENAI_API_KEY not set — skipping OpenAI integration test",
)

# Check Jaeger accessibility once at collection time (2 s timeout, best-effort).
try:
    httpx.get("http://localhost:16686/api/services", timeout=2.0)
    _JAEGER_UP = True
except Exception:
    _JAEGER_UP = False

needs_jaeger = pytest.mark.skipif(
    not _JAEGER_UP,
    reason="Jaeger not reachable at localhost:16686 — skipping trace tests",
)

# ---------------------------------------------------------------------------
# Shared request bodies
# ---------------------------------------------------------------------------
_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_OPENAI_MODEL = "gpt-4o-mini"
_SHORT_PROMPT = [{"role": "user", "content": "Reply with exactly one word: hello"}]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """AsyncClient backed by the real FastAPI app with a live provider.

    * ``timeout=30`` — generous enough for slow providers, not so long it hangs.
    * ``max_retries=1`` — fail fast in tests; we are not testing retry behaviour.
    * The provider is attached to ``app.state`` and cleaned up after each test.
    """
    provider = LLMGatewayProvider(timeout=30, max_retries=1)
    app.state.provider = provider

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
    ) as ac:
        yield ac

    if hasattr(app.state, "provider"):
        del app.state.provider


# ---------------------------------------------------------------------------
# 1. End-to-End Gateway Tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    # ------------------------------------------------------------------
    # OpenAI streaming
    # ------------------------------------------------------------------

    @needs_openai
    async def test_openai_streaming_sse_format(self, client: AsyncClient) -> None:
        """Every data line must be valid JSON in the OpenAI SSE envelope."""
        response = await client.post(
            "/v1/chat/completions",
            json={"model": _OPENAI_MODEL, "messages": _SHORT_PROMPT, "stream": True},
        )
        assert response.status_code == 200

        data_lines = [
            line[6:]  # strip "data: "
            for line in response.text.splitlines()
            if line.startswith("data: ") and "[DONE]" not in line
        ]
        assert data_lines, "Expected at least one SSE data line"
        for raw in data_lines:
            parsed = json.loads(raw)
            # If the provider returned an error (e.g. quota exceeded), skip rather
            # than fail — this is an environment issue, not a code defect.
            if "error" in parsed:
                pytest.skip(
                    f"Provider error during streaming: {parsed['error'].get('message', raw)}"
                )
            assert parsed["object"] == "chat.completion.chunk"
            assert "choices" in parsed
            assert "id" in parsed

    @needs_openai
    async def test_openai_streaming_done_marker(self, client: AsyncClient) -> None:
        """The final SSE event must be ``data: [DONE]``."""
        response = await client.post(
            "/v1/chat/completions",
            json={"model": _OPENAI_MODEL, "messages": _SHORT_PROMPT, "stream": True},
        )
        assert response.status_code == 200
        assert "data: [DONE]" in response.text

    @needs_openai
    async def test_openai_streaming_chunks_received_incrementally(
        self, client: AsyncClient
    ) -> None:
        """Use the streaming API to consume SSE chunks as they arrive."""
        received: list[str] = []

        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": _OPENAI_MODEL, "messages": _SHORT_PROMPT, "stream": True},
        ) as response:
            assert response.status_code == 200
            async for chunk in response.aiter_text():
                if chunk.strip():
                    received.append(chunk)

        full = "".join(received)
        assert "data: " in full, "No SSE data lines received"
        assert "[DONE]" in full, "Stream never terminated with [DONE]"

    @needs_openai
    async def test_openai_streaming_response_headers(self, client: AsyncClient) -> None:
        """Gateway must inject ``X-Request-ID`` and ``X-Provider`` headers."""
        response = await client.post(
            "/v1/chat/completions",
            json={"model": _OPENAI_MODEL, "messages": _SHORT_PROMPT, "stream": True},
        )
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Provider"] == "openai"
        assert response.headers.get("Cache-Control") == "no-cache"

    # ------------------------------------------------------------------
    # Anthropic non-streaming
    # ------------------------------------------------------------------

    @needs_anthropic
    async def test_anthropic_non_streaming_response_format(self, client: AsyncClient) -> None:
        """Non-streaming response must be a valid OpenAI-format JSON object."""
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": _ANTHROPIC_MODEL,
                "messages": _SHORT_PROMPT,
                "stream": False,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert isinstance(body["choices"], list)
        assert len(body["choices"]) == 1
        choice = body["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert len(choice["message"]["content"]) > 0

    @needs_anthropic
    async def test_anthropic_non_streaming_usage_tokens_present(self, client: AsyncClient) -> None:
        """``usage`` block must report positive token counts."""
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": _ANTHROPIC_MODEL,
                "messages": _SHORT_PROMPT,
                "stream": False,
            },
        )
        assert response.status_code == 200
        usage = response.json()["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    @needs_anthropic
    async def test_anthropic_non_streaming_response_time_under_10s(
        self, client: AsyncClient
    ) -> None:
        """A simple one-word reply from Claude Haiku must arrive within 10 seconds."""
        start = time.monotonic()
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": _ANTHROPIC_MODEL,
                "messages": _SHORT_PROMPT,
                "stream": False,
            },
        )
        elapsed = time.monotonic() - start
        assert response.status_code == 200
        assert elapsed < 10.0, f"Response took {elapsed:.2f}s — exceeded 10 s budget"

    @needs_anthropic
    async def test_anthropic_response_headers(self, client: AsyncClient) -> None:
        """Non-streaming response must carry gateway tracking headers."""
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": _ANTHROPIC_MODEL,
                "messages": _SHORT_PROMPT,
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Provider"] == "anthropic"

    # ------------------------------------------------------------------
    # Input validation (no API key needed — rejected before provider call)
    # ------------------------------------------------------------------

    async def test_missing_messages_field_returns_422(self, client: AsyncClient) -> None:
        """Pydantic validates the body; missing ``messages`` → 422."""
        response = await client.post(
            "/v1/chat/completions",
            json={"model": _ANTHROPIC_MODEL},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert any("messages" in str(err) for err in detail)

    async def test_invalid_temperature_returns_422(self, client: AsyncClient) -> None:
        """``temperature`` must be in [0.0, 2.0]; 5.0 → 422."""
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": _ANTHROPIC_MODEL,
                "messages": _SHORT_PROMPT,
                "temperature": 5.0,
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert any("temperature" in str(err) for err in detail)

    async def test_invalid_role_returns_400(self, client: AsyncClient) -> None:
        """Invalid message role passes Pydantic but fails CompletionRequest → 400."""
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": _ANTHROPIC_MODEL,
                "messages": [{"role": "badRole", "content": "hi"}],
            },
        )
        assert response.status_code == 400
        assert response.json()["detail"]["type"] == "invalid_request_error"

    @needs_anthropic
    async def test_invalid_model_name_returns_400(self, client: AsyncClient) -> None:
        """A well-formed but non-existent model string → 400 from LiteLLM."""
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-nonexistent-model-zzz-99999",
                "messages": _SHORT_PROMPT,
                "stream": False,
            },
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# 2. Observability Tests
# ---------------------------------------------------------------------------


class TestObservability:
    # ------------------------------------------------------------------
    # Prometheus
    # ------------------------------------------------------------------

    async def test_metrics_endpoint_is_accessible(self, client: AsyncClient) -> None:
        """/metrics returns 200 with Prometheus text format.

        The metrics sub-app is mounted at /metrics and Starlette redirects bare
        /metrics → /metrics/, so follow_redirects=True is required.
        """
        response = await client.get("/metrics", follow_redirects=True)
        assert response.status_code == 200
        ct = response.headers.get("content-type", "")
        assert "text/plain" in ct

    async def test_metrics_contains_process_metrics(self, client: AsyncClient) -> None:
        """Default prometheus_client process metrics are always present."""
        response = await client.get("/metrics", follow_redirects=True)
        body = response.text
        # python_info is emitted by prometheus_client for every Python process
        assert "python_info" in body or "process_virtual_memory_bytes" in body

    @needs_anthropic
    async def test_metrics_endpoint_still_works_after_completion_request(
        self, client: AsyncClient
    ) -> None:
        """Make a real completion, then verify /metrics is still healthy.

        When ``llm_requests_total`` is implemented as a custom Prometheus
        counter this test should be extended to assert the counter incremented:

            before = _read_counter(await client.get("/metrics"), "llm_requests_total")
            # ... make request ...
            after = _read_counter(await client.get("/metrics"), "llm_requests_total")
            assert after == before + 1
        """
        before = await client.get("/metrics", follow_redirects=True)
        assert before.status_code == 200

        await client.post(
            "/v1/chat/completions",
            json={
                "model": _ANTHROPIC_MODEL,
                "messages": _SHORT_PROMPT,
                "stream": False,
            },
        )

        after = await client.get("/metrics", follow_redirects=True)
        assert after.status_code == 200
        # Prometheus text body is non-empty and parseable
        assert len(after.text) > 0

    # ------------------------------------------------------------------
    # Jaeger distributed tracing
    # ------------------------------------------------------------------

    @needs_jaeger
    @needs_anthropic
    async def test_jaeger_trace_created_for_completion_request(self, client: AsyncClient) -> None:
        """A completion request must produce a trace visible in Jaeger.

        The ``BatchSpanProcessor`` exports asynchronously, so we wait up to
        5 seconds for the trace to appear before asserting.
        """
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": _ANTHROPIC_MODEL,
                "messages": _SHORT_PROMPT,
                "stream": False,
            },
        )
        assert response.status_code == 200
        response.headers.get("X-Request-ID", "")

        # Force the BatchSpanProcessor to flush pending spans to Jaeger
        # immediately rather than waiting for its default 5-second schedule.
        tracer_provider.force_flush(timeout_millis=5000)

        jaeger_resp = httpx.get(
            "http://localhost:16686/api/traces",
            params={"service": "llm-gateway", "limit": "10", "lookback": "5m"},
            timeout=5.0,
        )
        assert jaeger_resp.status_code == 200

        traces = jaeger_resp.json().get("data", [])
        assert len(traces) > 0, "No traces found in Jaeger for service llm-gateway"

        # Collect all span operation names across all recent traces.
        span_names: set[str] = set()
        for trace in traces:
            for span in trace.get("spans", []):
                span_names.add(span.get("operationName", ""))

        assert (
            "gateway.completions" in span_names
        ), f"Expected 'gateway.completions' span; found: {sorted(span_names)}"
        assert (
            "llm.generate" in span_names
        ), f"Expected 'llm.generate' span; found: {sorted(span_names)}"

    @needs_jaeger
    @needs_anthropic
    async def test_jaeger_span_attributes_present(self, client: AsyncClient) -> None:
        """GenAI semantic convention attributes must be set on the trace span."""
        await client.post(
            "/v1/chat/completions",
            json={
                "model": _ANTHROPIC_MODEL,
                "messages": _SHORT_PROMPT,
                "stream": False,
            },
        )
        tracer_provider.force_flush(timeout_millis=5000)

        jaeger_resp = httpx.get(
            "http://localhost:16686/api/traces",
            params={"service": "llm-gateway", "limit": "5", "lookback": "2m"},
            timeout=5.0,
        )
        assert jaeger_resp.status_code == 200
        traces = jaeger_resp.json().get("data", [])
        assert traces, "No traces in Jaeger"

        # Find the llm.generate span in the most recent trace.
        llm_span: dict | None = None
        for span in traces[0].get("spans", []):
            if span.get("operationName") == "llm.generate":
                llm_span = span
                break

        assert llm_span is not None, "llm.generate span not found in most recent trace"

        tag_keys = {tag["key"] for tag in llm_span.get("tags", [])}
        assert "gen_ai.system" in tag_keys
        assert "gen_ai.request.model" in tag_keys

    # ------------------------------------------------------------------
    # Health endpoints (quick sanity checks)
    # ------------------------------------------------------------------

    async def test_health_endpoint_returns_healthy(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    async def test_liveness_endpoint(self, client: AsyncClient) -> None:
        response = await client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"
