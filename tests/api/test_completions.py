"""Tests for POST /v1/chat/completions endpoint."""

import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from llmgateway.api.completions import _format_chunk, _provider_from_model
from llmgateway.cost import CostTracker
from llmgateway.main import app
from llmgateway.providers import (
    AuthError,
    CompletionChunk,
    InvalidRequestError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
)
from llmgateway.ratelimit import RateLimiter, RateLimitResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_BODY: dict = {
    "model": "claude-haiku-4-5-20251001",
    "messages": [{"role": "user", "content": "Hello"}],
}


def _make_chunks(*chunks: CompletionChunk) -> MagicMock:
    """Return a mock provider whose generate() yields *chunks*."""

    async def _gen(_req):
        for chunk in chunks:
            yield chunk

    mock = MagicMock()
    mock.generate = _gen
    return mock


def _make_error_provider(exc: Exception) -> MagicMock:
    """Return a mock provider whose generate() raises *exc* immediately."""

    async def _gen(_req):
        if True:  # always — makes the function an async generator via the yield below
            raise exc
        yield  # noqa: E402 — required to make _gen an async generator

    mock = MagicMock()
    mock.generate = _gen
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_provider():
    """Remove app.state provider, cache_manager, rate_limiter, cost_tracker after every test."""
    yield
    for attr in ("provider", "cache_manager", "rate_limiter", "cost_tracker"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# get_provider() dependency
# ---------------------------------------------------------------------------


class TestGetProvider:
    async def test_missing_provider_returns_503(self, client: AsyncClient) -> None:
        response = await client.post("/v1/chat/completions", json=_DEFAULT_BODY)
        assert response.status_code == 503

    async def test_provider_present_proceeds(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(
                content="hi",
                finish_reason="stop",
                usage={"input_tokens": 1, "output_tokens": 1},
                model="claude-haiku-4-5-20251001",
            )
        )
        response = await client.post("/v1/chat/completions", json=_DEFAULT_BODY)
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Non-streaming — success
# ---------------------------------------------------------------------------


class TestNonStreamingSuccess:
    async def test_returns_200(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(
                content="Hello!",
                finish_reason="stop",
                usage={"input_tokens": 5, "output_tokens": 3},
                model="claude-haiku-4-5-20251001",
            )
        )
        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )
        assert response.status_code == 200

    async def test_response_structure(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(
                content="Hello!",
                finish_reason="stop",
                usage={"input_tokens": 5, "output_tokens": 3},
                model="claude-haiku-4-5-20251001",
            )
        )
        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Hello!"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["prompt_tokens"] == 5
        assert body["usage"]["completion_tokens"] == 3
        assert body["usage"]["total_tokens"] == 8

    async def test_request_id_and_provider_headers(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(content="ok", finish_reason="stop", usage=None, model=None)
        )
        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )
        assert "X-Request-ID" in response.headers
        assert "X-Provider" in response.headers
        assert response.headers["X-Provider"] == "anthropic"

    async def test_no_usage_defaults_to_zero(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(content="hi", finish_reason="stop", usage=None, model=None)
        )
        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )
        body = response.json()
        assert body["usage"]["prompt_tokens"] == 0
        assert body["usage"]["completion_tokens"] == 0
        assert body["usage"]["total_tokens"] == 0

    async def test_multi_chunk_content_concatenated(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(content="He", finish_reason=None, usage=None, model=None),
            CompletionChunk(
                content="llo",
                finish_reason="stop",
                usage={"input_tokens": 2, "output_tokens": 2},
                model=None,
            ),
        )
        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )
        assert response.json()["choices"][0]["message"]["content"] == "Hello"

    async def test_chunk_model_used_in_response(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(
                content="ok",
                finish_reason="stop",
                usage=None,
                model="claude-3-5-sonnet-20241022",
            )
        )
        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )
        assert response.json()["model"] == "claude-3-5-sonnet-20241022"

    async def test_user_field_forwarded(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(content="ok", finish_reason="stop", usage=None, model=None)
        )
        response = await client.post(
            "/v1/chat/completions",
            json={**_DEFAULT_BODY, "stream": False, "user": "user-42"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Non-streaming — errors
# ---------------------------------------------------------------------------


class TestNonStreamingErrors:
    @pytest.mark.parametrize(
        "exc, expected_status",
        [
            (AuthError("bad key", provider="anthropic"), 401),
            (TimeoutError("timed out", provider="anthropic"), 504),
            (InvalidRequestError("bad request", provider="anthropic"), 400),
            (ProviderUnavailableError("down", provider="anthropic"), 502),
            (ProviderError("unknown", provider="anthropic"), 500),
        ],
    )
    async def test_error_status_codes(
        self, client: AsyncClient, exc: ProviderError, expected_status: int
    ) -> None:
        app.state.provider = _make_error_provider(exc)
        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )
        assert response.status_code == expected_status

    async def test_rate_limit_with_retry_after_header(self, client: AsyncClient) -> None:
        exc = RateLimitError("rate limited", retry_after=30.0, provider="anthropic")
        app.state.provider = _make_error_provider(exc)
        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )
        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "30"

    async def test_rate_limit_without_retry_after_omits_header(self, client: AsyncClient) -> None:
        exc = RateLimitError("rate limited", retry_after=None, provider="anthropic")
        app.state.provider = _make_error_provider(exc)
        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )
        assert response.status_code == 429
        assert "Retry-After" not in response.headers

    async def test_invalid_role_triggers_400_before_provider(self, client: AsyncClient) -> None:
        """InvalidRequestError from CompletionRequest validation → HTTP 400."""
        app.state.provider = MagicMock()  # never called
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "badRole", "content": "hi"}],
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert body["detail"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    async def test_returns_200_with_event_stream_content_type(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(content="Hello", finish_reason=None, usage=None, model=None),
            CompletionChunk(
                content="!",
                finish_reason="stop",
                usage={"input_tokens": 5, "output_tokens": 2},
                model=None,
            ),
        )
        response = await client.post("/v1/chat/completions", json={**_DEFAULT_BODY, "stream": True})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    async def test_contains_done_marker(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(content="hi", finish_reason="stop", usage=None, model=None),
        )
        response = await client.post("/v1/chat/completions", json={**_DEFAULT_BODY, "stream": True})
        assert "data: [DONE]" in response.text

    async def test_chunks_are_valid_sse_json(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(
                content="hello",
                finish_reason="stop",
                usage={"input_tokens": 1, "output_tokens": 1},
                model=None,
            ),
        )
        response = await client.post("/v1/chat/completions", json={**_DEFAULT_BODY, "stream": True})
        data_lines = [
            line[6:]  # strip "data: "
            for line in response.text.splitlines()
            if line.startswith("data: ") and "[DONE]" not in line
        ]
        for raw in data_lines:
            parsed = json.loads(raw)
            assert "id" in parsed
            assert "choices" in parsed

    async def test_streaming_error_yields_error_sse_event(self, client: AsyncClient) -> None:
        exc = RateLimitError("rate limited", provider="anthropic")
        app.state.provider = _make_error_provider(exc)
        response = await client.post("/v1/chat/completions", json={**_DEFAULT_BODY, "stream": True})
        # HTTP 200 because headers were already committed
        assert response.status_code == 200
        assert "RateLimitError" in response.text
        assert "data: [DONE]" in response.text

    async def test_streaming_usage_in_chunk(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(
                content="x",
                finish_reason="stop",
                usage={"input_tokens": 3, "output_tokens": 7},
                model=None,
            ),
        )
        response = await client.post("/v1/chat/completions", json={**_DEFAULT_BODY, "stream": True})
        data_lines = [
            line[6:]
            for line in response.text.splitlines()
            if line.startswith("data: ") and "[DONE]" not in line
        ]
        data = json.loads(data_lines[0])
        assert data["usage"]["prompt_tokens"] == 3
        assert data["usage"]["completion_tokens"] == 7


# ---------------------------------------------------------------------------
# _provider_from_model() unit tests
# ---------------------------------------------------------------------------


class TestProviderFromModel:
    @pytest.mark.parametrize(
        "model, expected",
        [
            ("groq/llama-3.1-70b-versatile", "groq"),
            ("together_ai/meta-llama/Llama-3-70b", "together_ai"),
            ("azure/gpt-4o", "azure"),
            ("gpt-4o", "openai"),
            ("gpt-4o-mini", "openai"),
            ("o1-mini", "openai"),
            ("o3-mini", "openai"),
            ("claude-haiku-4-5-20251001", "anthropic"),
            ("claude-3-5-sonnet-20241022", "anthropic"),
            ("gemini-pro", "google"),
            ("command-r-plus", "cohere"),
            ("mistral-large-latest", "mistral"),
            ("some-unknown-model", "unknown"),
        ],
    )
    def test_extraction(self, model: str, expected: str) -> None:
        assert _provider_from_model(model) == expected


# ---------------------------------------------------------------------------
# _format_chunk() unit tests
# ---------------------------------------------------------------------------


class TestFormatChunk:
    def test_content_in_delta(self) -> None:
        chunk = CompletionChunk(content="hello", finish_reason=None, usage=None, model=None)
        data = _format_chunk(chunk, "req-1", 1234, "claude-haiku-4-5-20251001")
        assert data["choices"][0]["delta"] == {"content": "hello"}

    def test_empty_content_yields_empty_delta(self) -> None:
        chunk = CompletionChunk(content="", finish_reason="stop", usage=None, model=None)
        data = _format_chunk(chunk, "req-1", 1234, "claude-haiku-4-5-20251001")
        assert data["choices"][0]["delta"] == {}

    def test_with_usage(self) -> None:
        chunk = CompletionChunk(
            content="x",
            finish_reason=None,
            usage={"input_tokens": 5, "output_tokens": 3},
            model=None,
        )
        data = _format_chunk(chunk, "req-1", 1234, "claude-haiku-4-5-20251001")
        assert data["usage"]["prompt_tokens"] == 5
        assert data["usage"]["completion_tokens"] == 3
        assert data["usage"]["total_tokens"] == 8

    def test_without_usage_no_usage_key(self) -> None:
        chunk = CompletionChunk(content="x", finish_reason=None, usage=None, model=None)
        data = _format_chunk(chunk, "req-1", 1234, "claude-haiku-4-5-20251001")
        assert "usage" not in data

    def test_chunk_model_overrides_default(self) -> None:
        chunk = CompletionChunk(
            content="x", finish_reason=None, usage=None, model="claude-3-5-sonnet-20241022"
        )
        data = _format_chunk(chunk, "req-1", 1234, "default-model")
        assert data["model"] == "claude-3-5-sonnet-20241022"

    def test_default_model_used_when_no_chunk_model(self) -> None:
        chunk = CompletionChunk(content="x", finish_reason=None, usage=None, model=None)
        data = _format_chunk(chunk, "req-1", 1234, "default-model")
        assert data["model"] == "default-model"

    def test_response_id_has_chatcmpl_prefix(self) -> None:
        chunk = CompletionChunk(content="x", finish_reason=None, usage=None, model=None)
        data = _format_chunk(chunk, "abc123", 1234, "model")
        assert data["id"] == "chatcmpl-abc123"

    def test_object_type_is_chunk(self) -> None:
        chunk = CompletionChunk(content="x", finish_reason=None, usage=None, model=None)
        data = _format_chunk(chunk, "req-1", 1234, "model")
        assert data["object"] == "chat.completion.chunk"


# ---------------------------------------------------------------------------
# Cache hit / miss paths
# ---------------------------------------------------------------------------

_CACHED_RESPONSE: dict = {
    "id": "chatcmpl-cached",
    "object": "chat.completion",
    "created": 1000,
    "model": "claude-haiku-4-5-20251001",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "cached!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


class TestCachePaths:
    async def test_exact_cache_hit_returns_hit_headers(self, client: AsyncClient) -> None:
        mock_cache = AsyncMock()
        mock_cache.get_cached_response.return_value = _CACHED_RESPONSE
        app.state.cache_manager = mock_cache
        app.state.provider = MagicMock()  # should not be reached

        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )

        assert response.status_code == 200
        assert response.headers.get("X-Cache-Status") == "HIT"
        assert response.headers.get("X-Cache-Type") == "EXACT"
        assert response.json()["choices"][0]["message"]["content"] == "cached!"

    async def test_semantic_cache_hit_returns_semantic_headers(self, client: AsyncClient) -> None:
        mock_cache = AsyncMock()
        mock_cache.get_cached_response.return_value = None  # exact miss
        mock_cache.get_semantic_match.return_value = (_CACHED_RESPONSE, 0.97)
        app.state.cache_manager = mock_cache
        app.state.provider = MagicMock()

        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )

        assert response.status_code == 200
        assert response.headers.get("X-Cache-Status") == "HIT"
        assert response.headers.get("X-Cache-Type") == "SEMANTIC"
        assert response.headers.get("X-Cache-Similarity") == "0.97"
        assert response.json()["choices"][0]["message"]["content"] == "cached!"

    async def test_cache_miss_type_header_is_miss(self, client: AsyncClient) -> None:
        mock_cache = AsyncMock()
        mock_cache.get_cached_response.return_value = None
        mock_cache.get_semantic_match.return_value = None
        app.state.cache_manager = mock_cache
        app.state.provider = _make_chunks(
            CompletionChunk(content="fresh", finish_reason="stop", usage=None, model=None)
        )

        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )

        assert response.status_code == 200
        assert response.headers.get("X-Cache-Type") == "MISS"

    async def test_cache_miss_stores_in_both_caches(self, client: AsyncClient) -> None:
        mock_cache = AsyncMock()
        mock_cache.get_cached_response.return_value = None
        mock_cache.get_semantic_match.return_value = None
        app.state.cache_manager = mock_cache
        app.state.provider = _make_chunks(
            CompletionChunk(
                content="fresh",
                finish_reason="stop",
                usage={"input_tokens": 2, "output_tokens": 2},
                model=None,
            )
        )

        await client.post("/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False})

        mock_cache.cache_response.assert_called_once()
        mock_cache.cache_with_embedding.assert_called_once()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

import time as _time  # noqa: E402


def _rl_result(*, allowed: bool = True, remaining: float = 19.0, retry_after: float = 0.0):
    return RateLimitResult(
        allowed=allowed,
        retry_after=retry_after,
        remaining=remaining,
        reset_time=_time.time() + 60,
        limit=20.0,
    )


class TestRateLimiting:
    async def test_exceeded_returns_429(self, client: AsyncClient) -> None:
        mock_rl = AsyncMock(spec=RateLimiter)
        mock_rl.check_rate_limit.return_value = _rl_result(
            allowed=False, remaining=0.0, retry_after=6.0
        )
        app.state.rate_limiter = mock_rl
        app.state.provider = MagicMock()

        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )

        assert response.status_code == 429

    async def test_exceeded_includes_retry_after_header(self, client: AsyncClient) -> None:
        mock_rl = AsyncMock(spec=RateLimiter)
        mock_rl.check_rate_limit.return_value = _rl_result(
            allowed=False, remaining=0.0, retry_after=6.0
        )
        app.state.rate_limiter = mock_rl
        app.state.provider = MagicMock()

        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )

        assert "Retry-After" in response.headers
        assert int(response.headers["Retry-After"]) >= 6

    async def test_exceeded_includes_ratelimit_headers(self, client: AsyncClient) -> None:
        mock_rl = AsyncMock(spec=RateLimiter)
        mock_rl.check_rate_limit.return_value = _rl_result(
            allowed=False, remaining=0.0, retry_after=10.0
        )
        app.state.rate_limiter = mock_rl
        app.state.provider = MagicMock()

        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )

        assert response.headers.get("X-RateLimit-Limit") == "20"
        assert response.headers.get("X-RateLimit-Remaining") == "0"
        assert "X-RateLimit-Reset" in response.headers

    async def test_success_includes_ratelimit_headers(self, client: AsyncClient) -> None:
        mock_rl = AsyncMock(spec=RateLimiter)
        mock_rl.check_rate_limit.return_value = _rl_result(allowed=True, remaining=18.0)
        app.state.rate_limiter = mock_rl
        app.state.provider = _make_chunks(
            CompletionChunk(content="ok", finish_reason="stop", usage=None, model=None)
        )

        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )

        assert response.status_code == 200
        assert response.headers.get("X-RateLimit-Limit") == "20"
        assert response.headers.get("X-RateLimit-Remaining") == "18"
        assert "X-RateLimit-Reset" in response.headers

    async def test_no_rate_limiter_returns_200(self, client: AsyncClient) -> None:
        """When rate_limiter is absent from app.state the endpoint still works."""
        app.state.provider = _make_chunks(
            CompletionChunk(content="ok", finish_reason="stop", usage=None, model=None)
        )

        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )

        assert response.status_code == 200

    async def test_user_field_used_as_user_id(self, client: AsyncClient) -> None:
        mock_rl = AsyncMock(spec=RateLimiter)
        mock_rl.check_rate_limit.return_value = _rl_result(allowed=True, remaining=19.0)
        app.state.rate_limiter = mock_rl
        app.state.provider = _make_chunks(
            CompletionChunk(content="ok", finish_reason="stop", usage=None, model=None)
        )

        await client.post(
            "/v1/chat/completions",
            json={**_DEFAULT_BODY, "stream": False, "user": "alice"},
        )

        mock_rl.check_rate_limit.assert_called_once()
        user_id_arg = mock_rl.check_rate_limit.call_args[0][0]
        assert user_id_arg == "alice"

    async def test_x_user_id_header_takes_precedence(self, client: AsyncClient) -> None:
        mock_rl = AsyncMock(spec=RateLimiter)
        mock_rl.check_rate_limit.return_value = _rl_result(allowed=True, remaining=19.0)
        app.state.rate_limiter = mock_rl
        app.state.provider = _make_chunks(
            CompletionChunk(content="ok", finish_reason="stop", usage=None, model=None)
        )

        await client.post(
            "/v1/chat/completions",
            json={**_DEFAULT_BODY, "stream": False, "user": "body-user"},
            headers={"X-User-ID": "header-user"},
        )

        user_id_arg = mock_rl.check_rate_limit.call_args[0][0]
        assert user_id_arg == "header-user"

    async def test_anonymous_fallback_when_no_user(self, client: AsyncClient) -> None:
        mock_rl = AsyncMock(spec=RateLimiter)
        mock_rl.check_rate_limit.return_value = _rl_result(allowed=True, remaining=19.0)
        app.state.rate_limiter = mock_rl
        app.state.provider = _make_chunks(
            CompletionChunk(content="ok", finish_reason="stop", usage=None, model=None)
        )

        await client.post("/v1/chat/completions", json=_DEFAULT_BODY)

        user_id_arg = mock_rl.check_rate_limit.call_args[0][0]
        assert user_id_arg == "anonymous"


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


def _mock_cost_tracker() -> CostTracker:
    tracker = AsyncMock(spec=CostTracker)
    tracker.record_usage = AsyncMock(return_value=None)
    tracker.get_daily_cost = AsyncMock(return_value=0.5)
    return tracker


class TestCostTracking:
    async def test_x_cost_header_present_on_live_response(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(
                content="hi",
                finish_reason="stop",
                usage={"input_tokens": 10, "output_tokens": 5},
                model="claude-haiku-4-5-20251001",
            )
        )
        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )
        assert response.status_code == 200
        assert "X-Cost" in response.headers

    async def test_x_cost_is_numeric(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(
                content="hi",
                finish_reason="stop",
                usage={"input_tokens": 100, "output_tokens": 50},
                model="claude-haiku-4-5-20251001",
            )
        )
        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )
        cost_val = float(response.headers["X-Cost"])
        assert cost_val >= 0.0

    async def test_x_cost_zero_when_no_usage(self, client: AsyncClient) -> None:
        app.state.provider = _make_chunks(
            CompletionChunk(content="hi", finish_reason="stop", usage=None, model=None)
        )
        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )
        assert float(response.headers["X-Cost"]) == 0.0

    async def test_x_cost_present_on_exact_cache_hit(self, client: AsyncClient) -> None:
        mock_cache = AsyncMock()
        mock_cache.get_cached_response.return_value = _CACHED_RESPONSE
        app.state.cache_manager = mock_cache
        app.state.provider = MagicMock()

        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )

        assert "X-Cost" in response.headers
        assert response.headers["X-Cost"] == "0.00000000"

    async def test_x_cost_present_on_semantic_cache_hit(self, client: AsyncClient) -> None:
        mock_cache = AsyncMock()
        mock_cache.get_cached_response.return_value = None
        mock_cache.get_semantic_match.return_value = (_CACHED_RESPONSE, 0.97)
        app.state.cache_manager = mock_cache
        app.state.provider = MagicMock()

        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )

        assert "X-Cost" in response.headers
        assert response.headers["X-Cost"] == "0.00000000"

    async def test_no_cost_tracker_still_returns_200(self, client: AsyncClient) -> None:
        """Endpoint works fine without a cost_tracker in app.state."""
        app.state.provider = _make_chunks(
            CompletionChunk(content="ok", finish_reason="stop", usage=None, model=None)
        )
        response = await client.post(
            "/v1/chat/completions", json={**_DEFAULT_BODY, "stream": False}
        )
        assert response.status_code == 200
