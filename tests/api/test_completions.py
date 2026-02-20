"""Tests for POST /v1/chat/completions endpoint."""

import json
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from llmgateway.api.completions import _format_chunk, _provider_from_model
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
        yield  # noqa: unreachable — required to make _gen an async generator

    mock = MagicMock()
    mock.generate = _gen
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_provider():
    """Remove app.state.provider after every test to prevent cross-test leakage."""
    yield
    if hasattr(app.state, "provider"):
        del app.state.provider


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
