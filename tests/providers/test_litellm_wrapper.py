"""Unit tests for LLMGatewayProvider (litellm_wrapper.py).

Mocking strategy
----------------
* ``litellm.acompletion`` is patched at
  ``llmgateway.providers.litellm_wrapper.litellm.acompletion`` to intercept
  every provider call without hitting a real API.
* ``litellm.token_counter`` is patched similarly for token-counting tests.
* ``tenacity.asyncio._portable_async_sleep`` is replaced with an AsyncMock to
  prevent real backoff delays during retry tests.
* The module-level ``_tracer`` in ``litellm_wrapper`` is replaced with a
  MagicMock to capture OpenTelemetry span operations.

No real API calls are made in this test suite.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import litellm
import pytest

from llmgateway.providers.errors import (
    AuthError,
    InvalidRequestError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
)
from llmgateway.providers.litellm_wrapper import LLMGatewayProvider
from llmgateway.providers.models import CompletionRequest

# ---------------------------------------------------------------------------
# Mock helpers – lightweight stand-ins for LiteLLM response objects
# ---------------------------------------------------------------------------


class _Delta:
    """Mimics litellm's streaming delta object."""

    def __init__(self, content: str | None = None) -> None:
        self.content = content


class _StreamChoice:
    def __init__(self, content: str | None, finish_reason: str | None) -> None:
        self.delta = _Delta(content)
        self.finish_reason = finish_reason


class _StreamUsage:
    """Token usage as it appears on the final streaming chunk."""

    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _StreamChunk:
    """Mimics one chunk emitted by a LiteLLM streaming response."""

    def __init__(
        self,
        content: str | None = None,
        finish_reason: str | None = None,
        usage: _StreamUsage | None = None,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.model = model
        self.choices = [_StreamChoice(content, finish_reason)]
        self.usage = usage


class _AsyncStreamResponse:
    """Async-iterable wrapper around a list of _StreamChunk objects.

    Returned by the mocked ``litellm.acompletion`` when ``stream=True`` so
    that ``_generate_streaming`` can iterate over it normally.
    """

    def __init__(self, chunks: list[_StreamChunk]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> "_AsyncStreamResponse":
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self) -> _StreamChunk:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _ResponseMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _ResponseChoice:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.message = _ResponseMessage(content)
        self.finish_reason = finish_reason


class _CompletionResponse:
    """Mimics a LiteLLM non-streaming completion response object."""

    def __init__(
        self,
        content: str = "Hello!",
        finish_reason: str = "stop",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.model = model
        self.choices = [_ResponseChoice(content, finish_reason)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


# ---------------------------------------------------------------------------
# LiteLLM exception factories
# Uses real constructors so ``isinstance`` checks in ``_map_error`` pass.
# ---------------------------------------------------------------------------

_DUMMY_REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
_DUMMY_RESP_401 = httpx.Response(401, request=_DUMMY_REQ)
_DUMMY_RESP_429 = httpx.Response(429, request=_DUMMY_REQ)
_DUMMY_RESP_400 = httpx.Response(400, request=_DUMMY_REQ)
_DUMMY_RESP_503 = httpx.Response(503, request=_DUMMY_REQ)


def _auth_error() -> litellm.AuthenticationError:
    return litellm.AuthenticationError(
        message="Missing or invalid API key",
        llm_provider="anthropic",
        model="claude-haiku-4-5-20251001",
        response=_DUMMY_RESP_401,
    )


def _rate_limit_error(retry_after: float | None = None) -> litellm.RateLimitError:
    exc = litellm.RateLimitError(
        message="Rate limit exceeded",
        llm_provider="anthropic",
        model="claude-haiku-4-5-20251001",
        response=_DUMMY_RESP_429,
    )
    exc.retry_after = retry_after
    return exc


def _timeout_error() -> litellm.Timeout:
    return litellm.Timeout(
        message="Request timed out",
        model="claude-haiku-4-5-20251001",
        llm_provider="anthropic",
    )


def _bad_request_error() -> litellm.BadRequestError:
    return litellm.BadRequestError(
        message="Invalid request",
        llm_provider="anthropic",
        model="claude-haiku-4-5-20251001",
        response=_DUMMY_RESP_400,
    )


def _service_unavailable_error() -> litellm.ServiceUnavailableError:
    return litellm.ServiceUnavailableError(
        message="Service unavailable",
        llm_provider="anthropic",
        model="claude-haiku-4-5-20251001",
        response=_DUMMY_RESP_503,
    )


def _api_connection_error() -> litellm.APIConnectionError:
    return litellm.APIConnectionError(
        message="Connection failed",
        llm_provider="anthropic",
        model="claude-haiku-4-5-20251001",
        request=_DUMMY_REQ,
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> LLMGatewayProvider:
    """Provider with a 3-retry budget and short timeout."""
    return LLMGatewayProvider(timeout=5, max_retries=3)


@pytest.fixture
def non_streaming_request() -> CompletionRequest:
    return CompletionRequest(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
    )


@pytest.fixture
def streaming_request() -> CompletionRequest:
    return CompletionRequest(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )


@pytest.fixture
def mock_span() -> MagicMock:
    """A MagicMock that behaves as an OTel span context manager."""
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    return span


@pytest.fixture
def mock_tracer(mock_span: MagicMock) -> MagicMock:
    """A MagicMock tracer whose start_as_current_span always returns mock_span."""
    tracer = MagicMock()
    tracer.start_as_current_span.return_value = mock_span
    return tracer


# ---------------------------------------------------------------------------
# 1. CompletionRequest validation
# ---------------------------------------------------------------------------


class TestCompletionRequestValidation:
    """CompletionRequest enforces invariants at construction time."""

    def test_valid_request_accepted(self) -> None:
        """A well-formed request is constructed without raising."""
        req = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=1.0,
            max_tokens=100,
            stream=True,
            user_id="user-42",
        )
        assert req.model == "claude-haiku-4-5-20251001"
        assert req.temperature == 1.0
        assert req.max_tokens == 100

    def test_temperature_above_maximum_raises(self) -> None:
        """temperature > 2.0 raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError, match="temperature"):
            CompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "x"}],
                temperature=2.01,
            )

    def test_temperature_below_minimum_raises(self) -> None:
        """temperature < 0.0 raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError, match="temperature"):
            CompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "x"}],
                temperature=-0.01,
            )

    def test_temperature_boundary_values_accepted(self) -> None:
        """Exact boundary values 0.0 and 2.0 are valid."""
        CompletionRequest(model="m", messages=[{"role": "user", "content": "x"}], temperature=0.0)
        CompletionRequest(model="m", messages=[{"role": "user", "content": "x"}], temperature=2.0)

    def test_empty_messages_list_raises(self) -> None:
        """An empty messages list raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError, match="messages"):
            CompletionRequest(model="m", messages=[])

    def test_invalid_message_role_raises(self) -> None:
        """An unrecognised role raises InvalidRequestError naming the bad role."""
        with pytest.raises(InvalidRequestError, match="robot"):
            CompletionRequest(
                model="m",
                messages=[{"role": "robot", "content": "Hi"}],
            )

    def test_message_missing_content_key_raises(self) -> None:
        """A message without a 'content' key raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError, match="content"):
            CompletionRequest(model="m", messages=[{"role": "user"}])

    def test_message_missing_role_key_raises(self) -> None:
        """A message without a 'role' key raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError, match="role"):
            CompletionRequest(model="m", messages=[{"content": "Hi"}])

    def test_all_valid_roles_accepted(self) -> None:
        """Every documented role (system, user, assistant, tool, function) is valid."""
        for role in ("system", "user", "assistant", "tool", "function"):
            CompletionRequest(
                model="m",
                messages=[{"role": role, "content": "x"}],
            )

    def test_zero_max_tokens_raises(self) -> None:
        """max_tokens=0 raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError, match="max_tokens"):
            CompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "x"}],
                max_tokens=0,
            )

    def test_negative_max_tokens_raises(self) -> None:
        """Negative max_tokens raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError, match="max_tokens"):
            CompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "x"}],
                max_tokens=-1,
            )

    def test_whitespace_only_model_raises(self) -> None:
        """A model string containing only whitespace raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError, match="model"):
            CompletionRequest(model="   ", messages=[{"role": "user", "content": "Hi"}])

    def test_multiple_messages_all_validated(self) -> None:
        """Validation is applied to every message in the list, not just the first."""
        with pytest.raises(InvalidRequestError, match="messages\\[1\\]"):
            CompletionRequest(
                model="m",
                messages=[
                    {"role": "user", "content": "ok"},
                    {"role": "bot", "content": "bad role"},
                ],
            )


# ---------------------------------------------------------------------------
# 2. Streaming success cases
# ---------------------------------------------------------------------------


class TestStreamingSuccess:
    """generate() yields CompletionChunk objects for streaming requests."""

    async def test_chunks_yielded_in_order(
        self, provider: LLMGatewayProvider, streaming_request: CompletionRequest, mocker: Any
    ) -> None:
        """Each streaming token becomes a separate CompletionChunk in arrival order."""
        raw_chunks = [
            _StreamChunk(content="Hello"),
            _StreamChunk(content=" world"),
            _StreamChunk(content=None, finish_reason="stop"),
        ]
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_AsyncStreamResponse(raw_chunks)),
        )

        chunks = [chunk async for chunk in provider.generate(streaming_request)]

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[2].finish_reason == "stop"

    async def test_intermediate_chunks_have_no_finish_reason(
        self, provider: LLMGatewayProvider, streaming_request: CompletionRequest, mocker: Any
    ) -> None:
        """finish_reason is None on intermediate chunks; only the final chunk sets it."""
        raw_chunks = [
            _StreamChunk(content="A"),
            _StreamChunk(content=None, finish_reason="stop"),
        ]
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_AsyncStreamResponse(raw_chunks)),
        )

        chunks = [chunk async for chunk in provider.generate(streaming_request)]

        assert chunks[0].finish_reason is None
        assert chunks[1].finish_reason == "stop"

    async def test_usage_extracted_from_final_chunk(
        self, provider: LLMGatewayProvider, streaming_request: CompletionRequest, mocker: Any
    ) -> None:
        """Token usage on the final streaming chunk is mapped to CompletionChunk.usage."""
        usage = _StreamUsage(prompt_tokens=15, completion_tokens=8)
        raw_chunks = [
            _StreamChunk(content="Hi"),
            _StreamChunk(content=None, finish_reason="stop", usage=usage),
        ]
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_AsyncStreamResponse(raw_chunks)),
        )

        chunks = [chunk async for chunk in provider.generate(streaming_request)]

        final = chunks[-1]
        assert final.usage is not None
        assert final.usage["input_tokens"] == 15
        assert final.usage["output_tokens"] == 8

    async def test_model_name_propagated_from_chunk(
        self, provider: LLMGatewayProvider, streaming_request: CompletionRequest, mocker: Any
    ) -> None:
        """The model name reported inside each streaming chunk is forwarded."""
        raw_chunks = [_StreamChunk(content="x", model="claude-haiku-4-5-20251001")]
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_AsyncStreamResponse(raw_chunks)),
        )

        chunks = [chunk async for chunk in provider.generate(streaming_request)]

        assert chunks[0].model == "claude-haiku-4-5-20251001"

    async def test_acompletion_called_with_stream_true(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """acompletion receives stream=True and the full request parameters."""
        mock_acompletion = mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(
                return_value=_AsyncStreamResponse(
                    [_StreamChunk(content=None, finish_reason="stop")]
                )
            ),
        )
        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Count"}],
            temperature=0.5,
            max_tokens=50,
            stream=True,
            user_id="u1",
        )

        async for _ in provider.generate(request):
            pass

        kw = mock_acompletion.call_args.kwargs
        assert kw["model"] == "claude-haiku-4-5-20251001"
        assert kw["temperature"] == 0.5
        assert kw["max_tokens"] == 50
        assert kw["stream"] is True
        assert kw["user"] == "u1"

    async def test_optional_params_omitted_when_none(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """max_tokens and user are not forwarded to acompletion when unset."""
        mock_acompletion = mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(
                return_value=_AsyncStreamResponse(
                    [_StreamChunk(content=None, finish_reason="stop")]
                )
            ),
        )
        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        async for _ in provider.generate(request):
            pass

        kw = mock_acompletion.call_args.kwargs
        assert "max_tokens" not in kw
        assert "user" not in kw

    async def test_empty_content_chunks_not_yielded(
        self, provider: LLMGatewayProvider, streaming_request: CompletionRequest, mocker: Any
    ) -> None:
        """Chunks with no content and no finish_reason are filtered out."""
        raw_chunks = [
            _StreamChunk(content=None, finish_reason=None),  # empty, no finish – filtered
            _StreamChunk(content="Hi"),
            _StreamChunk(content=None, finish_reason="stop"),
        ]
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_AsyncStreamResponse(raw_chunks)),
        )

        chunks = [chunk async for chunk in provider.generate(streaming_request)]

        assert len(chunks) == 2
        assert chunks[0].content == "Hi"
        assert chunks[1].finish_reason == "stop"


# ---------------------------------------------------------------------------
# 3. Non-streaming success cases
# ---------------------------------------------------------------------------


class TestNonStreamingSuccess:
    """generate() yields a single CompletionChunk for non-streaming requests."""

    async def test_exactly_one_chunk_yielded(
        self,
        provider: LLMGatewayProvider,
        non_streaming_request: CompletionRequest,
        mocker: Any,
    ) -> None:
        """Non-streaming generate always yields exactly one chunk."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_CompletionResponse(content="4")),
        )

        chunks = [chunk async for chunk in provider.generate(non_streaming_request)]

        assert len(chunks) == 1

    async def test_content_extracted_correctly(
        self,
        provider: LLMGatewayProvider,
        non_streaming_request: CompletionRequest,
        mocker: Any,
    ) -> None:
        """The full response text is placed in CompletionChunk.content."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_CompletionResponse(content="The answer is 42.")),
        )

        chunks = [chunk async for chunk in provider.generate(non_streaming_request)]

        assert chunks[0].content == "The answer is 42."

    async def test_usage_tokens_mapped_correctly(
        self,
        provider: LLMGatewayProvider,
        non_streaming_request: CompletionRequest,
        mocker: Any,
    ) -> None:
        """prompt_tokens and completion_tokens are mapped to input/output_tokens."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(
                return_value=_CompletionResponse(prompt_tokens=20, completion_tokens=10)
            ),
        )

        chunks = [chunk async for chunk in provider.generate(non_streaming_request)]

        assert chunks[0].usage == {"input_tokens": 20, "output_tokens": 10}

    async def test_finish_reason_extracted(
        self,
        provider: LLMGatewayProvider,
        non_streaming_request: CompletionRequest,
        mocker: Any,
    ) -> None:
        """The provider's finish_reason is forwarded unchanged."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_CompletionResponse(finish_reason="length")),
        )

        chunks = [chunk async for chunk in provider.generate(non_streaming_request)]

        assert chunks[0].finish_reason == "length"

    async def test_model_name_from_response_forwarded(
        self,
        provider: LLMGatewayProvider,
        non_streaming_request: CompletionRequest,
        mocker: Any,
    ) -> None:
        """The resolved model name in the LiteLLM response is preserved."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_CompletionResponse(model="claude-haiku-resolved")),
        )

        chunks = [chunk async for chunk in provider.generate(non_streaming_request)]

        assert chunks[0].model == "claude-haiku-resolved"

    async def test_acompletion_called_with_stream_false(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """acompletion receives stream=False for non-streaming requests."""
        mock_acompletion = mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_CompletionResponse()),
        )
        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )

        async for _ in provider.generate(request):
            pass

        assert mock_acompletion.call_args.kwargs["stream"] is False


# ---------------------------------------------------------------------------
# 4. Error mapping
# ---------------------------------------------------------------------------


class TestErrorMapping:
    """litellm exceptions are converted to the correct gateway ProviderError subclass."""

    async def _assert_maps_to(
        self,
        provider: LLMGatewayProvider,
        mocker: Any,
        litellm_exc: Exception,
        expected_type: type,
        model: str = "claude-haiku-4-5-20251001",
    ) -> ProviderError:
        """Helper: patch acompletion to raise *litellm_exc*, run generate, assert type."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(side_effect=litellm_exc),
        )
        request = CompletionRequest(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )
        with pytest.raises(expected_type) as exc_info:
            async for _ in provider.generate(request):
                pass
        return exc_info.value

    async def test_authentication_error_maps_to_auth_error(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """litellm.AuthenticationError → AuthError."""
        err = await self._assert_maps_to(provider, mocker, _auth_error(), AuthError)
        assert err.provider == "anthropic"
        assert isinstance(err.original_error, litellm.AuthenticationError)

    async def test_rate_limit_error_maps_with_retry_after(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """litellm.RateLimitError → RateLimitError; retry_after is extracted."""
        err = await self._assert_maps_to(
            provider, mocker, _rate_limit_error(retry_after=30.0), RateLimitError
        )
        assert err.provider == "anthropic"
        assert err.retry_after == 30.0

    async def test_rate_limit_error_without_retry_after(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """retry_after is None when the litellm exception does not carry it."""
        err = await self._assert_maps_to(
            provider, mocker, _rate_limit_error(), RateLimitError
        )
        assert err.retry_after is None

    async def test_timeout_maps_to_timeout_error(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """litellm.Timeout → TimeoutError."""
        err = await self._assert_maps_to(provider, mocker, _timeout_error(), TimeoutError)
        assert err.provider == "anthropic"

    async def test_bad_request_maps_to_invalid_request_error(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """litellm.BadRequestError → InvalidRequestError."""
        err = await self._assert_maps_to(
            provider, mocker, _bad_request_error(), InvalidRequestError
        )
        assert err.provider == "anthropic"

    async def test_service_unavailable_maps_to_provider_unavailable(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """litellm.ServiceUnavailableError → ProviderUnavailableError."""
        err = await self._assert_maps_to(
            provider, mocker, _service_unavailable_error(), ProviderUnavailableError
        )
        assert err.provider == "anthropic"

    async def test_api_connection_error_maps_to_provider_unavailable(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """litellm.APIConnectionError → ProviderUnavailableError."""
        await self._assert_maps_to(
            provider, mocker, _api_connection_error(), ProviderUnavailableError
        )

    async def test_unknown_exception_maps_to_base_provider_error(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """An unexpected exception not in the mapping table wraps as ProviderError."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(side_effect=RuntimeError("something strange")),
        )
        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )
        with pytest.raises(ProviderError) as exc_info:
            async for _ in provider.generate(request):
                pass
        # Must be the exact base class, not a subclass
        assert type(exc_info.value) is ProviderError

    async def test_provider_extracted_from_slash_prefix(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """Provider is taken from the part before '/' in a prefixed model string."""
        err = await self._assert_maps_to(
            provider, mocker, _auth_error(), AuthError, model="groq/llama-3.1-70b"
        )
        assert err.provider == "groq"

    async def test_provider_inferred_for_claude_models(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """Bare 'claude-*' model names are attributed to 'anthropic'."""
        err = await self._assert_maps_to(
            provider, mocker, _auth_error(), AuthError, model="claude-3-5-sonnet-20241022"
        )
        assert err.provider == "anthropic"

    async def test_provider_inferred_for_gpt_models(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """Bare 'gpt-*' model names are attributed to 'openai'."""
        err = await self._assert_maps_to(
            provider, mocker, _auth_error(), AuthError, model="gpt-4o"
        )
        assert err.provider == "openai"

    async def test_auth_error_message_includes_provider_name(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """AuthError.message contains the provider name for easy debugging."""
        err = await self._assert_maps_to(provider, mocker, _auth_error(), AuthError)
        assert "anthropic" in err.message

    async def test_original_error_preserved_on_all_mapped_types(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """original_error on every mapped exception points back to the litellm source."""
        cases: list[tuple[Exception, type]] = [
            (_rate_limit_error(), RateLimitError),
            (_timeout_error(), TimeoutError),
            (_bad_request_error(), InvalidRequestError),
            (_service_unavailable_error(), ProviderUnavailableError),
        ]
        for litellm_exc, expected_type in cases:
            err = await self._assert_maps_to(provider, mocker, litellm_exc, expected_type)
            assert err.original_error is litellm_exc, (
                f"{expected_type.__name__}.original_error should be the original litellm exc"
            )


# ---------------------------------------------------------------------------
# 5. Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Tenacity retries are applied only to transient errors."""

    @pytest.fixture(autouse=True)
    def suppress_backoff(self, mocker: Any) -> None:
        """Patch tenacity's async sleep so retries complete instantly in tests."""
        mocker.patch(
            "tenacity.asyncio._portable_async_sleep",
            new=AsyncMock(return_value=None),
        )

    async def test_retries_on_rate_limit_succeeds_on_third_attempt(
        self, mocker: Any
    ) -> None:
        """Two consecutive RateLimitErrors are retried; success on attempt 3 is returned."""
        provider = LLMGatewayProvider(timeout=5, max_retries=3)
        response = _CompletionResponse(content="ok")
        mock_acompletion = mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(
                side_effect=[_rate_limit_error(), _rate_limit_error(), response]
            ),
        )
        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )

        chunks = [chunk async for chunk in provider.generate(request)]

        assert mock_acompletion.call_count == 3
        assert chunks[0].content == "ok"

    async def test_retries_on_timeout_succeeds_on_second_attempt(
        self, mocker: Any
    ) -> None:
        """A single TimeoutError triggers one retry; the next success is returned."""
        provider = LLMGatewayProvider(timeout=5, max_retries=3)
        response = _CompletionResponse(content="done")
        mock_acompletion = mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(side_effect=[_timeout_error(), response]),
        )
        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )

        chunks = [chunk async for chunk in provider.generate(request)]

        assert mock_acompletion.call_count == 2
        assert chunks[0].content == "done"

    async def test_retries_on_provider_unavailable_succeeds(
        self, mocker: Any
    ) -> None:
        """A 503 ServiceUnavailableError triggers a retry that eventually succeeds."""
        provider = LLMGatewayProvider(timeout=5, max_retries=3)
        response = _CompletionResponse(content="recovered")
        mock_acompletion = mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(side_effect=[_service_unavailable_error(), response]),
        )
        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )

        async for chunk in provider.generate(request):
            assert chunk.content == "recovered"
        assert mock_acompletion.call_count == 2

    async def test_auth_error_is_not_retried(self, mocker: Any) -> None:
        """AuthError is permanent — acompletion is called exactly once."""
        provider = LLMGatewayProvider(timeout=5, max_retries=3)
        mock_acompletion = mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(side_effect=_auth_error()),
        )
        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )

        with pytest.raises(AuthError):
            async for _ in provider.generate(request):
                pass

        assert mock_acompletion.call_count == 1

    async def test_invalid_request_error_is_not_retried(self, mocker: Any) -> None:
        """InvalidRequestError (HTTP 400) is permanent — no retry attempts."""
        provider = LLMGatewayProvider(timeout=5, max_retries=3)
        mock_acompletion = mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(side_effect=_bad_request_error()),
        )
        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )

        with pytest.raises(InvalidRequestError):
            async for _ in provider.generate(request):
                pass

        assert mock_acompletion.call_count == 1

    async def test_exhausted_retries_raises_last_error(self, mocker: Any) -> None:
        """When max_retries is exhausted, the final mapped error is re-raised."""
        provider = LLMGatewayProvider(timeout=5, max_retries=2)
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(side_effect=_rate_limit_error()),
        )
        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )

        with pytest.raises(RateLimitError):
            async for _ in provider.generate(request):
                pass

    async def test_mid_stream_error_not_retried(self, mocker: Any) -> None:
        """Errors raised during stream iteration are mapped but never retried.

        Only the initial connection (inside _call_litellm) participates in
        retry logic.  Once the stream object is returned, any mid-stream
        failure propagates directly to the caller.
        """

        class _FailAfterFirstChunk:
            """Async iterator that yields one chunk then raises."""

            def __aiter__(self) -> "_FailAfterFirstChunk":
                self._count = 0
                return self

            async def __anext__(self) -> _StreamChunk:
                if self._count == 0:
                    self._count += 1
                    return _StreamChunk(content="partial")
                raise litellm.ServiceUnavailableError(
                    message="stream broke",
                    llm_provider="anthropic",
                    model="claude-haiku-4-5-20251001",
                    response=_DUMMY_RESP_503,
                )

        provider = LLMGatewayProvider(timeout=5, max_retries=3)
        mock_acompletion = mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_FailAfterFirstChunk()),
        )
        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        with pytest.raises(ProviderUnavailableError):
            async for _ in provider.generate(request):
                pass

        # The initial call succeeds; mid-stream failure is NOT a new acompletion call.
        assert mock_acompletion.call_count == 1


# ---------------------------------------------------------------------------
# 6. Token counting
# ---------------------------------------------------------------------------


class TestTokenCounting:
    """count_tokens returns litellm estimates or falls back to a word approximation."""

    async def test_returns_litellm_token_count(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """The raw token_counter result is returned without modification."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.token_counter",
            return_value=7,
        )
        count = await provider.count_tokens("Hello world", "claude-haiku-4-5-20251001")
        assert count == 7

    async def test_token_counter_called_with_correct_model_and_text(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """token_counter receives the exact model and text arguments."""
        mock_counter = mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.token_counter",
            return_value=5,
        )
        await provider.count_tokens("test text", "gpt-4o")
        mock_counter.assert_called_once_with(model="gpt-4o", text="test text")

    async def test_fallback_used_when_token_counter_raises(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """A word-count approximation is returned when token_counter raises."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.token_counter",
            side_effect=Exception("unsupported model"),
        )
        # "Hello world" → 2 words → round(2 * 1.3) = 3
        count = await provider.count_tokens("Hello world", "unknown-model")
        assert count > 0

    async def test_fallback_minimum_is_one(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """The word-count fallback always returns at least 1, even for empty input."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.token_counter",
            side_effect=Exception("fail"),
        )
        count = await provider.count_tokens("", "unknown-model")
        assert count >= 1

    async def test_fallback_scales_with_word_count(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """The fallback approximation grows with more words."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.token_counter",
            side_effect=Exception("fail"),
        )
        short = await provider.count_tokens("hi", "x")
        long_ = await provider.count_tokens("word " * 100, "x")
        assert long_ > short


# ---------------------------------------------------------------------------
# 7. OpenTelemetry span instrumentation
# ---------------------------------------------------------------------------


class TestOpenTelemetry:
    """generate() creates OTel spans and records attributes / errors correctly."""

    async def test_llm_generate_span_is_started(
        self,
        provider: LLMGatewayProvider,
        non_streaming_request: CompletionRequest,
        mock_tracer: MagicMock,
        mock_span: MagicMock,
        mocker: Any,
    ) -> None:
        """The 'llm.generate' span is opened for every request."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_CompletionResponse()),
        )
        mocker.patch("llmgateway.providers.litellm_wrapper._tracer", mock_tracer)

        async for _ in provider.generate(non_streaming_request):
            pass

        span_names = [c[0][0] for c in mock_tracer.start_as_current_span.call_args_list]
        assert "llm.generate" in span_names

    async def test_model_and_temperature_attributes_recorded(
        self,
        provider: LLMGatewayProvider,
        mock_tracer: MagicMock,
        mock_span: MagicMock,
        mocker: Any,
    ) -> None:
        """gen_ai.request.model and gen_ai.request.temperature are set on the span."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_CompletionResponse()),
        )
        mocker.patch("llmgateway.providers.litellm_wrapper._tracer", mock_tracer)

        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.9,
            stream=False,
        )
        async for _ in provider.generate(request):
            pass

        attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert attrs.get("gen_ai.request.model") == "claude-haiku-4-5-20251001"
        assert attrs.get("gen_ai.request.temperature") == 0.9

    async def test_gen_ai_system_attribute_set(
        self,
        provider: LLMGatewayProvider,
        mock_tracer: MagicMock,
        mock_span: MagicMock,
        mocker: Any,
    ) -> None:
        """gen_ai.system is set to the provider name derived from the model string."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_CompletionResponse()),
        )
        mocker.patch("llmgateway.providers.litellm_wrapper._tracer", mock_tracer)

        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )
        async for _ in provider.generate(request):
            pass

        attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert attrs.get("gen_ai.system") == "anthropic"

    async def test_user_id_span_attribute_set_when_provided(
        self,
        provider: LLMGatewayProvider,
        mock_tracer: MagicMock,
        mock_span: MagicMock,
        mocker: Any,
    ) -> None:
        """enduser.id is recorded on the span when user_id is present in the request."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_CompletionResponse()),
        )
        mocker.patch("llmgateway.providers.litellm_wrapper._tracer", mock_tracer)

        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            user_id="user-42",
            stream=False,
        )
        async for _ in provider.generate(request):
            pass

        attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert attrs.get("enduser.id") == "user-42"

    async def test_span_error_recorded_on_auth_failure(
        self,
        provider: LLMGatewayProvider,
        mock_tracer: MagicMock,
        mock_span: MagicMock,
        mocker: Any,
    ) -> None:
        """record_exception is called on the span when a ProviderError is raised."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(side_effect=_auth_error()),
        )
        mocker.patch("llmgateway.providers.litellm_wrapper._tracer", mock_tracer)

        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )
        with pytest.raises(AuthError):
            async for _ in provider.generate(request):
                pass

        mock_span.record_exception.assert_called()

    async def test_span_status_set_to_error_on_failure(
        self,
        provider: LLMGatewayProvider,
        mock_tracer: MagicMock,
        mock_span: MagicMock,
        mocker: Any,
    ) -> None:
        """set_status(ERROR, ...) is called on the span when a request fails."""
        from opentelemetry.trace import StatusCode

        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(side_effect=_auth_error()),
        )
        mocker.patch("llmgateway.providers.litellm_wrapper._tracer", mock_tracer)

        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )
        with pytest.raises(AuthError):
            async for _ in provider.generate(request):
                pass

        mock_span.set_status.assert_called()
        status_code_arg = mock_span.set_status.call_args[0][0]
        assert status_code_arg == StatusCode.ERROR

    async def test_max_tokens_span_attribute_set_when_provided(
        self,
        provider: LLMGatewayProvider,
        mock_tracer: MagicMock,
        mock_span: MagicMock,
        mocker: Any,
    ) -> None:
        """gen_ai.request.max_tokens is recorded when max_tokens is specified."""
        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_CompletionResponse()),
        )
        mocker.patch("llmgateway.providers.litellm_wrapper._tracer", mock_tracer)

        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=256,
            stream=False,
        )
        async for _ in provider.generate(request):
            pass

        attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert attrs.get("gen_ai.request.max_tokens") == 256


# ---------------------------------------------------------------------------
# 8. Provider extraction utility (_extract_provider)
# ---------------------------------------------------------------------------


class TestExtractProvider:
    """_extract_provider derives the correct provider name from a model string."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("groq/llama-3.1-70b-versatile", "groq"),
            ("azure/gpt-4o", "azure"),
            ("bedrock/claude-3-5-sonnet", "bedrock"),
            ("gpt-4o", "openai"),
            ("gpt-3.5-turbo", "openai"),
            ("o1-mini", "openai"),
            ("o3-mini", "openai"),
            ("claude-3-5-sonnet-20241022", "anthropic"),
            ("claude-haiku-4-5-20251001", "anthropic"),
            ("gemini-pro", "google"),
            ("gemini-1.5-flash", "google"),
            ("command-r-plus", "cohere"),
            ("mistral-7b-instruct", "mistral"),
            ("totally-unknown-model", "unknown"),
        ],
    )
    def test_provider_extracted_correctly(self, model: str, expected: str) -> None:
        """_extract_provider returns the expected provider string for each model pattern."""
        p = LLMGatewayProvider()
        assert p._extract_provider(model) == expected


# ---------------------------------------------------------------------------
# 9. Edge cases covering remaining branches
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Covers the three branches not reachable through the primary happy/error paths."""

    async def test_parse_error_in_non_streaming_caught_by_generate(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """An exception from _parse_response (not a ProviderError) is mapped in generate().

        This exercises lines 184-193: the ``except Exception`` handler in
        ``generate()`` that catches non-ProviderError exceptions which escape
        from ``_generate_non_streaming`` — specifically from ``_parse_response``
        when the response object is malformed.
        """

        class _MalformedResponse:
            """Response with no choices — _parse_response will raise IndexError."""

            model = "claude-haiku-4-5-20251001"
            choices: list = []
            usage = _Usage(10, 5)

        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_MalformedResponse()),
        )
        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )
        # The IndexError from choices[0] is caught and mapped to ProviderError.
        with pytest.raises(ProviderError):
            async for _ in provider.generate(request):
                pass

    async def test_provider_error_raised_mid_stream_reraises_unchanged(
        self, provider: LLMGatewayProvider, mocker: Any
    ) -> None:
        """A ProviderError raised during stream iteration is re-raised as-is.

        This exercises line 258: ``except ProviderError: raise`` in
        ``_generate_streaming``.  When the stream itself raises an already-mapped
        ProviderError (not a raw litellm exception), it must not be double-wrapped.
        """

        original_error = ProviderUnavailableError(
            message="upstream gone", provider="anthropic"
        )

        class _StreamRaisesProviderError:
            def __aiter__(self) -> "_StreamRaisesProviderError":
                return self

            async def __anext__(self) -> _StreamChunk:
                raise original_error

        mocker.patch(
            "llmgateway.providers.litellm_wrapper.litellm.acompletion",
            new=AsyncMock(return_value=_StreamRaisesProviderError()),
        )
        request = CompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        with pytest.raises(ProviderUnavailableError) as exc_info:
            async for _ in provider.generate(request):
                pass

        # Must be the exact same object — no re-wrapping.
        assert exc_info.value is original_error

    def test_map_error_returns_provider_error_unchanged(
        self, provider: LLMGatewayProvider
    ) -> None:
        """_map_error returns an already-mapped ProviderError as-is (no double-wrap).

        This exercises line 398: the early-exit guard that prevents wrapping a
        ProviderError inside another ProviderError when _map_error is called
        recursively or from multiple sites.
        """
        already_mapped = AuthError(message="was mapped", provider="anthropic")
        result = provider._map_error(already_mapped, "claude-haiku-4-5-20251001")
        assert result is already_mapped
