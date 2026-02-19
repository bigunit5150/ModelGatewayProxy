"""LiteLLM wrapper with structured error handling, observability, and retry logic.

LiteLLM already handles multi-provider normalisation, so this module focuses
exclusively on what LiteLLM does *not* provide out of the box:

* Typed exception hierarchy (:mod:`llmgateway.providers.errors`)
* OpenTelemetry spans using GenAI semantic conventions
* Structured logging via structlog with a per-request correlation ID
* Exponential-backoff retry (tenacity) on transient failures only
"""

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import litellm
import structlog
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llmgateway.providers.errors import (
    AuthError,
    InvalidRequestError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
)
from llmgateway.providers.models import CompletionChunk, CompletionRequest

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

# Suppress LiteLLM's own verbose logging — we emit our own structured logs.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)

_log = structlog.get_logger(__name__)
_tracer = trace.get_tracer(__name__)


def _before_sleep(retry_state: RetryCallState) -> None:
    """Tenacity ``before_sleep`` hook that emits a structured warning."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    next_action = retry_state.next_action
    wait_seconds = next_action.sleep if next_action is not None else 0.0
    _log.warning(
        "llm_request_retry",
        attempt=retry_state.attempt_number,
        wait_seconds=round(wait_seconds, 2),
        error_type=type(exc).__name__ if exc else None,
        error=str(exc) if exc else None,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class LLMGatewayProvider:
    """Thin, production-ready wrapper around LiteLLM.

    LiteLLM handles provider selection and request normalisation.  This class
    adds typed errors, retry logic, OpenTelemetry instrumentation, and
    structured logging on top.

    Example::

        provider = LLMGatewayProvider()
        request = CompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        async for chunk in provider.generate(request):
            print(chunk.content, end="", flush=True)

    Args:
        timeout: Per-request timeout in seconds passed to LiteLLM.
        max_retries: Maximum number of attempts for transient failures.
            Only :class:`~llmgateway.providers.errors.RateLimitError`,
            :class:`~llmgateway.providers.errors.TimeoutError`, and
            :class:`~llmgateway.providers.errors.ProviderUnavailableError`
            trigger retries.  Permanent errors
            (:class:`~llmgateway.providers.errors.AuthError`,
            :class:`~llmgateway.providers.errors.InvalidRequestError`) are
            raised immediately.
        enable_fallback: Reserved for future LiteLLM router fallback support.
    """

    def __init__(
        self,
        timeout: int = 60,
        max_retries: int = 3,
        enable_fallback: bool = False,
    ) -> None:
        self._timeout = timeout
        self._max_retries = max_retries
        self._enable_fallback = enable_fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        request: CompletionRequest,
    ) -> AsyncGenerator[CompletionChunk, None]:
        """Generate a completion via LiteLLM.

        Works for both streaming and non-streaming modes — the caller always
        consumes the same async iterator interface.  For streaming requests
        each token arrives as a separate :class:`~llmgateway.providers.models.CompletionChunk`;
        for non-streaming requests a single chunk with the full response is
        yielded.

        Args:
            request: Validated completion parameters.

        Yields:
            CompletionChunk: One chunk per token (streaming) or one chunk for
            the full response (non-streaming).

        Raises:
            RateLimitError: Provider returned HTTP 429.
            AuthError: API key missing or invalid (HTTP 401 / 403).
            TimeoutError: Request exceeded the configured timeout.
            InvalidRequestError: Request rejected as malformed (HTTP 400 / 422).
            ProviderUnavailableError: Provider down or unreachable (5xx / network).
        """
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        with _tracer.start_as_current_span("llm.generate") as span:
            span.set_attribute("gen_ai.system", self._extract_provider(request.model))
            span.set_attribute("gen_ai.request.model", request.model)
            span.set_attribute("gen_ai.request.temperature", request.temperature)
            span.set_attribute("llm.stream", request.stream)
            if request.max_tokens is not None:
                span.set_attribute("gen_ai.request.max_tokens", request.max_tokens)
            if request.user_id is not None:
                span.set_attribute("enduser.id", request.user_id)

            log = _log.bind(
                request_id=request_id,
                model=request.model,
                stream=request.stream,
                user_id=request.user_id,
            )
            log.info("llm_request_start", temperature=request.temperature)

            params = self._to_litellm_format(request)

            try:
                if request.stream:
                    async for chunk in self._generate_streaming(params):
                        yield chunk
                else:
                    yield await self._generate_non_streaming(params)

            except ProviderError as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, exc.message)
                log.error(
                    "llm_request_error",
                    error_type=type(exc).__name__,
                    error=exc.message,
                    provider=exc.provider,
                )
                raise

            except Exception as exc:
                mapped = self._map_error(exc, request.model)
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                log.error(
                    "llm_request_error",
                    error_type=type(mapped).__name__,
                    error=str(exc),
                )
                raise mapped from exc

            finally:
                duration_ms = round((time.monotonic() - start_time) * 1000, 2)
                log.info("llm_request_complete", duration_ms=duration_ms)

    async def count_tokens(self, text: str, model: str) -> int:
        """Estimate the token count for *text* using LiteLLM's token counter.

        Uses tiktoken for OpenAI-family models and provider-specific counters
        where available.  Falls back to a word-count approximation when the
        model is unsupported.

        Args:
            text: Input text to measure.
            model: Model name used to select the appropriate tokeniser.

        Returns:
            Estimated token count (always at least 1).
        """
        try:
            return litellm.token_counter(model=model, text=text)
        except Exception as exc:
            _log.warning(
                "token_counting_failed",
                model=model,
                error=str(exc),
                fallback="word_count_approximation",
            )
            return max(1, round(len(text.split()) * 1.3))

    # ------------------------------------------------------------------
    # Internal generation helpers
    # ------------------------------------------------------------------

    async def _generate_streaming(
        self,
        params: dict[str, Any],
    ) -> AsyncGenerator[CompletionChunk, None]:
        """Yield chunks from a streaming LiteLLM call.

        The initial connection attempt is covered by :meth:`_call_litellm`'s
        retry logic.  Errors during mid-stream iteration are mapped to gateway
        types but are *not* retried (resuming a partial stream is unsafe).
        """
        with _tracer.start_as_current_span("llm.api_call") as span:
            span.set_attribute("call_type", "streaming")

            response = await self._call_litellm(params)

            try:
                async for raw_chunk in response:
                    chunk = self._parse_chunk(raw_chunk)
                    if chunk.content or chunk.finish_reason:
                        if chunk.usage:
                            span.set_attribute(
                                "gen_ai.usage.input_tokens",
                                chunk.usage.get("input_tokens", 0),
                            )
                            span.set_attribute(
                                "gen_ai.usage.output_tokens",
                                chunk.usage.get("output_tokens", 0),
                            )
                        yield chunk
            except ProviderError:
                raise
            except Exception as exc:
                raise self._map_error(exc, str(params.get("model", ""))) from exc

    async def _generate_non_streaming(
        self,
        params: dict[str, Any],
    ) -> CompletionChunk:
        """Fetch a complete response from a non-streaming LiteLLM call."""
        with _tracer.start_as_current_span("llm.api_call") as span:
            span.set_attribute("call_type", "non_streaming")

            response = await self._call_litellm(params)
            chunk = self._parse_response(response)

            if chunk.usage:
                span.set_attribute(
                    "gen_ai.usage.input_tokens",
                    chunk.usage.get("input_tokens", 0),
                )
                span.set_attribute(
                    "gen_ai.usage.output_tokens",
                    chunk.usage.get("output_tokens", 0),
                )
            if chunk.finish_reason:
                span.set_attribute("gen_ai.response.finish_reasons", chunk.finish_reason)

            return chunk

    async def _call_litellm(self, params: dict[str, Any]) -> Any:
        """Call ``litellm.acompletion`` with exponential-backoff retry.

        Errors are mapped to gateway types *before* tenacity evaluates them so
        that the retry predicate matches on :class:`RateLimitError`,
        :class:`TimeoutError`, and :class:`ProviderUnavailableError`.

        Permanent errors (:class:`AuthError`, :class:`InvalidRequestError`) are
        mapped and re-raised immediately — tenacity will not retry them.
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            retry=retry_if_exception_type((RateLimitError, TimeoutError, ProviderUnavailableError)),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            before_sleep=_before_sleep,
            reraise=True,
        ):
            with attempt:
                try:
                    return await litellm.acompletion(timeout=self._timeout, **params)
                except Exception as exc:
                    raise self._map_error(exc, str(params.get("model", ""))) from exc

    # ------------------------------------------------------------------
    # Format conversion
    # ------------------------------------------------------------------

    def _to_litellm_format(self, request: CompletionRequest) -> dict[str, Any]:
        """Convert a :class:`CompletionRequest` to ``litellm.acompletion`` kwargs."""
        params: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "stream": request.stream,
        }
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.user_id is not None:
            # Forwarded to providers that support per-user attribution (e.g. OpenAI).
            params["user"] = request.user_id
        return params

    def _parse_chunk(self, raw: Any) -> CompletionChunk:
        """Convert a LiteLLM streaming chunk to :class:`CompletionChunk`."""
        content = ""
        finish_reason: str | None = None
        usage: dict[str, int] | None = None

        if hasattr(raw, "choices") and raw.choices:
            choice = raw.choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                content = getattr(delta, "content", None) or ""
            finish_reason = getattr(choice, "finish_reason", None)

        raw_usage = getattr(raw, "usage", None)
        if raw_usage is not None:
            usage = {
                "input_tokens": getattr(raw_usage, "prompt_tokens", 0),
                "output_tokens": getattr(raw_usage, "completion_tokens", 0),
            }

        return CompletionChunk(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            model=getattr(raw, "model", None),
        )

    def _parse_response(self, raw: Any) -> CompletionChunk:
        """Convert a LiteLLM non-streaming response to :class:`CompletionChunk`."""
        content: str = raw.choices[0].message.content or ""
        finish_reason: str | None = raw.choices[0].finish_reason
        usage: dict[str, int] = {
            "input_tokens": raw.usage.prompt_tokens,
            "output_tokens": raw.usage.completion_tokens,
        }
        return CompletionChunk(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            model=raw.model,
        )

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    def _map_error(self, error: Exception, model: str) -> ProviderError:
        """Map a LiteLLM exception to a typed gateway :class:`ProviderError`.

        Mapping table:

        ===================================  ==============================
        LiteLLM exception                    Gateway exception
        ===================================  ==============================
        ``litellm.RateLimitError``           :class:`RateLimitError`
        ``litellm.AuthenticationError``      :class:`AuthError`
        ``litellm.Timeout``                  :class:`TimeoutError`
        ``litellm.BadRequestError``          :class:`InvalidRequestError`
        ``litellm.ServiceUnavailableError``  :class:`ProviderUnavailableError`
        ``litellm.APIConnectionError``       :class:`ProviderUnavailableError`
        ``litellm.APIError`` (catch-all)     :class:`ProviderUnavailableError`
        ===================================  ==============================

        ``litellm.ContextWindowExceededError`` is a subclass of
        ``litellm.BadRequestError`` and is therefore also mapped to
        :class:`InvalidRequestError`.
        """
        # Already mapped — avoid double-wrapping (e.g. raised during streaming).
        if isinstance(error, ProviderError):
            return error

        provider = self._extract_provider(model)

        if isinstance(error, litellm.RateLimitError):
            return RateLimitError(
                message=str(error),
                provider=provider,
                retry_after=getattr(error, "retry_after", None),
                original_error=error,
            )

        if isinstance(error, litellm.AuthenticationError):
            return AuthError(
                message=f"Authentication failed for {provider}: {error}",
                provider=provider,
                original_error=error,
            )

        if isinstance(error, litellm.Timeout):
            return TimeoutError(
                message=f"Request to {provider} timed out: {error}",
                provider=provider,
                original_error=error,
            )

        # BadRequestError is the parent of ContextWindowExceededError in LiteLLM.
        if isinstance(error, litellm.BadRequestError):
            return InvalidRequestError(
                message=f"Invalid request to {provider}: {error}",
                provider=provider,
                original_error=error,
            )

        if isinstance(
            error,
            litellm.ServiceUnavailableError | litellm.APIConnectionError | litellm.APIError,
        ):
            return ProviderUnavailableError(
                message=f"{provider} is unavailable: {error}",
                provider=provider,
                original_error=error,
            )

        return ProviderError(
            message=f"Unexpected error from {provider}: {error}",
            provider=provider,
            original_error=error,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _extract_provider(self, model: str) -> str:
        """Derive the provider name from a LiteLLM model string.

        LiteLLM uses ``provider/model`` prefixes (e.g. ``groq/llama-3.1-70b``).
        Bare model names (e.g. ``gpt-4o``) are matched by well-known prefixes.
        """
        if "/" in model:
            return model.split("/")[0]
        if model.startswith(("gpt-", "o1-", "o3-", "text-embedding-", "davinci")):
            return "openai"
        if model.startswith("claude-"):
            return "anthropic"
        if model.startswith(("gemini", "palm")):
            return "google"
        if model.startswith("command"):
            return "cohere"
        if model.startswith("mistral"):
            return "mistral"
        return "unknown"
