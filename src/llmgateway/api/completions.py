"""OpenAI-compatible POST /v1/chat/completions endpoint.

Translates between the OpenAI wire format and the gateway's internal
:class:`~llmgateway.providers.CompletionRequest` / :class:`~llmgateway.providers.CompletionChunk`
types, streams Server-Sent Events for streaming requests, and maps gateway
errors to the appropriate HTTP status codes.
"""

import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from pydantic import BaseModel, Field

from llmgateway.providers import (
    AuthError,
    CompletionRequest,
    InvalidRequestError,
    LLMGatewayProvider,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
)

router = APIRouter(prefix="/v1", tags=["completions"])

_log = structlog.get_logger(__name__)
_tracer = trace.get_tracer(__name__)

# ---------------------------------------------------------------------------
# HTTP status codes for each gateway error type
# ---------------------------------------------------------------------------
_ERROR_STATUS: dict[type[ProviderError], int] = {
    RateLimitError: 429,
    AuthError: 401,
    TimeoutError: 504,
    InvalidRequestError: 400,
    ProviderUnavailableError: 502,
}


# ---------------------------------------------------------------------------
# Request / response models (OpenAI wire format)
# ---------------------------------------------------------------------------


class _Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request body."""

    model: str
    messages: list[_Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    user: str | None = None  # maps to CompletionRequest.user_id


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def get_provider(request: Request) -> LLMGatewayProvider:
    """Return the shared :class:`LLMGatewayProvider` from ``app.state``."""
    provider: LLMGatewayProvider | None = getattr(request.app.state, "provider", None)
    if provider is None:
        raise HTTPException(status_code=503, detail="Provider not initialised")
    return provider


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    provider: LLMGatewayProvider = Depends(get_provider),
) -> StreamingResponse | JSONResponse:
    """Generate a chat completion.

    Supports both streaming (``stream=true``) and non-streaming responses and
    is wire-compatible with the OpenAI ``/v1/chat/completions`` API so that
    any OpenAI SDK can point at this gateway without modification.

    Args:
        body: OpenAI-compatible request body.
        request: Raw FastAPI request (used to access ``app.state``).
        provider: Injected :class:`LLMGatewayProvider` instance.

    Returns:
        A ``text/event-stream`` :class:`StreamingResponse` when
        ``body.stream`` is ``True``, otherwise a :class:`JSONResponse`
        containing the full completion.
    """
    request_id = str(uuid.uuid4())
    created = int(time.time())
    start_time = time.monotonic()
    provider_name = _provider_from_model(body.model)

    log = _log.bind(
        request_id=request_id,
        model=body.model,
        stream=body.stream,
        user_id=body.user,
        provider=provider_name,
    )

    base_headers: dict[str, str] = {
        "X-Request-ID": request_id,
        "X-Provider": provider_name,
        "X-Cache-Status": "MISS",
    }

    with _tracer.start_as_current_span("gateway.completions") as span:
        span.set_attribute("gen_ai.system", provider_name)
        span.set_attribute("gen_ai.request.model", body.model)
        span.set_attribute("llm.stream", body.stream)
        if body.user:
            span.set_attribute("enduser.id", body.user)

        log.info("completion_request_start")

        # Validate + convert to our internal request type
        try:
            completion_request = CompletionRequest(
                model=body.model,
                messages=[{"role": m.role, "content": m.content} for m in body.messages],
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                stream=body.stream,
                user_id=body.user,
            )
        except InvalidRequestError as exc:
            span.set_status(StatusCode.ERROR, exc.message)
            raise HTTPException(
                status_code=400,
                detail={"message": exc.message, "type": "invalid_request_error"},
            ) from exc

        # ------------------------------------------------------------------
        # Streaming
        # ------------------------------------------------------------------
        if body.stream:
            # The gateway.completions span covers request setup only;
            # llm.generate / llm.api_call spans inside the provider carry
            # the full call lifecycle.
            return StreamingResponse(
                _stream_sse(provider, completion_request, request_id, created, log, start_time),
                media_type="text/event-stream",
                headers={
                    **base_headers,
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # ------------------------------------------------------------------
        # Non-streaming
        # ------------------------------------------------------------------
        try:
            response_data = await _build_json_response(
                provider, completion_request, request_id, created
            )
        except ProviderError as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, exc.message)
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)
            log.error(
                "completion_request_error",
                error_type=type(exc).__name__,
                error=exc.message,
                duration_ms=duration_ms,
            )
            status_code = _ERROR_STATUS.get(type(exc), 500)
            error_headers = dict(base_headers)
            if isinstance(exc, RateLimitError) and exc.retry_after is not None:
                error_headers["Retry-After"] = str(int(exc.retry_after))
            raise HTTPException(
                status_code=status_code,
                detail={"message": exc.message, "type": type(exc).__name__},
                headers=error_headers,
            ) from exc

        usage = response_data.get("usage", {})
        duration_ms = round((time.monotonic() - start_time) * 1000, 2)
        log.info("completion_request_complete", duration_ms=duration_ms, usage=usage)

        return JSONResponse(content=response_data, headers=base_headers)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _stream_sse(
    provider: LLMGatewayProvider,
    completion_request: CompletionRequest,
    request_id: str,
    created: int,
    log: Any,
    start_time: float,
) -> AsyncGenerator[str, None]:
    """Yield Server-Sent Event lines for each chunk from the provider.

    Errors during streaming are surfaced as a final SSE ``error`` event so the
    client can detect them even though the HTTP 200 header has already been
    sent.
    """
    usage_summary: dict[str, int] | None = None

    try:
        async for chunk in provider.generate(completion_request):
            if chunk.usage:
                usage_summary = chunk.usage
            payload = _format_chunk(chunk, request_id, created, completion_request.model)
            yield f"data: {json.dumps(payload)}\n\n"

        yield "data: [DONE]\n\n"

    except ProviderError as exc:
        log.error(
            "completion_stream_error",
            error_type=type(exc).__name__,
            error=exc.message,
        )
        error_payload = {"error": {"message": exc.message, "type": type(exc).__name__}}
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"

    finally:
        duration_ms = round((time.monotonic() - start_time) * 1000, 2)
        log.info("completion_request_complete", duration_ms=duration_ms, usage=usage_summary)


async def _build_json_response(
    provider: LLMGatewayProvider,
    completion_request: CompletionRequest,
    request_id: str,
    created: int,
) -> dict[str, Any]:
    """Collect all chunks and return a single OpenAI-format completion dict."""
    content = ""
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    model = completion_request.model

    async for chunk in provider.generate(completion_request):
        content += chunk.content
        if chunk.finish_reason:
            finish_reason = chunk.finish_reason
        if chunk.usage:
            usage = chunk.usage
        if chunk.model:
            model = chunk.model

    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage["input_tokens"] if usage else 0,
            "completion_tokens": usage["output_tokens"] if usage else 0,
            "total_tokens": ((usage["input_tokens"] + usage["output_tokens"]) if usage else 0),
        },
    }


def _format_chunk(
    chunk: Any,
    request_id: str,
    created: int,
    default_model: str,
) -> dict[str, Any]:
    """Format a :class:`~llmgateway.providers.CompletionChunk` as an OpenAI SSE data dict."""
    delta: dict[str, Any] = {"content": chunk.content} if chunk.content else {}

    data: dict[str, Any] = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": chunk.model or default_model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": chunk.finish_reason,
            }
        ],
    }

    if chunk.usage:
        data["usage"] = {
            "prompt_tokens": chunk.usage["input_tokens"],
            "completion_tokens": chunk.usage["output_tokens"],
            "total_tokens": chunk.usage["input_tokens"] + chunk.usage["output_tokens"],
        }

    return data


def _provider_from_model(model: str) -> str:
    """Derive a short provider name from a LiteLLM model string for response headers."""
    if "/" in model:
        return model.split("/")[0]
    prefixes = [
        ("gpt-", "openai"),
        ("o1-", "openai"),
        ("o3-", "openai"),
        ("claude-", "anthropic"),
        ("gemini", "google"),
        ("command", "cohere"),
        ("mistral", "mistral"),
    ]
    for prefix, name in prefixes:
        if model.startswith(prefix):
            return name
    return "unknown"
