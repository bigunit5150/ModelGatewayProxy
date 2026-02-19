# Provider System

The provider layer is a thin, production-ready wrapper around [LiteLLM](https://github.com/BerriAI/litellm) that adds typed error handling, OpenTelemetry instrumentation, structured logging, and exponential-backoff retry on top of LiteLLM's multi-provider normalisation.

---

## Table of Contents

- [Overview](#overview)
- [Supported Providers](#supported-providers)
- [Request Flow](#request-flow)
- [Usage Examples](#usage-examples)
- [Error Handling](#error-handling)
- [Observability](#observability)
- [Configuration Reference](#configuration-reference)

---

## Overview

### Why LiteLLM?

Every LLM provider exposes a slightly different API shape, authentication scheme, and error vocabulary. Rather than writing an adapter for each one, the gateway delegates that translation to LiteLLM and focuses on the concerns LiteLLM does *not* address:

| Concern | Who handles it |
|---|---|
| Provider API normalisation | LiteLLM |
| Authentication & credential routing | LiteLLM |
| Streaming / non-streaming unification | LiteLLM |
| Typed exception hierarchy | `LLMGatewayProvider` |
| Retry with exponential back-off | `LLMGatewayProvider` (tenacity) |
| OpenTelemetry spans & attributes | `LLMGatewayProvider` |
| Structured JSON logging | `LLMGatewayProvider` (structlog) |
| Request correlation IDs | `LLMGatewayProvider` |

### Core Components

```
src/llmgateway/providers/
├── __init__.py          # Public surface area (re-exports)
├── litellm_wrapper.py   # LLMGatewayProvider — main entry point
├── models.py            # CompletionRequest, CompletionChunk, CompletionResponse
└── errors.py            # Typed exception hierarchy
```

---

## Supported Providers

Any provider supported by LiteLLM works automatically. Set the corresponding environment variable in `.env` and use the model string shown below.

| Provider | Model string examples | Required env var |
|---|---|---|
| **Anthropic** | `claude-haiku-4-5-20251001` `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| **OpenAI** | `gpt-4o` `gpt-4o-mini` `o1-mini` `o3-mini` | `OPENAI_API_KEY` |
| **Azure OpenAI** | `azure/gpt-4o` `azure/gpt-4-turbo` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |
| **Google Gemini** | `gemini/gemini-1.5-pro` `gemini/gemini-1.5-flash` | `GEMINI_API_KEY` |
| **Groq** | `groq/llama-3.1-70b-versatile` `groq/mixtral-8x7b-32768` | `GROQ_API_KEY` |
| **Together AI** | `together_ai/meta-llama/Llama-3-70b-chat-hf` | `TOGETHER_API_KEY` |
| **Mistral** | `mistral/mistral-large-latest` `mistral/open-mistral-7b` | `MISTRAL_API_KEY` |
| **Cohere** | `command-r-plus` `command-r` | `COHERE_API_KEY` |
| **AWS Bedrock** | `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0` | AWS credentials |
| **Ollama (local)** | `ollama/llama3` `ollama/mistral` | *(none — runs locally)* |

> **Adding a new provider:** Install any provider-specific SDK LiteLLM requires, add the API key to `.env`, and use the model string directly in `CompletionRequest`. No code changes needed.

---

## Request Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Code                             │
│                                                                 │
│  request = CompletionRequest(model="claude-haiku-4-5-20251001", │
│                              messages=[...], stream=True)       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLMGatewayProvider.generate()                │
│                                                                 │
│  1. Validate request (CompletionRequest.__post_init__)          │
│  2. Open OTel span "llm.generate"                               │
│  3. Emit structured log "llm_request_start"                     │
│  4. Convert to LiteLLM params (_to_litellm_format)             │
│  5. Route to streaming or non-streaming path                    │
└───────────┬──────────────────────────────────┬──────────────────┘
            │ stream=True                      │ stream=False
            ▼                                  ▼
┌───────────────────────┐          ┌───────────────────────────┐
│  _generate_streaming  │          │  _generate_non_streaming  │
│                       │          │                           │
│  Open "llm.api_call"  │          │  Open "llm.api_call" span │
│  span                 │          │                           │
└───────────┬───────────┘          └──────────────┬────────────┘
            │                                     │
            └─────────────────┬───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       _call_litellm()                           │
│                                                                 │
│  tenacity AsyncRetrying (max 3 attempts, exponential backoff):  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  litellm.acompletion(model=..., messages=..., ...)         │ │
│  │                                                            │ │
│  │  On exception: _map_error() → typed ProviderError          │ │
│  │  Retry if: RateLimitError | TimeoutError |                 │ │
│  │            ProviderUnavailableError                        │ │
│  │  Raise immediately: AuthError | InvalidRequestError        │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                          LiteLLM                                │
│                                                                 │
│  Selects provider from model string prefix                      │
│  Handles auth, request serialisation, response parsing          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼────────────────┐
            ▼               ▼                ▼
      ┌──────────┐   ┌──────────┐   ┌──────────────┐
      │Anthropic │   │  OpenAI  │   │  Groq / etc. │
      │   API    │   │   API    │   │     API      │
      └──────────┘   └──────────┘   └──────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Back in generate()                           │
│                                                                 │
│  Parse raw response → CompletionChunk                           │
│  Yield chunk(s) to caller                                       │
│  Emit structured log "llm_request_complete" with duration_ms   │
│  Close OTel spans                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Usage Examples

### Streaming response

```python
from llmgateway.providers import LLMGatewayProvider, CompletionRequest

provider = LLMGatewayProvider()

request = CompletionRequest(
    model="claude-haiku-4-5-20251001",
    messages=[{"role": "user", "content": "Explain async/await in Python"}],
    temperature=0.7,
    stream=True,
)

async for chunk in provider.generate(request):
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if chunk.finish_reason:
        print(f"\n\nFinished: {chunk.finish_reason}")
    if chunk.usage:
        print(f"Tokens used: {chunk.usage}")
```

### Non-streaming response

```python
request = CompletionRequest(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
    temperature=0,
    stream=False,
)

async for chunk in provider.generate(request):
    # Exactly one chunk is yielded for non-streaming calls
    print(f"Response:  {chunk.content}")
    print(f"Model:     {chunk.model}")
    print(f"Usage:     {chunk.usage}")
    print(f"Finished:  {chunk.finish_reason}")
```

### Error handling

```python
from llmgateway.providers import (
    LLMGatewayProvider,
    CompletionRequest,
    AuthError,
    RateLimitError,
    TimeoutError,
    InvalidRequestError,
    ProviderUnavailableError,
    ProviderError,
)

provider = LLMGatewayProvider(timeout=30, max_retries=3)
request = CompletionRequest(
    model="claude-haiku-4-5-20251001",
    messages=[{"role": "user", "content": "Hello"}],
)

try:
    async for chunk in provider.generate(request):
        print(chunk.content, end="")

except AuthError as e:
    # API key missing or revoked — do not retry
    print(f"Auth failed for {e.provider}: {e.message}")

except RateLimitError as e:
    # Provider returned HTTP 429
    wait = e.retry_after or 60
    print(f"Rate limited by {e.provider}. Retry after {wait}s")

except TimeoutError as e:
    # Request exceeded provider.timeout seconds
    print(f"Timed out calling {e.provider}: {e.message}")

except InvalidRequestError as e:
    # Bad request — wrong model name, context too long, etc.
    print(f"Invalid request to {e.provider}: {e.message}")

except ProviderUnavailableError as e:
    # 5xx or network error — retries already exhausted
    print(f"{e.provider} is down: {e.message}")

except ProviderError as e:
    # Catch-all for unexpected provider errors
    print(f"Unexpected error from {e.provider}: {e.message}")
```

### Token counting

```python
provider = LLMGatewayProvider()

text = "Estimate how many tokens this text will consume."
count = await provider.count_tokens(text, model="claude-haiku-4-5-20251001")
print(f"Estimated tokens: {count}")
```

### Multi-turn conversation

```python
messages = [{"role": "system", "content": "You are a helpful assistant."}]

for user_input in ["Hello", "What can you do?", "Tell me a joke"]:
    messages.append({"role": "user", "content": user_input})

    request = CompletionRequest(
        model="claude-haiku-4-5-20251001",
        messages=messages,
        stream=False,
    )

    response_text = ""
    async for chunk in provider.generate(request):
        response_text = chunk.content

    messages.append({"role": "assistant", "content": response_text})
    print(f"Assistant: {response_text}\n")
```

### Provider-prefixed models (Groq, Together AI, etc.)

```python
# Groq — use "groq/" prefix
request = CompletionRequest(
    model="groq/llama-3.1-70b-versatile",
    messages=[{"role": "user", "content": "Hello from Groq!"}],
    stream=True,
)

# Together AI — use "together_ai/" prefix
request = CompletionRequest(
    model="together_ai/meta-llama/Llama-3-70b-chat-hf",
    messages=[{"role": "user", "content": "Hello from Together!"}],
    stream=True,
)
```

---

## Error Handling

### Exception hierarchy

```
Exception
└── ProviderError                  Base for all gateway errors
    ├── RateLimitError             HTTP 429 — retried automatically
    │     └── .retry_after         Seconds to wait (if provider sends it)
    ├── AuthError                  HTTP 401/403 — never retried
    ├── TimeoutError               Request exceeded timeout — retried
    ├── InvalidRequestError        HTTP 400/422 — never retried
    │     (includes context-window exceeded)
    └── ProviderUnavailableError   HTTP 5xx / network — retried
```

All errors carry:
- `.message` — human-readable description
- `.provider` — provider name (`"anthropic"`, `"openai"`, etc.)
- `.original_error` — the original LiteLLM exception for deeper inspection

### Retry policy

| Error type | Retried? | Default attempts |
|---|---|---|
| `RateLimitError` | Yes | Up to `max_retries` |
| `TimeoutError` | Yes | Up to `max_retries` |
| `ProviderUnavailableError` | Yes | Up to `max_retries` |
| `AuthError` | **No** | 1 (fail immediately) |
| `InvalidRequestError` | **No** | 1 (fail immediately) |

Backoff formula: `min(2^attempt, 30)` seconds between attempts.

```python
# Customise retry behaviour at construction time
provider = LLMGatewayProvider(
    timeout=60,       # Per-request timeout in seconds (default: 60)
    max_retries=5,    # Maximum attempts for transient errors (default: 3)
)
```

---

## Observability

### OpenTelemetry spans

Every `generate()` call produces two nested spans:

```
llm.generate                          ← outer span, always present
│  gen_ai.system        = "anthropic"
│  gen_ai.request.model = "claude-haiku-4-5-20251001"
│  gen_ai.request.temperature = 0.7
│  gen_ai.request.max_tokens  = 1024   (if set)
│  llm.stream           = true
│  enduser.id           = "user-42"    (if request.user_id set)
│
└── llm.api_call                       ← inner span per litellm call
       call_type               = "streaming" | "non_streaming"
       gen_ai.usage.input_tokens  = 42
       gen_ai.usage.output_tokens = 128
       gen_ai.response.finish_reasons = "stop"   (non-streaming only)
```

Spans are exported to Jaeger via OTLP at `http://jaeger:4318` (configurable via `OTEL_EXPORTER_OTLP_ENDPOINT`). View traces at [http://localhost:16686](http://localhost:16686).

### Structured logs

Every request emits three log events via structlog (JSON in production):

```json
// Request start
{"event": "llm_request_start", "model": "claude-haiku-4-5-20251001",
 "request_id": "e614a8f7-...", "stream": true, "temperature": 0.7,
 "user_id": null, "level": "info", "timestamp": "2026-02-19T04:48:11Z"}

// On error only
{"event": "llm_request_error", "error_type": "RateLimitError",
 "error": "Rate limit exceeded", "provider": "anthropic",
 "request_id": "e614a8f7-...", "level": "error"}

// Always — success or failure
{"event": "llm_request_complete", "duration_ms": 934.19,
 "request_id": "e614a8f7-...", "level": "info"}
```

`request_id` is a UUID generated per call and appears in all three events, making it trivial to correlate logs for a single request.

---

## Configuration Reference

| Environment variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `AZURE_OPENAI_API_KEY` | — | Azure OpenAI key |
| `AZURE_OPENAI_ENDPOINT` | — | Azure endpoint URL |
| `GROQ_API_KEY` | — | Groq API key |
| `TOGETHER_API_KEY` | — | Together AI API key |
| `MISTRAL_API_KEY` | — | Mistral API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://jaeger:4318` | OTLP trace export endpoint |
| `OTEL_SERVICE_NAME` | `llm-gateway` | Service name in traces |
| `LOG_LEVEL` | `INFO` | Minimum log level |

Add keys to `.env` at the repo root. Only keys for providers you intend to use are required.
