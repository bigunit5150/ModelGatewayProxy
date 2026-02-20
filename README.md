# LLM Gateway

A production-grade API gateway for large language models. Routes requests to multiple providers (Anthropic, OpenAI, Groq, Together AI, and more) through a unified interface, with structured error handling, exponential-backoff retry, OpenTelemetry distributed tracing, and Prometheus metrics.

---

## Architecture

```
┌───────────────┐     HTTP      ┌──────────────────────────────────┐
│    Client     │ ────────────► │  FastAPI  (:8000)                │
└───────────────┘               │    POST /v1/chat/completions     │
                                │    /health   /health/live        │
                                │    /health/ready  /metrics       │
                                └──────────────┬───────────────────┘
                                               │
                                               ▼
                                ┌──────────────────────────────────┐
                                │  LLMGatewayProvider              │
                                │    Retry · OTel · Logging        │
                                └──────────────┬───────────────────┘
                                               │
                                               ▼
                                ┌──────────────────────────────────┐
                                │  LiteLLM                         │
                                │  (provider normalisation)        │
                                └──────┬──────────┬───────┬────────┘
                                       │          │       │
                                  Anthropic   OpenAI   Groq / etc.
```

**Infrastructure services** (Docker Compose):

| Service | URL | Purpose |
|---|---|---|
| Gateway | http://localhost:8000 | FastAPI app |
| Jaeger | http://localhost:16686 | Distributed trace UI |
| Prometheus | http://localhost:9090 | Metrics scraping |
| Grafana | http://localhost:3000 | Dashboards |
| Redis | localhost:6379 | Cache / rate limiting |
| PostgreSQL | localhost:5432 | Persistent storage |

---

## Quick Start

### 1. Clone and open in Dev Container

```bash
git clone <repo-url>
cd llm-gateway
# Open in VS Code → "Reopen in Container"
```

### 2. Configure API keys

Copy the example env file and add your keys:

```bash
cp .env.example .env   # if it exists, otherwise edit .env directly
```

Edit `.env`:

```bash
# Required for Anthropic (Claude models)
ANTHROPIC_API_KEY=sk-ant-...

# Optional — add keys for any providers you want to use
# OPENAI_API_KEY=sk-...
# GROQ_API_KEY=gsk_...
# TOGETHER_API_KEY=...
# MISTRAL_API_KEY=...
# GEMINI_API_KEY=...
```

### 3. Verify the setup

```bash
# Run the smoke test — calls the real API
python scripts/test_litellm_wrapper.py

# Run tests (no real API calls)
make test
```

### 4. Start the development server

```bash
make dev
# Server running at http://localhost:8000
```

---

## HTTP API

The gateway exposes an **OpenAI-compatible** REST API, so any client that works with OpenAI can point at this gateway without modification.

### `POST /v1/chat/completions`

**Request body** (OpenAI wire format):

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | required | LiteLLM model string (e.g. `claude-haiku-4-5-20251001`, `gpt-4o`, `groq/llama-3.1-70b-versatile`) |
| `messages` | array | required | Conversation history — each item must have `role` and `content` |
| `temperature` | float | `0.7` | Sampling temperature `[0.0, 2.0]` |
| `max_tokens` | int | `null` | Maximum tokens to generate (provider default if omitted) |
| `stream` | bool | `false` | Stream tokens as Server-Sent Events |
| `user` | string | `null` | Opaque user identifier forwarded to the provider |

**Response headers** included on every response:

| Header | Description |
|---|---|
| `X-Request-ID` | UUID identifying this specific request (appears in logs and traces) |
| `X-Provider` | Detected provider name (e.g. `anthropic`, `openai`, `groq`) |
| `X-Cache-Status` | `MISS` (caching not yet implemented) |

**curl — non-streaming:**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-haiku-4-5-20251001",
    "messages": [{"role": "user", "content": "Hello!"}]
  }' | jq .
```

**curl — streaming:**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Count to five."}],
    "stream": true
  }'
```

**OpenAI Python SDK** (pointing at the gateway):

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = await client.chat.completions.create(
    model="claude-haiku-4-5-20251001",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

**Error responses** follow the OpenAI format and use standard HTTP status codes:

| Error | Status |
|---|---|
| Invalid request | 400 |
| Authentication failure | 401 |
| Rate limited (`Retry-After` header set) | 429 |
| Gateway timeout | 504 |
| Provider unavailable | 502 |

---

## Multi-Provider Support

The gateway supports every provider that [LiteLLM](https://github.com/BerriAI/litellm) supports. Add the API key to `.env` and use the model string directly — no code changes required.

### Supported providers

| Provider | Model examples | `.env` key |
|---|---|---|
| **Anthropic** | `claude-haiku-4-5-20251001` `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| **OpenAI** | `gpt-4o` `gpt-4o-mini` `o1-mini` | `OPENAI_API_KEY` |
| **Azure OpenAI** | `azure/gpt-4o` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |
| **Groq** | `groq/llama-3.1-70b-versatile` | `GROQ_API_KEY` |
| **Together AI** | `together_ai/meta-llama/Llama-3-70b-chat-hf` | `TOGETHER_API_KEY` |
| **Google Gemini** | `gemini/gemini-1.5-pro` | `GEMINI_API_KEY` |
| **Mistral** | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` |
| **Cohere** | `command-r-plus` | `COHERE_API_KEY` |
| **Ollama (local)** | `ollama/llama3` | *(none)* |

### Quick usage examples

**Streaming (Anthropic)**

```python
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))

from llmgateway.providers import LLMGatewayProvider, CompletionRequest

provider = LLMGatewayProvider()

request = CompletionRequest(
    model="claude-haiku-4-5-20251001",
    messages=[{"role": "user", "content": "Explain async/await in three sentences."}],
    temperature=0.7,
    stream=True,
)

async for chunk in provider.generate(request):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

**Non-streaming (OpenAI)**

```python
request = CompletionRequest(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
    temperature=0,
    stream=False,
)

async for chunk in provider.generate(request):
    print(chunk.content)
    print(f"Tokens: {chunk.usage}")
```

**Groq (fast inference)**

```python
request = CompletionRequest(
    model="groq/llama-3.1-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
```

**Token counting**

```python
count = await provider.count_tokens(
    text="How many tokens is this?",
    model="claude-haiku-4-5-20251001",
)
print(f"Estimated tokens: {count}")
```

### Error handling

```python
from llmgateway.providers import (
    AuthError, RateLimitError, TimeoutError,
    InvalidRequestError, ProviderUnavailableError,
)

try:
    async for chunk in provider.generate(request):
        process(chunk)
except AuthError as e:
    print(f"Bad API key for {e.provider}")
except RateLimitError as e:
    print(f"Rate limited — retry after {e.retry_after}s")
except TimeoutError as e:
    print(f"Request timed out")
except InvalidRequestError as e:
    print(f"Bad request: {e.message}")
except ProviderUnavailableError as e:
    print(f"{e.provider} is down")
```

Transient errors (`RateLimitError`, `TimeoutError`, `ProviderUnavailableError`) are retried automatically with exponential backoff before the exception reaches your code.

---

## Development Commands

```bash
make dev          # Start dev server with hot-reload (port 8000)
make test         # Run test suite with coverage report
make lint         # Check code style with Ruff
make format       # Auto-format code with Ruff
make type-check   # Run mypy static analysis
make clean        # Remove build artefacts and caches
```

### Running tests

```bash
# All unit tests (integration tests excluded by default)
pytest

# Provider layer tests only
pytest tests/providers/ -v

# HTTP API endpoint tests only
pytest tests/api/ -v

# With coverage report
pytest --cov=src/llmgateway --cov-report=term-missing
```

### Integration tests (real API calls)

Integration tests make live API calls and require valid keys in `.env`. They are excluded from the default `pytest` run.

```bash
# Run all integration tests
pytest -m integration -v

# Run with a specific provider only
ANTHROPIC_API_KEY=sk-ant-... pytest -m integration -v
```

Jaeger and Prometheus integration tests additionally require the Docker Compose stack (`docker compose up -d`).

### Manual provider smoke tests

```bash
# Test Anthropic only
python scripts/test_litellm_wrapper.py

# Test all configured providers
python scripts/test_all_providers.py
```

---

## Observability

### Health endpoints

```
GET /health         → {"status": "healthy", "version": "0.1.0", "timestamp": "..."}
GET /health/live    → {"status": "alive"}          (Kubernetes liveness)
GET /health/ready   → {"status": "ready", "checks": {...}}  (checks Redis + Postgres)
GET /metrics        → Prometheus metrics
```

### Distributed tracing

Every LLM request produces an `llm.generate` span with:
- Provider name, model, temperature
- Input/output token counts
- Request duration
- Error details (on failure)

View traces at [http://localhost:16686](http://localhost:16686) (Jaeger UI).

### Structured logging

All logs are emitted as JSON with a `request_id` field that links the start, error, and completion events for each request:

```json
{"event": "llm_request_start",    "request_id": "...", "model": "claude-haiku-4-5-20251001"}
{"event": "llm_request_complete", "request_id": "...", "duration_ms": 934.19}
```

---

## Project Structure

```
llm-gateway/
├── src/llmgateway/
│   ├── api/
│   │   ├── completions.py     # POST /v1/chat/completions (OpenAI-compatible)
│   │   └── health.py          # Health check endpoints
│   ├── providers/
│   │   ├── litellm_wrapper.py # LLMGatewayProvider — main entry point
│   │   ├── models.py          # Request/response types
│   │   └── errors.py          # Typed exception hierarchy
│   ├── config.py              # Pydantic settings (reads .env)
│   └── main.py                # FastAPI app + OTel setup
├── tests/
│   ├── api/
│   │   └── test_completions.py       # Unit tests for /v1/chat/completions
│   ├── integration/
│   │   └── test_gateway_integration.py  # End-to-end tests (real API calls)
│   ├── providers/
│   │   └── test_litellm_wrapper.py   # Unit tests for the provider layer
│   ├── conftest.py                   # Shared fixtures
│   └── test_health.py                # Health endpoint tests
├── scripts/
│   ├── test_litellm_wrapper.py       # Anthropic smoke test
│   └── test_all_providers.py         # Multi-provider smoke test
├── docs/
│   └── providers/
│       ├── README.md              # Provider system overview
│       ├── design-decisions.md    # Architecture decision records
│       └── troubleshooting.md     # Common issues and fixes
├── .env                           # API keys (not committed)
├── pyproject.toml                 # Dependencies and tool config
└── Makefile                       # Dev commands
```

---

## Documentation

- [Provider System Overview](docs/providers/README.md) — supported providers, request flow, usage examples, error types, observability
- [Design Decisions](docs/providers/design-decisions.md) — why LiteLLM, custom error types, OTel integration, retry strategy
- [Troubleshooting](docs/providers/troubleshooting.md) — solutions for auth errors, rate limits, timeouts, and unexpected responses
