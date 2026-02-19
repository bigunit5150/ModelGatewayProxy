# Provider Layer — Troubleshooting Guide

---

## Quick Diagnostics

Before diving into specific errors, run the provider smoke test to confirm your environment is wired up correctly:

```bash
python scripts/test_litellm_wrapper.py   # Anthropic only
python scripts/test_all_providers.py     # All configured providers
```

Check which keys are loaded:

```bash
python -c "
from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv(Path('.env'))
for key in ['ANTHROPIC_API_KEY','OPENAI_API_KEY','GROQ_API_KEY','TOGETHER_API_KEY']:
    val = os.environ.get(key, '')
    print(f'{key}: {\"SET (\" + val[:12] + \"...)\" if val else \"NOT SET\"}')"
```

---

## AuthError: Authentication Failed

### Symptoms

```
llmgateway.providers.errors.AuthError: Authentication failed for anthropic:
  litellm.AuthenticationError: Missing Anthropic API Key
```

```
llmgateway.providers.errors.AuthError: Authentication failed for openai:
  litellm.AuthenticationError: Incorrect API key provided: sk-...
```

### Cause 1 — API key not in environment

The most common cause. The key is in `.env` but was never loaded into the process environment.

**Fix:** Ensure `load_dotenv` is called before any imports that use the key, using an explicit path:

```python
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Imports that use the provider come AFTER load_dotenv
from llmgateway.providers import LLMGatewayProvider
```

Verify it is actually set:

```python
import os
print(os.environ.get("ANTHROPIC_API_KEY", "NOT SET"))
```

### Cause 2 — Key is commented out in `.env`

```bash
# ANTHROPIC_API_KEY=sk-ant-...   ← still commented out
ANTHROPIC_API_KEY=sk-ant-...     ← this is what you want
```

**Fix:** Remove the leading `#`.

### Cause 3 — Key is expired or revoked

The key exists in the environment but the provider rejects it. The error message will say "invalid" rather than "missing".

**Fix:** Generate a new key in the provider console and update `.env`:
- Anthropic: [console.anthropic.com](https://console.anthropic.com)
- OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Groq: [console.groq.com](https://console.groq.com)

### Cause 4 — Wrong key for the provider

Using an OpenAI key for an Anthropic model, or vice versa.

**Fix:** Check that the key prefix matches the provider:
- Anthropic keys start with `sk-ant-`
- OpenAI keys start with `sk-` (no `ant`)

### Note: AuthError is never retried

Unlike rate limits and timeouts, authentication failures are permanent. The gateway raises `AuthError` immediately on the first attempt without retrying. Fix the credential before retrying.

---

## RateLimitError: Rate Limit Exceeded

### Symptoms

```
llmgateway.providers.errors.RateLimitError: Rate limit exceeded
```

Logs will show retry attempts:

```json
{"event": "llm_request_retry", "attempt": 1, "wait_seconds": 2.0, "error_type": "RateLimitError"}
{"event": "llm_request_retry", "attempt": 2, "wait_seconds": 4.0, "error_type": "RateLimitError"}
```

### What the gateway does automatically

`RateLimitError` is retried up to `max_retries` times (default: 3) with exponential backoff (2s, 4s, up to 30s). If `retry_after` is present in the provider response, the gateway waits at least that long.

If all retries are exhausted, `RateLimitError` is raised to the caller.

### Fixes

**Reduce concurrency.** If you are sending many requests in parallel, spread them out:

```python
import asyncio

async def rate_limited_generate(requests, concurrency=5):
    semaphore = asyncio.Semaphore(concurrency)
    async def bounded(request):
        async with semaphore:
            return [chunk async for chunk in provider.generate(request)]
    return await asyncio.gather(*[bounded(r) for r in requests])
```

**Increase `max_retries`** for bursty workloads:

```python
provider = LLMGatewayProvider(max_retries=5)
```

**Check your tier limits** in the provider console:
- Anthropic: [console.anthropic.com/settings/limits](https://console.anthropic.com/settings/limits)
- OpenAI: [platform.openai.com/account/limits](https://platform.openai.com/account/limits)

**Use a lower-tier model** for high-volume use cases. `claude-haiku-4-5-20251001` and `gpt-4o-mini` have higher rate limits than their larger counterparts.

---

## TimeoutError: Request Timed Out

### Symptoms

```
llmgateway.providers.errors.TimeoutError: Request to anthropic timed out
```

### What the gateway does automatically

`TimeoutError` is retried with exponential backoff up to `max_retries` times. The timeout clock resets on each attempt.

### Cause 1 — Timeout too short for the model / prompt

Large models generating long outputs take longer. Default timeout is 60 seconds.

**Fix:** Increase the timeout for long-running requests:

```python
provider = LLMGatewayProvider(timeout=120)  # 2 minutes
```

For very long generations, use streaming. The first token arrives quickly and keeps the connection alive:

```python
request = CompletionRequest(..., stream=True)
async for chunk in provider.generate(request):
    process(chunk)  # tokens arrive incrementally
```

### Cause 2 — Provider degradation

The provider is slow or partially unavailable. Check the provider status page:
- Anthropic: [status.anthropic.com](https://status.anthropic.com)
- OpenAI: [status.openai.com](https://status.openai.com)

**Fix:** Either wait for the provider to recover (retries will handle brief blips) or switch to a different provider:

```python
# Fallback to a different model on timeout
try:
    async for chunk in provider.generate(primary_request):
        yield chunk
except TimeoutError:
    fallback_request = CompletionRequest(
        model="groq/llama-3.1-70b-versatile",  # faster, different provider
        messages=primary_request.messages,
        stream=primary_request.stream,
    )
    async for chunk in provider.generate(fallback_request):
        yield chunk
```

### Cause 3 — Network issues in Docker

If running inside Docker, the container may not have egress to the provider APIs.

**Fix:** Check container networking:

```bash
docker exec <container> curl -s https://api.anthropic.com/v1/models \
  -H "x-api-key: $ANTHROPIC_API_KEY" | head -c 200
```

---

## InvalidRequestError: Request Rejected

### Symptoms

```
llmgateway.providers.errors.InvalidRequestError: Invalid request to anthropic:
  litellm.BadRequestError: ...
```

### Cause 1 — Model name does not exist

```
litellm.BadRequestError: model: claude-3-5-haiku-20241022
```

**Fix:** Use the exact model ID. Check currently available models:

```python
import litellm
# For Anthropic
models = litellm.get_valid_models()
anthropic_models = [m for m in models if "claude" in m.lower()]
print(anthropic_models)
```

Common correct names:
```
claude-haiku-4-5-20251001          ← Claude Haiku (current)
claude-3-5-sonnet-20241022         ← Claude Sonnet 3.5
gpt-4o                             ← OpenAI GPT-4o
gpt-4o-mini                        ← OpenAI GPT-4o mini
groq/llama-3.1-70b-versatile       ← Groq Llama 3.1 70B
```

### Cause 2 — Context window exceeded

```
litellm.ContextWindowExceededError: This model's maximum context length is
  8192 tokens. Your messages resulted in 9841 tokens.
```

**Fix:** Truncate or summarise the conversation history before sending, or switch to a model with a larger context window:

```python
# Count tokens before sending
token_count = await provider.count_tokens(
    str(messages), model="claude-haiku-4-5-20251001"
)
if token_count > 7000:
    # Truncate oldest messages, keeping system prompt
    messages = [messages[0]] + messages[-10:]
```

### Cause 3 — Invalid message structure

```
InvalidRequestError: messages[0] must contain both 'role' and 'content' keys
```

`CompletionRequest` validates messages at construction time. The error shows exactly which message index is malformed.

**Fix:** Ensure every message dict has both `role` and `content` keys with a valid role (`system`, `user`, `assistant`, `tool`, `function`).

### Cause 4 — Invalid temperature

```
InvalidRequestError: temperature must be in [0.0, 2.0], got 3.5
```

**Fix:** Temperature must be between 0.0 and 2.0 inclusive.

---

## ProviderUnavailableError: Provider Down or Unreachable

### Symptoms

```
llmgateway.providers.errors.ProviderUnavailableError: anthropic is unavailable:
  litellm.ServiceUnavailableError: ...
```

### What the gateway does automatically

`ProviderUnavailableError` is retried with exponential backoff. After `max_retries` exhausted, the error is raised to the caller.

### Cause 1 — Provider outage

Check the status page for the affected provider.

**Fix:** Implement a fallback to a different provider:

```python
from llmgateway.providers import ProviderUnavailableError

async def resilient_generate(messages):
    providers_to_try = [
        ("claude-haiku-4-5-20251001", LLMGatewayProvider()),
        ("groq/llama-3.1-70b-versatile", LLMGatewayProvider()),
        ("gpt-4o-mini", LLMGatewayProvider()),
    ]
    for model, provider in providers_to_try:
        try:
            request = CompletionRequest(model=model, messages=messages)
            async for chunk in provider.generate(request):
                yield chunk
            return
        except ProviderUnavailableError:
            continue  # try next provider
    raise ProviderUnavailableError("All providers unavailable")
```

### Cause 2 — DNS / network failure in Docker

The container cannot reach `api.anthropic.com`.

**Fix:**

```bash
# Test DNS from inside the container
docker exec <container> nslookup api.anthropic.com

# Test connectivity
docker exec <container> curl -I https://api.anthropic.com
```

If DNS fails, the Docker daemon's DNS may be misconfigured. Add a public DNS server to the Docker daemon config:

```json
// /etc/docker/daemon.json
{"dns": ["8.8.8.8", "1.1.1.1"]}
```

---

## Provider Returns Unexpected Format

### Symptoms

```
llmgateway.providers.errors.ProviderError: Unexpected error from anthropic: ...
IndexError: list index out of range
```

or chunks are yielded with empty content when content is expected.

### Cause — Response shape mismatch

The provider returned a response that does not match the expected LiteLLM structure (rare, but can happen with beta features, fine-tuned models, or during provider API migrations).

### Debugging steps

**Step 1:** Enable LiteLLM debug logging temporarily:

```python
import litellm
litellm._turn_on_debug()
```

**Step 2:** Inspect the raw response by calling litellm directly:

```python
import litellm, os

response = await litellm.acompletion(
    model="claude-haiku-4-5-20251001",
    messages=[{"role": "user", "content": "hi"}],
    stream=False,
)
print(response.model_dump())
```

**Step 3:** Check the LiteLLM GitHub issues for known regressions with that model/provider combination.

**Step 4:** Pin to a known-good LiteLLM version in `pyproject.toml`:

```toml
"litellm==1.81.13",  # pin until upstream fix
```

---

## Structured Log Reference

All provider log events share these fields:

| Field | Type | Description |
|---|---|---|
| `request_id` | UUID string | Unique ID for this generate() call — present on all events |
| `model` | string | Model identifier as passed in the request |
| `stream` | bool | Whether streaming was requested |
| `user_id` | string \| null | From `CompletionRequest.user_id` |

**`llm_request_start`** — emitted at the start of every call:

| Field | Description |
|---|---|
| `temperature` | Sampling temperature |

**`llm_request_error`** — emitted only on failure:

| Field | Description |
|---|---|
| `error_type` | Gateway exception class name |
| `error` | Human-readable error message |
| `provider` | Provider name |

**`llm_request_retry`** — emitted before each retry sleep:

| Field | Description |
|---|---|
| `attempt` | Attempt number (1-indexed) |
| `wait_seconds` | Backoff duration before next attempt |
| `error_type` | Exception type that triggered the retry |

**`llm_request_complete`** — emitted at the end of every call (success or failure):

| Field | Description |
|---|---|
| `duration_ms` | Wall-clock time from start to finish in milliseconds |

### Filtering logs for a specific request

```bash
# In production (JSON logs), filter by request_id
journalctl -u llm-gateway | jq 'select(.request_id == "e614a8f7-...")'

# Or in development (plain text)
python scripts/test_litellm_wrapper.py 2>&1 | grep "e614a8f7"
```
