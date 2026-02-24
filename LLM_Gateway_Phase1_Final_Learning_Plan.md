# LLM Gateway Learning Plan - Phase 1 (Final Version)
## Claude Code + VS Code + Dev Containers Edition

**Status:** Week 2 In Progress — Caching, Rate Limiting & Cost Tracking
**Environment:** Docker Dev Containers (macOS/Windows)
**Tools:** VS Code + Claude Code Extension + Dev Containers
**Timeline:** 3-4 weeks to production deployment

### Progress Snapshot (as of 2026-02-24)
| Week | Topic | Status |
|------|-------|--------|
| 0 | Environment Setup | ✅ Complete |
| 1 | Provider Abstraction (LiteLLM) | ✅ Complete |
| 2 | Caching, Rate Limiting, Cost Tracking | 🔄 In Progress |
| 3 | Observability & Production Deployment | ⏳ Upcoming |
| 4 | Advanced Features (optional) | ⏳ Upcoming |

---

## Strategic Context

**Goal:** Build production-grade AI infrastructure expertise to strengthen platform engineering candidacy for Director-level roles.

**Why This Project:**
- Addresses AI-native development skill gaps identified in recent interviews (Binti feedback)
- Demonstrates modern platform engineering patterns at scale
- Creates concrete talking points: "I built an LLM gateway handling 10K req/day with p99 latency under 200ms"
- Shows both technical depth AND AI-assisted development proficiency

**Interview Value:**
After Phase 1, you'll confidently discuss:
- Rate limiting strategies for distributed systems
- Semantic caching trade-offs (cost vs. freshness)
- Observability patterns (RED metrics, distributed tracing)
- Real production metrics and operational learnings
- AI-native development workflows

---

## ✅ Day 0: Development Environment Setup (COMPLETE)

### What Was Accomplished

#### Infrastructure Created
- ✅ Dev Container with Python 3.11, FastAPI, full observability stack
- ✅ Docker Compose orchestrating 6 services (app, Redis, Postgres, Jaeger, Prometheus, Grafana)
- ✅ VS Code configuration with auto-formatting (Ruff), linting, type checking
- ✅ Pre-commit hooks for code quality gates

#### Application Foundation
- ✅ FastAPI app with health check endpoints (`/health`, `/health/live`, `/health/ready`)
- ✅ Configuration management using Pydantic Settings
- ✅ OpenTelemetry instrumentation connected to Jaeger
- ✅ Prometheus metrics endpoint at `/metrics`
- ✅ Structured logging with structlog (JSON output)

#### Development Tooling
- ✅ Makefile with common commands (`make dev`, `make test`, `make lint`)
- ✅ pytest setup with async support and coverage reporting
- ✅ Complete documentation (README, architecture overview)

#### Validation Completed
- ✅ Health checks return 200 OK
- ✅ Redis and Postgres connectivity verified
- ✅ Tests passing
- ✅ Observability stack accessible (Jaeger UI, Prometheus, Grafana)

### Working Environment Reference

**Services Running:**
```
FastAPI:     http://localhost:8000
Jaeger UI:   http://localhost:16686
Prometheus:  http://localhost:9090
Grafana:     http://localhost:3000
Redis:       localhost:6379 (internal)
PostgreSQL:  localhost:5432 (internal)
```

**Key Commands:**
```bash
make dev          # Start development server
make test         # Run tests with coverage
make lint         # Check code quality
make format       # Format code with Ruff
```

---

## Working with Claude Code in Dev Containers

### Understanding Your Environment

**Two VS Code Instances:**
1. **Host VS Code** - Runs on your Mac/Windows, manages the container
2. **Container VS Code** - Runs inside Docker, where you do all development

**When you "Reopen in Container":**
- Host VS Code closes
- Container VS Code opens (new window)
- All your files are mounted at `/workspace`
- Extensions need to be installed in the container
- Claude Code context is reset (this is normal)

### Claude Code Extension Setup

**First Time in Container:**
```bash
# 1. Open Extensions (Cmd+Shift+X)
# 2. Search "Claude Code"
# 3. Click "Install in Container: LLM Gateway"
```

**Make It Permanent:**
Add to `.devcontainer/devcontainer.json`:
```json
{
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff",
        "ms-azuretools.vscode-docker",
        "Anthropic.claude-code"  // Claude Code auto-installs
      ]
    }
  }
}
```

Then rebuild: **Cmd+Shift+P → "Dev Containers: Rebuild Container"**

### Claude Code Workflow Pattern

**For Each New Feature:**

1. **Open Claude Code Chat** (Cmd+Shift+P → "Claude Code: Open Chat")
2. **Provide context** from your learning plan
3. **Be specific** about files and requirements
4. **Review generated code** carefully
5. **Test immediately** (`make test`)
6. **Iterate** if needed ("Add error handling for X")
7. **Commit** when working (`git commit -m "feat: provider abstraction"`)

**Important:** Each Claude Code conversation is independent. Always provide:
- What you're building
- Where it fits in the architecture
- Key requirements
- Files that should be created/modified

---

## Week 1: Provider Abstraction & Core Gateway

### Goals
- ✅ Abstract provider interface supporting multiple LLM APIs
- ✅ OpenAI and Anthropic implementations with streaming
- ✅ Robust error handling and retry logic
- ✅ Basic gateway endpoint routing by model name
- ✅ Integration tests with real API calls (marked for CI skip)

### Day 1: LiteLLM Integration & Error Mapping

**Morning: Architecture Decision (YOU decide)**

Before coding, document your build vs buy decision:

```markdown
## Provider Abstraction Design Decision

### The Choice: LiteLLM vs Custom Abstraction

EVALUATED OPTIONS:
1. Custom provider abstraction (Protocol-based)
   - PRO: Full control over error handling, retry logic
   - PRO: Learning experience building provider layer
   - CON: 2+ weeks to implement and test all providers
   - CON: Maintenance burden when provider APIs change

2. LiteLLM wrapper
   - PRO: 20+ providers supported out of box
   - PRO: Battle-tested, community-maintained
   - PRO: Accurate token counting (tiktoken for OpenAI)
   - PRO: Focus engineering time on unique value-adds
   - CON: Less control over provider internals
   - CON: Dependency on external library

### DECISION: Use LiteLLM with Custom Wrapper

RATIONALE:
- Don't reinvent the wheel for commodity functionality
- Focus on unique value: caching, rate limiting, observability
- LiteLLM handles provider quirks and API changes
- Still build custom error types, retry logic, instrumentation
- Demonstrates good engineering judgment (build vs buy)

### What We Build On Top of LiteLLM:
1. Custom error type hierarchy (gateway's contract)
2. Enhanced retry logic with exponential backoff
3. OpenTelemetry instrumentation for every call
4. Structured logging with correlation IDs
5. Request/response validation
6. Our caching layer (Week 2)
7. Our rate limiting (Week 2)

### Interview Story:
"I evaluated building a custom provider abstraction versus using LiteLLM.
Building custom would take 2 weeks and require ongoing maintenance.
LiteLLM gives me 20+ providers instantly and is battle-tested by the community.
I chose to leverage LiteLLM and focus my engineering effort on the unique
value-adds: semantic caching, distributed rate limiting, and comprehensive
observability. This saved 2 weeks while demonstrating good architectural judgment."
```

**Afternoon: Implementation with Claude Code**

**Claude Code Prompt:**
```
I'm integrating LiteLLM into my LLM Gateway to handle multi-provider support.

CONTEXT:
- Production LLM Gateway in /workspace
- Decision: Use LiteLLM (don't reinvent the wheel)
- Focus: Custom error handling + observability on top of LiteLLM

WHAT LITELLM GIVES US:
- Multi-provider support (OpenAI, Anthropic, Together, Groq, etc.)
- Request format standardization
- Token counting

WHAT WE BUILD:
- Custom error type hierarchy
- Enhanced retry logic with exponential backoff
- OpenTelemetry instrumentation
- Structured logging with correlation IDs
- Request/response validation

CREATE:

1. src/llmgateway/providers/models.py
   - CompletionRequest dataclass: model, messages, temperature, max_tokens, stream, user_id
   - CompletionChunk dataclass: content, finish_reason, usage (tokens dict), model
   - CompletionResponse dataclass: full non-streaming response
   - Validation in __post_init__: temperature 0-2, non-empty messages

2. src/llmgateway/providers/errors.py
   - ProviderError base exception: message, provider name, original error
   - RateLimitError: add retry_after field
   - AuthError, TimeoutError, InvalidRequestError, ProviderUnavailableError

3. src/llmgateway/providers/litellm_wrapper.py
   - LLMGatewayProvider class wrapping litellm.acompletion
   - async generate(request) → AsyncIterator[CompletionChunk]
   - Map litellm exceptions to our error types
   - Use tenacity: retry on transient errors (RateLimit, Timeout), not permanent (Auth, InvalidRequest)
   - Add OpenTelemetry spans with attributes: model, stream, temperature, user_id
   - Structured logging (structlog): request start/end/errors
   - count_tokens() method using litellm.token_counter
   - Handle both streaming and non-streaming responses

4. src/llmgateway/providers/__init__.py
   - Export all key classes

5. Add litellm>=1.17.0 to pyproject.toml dependencies

REQUIREMENTS:
- Production quality: full type hints, docstrings, error handling
- Async/await throughout
- Support streaming (AsyncIterator) and non-streaming

Generate these files. After generation, I'll test with real API keys.
```

**Your Tasks:**
- [ ] Review generated code, especially error mapping
- [ ] Add litellm to dependencies if not present
- [ ] Check retry logic is correctly configured
- [ ] Verify OpenTelemetry spans are properly nested
- [ ] Test dataclass validation (invalid temperature, empty messages)

### Day 2: Testing & Validation

**Morning: Manual Testing with Real APIs**

**Note:** We're testing the LiteLLM wrapper directly (no HTTP routes yet - those come Day 4). This validates the provider layer works before integrating with FastAPI.

**Create Test Script:**
```python
# scripts/test_litellm_wrapper.py
import asyncio
import os
from src.llmgateway.providers.litellm_wrapper import LLMGatewayProvider
from src.llmgateway.providers.models import CompletionRequest

async def test_openai_streaming():
    """Test OpenAI streaming with LiteLLM wrapper"""
    provider = LLMGatewayProvider()

    request = CompletionRequest(
        model="gpt-4o-mini",  # Cheaper for testing
        messages=[{"role": "user", "content": "Count to 5"}],
        temperature=0.7,
        stream=True
    )

    print("Testing OpenAI streaming...")
    async for chunk in provider.generate(request):
        print(chunk.content, end="", flush=True)
        if chunk.finish_reason:
            print(f"\nFinish reason: {chunk.finish_reason}")
        if chunk.usage:
            print(f"Tokens: {chunk.usage}")
    print()

async def test_anthropic_non_streaming():
    """Test Anthropic non-streaming with LiteLLM wrapper"""
    provider = LLMGatewayProvider()

    request = CompletionRequest(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        temperature=0,
        stream=False
    )

    print("Testing Anthropic non-streaming...")
    async for chunk in provider.generate(request):
        print(f"Response: {chunk.content}")
        print(f"Tokens: {chunk.usage}")
    print()

async def test_error_handling():
    """Test error mapping with invalid API key"""
    # Save real key
    real_key = os.environ.get("OPENAI_API_KEY")

    # Temporarily set bad API key
    os.environ["OPENAI_API_KEY"] = "sk-invalid-key-for-testing"

    provider = LLMGatewayProvider()
    request = CompletionRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False
    )

    print("Testing error handling (expect AuthError)...")
    try:
        async for chunk in provider.generate(request):
            print(chunk.content)
    except Exception as e:
        print(f"✅ Caught expected error: {type(e).__name__}: {e}")

    # Restore real key
    if real_key:
        os.environ["OPENAI_API_KEY"] = real_key
    print()

async def test_token_counting():
    """Test token counting functionality"""
    provider = LLMGatewayProvider()

    text = "Hello world, this is a test message"
    count = await provider.count_tokens(text, "gpt-4o-mini")

    print(f"Token counting test:")
    print(f"  Text: '{text}'")
    print(f"  Token count: {count}")
    print()

async def main():
    print("=" * 60)
    print("LiteLLM Wrapper Manual Testing")
    print("=" * 60)
    print()

    await test_openai_streaming()
    await test_anthropic_non_streaming()
    await test_token_counting()
    await test_error_handling()

    print("=" * 60)
    print("Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
```

**Your Testing Tasks:**
- [ ] Create `.env` file with real API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)
- [ ] Run test script: `python scripts/test_litellm_wrapper.py`
- [ ] Verify OpenAI streaming works (prints content incrementally)
- [ ] Verify Anthropic non-streaming works (single response)
- [ ] Verify token counting returns reasonable number
- [ ] Verify error handling catches AuthError
- [ ] Check Jaeger traces at http://localhost:16686
- [ ] Verify traces show nested spans (generate → api_call)

**What This Tests:**
- ✅ LiteLLM wrapper works with real APIs
- ✅ Streaming and non-streaming both work
- ✅ Error mapping converts LiteLLM errors to our custom types
- ✅ Token counting uses LiteLLM's token_counter
- ✅ OpenTelemetry instrumentation creates traces

**What This Does NOT Test:**
- ❌ HTTP endpoints (not created until Day 4)
- ❌ FastAPI integration (not created until Day 4)
- ❌ Caching (not created until Week 2)
- ❌ Rate limiting (not created until Week 2)

**Afternoon: Unit Tests with Mocking**

**Claude Code Prompt:**
```
Create comprehensive unit tests for the LiteLLM wrapper.

CONTEXT:
- LLMGatewayProvider wraps litellm.acompletion
- Need mocked tests (no real API calls during unit tests)
- Test error mappings and retry logic

CREATE:

tests/providers/test_litellm_wrapper.py

Use pytest-asyncio and pytest-mock for mocking.

TEST COVERAGE:

1. CompletionRequest Validation
   - Valid request accepted
   - Invalid temperature (>2) raises error
   - Empty messages raises error
   - Invalid message role raises error

2. LLMGatewayProvider - Success Cases
   - Streaming response: mock litellm to return async generator with chunks
   - Non-streaming response: mock litellm to return single response
   - Verify CompletionChunk conversion
   - Verify usage tokens extracted correctly

3. Error Mapping (mock litellm exceptions)
   - litellm.RateLimitError → RateLimitError (verify retry_after extracted)
   - litellm.AuthenticationError → AuthError (verify NO retries)
   - litellm.Timeout → TimeoutError (verify retries attempted)
   - litellm.BadRequestError → InvalidRequestError (verify NO retries)
   - litellm.ServiceUnavailableError → ProviderUnavailableError (verify retries)

4. Retry Logic
   - Mock failure → failure → success (verify 3 attempts total)
   - Verify exponential backoff delays
   - Verify retries only on transient errors

5. Token Counting
   - Mock litellm.token_counter
   - Verify correct count returned
   - Test fallback when counting fails

6. OpenTelemetry
   - Verify spans created with correct attributes
   - Mock tracer to capture span data

Include mock helper classes for LiteLLM response structure.

REQUIREMENTS:
- Use pytest.mark.asyncio for async tests
- Use mocker.patch to mock litellm.acompletion
- Aim for >90% coverage of litellm_wrapper.py
- Include docstrings explaining test scenarios

Generate this test file.
```

**Your Tasks:**
- [ ] Run tests: `pytest tests/providers/test_litellm_wrapper.py -v`
- [ ] Check coverage: `pytest --cov=src/llmgateway/providers tests/providers/`
- [ ] Fix any failing tests
- [ ] Add more edge case tests if needed
- [ ] Verify all error types are tested

### Day 3: Multi-Provider Testing & Documentation

**Morning: Test Multiple Providers**

LiteLLM supports 20+ providers out of the box. Test several to verify your wrapper works universally.

**Providers to Test:**

1. **OpenAI** (already tested Day 2)
   - Models: gpt-4o, gpt-4o-mini, gpt-3.5-turbo

2. **Anthropic**
   - Models: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
   - Add ANTHROPIC_API_KEY to .env

3. **Together AI** (optional, free tier available)
   - Models: together/meta-llama/Llama-3-70b-chat-hf
   - Add TOGETHER_API_KEY to .env

4. **Groq** (optional, very fast inference)
   - Models: groq/llama-3-70b-8192
   - Add GROQ_API_KEY to .env

**Extended Test Script:**

```python
# scripts/test_all_providers.py
import asyncio
import os
from src.llmgateway.providers.litellm_wrapper import LLMGatewayProvider
from src.llmgateway.providers.models import CompletionRequest

PROVIDERS_TO_TEST = [
    {
        "name": "OpenAI GPT-4o-mini",
        "model": "gpt-4o-mini",
        "required_env": "OPENAI_API_KEY"
    },
    {
        "name": "Anthropic Claude Haiku",
        "model": "claude-3-5-haiku-20241022",
        "required_env": "ANTHROPIC_API_KEY"
    },
    {
        "name": "Together Llama 3 70B",
        "model": "together/meta-llama/Llama-3-70b-chat-hf",
        "required_env": "TOGETHER_API_KEY"
    },
]

async def test_provider(provider_info: dict):
    """Test a single provider"""

    # Check if API key is available
    if not os.getenv(provider_info["required_env"]):
        print(f"⏭️  Skipping {provider_info['name']} (no API key)")
        return

    print(f"\n🧪 Testing {provider_info['name']}...")

    provider = LLMGatewayProvider()
    request = CompletionRequest(
        model=provider_info["model"],
        messages=[{"role": "user", "content": "Say 'Hello from LiteLLM!' in one sentence."}],
        temperature=0.7,
        stream=True
    )

    try:
        print("   Response: ", end="")
        async for chunk in provider.generate(request):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if chunk.usage:
                print(f"\n   Tokens: {chunk.usage}")
        print(f"\n   ✅ {provider_info['name']} working!")

    except Exception as e:
        print(f"\n   ❌ Error: {type(e).__name__}: {e}")

async def main():
    print("=" * 60)
    print("Multi-Provider Test Suite")
    print("=" * 60)

    for provider_info in PROVIDERS_TO_TEST:
        await test_provider(provider_info)

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
```

**Your Testing Tasks:**
- [ ] Run multi-provider test: `python scripts/test_all_providers.py`
- [ ] Verify at least OpenAI and Anthropic work
- [ ] Compare response quality between providers
- [ ] Check Jaeger traces show correct provider attribution
- [ ] Test token counting across providers
- [ ] Document any provider-specific quirks you notice

**Afternoon: Documentation & Design Decisions**

**Claude Code Prompt:**
```
Create comprehensive documentation for the LiteLLM wrapper and provider system.

CONTEXT:
- We've implemented a wrapper around LiteLLM
- Supports multiple providers (OpenAI, Anthropic, Together, Groq, etc.)
- Custom error handling and observability

CREATE:

1. docs/providers/README.md
   - System overview (LiteLLM wrapper approach and rationale)
   - Supported providers table (provider name, model examples, required API keys)
   - Request flow diagram (ASCII art: Client → Gateway → LiteLLM → Provider)
   - Error handling (document each custom error type)
   - Observability (OpenTelemetry spans, structured logging)
   - Usage examples (streaming, non-streaming, error handling, token counting)

2. docs/providers/design-decisions.md
   - Decision: LiteLLM vs Custom Abstraction (date, status, rationale, consequences)
   - Decision: Custom Error Types (why we built on top of LiteLLM)
   - Decision: OpenTelemetry Integration (what we track and why)

3. docs/providers/troubleshooting.md
   - Common issues with solutions:
     - "AuthError: Invalid API key"
     - "RateLimitError: Rate limit exceeded"
     - "TimeoutError: Request timed out"
     - "Provider returns unexpected format"

4. Update README.md
   - Add multi-provider support section
   - Quick start with different providers
   - Environment variable setup
   - Available models list

REQUIREMENTS:
- Clear explanations with code examples
- ASCII diagrams acceptable
- Practical troubleshooting guidance

Generate these documentation files.
```

**Your Tasks:**
- [ ] Review generated documentation
- [ ] Add any provider-specific notes from your testing
- [ ] Create ASCII diagram of request flow
- [ ] Document which providers you tested successfully
- [ ] Add examples from your manual testing

### Day 4: Gateway Routing & FastAPI Integration

**Morning: Architecture Decision**

With LiteLLM, routing is simpler - LiteLLM automatically routes based on model name. But you still need to build the gateway layer.

```markdown
## Gateway Routing Design

### LiteLLM's Built-in Routing
- Model name determines provider: "gpt-4" → OpenAI, "claude-3" → Anthropic
- No custom router needed for basic routing
- We can focus on gateway-level concerns

### What We Build:
1. FastAPI endpoint that accepts requests
2. Request validation (our CompletionRequest)
3. Call LiteLLM wrapper
4. Handle streaming responses properly
5. Error handling → HTTP status codes
6. Add observability and metrics
7. Future: Add routing strategies (cost, latency, fallback)

### Request Flow:
Client → FastAPI /v1/chat/completions → LLMGatewayProvider → LiteLLM → Provider API
```

**Afternoon: FastAPI Integration**

**Claude Code Prompt:**
```
Create the FastAPI gateway endpoint that integrates with LiteLLM wrapper.

CONTEXT:
- LLMGatewayProvider already handles provider selection automatically
- Need OpenAI-compatible /v1/chat/completions endpoint
- Support streaming (Server-Sent Events) and non-streaming

CREATE:

1. src/llmgateway/api/completions.py
   - APIRouter with prefix="/v1"
   - POST /v1/chat/completions endpoint
   - Request body: CompletionRequest from providers.models
   - Response: StreamingResponse (SSE format) or JSONResponse
   - SSE format: "data: {json}\n\n" for each chunk, "data: [DONE]\n\n" at end
   - Response headers: X-Provider, X-Request-ID (UUID), X-Cache-Status (MISS for now)
   - Error handling → HTTP status codes:
     - RateLimitError → 429 (with Retry-After header)
     - AuthError → 401
     - TimeoutError → 504
     - InvalidRequestError → 400
     - ProviderUnavailableError → 502
     - ValidationError → 422
   - OpenTelemetry span: "gateway.completions" with attributes (model, stream, user_id, provider)
   - Structured logging: request start/end with duration, tokens

2. Update src/llmgateway/main.py
   - Initialize LLMGatewayProvider in app.state at startup
   - Create dependency: get_provider() returns provider from app.state
   - Include completions router
   - Add startup event: log "LLM Gateway ready"

3. Update src/llmgateway/config.py
   - Add environment variables:
     - OPENAI_API_KEY
     - ANTHROPIC_API_KEY
     - TOGETHER_API_KEY (optional)
     - GROQ_API_KEY (optional)
     - LLM_TIMEOUT (default: 60)
     - LLM_MAX_RETRIES (default: 3)

REQUIREMENTS:
- OpenAI API compatible format
- Proper SSE formatting for streaming
- All errors mapped to HTTP status codes
- Comprehensive logging with correlation IDs
- OpenTelemetry instrumentation
- Full type hints and docstrings

Generate these files and updates.
```

**Your Manual Testing Tasks:**
- [ ] Start server: `make dev`
- [ ] Test non-streaming (curl)
- [ ] Test streaming (curl)
- [ ] Test with FastAPI docs UI: http://localhost:8000/docs
- [ ] Test error cases (bad API key, invalid model)
- [ ] Verify Jaeger traces show complete flow
- [ ] Check Prometheus metrics increment

**Test Commands:**

```bash
# 1. Non-streaming request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": false
  }'

# 2. Streaming request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-haiku-20241022",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'

# 3. Test with Anthropic
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Write a haiku"}],
    "stream": false
  }'

# 4. Test invalid model (should 400)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "invalid-model-9000",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'

# 5. Check response headers
curl -v http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": false
  }' | grep -i "x-"
# Should see: X-Provider, X-Request-ID, X-Cache-Status
```

**Validation Checklist:**
- [ ] Streaming works (incremental output)
- [ ] Non-streaming works (single response)
- [ ] Multiple providers work (OpenAI, Anthropic)
- [ ] Error responses have correct status codes
- [ ] Response headers present (X-Provider, X-Request-ID)
- [ ] Jaeger shows spans: gateway.completions → litellm.generate → litellm.api_call
- [ ] Logs show request start and end
- [ ] FastAPI docs UI works (can test from browser)

### Day 5: Integration Tests & Week 1 Review

**Morning: Integration Tests**

**Claude Code Prompt:**
```
Create integration tests for the complete gateway flow.

CONTEXT:
- Gateway uses LiteLLM wrapper for provider access
- End-to-end tests from HTTP request to provider response
- Use FastAPI TestClient for HTTP testing

CREATE:

tests/integration/test_gateway_integration.py

Mark all tests with @pytest.mark.integration (skip in CI by default)

FIXTURES:
- test_client: FastAPI TestClient
- Set up environment variables with API keys

TEST COVERAGE:

1. End-to-End Gateway Tests
   - OpenAI streaming request:
     - Verify SSE format
     - Verify chunks received incrementally
     - Verify [DONE] message at end
     - Check response headers (X-Provider, X-Request-ID)

   - Anthropic non-streaming request:
     - Verify JSON response format
     - Verify usage tokens present
     - Check response time < 5 seconds

   - Invalid model returns 400:
     - Send request with invalid model name
     - Verify error message in response

   - Missing messages returns 422:
     - Send request without messages field
     - Verify validation error

   - Invalid temperature returns 422:
     - Send request with temperature = 5
     - Verify validation error

2. Observability Tests
   - Jaeger trace created:
     - Make request
     - Query Jaeger API for trace (if accessible)
     - Verify spans present: gateway.completions, litellm.generate

   - Prometheus metrics incremented:
     - Get current metric count
     - Make request
     - Verify llm_requests_total incremented

REQUIREMENTS:
- Use real API calls (mark with @pytest.mark.integration)
- Test both OpenAI and Anthropic
- Test streaming and non-streaming
- Verify response format (SSE vs JSON)
- Reasonable timeouts (don't hang)

Generate this test file.
```

**Your Testing Tasks:**
- [ ] Set up API keys in test environment
- [ ] Run integration tests: `pytest -m integration -v`
- [ ] Verify all tests pass
- [ ] Check Jaeger for test traces
- [ ] Monitor Prometheus metrics during tests

**Afternoon: Documentation & Week 1 Review**

**Create Week 1 Summary Document:**

```markdown
# Week 1 Summary: Provider Abstraction with LiteLLM

## What We Built

### Architecture Decision
- Chose LiteLLM over custom provider abstraction
- Rationale: Focus on unique value (caching, rate limiting)
- Supports 20+ providers out of the box

### Components Delivered
1. **LiteLLM Wrapper** (src/llmgateway/providers/litellm_wrapper.py)
   - Custom error mapping
   - OpenTelemetry instrumentation
   - Enhanced retry logic
   - Structured logging

2. **Data Models** (src/llmgateway/providers/models.py)
   - CompletionRequest (with validation)
   - CompletionChunk
   - CompletionResponse

3. **Error Types** (src/llmgateway/providers/errors.py)
   - ProviderError (base)
   - RateLimitError (retriable)
   - AuthError (permanent)
   - TimeoutError (retriable)
   - InvalidRequestError (permanent)
   - ProviderUnavailableError (retriable)

4. **Gateway Endpoint** (src/llmgateway/api/completions.py)
   - OpenAI-compatible /v1/chat/completions
   - Streaming (SSE) and non-streaming support
   - Error handling → HTTP status codes
   - Response headers (X-Provider, X-Request-ID, X-Cache-Status)

5. **Tests**
   - Unit tests with mocking (>90% coverage)
   - Integration tests with real API calls
   - Multi-provider testing

### Providers Tested
- ✅ OpenAI (gpt-4o, gpt-4o-mini)
- ✅ Anthropic (claude-3-5-sonnet, claude-3-5-haiku)
- ⚠️ Together AI (optional, if API key available)
- ⚠️ Groq (optional, if API key available)

## Metrics

### Code Quality
- Test coverage: >85%
- Type hints: 100% of functions
- Docstrings: All public functions
- Linting: Ruff passes

### Performance
- Response time: ~1-2s (provider latency)
- Streaming latency: <100ms first chunk
- Error rate: 0% (in tests)

### Observability
- OpenTelemetry spans: ✅
  - gateway.completions
  - litellm.generate
  - litellm.api_call
- Structured logging: ✅
  - Request start/end
  - Errors with context
- Prometheus metrics: ✅ (basic, will expand Week 2)

## What We Learned

### Technical Learnings
1. **LiteLLM Integration**
   - How to wrap third-party library with custom logic
   - Error mapping patterns
   - Async streaming with AsyncIterator

2. **FastAPI Streaming**
   - Server-Sent Events (SSE) format
   - StreamingResponse usage
   - Proper async context handling

3. **Observability Patterns**
   - OpenTelemetry span hierarchy
   - Correlation IDs for request tracking
   - Structured logging best practices

4. **Testing Strategies**
   - Mocking external APIs
   - Integration test setup
   - Coverage measurement

### Design Decisions
1. **Build vs Buy**: Chose LiteLLM (pragmatic)
2. **Error Handling**: Custom types for gateway contract
3. **Retry Logic**: Exponential backoff with jitter
4. **Streaming**: SSE for compatibility

## Interview Talking Points

### Technical Depth
> "I integrated LiteLLM for multi-provider support but added a custom wrapper with enhanced error handling, OpenTelemetry instrumentation, and retry logic with exponential backoff. This gave me instant support for 20+ providers while maintaining full control over observability and error behavior."

### Build vs Buy Decision
> "I evaluated building a custom provider abstraction versus using LiteLLM. Building custom would take 2 weeks and require ongoing maintenance. LiteLLM is battle-tested and community-maintained. I chose to leverage it and focus on unique value-adds: semantic caching, rate limiting, and observability."

### Error Handling
> "I mapped all LiteLLM exceptions to custom error types that align with our gateway's retry strategy. RateLimitError and TimeoutError are retriable with exponential backoff, while AuthError and InvalidRequestError fail immediately. This gives us consistent error behavior across all providers."

### Observability
> "Every LLM call has OpenTelemetry spans showing the complete flow: gateway endpoint → LiteLLM wrapper → provider API call. With structured logging, I can correlate all logs for a single request using correlation IDs. This makes debugging production issues trivial."

## Next Week Preview: Caching & Rate Limiting

Week 2 will add:
- ✅ Exact match caching (Redis)
- ✅ Semantic caching (embedding similarity)
- ✅ Token bucket rate limiting (distributed)
- ✅ Real-time cost tracking

These are the unique value-adds that differentiate our gateway from just using LiteLLM directly.

## Outstanding Issues
- None! Week 1 complete and working.

## Time Spent
- Day 1: LiteLLM integration (3 hours)
- Day 2: Testing and validation (3 hours)
- Day 3: Multi-provider testing (2 hours)
- Day 4: Gateway endpoint (3 hours)
- Day 5: Integration tests & review (2 hours)
- **Total: 13 hours**

Ahead of schedule! Used time saved from not building custom providers.
```

**Your Tasks:**
- [ ] Review all Week 1 code
- [ ] Update README with current functionality
- [ ] Commit all changes: `git commit -m "feat: Week 1 complete - LiteLLM integration"`
- [ ] Tag release: `git tag v0.1.0-week1`
- [ ] Push to GitHub: `git push && git push --tags`
- [ ] Update project documentation
- [ ] Celebrate! 🎉 Week 1 is complete!

### Week 1 Deliverables

**Code:**
- ✅ LiteLLM wrapper with custom error handling and observability
- ✅ Custom error type hierarchy (6 error types)
- ✅ Data models with validation (CompletionRequest, CompletionChunk, CompletionResponse)
- ✅ FastAPI completions endpoint (`/v1/chat/completions`)
- ✅ Support for 20+ providers via LiteLLM (tested: OpenAI, Anthropic)
- ✅ Streaming (SSE) and non-streaming responses
- ✅ Comprehensive test suite (>85% coverage)

**Validation:**
```bash
# All tests pass
make test
pytest -m integration  # Integration tests with real APIs

# Can proxy requests to multiple providers
curl localhost:8000/v1/chat/completions -d '{"model":"gpt-4o-mini",...}'
curl localhost:8000/v1/chat/completions -d '{"model":"claude-3-5-haiku-20241022",...}'

# Streaming works
curl localhost:8000/v1/chat/completions -d '{"model":"gpt-4o-mini","stream":true,...}'
# Outputs SSE format: data: {...}\n\n

# Observability working
# - Jaeger shows traces: gateway.completions → litellm.generate → litellm.api_call
# - Logs show structured output with correlation IDs
# - Prometheus shows basic metrics
```

**Documentation:**
- ✅ Provider architecture documented
- ✅ Design decision: LiteLLM vs custom abstraction
- ✅ Multi-provider testing results
- ✅ Usage examples in README
- ✅ Troubleshooting guide

**Interview Talking Points:**
- "I chose LiteLLM over building custom provider abstraction, saving 2 weeks while getting 20+ providers"
- "Built custom wrapper with enhanced error handling, retry logic, and OpenTelemetry instrumentation"
- "Custom error types enable intelligent retry strategy: transient vs permanent failures"
- "Every LLM call has distributed tracing showing request flow and latency breakdown"
- "Supports streaming via Server-Sent Events, compatible with OpenAI API format"

**Time Saved:**
Using LiteLLM saved ~1.5 weeks compared to building custom provider implementations. This time will be invested in Week 2's caching and rate limiting features.

---

## Week 2: Caching & Rate Limiting

### Goals
- ✅ Exact match caching (hash-based, Redis) — **DONE**
- ✅ Semantic caching (embedding similarity, Redis) — **DONE**
- ✅ Token bucket rate limiting (distributed, Redis Lua) — **DONE**
- ✅ Cost tracking per request — **DONE**
- ⏳ Cache hit rate >30% — measured after load testing

### Week 2 Actual Components Built

```
src/llmgateway/
├── cache/
│   ├── base.py           CacheEntry dataclass + CacheBackend Protocol
│   ├── redis_cache.py    RedisCache (redis.asyncio, OTel spans, structlog)
│   ├── cache_manager.py  CacheManager (exact-match + semantic, Prometheus metrics)
│   └── embeddings.py     EmbeddingModel (sentence-transformers, cosine similarity)
├── ratelimit/
│   ├── limiter.py        RateLimiter (token bucket via Redis Lua)
│   └── scripts/
│       └── token_bucket.lua  Atomic Lua script
├── cost/
│   ├── pricing.py        PRICING_TABLE + calculate_cost()
│   └── tracker.py        CostTracker (SQLAlchemy async → PostgreSQL)
└── api/
    ├── completions.py    Updated: cache check, rate limit, cost record, headers
    └── admin.py          GET /admin/usage (cost summary endpoint)

scripts/
└── test_cache.py         Manual test script (component + HTTP tests)
```

### Day 1: Cache Architecture Design

**Morning: Design Decisions (YOU make these)**

```markdown
## Caching Strategy Design

### Two-Layer Approach

Layer 1: Exact Match Cache
- Key: SHA256(model + messages + temperature + max_tokens)
- TTL: 1 hour
- Hit rate: ~20-25% (users repeat exact queries)
- Latency: 5ms
- Cost savings: 100%

Layer 2: Semantic Cache
- Key: Embedding of user message + threshold match
- TTL: 1 hour
- Hit rate: ~15-20% (semantically similar queries)
- Latency: 50ms (embedding + vector search)
- Cost savings: 80% (still count as "used tokens")

### Why Two Layers?
- Exact match is fast, cheap, high confidence
- Semantic catch broader set of queries
- Can disable semantic if cost/latency concern

### Storage: Redis
- Fast (microsecond latency)
- TTL built-in (automatic expiration)
- Lua scripts for atomic operations
- Scales with Redis Cluster if needed

Alternative Considered: Qdrant/Pinecone for semantic
- More powerful vector search
- Overkill for <10K cached entries
- Decision: Start with Redis, migrate if needed

### Cache Invalidation
- Time-based: TTL handles most cases
- Manual: Admin endpoint to clear cache
- Selective: Option to bypass cache per request
```

### Day 1 Afternoon: Exact Match Cache ✅ COMPLETE

**Claude Code Prompt:**
```
Implement exact match caching using Redis.

CONTEXT:
- Cache responses to reduce API calls and costs
- Use Redis for fast lookups
- Only cache deterministic requests (temperature=0)

CREATE:

1. src/llmgateway/cache/base.py
   - CacheEntry dataclass: key, value (JSON-serialized), created_at, ttl, metadata
   - CacheBackend Protocol: async get(), set(), delete(), exists()

2. src/llmgateway/cache/redis_cache.py
   - RedisCache class implementing CacheBackend
   - Use aioredis for async operations
   - Serialize CacheEntry as JSON
   - Add OpenTelemetry spans for cache operations
   - Log cache hits/misses with structlog

3. src/llmgateway/cache/cache_manager.py
   - CacheManager class
   - generate_cache_key(): SHA256 hash of (model + messages + temp + max_tokens)
   - Only cache temperature=0 requests (return None for others)
   - get_cached_response(): check cache, log hit/miss, update metrics
   - cache_response(): store with TTL (default 1 hour)

4. Add Prometheus metrics
   - llm_cache_hits_total
   - llm_cache_misses_total
   - llm_cache_lookup_duration_seconds

5. Update src/llmgateway/api/completions.py
   - Inject CacheManager dependency
   - Check cache before calling provider
   - Store response in cache after provider call
   - Add X-Cache-Status header (HIT or MISS)

REQUIREMENTS:
- Only cache temperature=0 requests
- TTL configurable via environment (default 3600s)
- Cache failures shouldn't break requests (fail open)
- Comprehensive logging

Generate these files.
```

**Your Tasks:**
- [x] Add redis[hiredis] to dependencies (already present in pyproject.toml)
- [x] Test cache hit/miss behavior
- [x] Verify X-Cache-Status header
- [x] Check Prometheus metrics: curl localhost:8000/metrics | grep cache
- [ ] Load test: send same request 100 times, verify 1 API call

#### What Was Actually Built — Day 1

**`cache/base.py`**
- `CacheEntry` dataclass: `key`, `value` (JSON string), `created_at` (Unix ts), `ttl`, `metadata`, `is_expired()` helper
- `CacheBackend` Protocol (`@runtime_checkable`): `async get / set / delete / exists`

**`cache/redis_cache.py`**
- Uses `redis.asyncio` (bundled in `redis[hiredis]≥4.2`) — **not** a separate `aioredis` package
- Every operation wrapped in an OTel span with `db.system=redis`, `cache.key`, `cache.hit` attributes
- All exceptions caught → returns `None`/`False` (fail-open)
- Keys namespaced as `llmgw:cache:<sha256>`

**`cache/cache_manager.py`**
- `generate_cache_key()` → `SHA256(json.dumps({model, messages, temperature, max_tokens}, sort_keys=True))`
  - `sort_keys=True` ensures key is stable regardless of dict insertion order
- `get_cached_response()` — skips immediately for `temperature != 0.0`
- `cache_response()` — skips for `temperature != 0.0`, stores with configurable TTL
- Three Prometheus metrics: `llm_cache_hits_total`, `llm_cache_misses_total`, `llm_cache_lookup_duration_seconds`

**`config.py`** — added `cache_ttl: int = Field(default=3600)` (override with `CACHE_TTL` env var)

**`main.py`** — startup initializes `aioredis.from_url()` → `RedisCache` → `CacheManager` on `app.state`; Redis failure is non-fatal (logs warning, sets `cache_manager=None`)

**`api/completions.py`** — non-streaming path: `get_cached_response()` → return `HIT`; else call provider → `cache_response()` → return `MISS`. Streaming always `MISS`.

**Design Decisions Made:**
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Only cache `temperature=0` | Hard gate, not configurable | Non-zero temp is stochastic; serving stale would be semantically wrong |
| Fail-open on Redis error | Return `None` (treat as miss) | Cache is a performance optimisation, not required for correctness |
| `redis.asyncio` not `aioredis` | Use bundled module | `aioredis` was merged into `redis-py≥4.2`; no extra dep needed |
| SHA-256 over all params | Whole request fingerprint | Ensures model, messages, temp, max_tokens all affect the key |
| `sort_keys=True` | Canonical JSON | Prevents dict-ordering differences generating different keys |

**`scripts/test_cache.py`** created — exercises component tests (no server needed) AND HTTP tests (gateway needed), with graceful skip when either is unavailable.

**Ruff pre-commit hook fix:**
`scripts/test_all_providers.py` was missing `# noqa: E402` on one import, and `tests/api/test_completions.py` had an invalid noqa label (`unreachable` instead of a code). Both fixed before commit.

### Day 2: Semantic Caching ✅ COMPLETE

**Claude Code Prompt:**
```
Implement semantic caching using embedding similarity.

CONTEXT:
- Catch similar queries: "What's the weather?" ≈ "Tell me about weather"
- Use sentence-transformers for embeddings
- Store embeddings in Redis, search with cosine similarity
- Only check semantic cache if exact match misses

CREATE:

1. src/llmgateway/cache/embeddings.py
   - EmbeddingModel class
   - Use sentence-transformers model: "all-MiniLM-L6-v2" (fast, 384 dimensions)
   - encode(text) → numpy array (cache model instance, don't reload)
   - Add OpenTelemetry span for encoding time
   - cosine_similarity(vec1, vec2) → float (0-1)

2. Update src/llmgateway/cache/cache_manager.py
   - Add semantic cache methods:
   - get_semantic_match(): extract user message, generate embedding, search Redis for similar embeddings, calculate cosine similarity, return if >threshold (0.95)
   - cache_with_embedding(): store response + embedding in Redis
   - Log similarity scores

3. Update src/llmgateway/api/completions.py
   - Check exact match first (fast path)
   - If miss and semantic enabled, check semantic match
   - Add X-Cache-Type header (EXACT, SEMANTIC, or MISS)

4. Configuration
   - ENABLE_SEMANTIC_CACHE (default: true)
   - SEMANTIC_CACHE_THRESHOLD (default: 0.95)
   - SEMANTIC_CACHE_MAX_ENTRIES (default: 1000)

5. Metrics
   - llm_semantic_cache_hits_total
   - llm_semantic_cache_lookups_duration_seconds
   - llm_embedding_generation_duration_seconds

REQUIREMENTS:
- Semantic cache opt-in (can disable)
- Limit to most recent N entries
- Embedding generation <50ms p99
- Graceful fallback if embedding fails

Generate these files and updates.
```

**Your Tasks:**
- [x] Add sentence-transformers to dependencies
- [ ] Test with similar queries: "hello" vs "hi there"
- [ ] Verify semantic matches when similarity >0.95
- [ ] Measure embedding latency (should be <50ms)
- [x] Test with semantic cache disabled (should still work — `ENABLE_SEMANTIC_CACHE=false`)

#### What Was Actually Built — Day 2

**`cache/embeddings.py`**
- `EmbeddingModel` wraps `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dims)
- `encode(text)` runs in a thread pool (`asyncio.get_event_loop().run_in_executor`) to avoid blocking the event loop; OTel span wraps the call
- `cosine_similarity(vec1, vec2)` — pure numpy dot product on L2-normalised vectors

**`cache/cache_manager.py`** extended with semantic methods:
- `get_semantic_match(request)` — extracts last user message → generates embedding → `ZRANGE llmgw:sem:idx:{model}` → `MGET` all embeddings in one round-trip → cosine similarity scan → fetch response if best score ≥ `semantic_threshold`
- `cache_with_embedding(request, response_data)` — generate embedding → `SET llmgw:sem:emb:{uuid}` + `SET llmgw:sem:resp:{uuid}` with TTL → `ZADD llmgw:sem:idx:{model}` (score = Unix timestamp) → evict oldest with `ZPOPMIN` if index exceeds `semantic_max_entries`

**Redis key layout for semantic index:**
```
llmgw:sem:emb:{uuid}   JSON float list (embedding vector), TTL-expiring
llmgw:sem:resp:{uuid}  JSON response dict, TTL-expiring
llmgw:sem:idx:{model}  Sorted set of UUIDs, scored by creation time (FIFO eviction)
```

**New Prometheus metrics:** `llm_semantic_cache_hits_total`, `llm_semantic_cache_lookups_duration_seconds`

**`completions.py`** request flow (non-streaming):
```
1. get_cached_response()    → exact-match check (SHA-256 lookup)
   ├─ HIT  → return (X-Cache-Status: HIT, X-Cache-Type: EXACT)
   └─ MISS ↓
2. get_semantic_match()     → embedding + cosine scan
   ├─ HIT  → return (X-Cache-Status: HIT, X-Cache-Type: SEMANTIC)
   └─ MISS ↓
3. provider.generate()      → LLM API call
4. cache_response()         → store exact-match entry
5. cache_with_embedding()   → store semantic entry
6. return (X-Cache-Status: MISS)
```

**`config.py`** additions:
- `enable_semantic_cache: bool = Field(default=True)` → `ENABLE_SEMANTIC_CACHE`
- `semantic_cache_threshold: float = Field(default=0.95)` → `SEMANTIC_CACHE_THRESHOLD`
- `semantic_cache_max_entries: int = Field(default=1000)` → `SEMANTIC_CACHE_MAX_ENTRIES`

**Design Decisions Made:**
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embed only the last user message | Single string | System prompts and history are handled by exact-match; semantic targets user intent |
| MGET batch for similarity scan | One Redis round-trip | Avoids N round-trips for N stored embeddings; critical for latency |
| Redis ZSET for index (not a list) | Sorted set by timestamp | Enables FIFO eviction with `ZPOPMIN` in O(log N); no separate expiry tracking needed |
| Thread pool for embedding | `run_in_executor` | `sentence-transformers` is CPU-bound; blocking the event loop would stall all concurrent requests |
| Threshold default 0.95 | High confidence | At 0.95 cosine similarity the queries are nearly identical; lower values risk semantically-different questions getting the same answer |

### Day 3: Rate Limiting ✅ COMPLETE

**Claude Code Prompt:**
```
Implement token bucket rate limiting using Redis.

CONTEXT:
- Prevent abuse and manage costs
- Token bucket algorithm (allows bursts)
- Distributed via Redis Lua scripts
- Per-user rate limits

CREATE:

1. src/llmgateway/ratelimit/token_bucket.py
   - TokenBucket class
   - async consume(user_id, tokens) → (allowed: bool, retry_after: float)
   - Use Redis Lua script for atomic operations (provided below)
   - Add OpenTelemetry span

2. src/llmgateway/ratelimit/limiter.py
   - RateLimiter class
   - RateLimitResult dataclass: allowed, retry_after, remaining, reset_time
   - check_rate_limit(user_id, cost) → RateLimitResult
   - get_rate_limit_info(user_id) → dict (for debugging/admin)

3. src/llmgateway/ratelimit/scripts/token_bucket.lua
   Use this Lua script for atomic token bucket operations:

```lua
-- Token bucket implementation
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

-- Get current state
local state = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(state[1]) or capacity
local last_refill = tonumber(state[2]) or now

-- Calculate refill
local elapsed = math.max(0, now - last_refill)
local refill = elapsed * rate
tokens = math.min(capacity, tokens + refill)

-- Check if enough tokens
if tokens >= requested then
  tokens = tokens - requested
  redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
  redis.call('EXPIRE', key, 3600)
  return {1, tokens, 0}  -- allowed, remaining, retry_after
else
  local needed = requested - tokens
  local retry_after = needed / rate
  return {0, tokens, retry_after}  -- denied, remaining, retry_after
end
```

4. Update src/llmgateway/api/completions.py
   - Extract user_id from request header or API key
   - Check rate limit before processing
   - If exceeded: return 429 with Retry-After and X-RateLimit-* headers
   - Add rate limit info to all response headers

5. Configuration
   - RATE_LIMIT_ENABLED (default: true)
   - RATE_LIMIT_DEFAULT_RATE (default: 10/minute)
   - RATE_LIMIT_DEFAULT_CAPACITY (default: 20)

6. Metrics
   - llm_rate_limit_exceeded_total{user_id}
   - llm_rate_limit_checks_total

REQUIREMENTS:
- Atomic operations via Lua script
- Handle Redis failures gracefully (fail open with warning)
- Per-user limits configurable
- Support different tiers (free, pro, enterprise)

Generate these files.
```

**Your Tasks:**
- [ ] Test rate limiting with rapid requests
- [ ] Verify Retry-After header is accurate
- [ ] Check X-RateLimit-* headers
- [ ] Test with multiple "users" (don't interfere with each other)
- [ ] Load test: `ab -n 200 -c 10 http://localhost:8000/v1/chat/completions`

#### What Was Actually Built — Day 3

**`ratelimit/limiter.py`** — `RateLimiter` class:
- Constructor: `redis_client`, `default_capacity` (bucket size), `default_rate` (tokens/sec), `enabled` flag
- `check(user_id)` → runs the Lua script via `redis.evalsha()`, returns `(allowed: bool, retry_after: float, remaining: float)`
- Redis key: `llmgw:rl:{user_id}`, hash with `tokens` and `last_refill` fields; 1-hour TTL via `EXPIRE`
- Fails open: Redis error → allow request, log warning

**`ratelimit/scripts/token_bucket.lua`** — atomic Lua script:
- Calculates tokens refilled since `last_refill` using `elapsed * rate`
- Caps at `capacity` (prevents "saving up" tokens indefinitely)
- Returns `{1, remaining, 0}` (allowed) or `{0, remaining, retry_after}` (denied)
- All operations in one Lua execution → atomically safe, no race conditions

**`completions.py`** rate-limit integration:
- `user_id` extracted from `body.user` (falls back to `"anonymous"`)
- Check runs before cache lookup — rate-limited requests never hit the LLM or cache
- 429 response includes `Retry-After` (seconds) and `X-RateLimit-Remaining` headers

**`config.py`** additions:
- `rate_limit_enabled: bool = Field(default=True)` → `RATE_LIMIT_ENABLED`
- `rate_limit_default_capacity: int = Field(default=20)` → `RATE_LIMIT_DEFAULT_CAPACITY`
- `rate_limit_default_rate: float = Field(default=10.0)` → `RATE_LIMIT_DEFAULT_RATE` (requests/minute, converted to/sec internally)

**`main.py`** — `RateLimiter` shares the same `redis_client` instance as `CacheManager` (no extra connection pool needed)

**Design Decisions Made:**
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Token bucket over sliding window | Token bucket | Allows short bursts (natural user behaviour) while enforcing sustained rate; simpler Lua script |
| Lua script for atomicity | Single `EVAL` call | No WATCH/MULTI/EXEC dance; Lua runs as a single Redis command — inherently atomic |
| Fail-open on Redis error | Allow request | Rate limiting is a protection, not a hard requirement; outage should not become a user-visible failure |
| Shared Redis client | One connection pool | Rate limiter + cache already run on the same Redis instance; sharing avoids doubling connections |
| `user_id` from request body | `body.user` field | OpenAI-compatible field; falls back to `"anonymous"` — no auth header required for now |

### Day 4: Cost Tracking ✅ COMPLETE

**Claude Code Prompt:**
```
Implement real-time cost tracking per request.

CONTEXT:
- Track costs to prevent budget overruns
- Store usage in PostgreSQL for analytics
- Real-time cost calculation using provider pricing

CREATE:

1. src/llmgateway/cost/pricing.py
   - PRICING_TABLE dict with per-1K-token rates:
     - gpt-4o: input $0.0025, output $0.01
     - gpt-4o-mini: input $0.00015, output $0.0006
     - claude-3-5-sonnet: input $0.003, output $0.015
     - claude-3-5-haiku: input $0.0008, output $0.004
   - calculate_cost(model, input_tokens, output_tokens) → float (USD)

2. src/llmgateway/cost/tracker.py
   - CostTracker class with async SQLAlchemy session
   - record_usage(user_id, model, tokens, cost, cached)
   - get_user_cost(user_id, start_date, end_date) → dict (breakdown by model)
   - get_daily_cost() → float (for monitoring/alerts)
   - Add OpenTelemetry spans

3. Database migration
   Create usage_records table with columns:
   - id, timestamp (indexed), user_id (indexed), model
   - input_tokens, output_tokens, cost_usd
   - cached (boolean), cache_type (EXACT/SEMANTIC/null)
   - Composite index: (user_id, timestamp)

4. Update src/llmgateway/api/completions.py
   - Calculate cost after response
   - Record usage in database (async, don't block response)
   - Add X-Cost header with USD amount
   - Update Prometheus gauge: llm_cost_usd_total

5. Admin endpoint
   - GET /admin/costs/summary with params: user_id, start_date, end_date
   - Return: total_cost, cost_by_model, request_count
   - Simple admin authentication

6. Budget alerts
   - Check daily cost after each request
   - Log warning if > DAILY_COST_ALERT_THRESHOLD
   - Future: webhook to Slack/email

REQUIREMENTS:
- Accurate cost calculation using current pricing
- Cached requests: cost = 0 for exact match
- Async database writes (don't block response)
- Cost metrics in Prometheus

Generate these files.
```

**Your Tasks:**
- [ ] Run database migration: `alembic upgrade head`
- [ ] Test cost calculation with different models
- [ ] Verify X-Cost header on responses
- [ ] Query database: `SELECT SUM(cost_usd) FROM usage_records WHERE timestamp > NOW() - INTERVAL '1 day'`
- [ ] Check Grafana: cost per hour, cost by model

#### What Was Actually Built — Day 4

**`cost/pricing.py`** — `PRICING_TABLE`:
- Exact model → `(input_$/1k, output_$/1k)` for OpenAI, Anthropic, Google, Together, Groq, Mistral
- Prefix fallback (`_PREFIX_FALLBACKS` list, most-specific first) for versioned model strings not in the table
- Conservative unknown default: `$0.002/1k` for both input and output
- `calculate_cost(model, input_tokens, output_tokens) → float` rounded to 8 decimal places

**`cost/tracker.py`** — `CostTracker` (SQLAlchemy async):
- ORM model `UsageRecord` → `usage_records` table with columns: `id`, `timestamp` (indexed), `user_id` (indexed), `model`, `input_tokens`, `output_tokens`, `cost_usd`, `cached` (bool), `cache_type` (`"EXACT"` / `"SEMANTIC"` / `None`)
- `record_usage()` — INSERT with `AsyncSession`; exceptions caught and logged (never propagated)
- `get_summary(user_id, start, end)` — `GROUP BY model` aggregate with optional date filter
- `get_daily_cost()` — `SUM(cost_usd)` for today UTC (used for budget alert)
- `close()` — `await engine.dispose()` called in shutdown event

**`completions.py`** cost integration:
- `_schedule_cost_record()` wraps DB write in `asyncio.create_task()` — **fire-and-forget**, response is never delayed
- `_record_cost_async()` calls `record_usage()` then checks `get_daily_cost()` vs `settings.daily_cost_alert_threshold` — logs WARNING if exceeded
- `X-Cost` header added with 8-decimal USD amount
- Prometheus counter `llm_cost_total{model, user_id}` incremented synchronously

**`api/admin.py`** — `GET /admin/usage`:
- Query params: `user_id` (optional), `start` / `end` (ISO datetime, optional)
- Returns `{total_cost_usd, request_count, cost_by_model}`
- Uses `_get_cost_tracker()` dependency; returns 503 if tracker not initialised

**`config.py`** addition:
- `daily_cost_alert_threshold: float = Field(default=50.0)` → `DAILY_COST_ALERT_THRESHOLD` (USD)

**Design Decisions Made:**
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Fire-and-forget DB write | `asyncio.create_task()` | Cost recording must never add latency to the user response; Postgres is slower than Redis |
| Cost = $0.00 for cache hits | `cost_usd=0.0`, `cached=True` | Cached responses consumed no tokens; still log the row so cache savings are measurable |
| Prefix-based fallback pricing | `_PREFIX_FALLBACKS` list | Model version strings (`claude-sonnet-4-6`, `gpt-4o-2024-05`) change frequently; prefix match keeps coverage without constant table updates |
| Daily alert in same task | After `record_usage()` | No need for a separate cron job; every non-cached request checks the daily total naturally |
| Admin endpoint (no auth yet) | Simple GET with query params | Intentionally deferred auth to keep scope focused; noted as a Phase 2 security task |

### Day 5: Testing & Week 2 Review

**Tasks:**
- [ ] Write tests for caching (exact and semantic)
- [ ] Write tests for rate limiting (bucket refill, multiple users)
- [ ] Write tests for cost tracking (accurate calculations)
- [ ] Integration tests: cache + rate limit + cost together
- [ ] Load test with Locust: measure cache hit rate
- [ ] Document caching strategy in docs/caching.md
- [ ] Update README with rate limiting documentation

**Week 2 Validation:**
```bash
# Test cache
curl localhost:8000/v1/chat/completions -d '{"model":"gpt-4o-mini","messages":[...],"temperature":0}'
# First call: X-Cache-Status: MISS
# Second call: X-Cache-Status: HIT

# Test rate limiting
for i in {1..30}; do
  curl localhost:8000/v1/chat/completions -d '...'
done
# Should see 429 after hitting limit

# Check costs
curl localhost:8000/admin/costs/summary?start_date=2025-02-18
# Should show total cost and breakdown by model

# Metrics
curl localhost:8000/metrics | grep -E "llm_(cache|rate_limit|cost)"
```

### Week 2 Deliverables

**Features:**
- ✅ Exact match caching (`SHA-256` key, `temperature=0` gate, Redis TTL, OTel spans)
- ✅ Semantic caching (`all-MiniLM-L6-v2` embeddings, cosine similarity, Redis ZSET index, FIFO eviction)
- ✅ Token bucket rate limiting (Lua script, per-user, `X-RateLimit-*` headers)
- ✅ Real-time cost tracking (static pricing table, async PostgreSQL, fire-and-forget writes)
- ✅ Admin API (`GET /admin/usage` — cost summary by model, user, date range)
- ✅ Manual test script (`scripts/test_cache.py` — component + HTTP, skip-aware)
- ✅ `X-Cache-Status` header (`HIT` / `MISS`) and `X-Cache-Type` (`EXACT` / `SEMANTIC` / absent)
- ✅ `X-Cost` header (8-decimal USD per response)

**All components fail-open:** Redis unavailable → cache miss, rate limit pass-through; Postgres unavailable → no cost record logged; none of these failures surface as HTTP errors.

**Response headers added (Week 2):**
```
X-Cache-Status: HIT | MISS
X-Cache-Type:   EXACT | SEMANTIC      (present only on HIT)
X-Cost:         0.00003750            (USD, 8 decimal places)
X-RateLimit-Remaining: 17.3           (tokens left in bucket)
Retry-After:    4                     (seconds; present only on 429)
```

**Metrics:**
- ⏳ Cache hit rate: to be measured after load testing
- ✅ Rate limiting: working, tunable via `RATE_LIMIT_DEFAULT_CAPACITY` / `RATE_LIMIT_DEFAULT_RATE`
- ✅ Cost tracking: per-request USD, daily total, model breakdown

**Interview Talking Points (Week 2):**
> "I built a two-layer cache: exact-match via SHA-256 in Redis handles identical requests in ~5ms, and a semantic layer using `all-MiniLM-L6-v2` embeddings catches similar queries with cosine similarity above 0.95. The semantic index uses a Redis sorted-set for O(log N) FIFO eviction and a single MGET round-trip to fetch all stored vectors — avoiding N serial lookups."

> "Rate limiting is a token bucket algorithm implemented in a Redis Lua script. The entire check-and-update runs atomically in one EVAL call, so there's no TOCTOU race between reading the bucket state and writing it back. The rate limiter shares the same Redis client as the cache — no extra connection pool."

> "Cost records are written to PostgreSQL via SQLAlchemy async, but the write is wrapped in `asyncio.create_task()` — fire-and-forget. The HTTP response never waits for the database. If Postgres is down the request still succeeds; a warning is logged. Cached responses record `cost_usd=0.0` so you can measure cache savings in the analytics query."

> "Every component is fail-open by design. Cache failures are cache misses. Rate-limiter failures are rate-limit passes. Cost-tracker failures are silent warnings. The gateway degrades gracefully rather than cascading."

---

## Week 3: Observability & Production Deployment

### Goals
- ✅ Comprehensive metrics (RED method)
- ✅ Distributed tracing with detailed spans
- ✅ Grafana dashboards showing key metrics
- ✅ Deployed to Fly.io with managed services
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ Load tested to 100 req/sec

### Day 1-2: Observability Deep Dive

**Claude Code Prompt:**
```
Enhance observability with comprehensive metrics and dashboards.

CONTEXT:
- OpenTelemetry already configured
- Need detailed metrics for production monitoring
- Create Grafana dashboards for visualization

CREATE:

1. src/llmgateway/observability/metrics.py
   - Define all Prometheus metrics:
     - Counters: llm_requests_total, llm_cache_hits_total, llm_cost_usd_total, llm_rate_limit_exceeded_total
     - Histograms: llm_request_duration_seconds, llm_token_count, llm_provider_api_duration_seconds, llm_cache_lookup_duration_seconds
     - Gauges: llm_cache_hit_rate, llm_active_requests
   - MetricsCollector class (singleton pattern)
   - Methods to record each metric with automatic label extraction
   - Thread-safe operations

2. Update all components to use MetricsCollector
   - completions.py, cache_manager.py, rate_limiter.py, providers

3. Grafana dashboard JSON (.devcontainer/grafana/dashboards/llm-gateway.json)
   - 6 rows covering:
     - Request rate (by model, by status code)
     - Latency (P50/P95/P99, by model, provider vs total)
     - Cache performance (hit rate, lookups/sec, duration)
     - Cost tracking (per hour, per 1K requests, daily total)
     - Rate limiting (exceeded/min, top users)
     - Errors (error rate %, by type)

4. Update Prometheus config (.devcontainer/prometheus.yml)
   - Add recording rules for common queries
   - Add alerting rules: error rate >5%, p99 latency >2s, daily cost >$50

5. src/llmgateway/observability/tracing.py
   - Add custom spans with hierarchy:
     - llm_gateway.request (root)
       - cache.lookup
       - rate_limit.check
       - provider.generate
         - provider.api_call
       - cost.calculate
       - cost.record
   - Rich attributes on each span

REQUIREMENTS:
- All metrics have help text
- Dashboard visually clear
- Tracing shows complete request flow with timing

Generate these files and updates.
```

**Your Tasks:**
- [ ] Restart Grafana to load dashboard
- [ ] Generate traffic: run load test
- [ ] View dashboard: http://localhost:3000
- [ ] Verify all panels show data
- [ ] Check Jaeger: http://localhost:16686
- [ ] Verify spans show complete flow with timing breakdown

### Day 3-4: Production Deployment

**Claude Code Prompt:**
```
Prepare for production deployment on Fly.io.

CONTEXT:
- Need production-ready Dockerfile
- Use Fly.io managed Redis and Postgres
- Set up GitHub Actions for CI/CD

CREATE:

1. Dockerfile (production, multi-stage build):
```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir build && \
    python -m build

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app/dist/*.whl .
RUN pip install --no-cache-dir *.whl
COPY src/ src/
EXPOSE 8000
CMD ["uvicorn", "src.llmgateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. fly.toml:
```toml
app = "llm-gateway-ksmith"

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = false
  auto_start_machines = true
  min_machines_running = 2
  [http_service.concurrency]
    type = "requests"
    soft_limit = 200
    hard_limit = 250

[[vm]]
  cpu_kind = "shared"
  cpus = 2
  memory_mb = 1024

[env]
  LOG_LEVEL = "INFO"
  OTEL_SERVICE_NAME = "llm-gateway"

[metrics]
  port = 8000
  path = "/metrics"
```

3. .github/workflows/ci.yml:
```yaml
name: CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: llmgateway_test
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e '.[dev]'
      - name: Run linting
        run: ruff check src/ tests/
      - name: Run type checking
        run: mypy src/
      - name: Run tests
        run: pytest --cov --cov-report=xml
        env:
          REDIS_URL: redis://redis:6379
          DATABASE_URL: postgresql://postgres:postgres@postgres:5432/llmgateway_test
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: superfly/flyctl-actions/setup-flyctl@master
      - name: Deploy to Fly.io
        run: flyctl deploy --remote-only
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```

4. Create deployment guide: docs/deployment.md
- Prerequisites: Fly.io account, flyctl installed
- Setup Fly.io app: `flyctl launch`
- Create Redis: `flyctl redis create`
- Create Postgres: `flyctl postgres create`
- Set secrets: `flyctl secrets set OPENAI_API_KEY=...`
- Deploy: `flyctl deploy`
- Monitor: `flyctl logs`

REQUIREMENTS:
- Production Dockerfile optimized for size
- Health checks configured
- Auto-scaling based on load
- Secrets managed securely

Generate these files.
```

**Your Tasks:**
- [ ] Create Fly.io account (free tier)
- [ ] Install flyctl: `brew install flyctl`
- [ ] Deploy: follow docs/deployment.md steps
- [ ] Verify deployment: `curl https://llm-gateway-ksmith.fly.dev/health`
- [ ] Check logs: `flyctl logs`
- [ ] Set up GitHub secrets for CI/CD

### Day 5: Load Testing & Optimization

**Claude Code Prompt:**
```
Create load testing suite and optimize performance.

CONTEXT:
- Need to validate system handles production load
- Target: 100 req/sec sustained
- Measure: latency, throughput, error rate

CREATE:

locustfile.py:
```python
from locust import HttpUser, task, between
import random

class LLMGatewayUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def cached_request(self):
        # Deterministic request (should hit cache)
        self.client.post("/v1/chat/completions", json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "temperature": 0,
            "stream": False
        })

    @task(1)
    def uncached_request(self):
        # Non-deterministic request
        self.client.post("/v1/chat/completions", json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"Random: {random.random()}"}],
            "temperature": 0.7,
            "stream": False
        })

    @task(1)
    def streaming_request(self):
        # Streaming request
        self.client.post("/v1/chat/completions", json={
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "Count to 10"}],
            "stream": True
        }, stream=True)
```

Run: `locust -f locustfile.py --host http://localhost:8000`
```

Create performance testing report template: docs/performance.md

**Your Tasks:**
- [ ] Install locust: `pip install locust`
- [ ] Run load test: 10 users → 50 users → 100 users
- [ ] Monitor Grafana during load test
- [ ] Identify bottlenecks from Jaeger traces
- [ ] Optimize (add connection pooling, increase Redis pool size, etc.)
- [ ] Re-test and document improvements
- [ ] Create performance report with metrics

**Optimization Checklist:**
- [ ] httpx connection pooling configured (limits)
- [ ] Redis connection pool size appropriate
- [ ] Database connection pool tuned
- [ ] Caching working correctly (high hit rate)
- [ ] No N+1 queries in database
- [ ] Proper indexes on usage_records table

### Week 3 Deliverables

**Observability:**
- ✅ Grafana dashboard with 15+ panels
- ✅ Distributed tracing showing full request flow
- ✅ Alerts configured (error rate, latency, cost)

**Production Deployment:**
- ✅ Deployed to Fly.io: https://llm-gateway-ksmith.fly.dev
- ✅ Using managed Redis and Postgres
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Health checks and monitoring configured

**Performance:**
- ✅ Load tested to 100 req/sec
- ✅ P99 latency: <200ms (cached) / <1.5s (uncached)
- ✅ Error rate: <0.5%
- ✅ Cache hit rate: 35-45%

**Documentation:**
- ✅ Deployment guide
- ✅ Performance report
- ✅ Operational runbook

**Interview Talking Points:**
- "Deployed to production with full observability stack"
- "Grafana dashboard shows RED metrics in real-time"
- "Load tested to 100 req/sec with p99 latency under 200ms"
- "Used distributed tracing to identify Redis connection pooling as bottleneck"
- "Achieved 40% cache hit rate reducing costs by $X per day"

---

## Week 4 (Optional): Advanced Features

### Goals
- Multi-provider fallback routing
- Prompt template management
- Evaluation framework (RAGAS)
- Interview prep materials

**Note:** Week 4 is optional but highly recommended for Director-level depth.

### Day 1-2: Intelligent Routing

**Claude Code Prompt:**
```
Implement multi-provider fallback and intelligent routing.

CONTEXT:
- Primary provider might be down or rate limited
- Want automatic fallback to secondary provider
- Route based on: cost, latency, or quality

CREATE:

1. src/llmgateway/routing/strategy.py:

Enum: RoutingStrategy
- COST: Choose cheapest provider
- LATENCY: Choose fastest provider
- QUALITY: Choose best for task type
- ROUND_ROBIN: Distribute evenly
- FAILOVER: Primary with fallback

Class: ModelRoute
- model: str
- primary_provider: str
- fallback_providers: list[str]
- strategy: RoutingStrategy

2. src/llmgateway/routing/smart_router.py:

Class: SmartRouter
- Constructor: routes config, provider_map
- Method: async select_provider(model, strategy) -> LLMProvider
  - Check provider health (circuit breaker)
  - Apply routing strategy
  - Return selected provider

- Method: async generate_with_fallback(request) -> AsyncIterator
  - Try primary provider
  - If fails, try fallback providers in order
  - Log fallback usage
  - Add X-Provider header to response

3. Circuit breaker pattern:

Class: CircuitBreaker
- States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing)
- Track failures per provider
- If failure rate > threshold: OPEN (stop sending traffic)
- After timeout: HALF_OPEN (send test request)
- If success: CLOSED (resume normal traffic)

4. Update configuration:
Define routes for each model in config/routes.yaml

REQUIREMENTS:
- Graceful fallback without user noticing
- Track fallback usage in metrics
- Circuit breaker prevents cascading failures

Generate these files.
```

### Day 3: Prompt Management

**Claude Code Prompt:**
```
Implement prompt template management and versioning.

CONTEXT:
- Store reusable prompt templates
- Version control for prompts
- A/B testing different prompt versions

CREATE:

1. Database migration: Add prompts table

2. src/llmgateway/prompts/manager.py:

Class: PromptTemplate
- id, name, version, template (Jinja2), created_at
- Variables: list[str] (extracted from template)

Class: PromptManager
- Method: async get_template(name, version) -> PromptTemplate
- Method: async render(name, version, variables) -> str
- Method: async create_template(name, template) -> PromptTemplate
- Method: async list_versions(name) -> list[PromptTemplate]

3. API endpoints:
GET /admin/prompts
POST /admin/prompts
GET /admin/prompts/{name}/versions

4. A/B testing:
- Randomly route 50% to version A, 50% to version B
- Track which version performed better

Generate these files.
```

### Day 4: Evaluation Framework

**Claude Code Prompt:**
```
Integrate RAGAS for response quality evaluation.

CONTEXT:
- Need to measure response quality
- Compare providers and prompt versions
- Create evaluation dataset

CREATE:

1. src/llmgateway/eval/evaluator.py:

Class: ResponseEvaluator
- Use RAGAS metrics: answer_relevancy, faithfulness
- Method: async evaluate_response(query, response, context) -> dict
- Return scores for each metric

2. Create evaluation dataset:
tests/eval/dataset.yaml with 50 test queries

3. Script: scripts/run_evaluation.py
- Load dataset
- Generate responses with different providers
- Evaluate all responses
- Generate report comparing providers

REQUIREMENTS:
- Automated evaluation pipeline
- Compare GPT-4 vs Claude Sonnet
- Report shows which provider performs better

Generate these files.
```

### Day 5: Interview Prep

**Create comprehensive interview prep document:**

docs/interview-prep.md:
- Architecture diagram (high-level)
- Design decisions log (why token bucket? why Redis?)
- Performance benchmarks (latency, throughput, cost)
- Production war stories (what broke? how did you debug?)
- Trade-off analyses (caching vs freshness, cost vs latency)
- What you'd do differently at scale
- Future improvements (Phase 2 features)

**Talking Points Template:**

```markdown
## Project Overview (2 minutes)
"I built a production LLM gateway to demonstrate AI infrastructure expertise..."

## Technical Deep Dive (5 minutes)
Choose 2-3 areas based on interviewer interest:
1. Caching Strategy
2. Rate Limiting
3. Observability
4. Production Deployment

## Metrics & Results (1 minute)
- 10K req/day
- P99 latency: 185ms (cached) / 1.2s (uncached)
- Cache hit rate: 40%
- Cost savings: 38%
- Throughput: 100 req/sec

## What I Learned (2 minutes)
- AI-native development with Claude Code
- Production observability patterns
- Distributed systems trade-offs
- Cost optimization strategies

## What I'd Do Differently (1 minute)
- Use vector DB for semantic cache at scale
- Implement request batching
- Add more sophisticated routing
```

---

## Phase 1 Success Criteria

### ✅ Technical Deliverables

**Code Quality:**
- [ ] Production-ready codebase (not tutorial code)
- [ ] Test coverage >80%
- [ ] Type hints throughout
- [ ] Comprehensive error handling
- [ ] Structured logging everywhere

**Features:**
- [ ] Multi-provider support (OpenAI, Anthropic)
- [ ] Two-layer caching (exact + semantic)
- [ ] Token bucket rate limiting
- [ ] Real-time cost tracking
- [ ] Full observability stack

**Production Deployment:**
- [ ] Deployed to Fly.io
- [ ] Managed Redis and Postgres
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Load tested to 100 req/sec

**Documentation:**
- [ ] Complete README
- [ ] Architecture docs
- [ ] API documentation
- [ ] Deployment guide
- [ ] Performance report

### ✅ Metrics Achieved

**Performance:**
- P50 latency: <100ms (cached), <800ms (uncached)
- P99 latency: <200ms (cached), <1.5s (uncached)
- Throughput: 100 req/sec sustained
- Error rate: <0.5%

**Cost Optimization:**
- Cache hit rate: 35-45%
- Cost savings vs direct API: 35-40%
- Semantic cache contribution: 10-15%

**Reliability:**
- Uptime: 99.9%
- Health checks: passing
- Rate limit effectiveness: <5% legitimate requests blocked

### ✅ Interview Readiness

**Can Confidently Discuss:**
- [ ] Provider abstraction design choices
- [ ] Caching strategy trade-offs (cost vs freshness vs latency)
- [ ] Rate limiting algorithm selection (token bucket vs alternatives)
- [ ] Observability patterns (RED metrics, distributed tracing)
- [ ] Production debugging with tracing (specific example)
- [ ] Cost optimization strategies
- [ ] What would change at 10x scale
- [ ] AI-native development practices

**Have Prepared:**
- [ ] Architecture diagram you can draw on whiteboard
- [ ] 3 specific "production war stories"
- [ ] Exact performance numbers memorized
- [ ] Design decisions document
- [ ] What you'd do differently (shows growth mindset)

**GitHub Presence:**
- [ ] Clean commit history
- [ ] Comprehensive README
- [ ] Live deployment link
- [ ] Documentation website (optional: MkDocs)

---

## Daily Workflow with Claude Code

### Morning Routine (15 minutes)
```bash
# 1. Start container if not running
# VS Code → Reopen in Container

# 2. Check services
docker ps
make dev  # Start app

# 3. Review yesterday's progress
git log --oneline -5
git diff HEAD~1

# 4. Read today's learning plan section
# Focus on understanding WHY before implementing
```

### Implementation Session (2-3 hours)
```
1. Make architectural decision (YOU decide)
   └─ Document in docs/decisions/

2. Open Claude Code
   └─ Provide context from learning plan
   └─ Be specific about requirements

3. Review generated code
   └─ Check error handling
   └─ Verify type hints
   └─ Understand implementation

4. Test immediately
   └─ Unit tests: make test
   └─ Manual testing: curl commands
   └─ Check observability: Jaeger, Grafana

5. Iterate if needed
   └─ "Add error handling for X"
   └─ "Improve docstring for Y"

6. Commit when working
   └─ git commit -m "feat: semantic caching"
```

### End of Day (15 minutes)
```
1. Document what you learned
   └─ Update docs/learnings.md
   └─ Note any issues or questions

2. Update checklist
   └─ Mark completed tasks
   └─ Plan tomorrow's focus

3. Push to GitHub
   └─ git push origin main

4. Quick retro
   └─ What worked well with Claude Code?
   └─ What would you do differently?
```

---

## Troubleshooting Guide

### Container Issues

**"Services not starting"**
```bash
docker-compose -f .devcontainer/docker-compose.yml ps
docker-compose -f .devcontainer/docker-compose.yml logs redis
docker-compose -f .devcontainer/docker-compose.yml up -d --force-recreate
```

**"Can't connect to Redis/Postgres"**
- Check hostnames in config.py (should be `redis`, `postgres`, NOT `localhost`)
- Verify services running: `docker ps`
- Test connection: `redis-cli -h redis ping` or `psql -h postgres -U postgres`

**"Port already in use"**
```bash
lsof -i :8000  # Find what's using port
kill -9 <PID>  # Kill process
# Or change port in config
```

### Claude Code Issues

**"Extension not in container"**
- Install manually: Extensions → Search "Claude Code" → Install in Container
- Or add to devcontainer.json extensions list and rebuild

**"Lost context from previous conversation"**
- This is expected! Container restart = fresh environment
- Context is in your files, not conversation
- Start new conversation with clear context from learning plan

**"Generated code has errors"**
- Review carefully before accepting
- Ask Claude to fix: "Add error handling for timeout"
- Don't blindly accept - you're the architect

### Application Issues

**"Tests failing"**
```bash
pytest -v  # See which test fails
pytest tests/test_file.py::test_name -s  # Debug specific test
# Check logs for errors
# Verify services are running
```

**"Do I need to rebuild the Docker container for new Python files?"**
No. The workspace is volume-mounted (`..:/workspace:cached`) into the container. `make dev` runs uvicorn with `--reload` directly inside the container and picks up all file changes instantly. Only rebuild (`docker compose build`) when the `Dockerfile` itself changes (e.g. new system packages) or after modifying `pyproject.toml` and needing `pip install` to re-run.

**"Cache not working"**
```bash
# Check Redis — note the actual key prefix used is "llmgw:cache:"
redis-cli -h redis
> KEYS llmgw:cache:*          # See exact-match entries
> KEYS llmgw:sem:*            # See semantic index entries
> GET llmgw:cache:<sha256>    # Inspect a cached response

# Check logs (watch for cache.hit / cache.miss events)
make dev

# Verify temperature=0  — only temperature=0 requests are cached
# Check X-Cache-Status header in response
```

**"Rate limiting not working"**
```bash
# Check Redis
redis-cli -h redis
> KEYS ratelimit:*

# Send rapid requests
for i in {1..20}; do curl localhost:8000/v1/chat/completions ...; done

# Should see 429 responses
```

---

## Resources & References

### Documentation
- **FastAPI:** https://fastapi.tiangolo.com
- **OpenTelemetry Python:** https://opentelemetry.io/docs/languages/python/
- **Prometheus:** https://prometheus.io/docs/
- **Redis:** https://redis.io/docs/
- **SQLAlchemy:** https://docs.sqlalchemy.org/

### Learning Materials
- **Latent Space Podcast:** AI infrastructure episodes
- **Software Engineering Daily:** Platform engineering topics
- **The Pragmatic Engineer:** Newsletter on engineering systems

### Inspiration
- **LiteLLM:** Open source LLM proxy (study their patterns)
- **Portkey:** LLM gateway company (read their blog)
- **Helicone:** Observability for LLMs (learn from their approach)

---

## Phase 2 Preview (Weeks 5-8)

After completing Phase 1, consider these advanced topics:

**Week 5-6: Advanced Features**
- Request batching for cost optimization
- Multi-modal support (images, audio)
- Streaming with server-sent events
- Webhook support for async processing

**Week 7: Security & Compliance**
- API key management (creation, rotation, scoping)
- Request signing and verification
- Audit logging for compliance
- PII detection and redaction

**Week 8: Scale & Optimization**
- Kubernetes deployment
- Horizontal autoscaling
- Advanced caching (vector DB)
- Query rewriting and optimization

But first: **nail Phase 1**. These fundamentals are what every interview will cover.

---

## Final Thoughts

**This is a Director-level project.** You're not just coding - you're making architectural decisions, evaluating trade-offs, and demonstrating production thinking.

**Key Success Factors:**
1. **YOU make design decisions** (document them!)
2. **Test everything** (don't trust generated code)
3. **Use observability** (Jaeger, Grafana are your debugging tools)
4. **Document learnings** (interview prep happens daily)
5. **Ship incrementally** (commit often, deploy early)

**In 3-4 weeks, you'll say:**
> "I built a production LLM gateway handling 10K req/day with semantic caching, distributed rate limiting, and full OpenTelemetry observability. When p99 latency spiked, I used Jaeger to identify the bottleneck and optimized it from 2s to 185ms. The system achieves 40% cost savings through intelligent caching."

That's the depth that gets Director-level offers.

**Now go build it.** Start with Week 1, Day 1, and work through systematically. You've got this. 🚀
