# LLM Gateway Learning Plan - Phase 1 (Final Version)
## Claude Code + VS Code + Dev Containers Edition

**Status:** In Progress
**Environment:** Docker Dev Containers (macOS/Windows)
**Tools:** VS Code + Claude Code Extension + Dev Containers
**Timeline:** 3-4 weeks to production deployment

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
I'm integrating LiteLLM into my LLM Gateway to handle multi-provider support. Create a thin wrapper around LiteLLM with custom error handling and observability.

CONTEXT:
- Project: Production LLM Gateway
- Location: src/llmgateway/providers/
- Decision: Use LiteLLM for provider abstraction (pragmatic choice)
- Focus: Error mapping, observability, and retry logic on top of LiteLLM

DESIGN RATIONALE:
LiteLLM already handles:
- Multi-provider support (OpenAI, Anthropic, Together, Groq, etc.)
- Request format standardization
- Token counting (tiktoken for OpenAI)
- Basic error handling

We add:
- Custom error types for our gateway
- OpenTelemetry instrumentation
- Structured logging with correlation IDs
- Enhanced retry logic with exponential backoff
- Request/response validation

CREATE:

1. src/llmgateway/providers/__init__.py:
   - Export: CompletionRequest, CompletionChunk, CompletionResponse, LLMGatewayProvider
   - Export: ProviderError, RateLimitError, AuthError, TimeoutError, InvalidRequestError, ProviderUnavailableError

2. src/llmgateway/providers/models.py:

   @dataclass(frozen=True)
   class CompletionRequest:
       model: str
       messages: list[dict[str, str]]
       temperature: float = 0.7
       max_tokens: int | None = None
       stream: bool = False
       user_id: str | None = None  # For rate limiting

       def __post_init__(self):
           # Validate model name format
           # Validate messages structure (non-empty, valid roles)
           # Validate temperature range (0-2)
           # Raise InvalidRequestError if validation fails

   @dataclass(frozen=True)
   class CompletionChunk:
       content: str
       finish_reason: str | None = None
       usage: dict[str, int] | None = None  # {"input_tokens": X, "output_tokens": Y}
       model: str | None = None

   @dataclass(frozen=True)
   class CompletionResponse:
       """Full response for non-streaming requests"""
       content: str
       usage: dict[str, int]
       model: str
       finish_reason: str

3. src/llmgateway/providers/errors.py:

   Complete exception hierarchy:

   class ProviderError(Exception):
       """Base exception for all provider errors"""
       def __init__(self, message: str, provider: str | None = None,
                    original_error: Exception | None = None):
           self.message = message
           self.provider = provider
           self.original_error = original_error
           super().__init__(message)

   class RateLimitError(ProviderError):
       """Raised when provider rate limit exceeded (429)"""
       def __init__(self, message: str, retry_after: float | None = None, **kwargs):
           super().__init__(message, **kwargs)
           self.retry_after = retry_after

   class AuthError(ProviderError):
       """Raised for authentication failures (401, 403)"""
       pass

   class TimeoutError(ProviderError):
       """Raised when request times out"""
       pass

   class InvalidRequestError(ProviderError):
       """Raised for malformed requests (400)"""
       pass

   class ProviderUnavailableError(ProviderError):
       """Raised when provider is down (502, 503, 504)"""
       pass

4. src/llmgateway/providers/litellm_wrapper.py:

   Main provider class wrapping LiteLLM:

   class LLMGatewayProvider:
       """
       Wrapper around LiteLLM providing:
       - Standardized error handling
       - OpenTelemetry instrumentation
       - Structured logging
       - Enhanced retry logic

       Example:
       ```python
       provider = LLMGatewayProvider()
       request = CompletionRequest(
           model="gpt-4o",
           messages=[{"role": "user", "content": "Hello"}],
           stream=True
       )
       async for chunk in provider.generate(request):
           print(chunk.content)
       ```
       """

       def __init__(
           self,
           timeout: int = 60,
           max_retries: int = 3,
           enable_fallback: bool = False
       ):
           # Initialize LiteLLM settings
           # Set timeout using litellm.timeout
           # Configure retry behavior
           # Initialize OpenTelemetry tracer (from opentelemetry import trace)
           # Initialize structlog logger

       async def generate(
           self,
           request: CompletionRequest
       ) -> AsyncIterator[CompletionChunk]:
           """
           Generate completion using LiteLLM.

           Automatically handles:
           - Provider selection based on model name
           - Request format conversion
           - Streaming and non-streaming responses
           - Error mapping to our custom types

           Args:
               request: Completion request parameters

           Yields:
               CompletionChunk: Streamed response chunks

           Raises:
               RateLimitError: When rate limited by provider
               AuthError: When API key is invalid
               TimeoutError: When request times out
               InvalidRequestError: When request is malformed
               ProviderUnavailableError: When provider is down
           """

           # Start OpenTelemetry span
           with tracer.start_as_current_span("litellm.generate") as span:
               span.set_attribute("model", request.model)
               span.set_attribute("stream", request.stream)
               span.set_attribute("temperature", request.temperature)
               if request.user_id:
                   span.set_attribute("user_id", request.user_id)

               # Log request start with structlog
               logger.info("llm_request_start",
                          model=request.model,
                          stream=request.stream,
                          user_id=request.user_id,
                          temperature=request.temperature)

               try:
                   # Convert to LiteLLM format
                   litellm_params = self._to_litellm_format(request)

                   # Call LiteLLM with retry decorator
                   if request.stream:
                       async for chunk in self._generate_streaming(litellm_params, span):
                           yield chunk
                   else:
                       response = await self._generate_non_streaming(litellm_params, span)
                       yield response

               except Exception as e:
                   # Log error
                   logger.error("llm_request_error",
                               model=request.model,
                               error=str(e),
                               error_type=type(e).__name__)
                   # Map LiteLLM exceptions to our error types
                   raise self._map_error(e, request.model)

               finally:
                   # Log request completion with duration
                   logger.info("llm_request_complete",
                              model=request.model,
                              duration_ms=span.get_span_context().trace_id if span else 0)

       @retry(
           stop=stop_after_attempt(3),
           retry=retry_if_exception_type((RateLimitError, TimeoutError, ProviderUnavailableError)),
           wait=wait_exponential(multiplier=1, min=2, max=30),
           before_sleep=before_sleep_log(logger, logging.WARNING)
       )
       async def _generate_streaming(
           self,
           litellm_params: dict,
           parent_span
       ) -> AsyncIterator[CompletionChunk]:
           """Internal method: streaming generation with retries"""

           from litellm import acompletion

           with tracer.start_as_current_span("litellm.api_call", parent_span) as span:
               span.set_attribute("call_type", "streaming")

               response = await acompletion(**litellm_params, stream=True)

               async for chunk in response:
                   # Convert LiteLLM chunk format to our CompletionChunk
                   parsed_chunk = self._parse_chunk(chunk)
                   if parsed_chunk.content or parsed_chunk.finish_reason:
                       yield parsed_chunk

       @retry(
           stop=stop_after_attempt(3),
           retry=retry_if_exception_type((RateLimitError, TimeoutError, ProviderUnavailableError)),
           wait=wait_exponential(multiplier=1, min=2, max=30),
           before_sleep=before_sleep_log(logger, logging.WARNING)
       )
       async def _generate_non_streaming(
           self,
           litellm_params: dict,
           parent_span
       ) -> CompletionChunk:
           """Internal method: non-streaming generation with retries"""

           from litellm import acompletion

           with tracer.start_as_current_span("litellm.api_call", parent_span) as span:
               span.set_attribute("call_type", "non_streaming")

               response = await acompletion(**litellm_params, stream=False)

               # Convert to our CompletionChunk format
               return self._parse_response(response)

       def _to_litellm_format(self, request: CompletionRequest) -> dict:
           """Convert our request format to LiteLLM format"""
           params = {
               "model": request.model,
               "messages": request.messages,
               "temperature": request.temperature,
               "stream": request.stream,
           }
           if request.max_tokens:
               params["max_tokens"] = request.max_tokens
           return params

       def _parse_chunk(self, litellm_chunk) -> CompletionChunk:
           """Convert LiteLLM streaming chunk to our format"""
           # LiteLLM chunk structure: chunk.choices[0].delta.content
           content = ""
           finish_reason = None
           usage = None

           if hasattr(litellm_chunk, 'choices') and len(litellm_chunk.choices) > 0:
               choice = litellm_chunk.choices[0]
               if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                   content = choice.delta.content or ""
               if hasattr(choice, 'finish_reason'):
                   finish_reason = choice.finish_reason

           # Usage appears in final chunk for some providers
           if hasattr(litellm_chunk, 'usage') and litellm_chunk.usage:
               usage = {
                   "input_tokens": litellm_chunk.usage.prompt_tokens,
                   "output_tokens": litellm_chunk.usage.completion_tokens
               }

           return CompletionChunk(
               content=content,
               finish_reason=finish_reason,
               usage=usage,
               model=getattr(litellm_chunk, 'model', None)
           )

       def _parse_response(self, litellm_response) -> CompletionChunk:
           """Convert LiteLLM non-streaming response to our format"""
           content = litellm_response.choices[0].message.content
           finish_reason = litellm_response.choices[0].finish_reason

           usage = {
               "input_tokens": litellm_response.usage.prompt_tokens,
               "output_tokens": litellm_response.usage.completion_tokens
           }

           return CompletionChunk(
               content=content,
               finish_reason=finish_reason,
               usage=usage,
               model=litellm_response.model
           )

       def _map_error(self, error: Exception, model: str) -> ProviderError:
           """
           Map LiteLLM exceptions to our custom error types.

           LiteLLM error types:
           - litellm.RateLimitError → RateLimitError
           - litellm.AuthenticationError → AuthError
           - litellm.Timeout → TimeoutError
           - litellm.BadRequestError → InvalidRequestError
           - litellm.ServiceUnavailableError → ProviderUnavailableError
           - litellm.APIError → ProviderError
           """

           import litellm

           provider = self._extract_provider(model)

           if isinstance(error, litellm.RateLimitError):
               # Try to extract retry_after from error
               retry_after = getattr(error, 'retry_after', None)
               return RateLimitError(
                   message=str(error),
                   provider=provider,
                   retry_after=retry_after,
                   original_error=error
               )

           elif isinstance(error, litellm.AuthenticationError):
               return AuthError(
                   message=f"Authentication failed for {provider}: {str(error)}",
                   provider=provider,
                   original_error=error
               )

           elif isinstance(error, litellm.Timeout):
               return TimeoutError(
                   message=f"Request to {provider} timed out: {str(error)}",
                   provider=provider,
                   original_error=error
               )

           elif isinstance(error, litellm.BadRequestError):
               return InvalidRequestError(
                   message=f"Invalid request to {provider}: {str(error)}",
                   provider=provider,
                   original_error=error
               )

           elif isinstance(error, (litellm.ServiceUnavailableError, litellm.APIError)):
               return ProviderUnavailableError(
                   message=f"{provider} is unavailable: {str(error)}",
                   provider=provider,
                   original_error=error
               )

           else:
               # Unknown error, wrap in base ProviderError
               return ProviderError(
                   message=f"Unexpected error from {provider}: {str(error)}",
                   provider=provider,
                   original_error=error
               )

       def _extract_provider(self, model: str) -> str:
           """Extract provider name from model string"""
           # gpt-4 → openai, claude-3 → anthropic, etc.
           if model.startswith("gpt-"):
               return "openai"
           elif model.startswith("claude-"):
               return "anthropic"
           elif model.startswith("together/"):
               return "together"
           elif model.startswith("groq/"):
               return "groq"
           else:
               return "unknown"

       async def count_tokens(self, text: str, model: str) -> int:
           """
           Count tokens using LiteLLM's token counter.

           Uses tiktoken for OpenAI models, approximation for others.
           """
           from litellm import token_counter

           try:
               return token_counter(model=model, text=text)
           except Exception as e:
               logger.warning("token_counting_failed",
                            model=model,
                            error=str(e))
               # Fallback: rough approximation
               return len(text.split()) * 1.3

5. Update pyproject.toml dependencies:
   Add to [project] dependencies: "litellm>=1.17.0"

REQUIREMENTS:
- Use LiteLLM's acompletion (async) for all calls
- Map ALL LiteLLM exceptions to our custom types
- Add OpenTelemetry span for every LLM call with detailed attributes
- Log request start, completion, and errors with structlog
- Use tenacity for retry logic with exponential backoff
- Retry on: RateLimitError, TimeoutError, ProviderUnavailableError (transient failures)
- Don't retry on: AuthError, InvalidRequestError (permanent failures)
- Include comprehensive docstrings with examples
- Full type hints throughout
- Handle both streaming and non-streaming responses
- Proper error context (provider name, original error)

Generate these files with production-quality code.
```

**Your Tasks:**
- [ ] Review generated code, especially error mapping
- [ ] Add litellm to dependencies if not present
- [ ] Check retry logic is correctly configured
- [ ] Verify OpenTelemetry spans are properly nested
- [ ] Test dataclass validation (invalid temperature, empty messages)

### Day 2: Testing & Validation

**Morning: Manual Testing with Real APIs**

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
    provider = LLMGatewayProvider()

    # Temporarily set bad API key
    os.environ["OPENAI_API_KEY"] = "invalid-key"

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
        print(f"Caught expected error: {type(e).__name__}: {e}")
    print()

async def main():
    await test_openai_streaming()
    await test_anthropic_non_streaming()
    await test_error_handling()

if __name__ == "__main__":
    asyncio.run(main())
```

**Your Testing Tasks:**
- [ ] Create `.env` file with real API keys
- [ ] Run test script: `python scripts/test_litellm_wrapper.py`
- [ ] Verify OpenAI streaming works (prints content incrementally)
- [ ] Verify Anthropic non-streaming works (single response)
- [ ] Test token counting: `await provider.count_tokens("Hello world", "gpt-4o")`
- [ ] Verify Jaeger traces appear at http://localhost:16686
- [ ] Check traces show nested spans (generate → api_call)

**Afternoon: Unit Tests with Mocking**

**Claude Code Prompt:**
```
Create comprehensive unit tests for the LiteLLM wrapper.

CONTEXT:
- LLMGatewayProvider wraps litellm.acompletion
- Need to mock LiteLLM to test error handling without real API calls
- Test both streaming and non-streaming responses
- Test all error mappings

CREATE:

tests/providers/test_litellm_wrapper.py:

Use pytest-asyncio and pytest-mock for mocking.

Test Classes:

1. TestCompletionRequest:
   - test_valid_request: Valid request is accepted
   - test_invalid_temperature: Temperature >2 raises ValueError
   - test_empty_messages: Empty messages raises ValueError
   - test_invalid_role: Invalid message role raises ValueError

2. TestLLMGatewayProvider:

   Fixtures:
   - mock_litellm: Mock litellm.acompletion using mocker.patch
   - provider: LLMGatewayProvider instance

   Tests:
   - test_generate_streaming_success:
     - Mock litellm to return async generator with 3 chunks
     - Verify chunks are converted correctly
     - Verify OpenTelemetry span is created

   - test_generate_non_streaming_success:
     - Mock litellm to return single response
     - Verify response conversion
     - Verify usage tokens are extracted

   - test_rate_limit_error_mapping:
     - Mock litellm to raise litellm.RateLimitError
     - Verify RateLimitError is raised with retry_after
     - Verify retry logic attempts 3 times

   - test_auth_error_mapping:
     - Mock litellm to raise litellm.AuthenticationError
     - Verify AuthError is raised
     - Verify NO retries (permanent failure)

   - test_timeout_error_mapping:
     - Mock litellm to raise litellm.Timeout
     - Verify TimeoutError is raised
     - Verify retry attempts

   - test_invalid_request_error_mapping:
     - Mock litellm to raise litellm.BadRequestError
     - Verify InvalidRequestError is raised
     - Verify NO retries

   - test_provider_unavailable_error_mapping:
     - Mock litellm to raise litellm.ServiceUnavailableError
     - Verify ProviderUnavailableError is raised
     - Verify retry attempts

   - test_retry_success_after_failure:
     - Mock litellm to fail twice (RateLimitError), succeed third time
     - Verify successful response after retries
     - Verify retry delays (exponential backoff)

   - test_token_counting:
     - Mock litellm.token_counter
     - Verify correct token count returned
     - Test fallback when token counting fails

Mock Helper Classes:

class MockLiteLLMChunk:
    """Mock LiteLLM streaming chunk"""
    def __init__(self, content: str, finish_reason: str | None = None):
        self.choices = [MockChoice(content, finish_reason)]
        self.model = "gpt-4o"
        self.usage = None

class MockChoice:
    def __init__(self, content: str, finish_reason: str | None):
        self.delta = MockDelta(content)
        self.finish_reason = finish_reason

class MockDelta:
    def __init__(self, content: str):
        self.content = content

REQUIREMENTS:
- Use pytest.mark.asyncio for async tests
- Use mocker.patch to mock litellm.acompletion
- Test all error mappings thoroughly
- Verify retry logic with multiple failure scenarios
- Test OpenTelemetry span creation (mock tracer)
- Aim for >90% coverage of litellm_wrapper.py

Generate this test file with comprehensive coverage.
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

1. docs/providers/README.md:

# Provider System Architecture

## Overview
Explanation of LiteLLM wrapper approach and rationale.

## Supported Providers
Table of supported providers with:
- Provider name
- Model examples
- Required API key environment variable
- Capabilities (streaming, function calling, etc.)

## Request Flow
Diagram showing: Client → Gateway → LiteLLM Wrapper → Provider API

## Error Handling
Document custom error types and when each is raised.

## Observability
Explain OpenTelemetry spans and what they track.

## Usage Examples
Code examples for:
- Streaming request
- Non-streaming request
- Error handling
- Token counting

2. docs/providers/design-decisions.md:

# Provider System Design Decisions

## Decision 1: LiteLLM vs Custom Abstraction
- Date: [Today's date]
- Status: Accepted
- Context: Need to support multiple LLM providers
- Decision: Use LiteLLM with custom wrapper
- Consequences:
  - PRO: 20+ providers instantly supported
  - PRO: Community-maintained, handles API changes
  - PRO: Focus engineering time on caching, rate limiting
  - CON: Dependency on external library
  - CON: Less control over provider internals

## Decision 2: Custom Error Types
- Why we built custom error hierarchy on top of LiteLLM
- How errors map for retry logic

## Decision 3: OpenTelemetry Integration
- Why we instrument every LLM call
- What attributes we track

3. docs/providers/troubleshooting.md:

Common issues and solutions:
- "AuthError: Invalid API key"
- "RateLimitError: Rate limit exceeded"
- "TimeoutError: Request timed out"
- "Provider returns unexpected format"

4. Update README.md:

Add section on multi-provider support:
- Quick start with different providers
- Environment variable setup
- Available models

REQUIREMENTS:
- Clear explanations with examples
- Architecture diagrams (ASCII art is fine)
- Troubleshooting section
- Design rationale documented

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
- LLMGatewayProvider already handles provider selection (via LiteLLM)
- Need FastAPI endpoint compatible with OpenAI API format
- Support both streaming and non-streaming responses
- Map errors to appropriate HTTP status codes

CREATE:

1. src/llmgateway/api/completions.py:

APIRouter with prefix="/v1"

Dependencies:
- get_provider: Returns LLMGatewayProvider instance (FastAPI dependency)

POST /v1/chat/completions endpoint:

Request body: Use CompletionRequest from providers.models
Response: StreamingResponse for streaming, JSONResponse for non-streaming

Functionality:
- Validate request (Pydantic does this automatically)
- Get provider instance from dependency
- Call provider.generate(request)
- If streaming:
  - Use FastAPI StreamingResponse
  - Format as Server-Sent Events (SSE)
  - Each chunk: "data: {json}\n\n"
  - Final: "data: [DONE]\n\n"
- If non-streaming:
  - Collect full response
  - Return as JSON
- Add response headers:
  - X-Provider: extracted from model name
  - X-Request-ID: correlation ID (generate UUID)
  - X-Cache-Status: MISS (for now, will add caching Week 2)
- Handle errors:
  - RateLimitError → 429 with Retry-After header
  - AuthError → 401 with error message
  - TimeoutError → 504 Gateway Timeout
  - InvalidRequestError → 400 Bad Request
  - ProviderUnavailableError → 502 Bad Gateway
  - Other ProviderError → 502
  - ValidationError → 422 Unprocessable Entity
- Add OpenTelemetry span:
  - Span name: "gateway.completions"
  - Attributes: model, stream, user_id, provider
  - Record duration
- Log request and response:
  - Request start: model, stream, user_id
  - Request end: status, duration, tokens
  - Errors: error type, message

Example SSE output format:
```
data: {"content":"Hello","finish_reason":null,"usage":null}

data: {"content":" world","finish_reason":null,"usage":null}

data: {"content":"","finish_reason":"stop","usage":{"input_tokens":10,"output_tokens":5}}

data: [DONE]

```

2. Update src/llmgateway/main.py:

- Initialize LLMGatewayProvider as application state
- Add dependency function get_provider() that returns provider from app.state
- Include completions router:
  ```python
  from src.llmgateway.api import completions
  app.include_router(completions.router)
  ```
- Add startup event to log "LLM Gateway ready"

3. Add to src/llmgateway/config.py:

Settings for LiteLLM:
- OPENAI_API_KEY (from environment)
- ANTHROPIC_API_KEY (from environment)
- TOGETHER_API_KEY (optional)
- GROQ_API_KEY (optional)
- LLM_TIMEOUT (default: 60 seconds)
- LLM_MAX_RETRIES (default: 3)

REQUIREMENTS:
- OpenAI API compatible endpoint format
- Proper SSE formatting for streaming
- All errors mapped to HTTP status codes
- Comprehensive logging with correlation IDs
- OpenTelemetry instrumentation
- Type hints and docstrings
- Handle edge cases (empty responses, connection errors)

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
- Need end-to-end tests from HTTP request to provider response
- Use TestClient from FastAPI for integration tests

CREATE:

tests/integration/test_gateway_integration.py:

Mark all tests with @pytest.mark.integration (skip in CI by default)

Fixtures:
- test_client: FastAPI TestClient
- mock_env: Set up environment variables (API keys)

Test Classes:

1. TestGatewayEndToEnd:

   tests/integration/test_gateway_e2e.py:

   - test_openai_streaming_request:
     - Send streaming request for gpt-4o-mini
     - Verify SSE format
     - Verify chunks received incrementally
     - Verify [DONE] message at end
     - Check response headers (X-Provider, X-Request-ID)

   - test_anthropic_non_streaming_request:
     - Send non-streaming request for claude-3-5-haiku
     - Verify JSON response format
     - Verify usage tokens present
     - Check response time < 5 seconds

   - test_invalid_model_returns_400:
     - Send request with invalid model name
     - Verify 400 status code
     - Verify error message in response

   - test_missing_messages_returns_422:
     - Send request without messages field
     - Verify 422 validation error

   - test_invalid_temperature_returns_422:
     - Send request with temperature = 5 (invalid)
     - Verify 422 validation error

2. TestProviderFallback (optional, for Week 4):

   - test_primary_provider_down_fallback_works:
     - Simulate primary provider unavailable
     - Verify fallback to secondary provider
     - Check logs show fallback attempt

3. TestObservability:

   - test_jaeger_trace_created:
     - Make request
     - Query Jaeger API for trace
     - Verify spans present: gateway.completions, litellm.generate

   - test_prometheus_metrics_incremented:
     - Get current metric count
     - Make request
     - Verify llm_requests_total incremented
     - Verify llm_request_duration_seconds recorded

REQUIREMENTS:
- Use real API calls (mark with @pytest.mark.integration)
- Test both OpenAI and Anthropic
- Test streaming and non-streaming
- Verify response format (SSE for streaming, JSON for non-streaming)
- Check observability (traces, metrics)
- Reasonable timeouts (don't hang indefinitely)

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
- ✅ Exact match caching (hash-based, Redis)
- ✅ Semantic caching (embedding similarity, Redis)
- ✅ Token bucket rate limiting (distributed, Redis Lua)
- ✅ Cost tracking per request
- ✅ Cache hit rate >30%

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

### Day 1 Afternoon: Exact Match Cache

**Claude Code Prompt:**
```
Implement exact match caching using Redis.

CONTEXT:
- Cache responses to reduce API calls and costs
- Use Redis for fast lookups
- Only cache deterministic requests (temperature=0)

CREATE:

1. src/llmgateway/cache/base.py:

Dataclass: CacheEntry
- key: str
- value: str (JSON-serialized CompletionChunk)
- created_at: datetime
- ttl: int (seconds)
- metadata: dict (model, tokens, cost)

Protocol: CacheBackend
- async def get(key: str) -> CacheEntry | None
- async def set(key: str, value: CacheEntry, ttl: int) -> None
- async def delete(key: str) -> None
- async def exists(key: str) -> bool

2. src/llmgateway/cache/redis_cache.py:

Class: RedisCache(CacheBackend)
- Constructor: redis_url from config
- Use aioredis for async Redis client
- Implement all methods from protocol
- Serialize CacheEntry as JSON before storing
- Add OpenTelemetry spans for cache operations
- Log cache hits/misses

3. src/llmgateway/cache/cache_manager.py:

Class: CacheManager
- Constructor: cache_backend, enable_semantic=False
- Method: generate_cache_key(request: CompletionRequest) -> str
  - Hash: SHA256(model + messages + temp + max_tokens)
  - Only for temperature=0 (deterministic)
  - Return None if not cacheable

- Method: async get_cached_response(request) -> CompletionChunk | None
  - Generate cache key
  - Check exact match cache
  - If hit: log, update metrics, return
  - If miss: return None

- Method: async cache_response(request, response) -> None
  - Generate cache key
  - Store with TTL (default 1 hour)
  - Update metrics

4. Add Prometheus metrics:
- llm_cache_hits_total (counter)
- llm_cache_misses_total (counter)
- llm_cache_lookup_duration_seconds (histogram)

5. Update src/llmgateway/api/completions.py:
- Inject CacheManager as dependency
- Before calling provider, check cache
- After provider response, store in cache
- Add X-Cache-Status header (HIT or MISS)

REQUIREMENTS:
- Only cache temperature=0 requests
- TTL configurable via environment (default 3600s)
- Atomic operations (get-or-set pattern)
- Proper error handling (cache failures shouldn't break requests)
- Comprehensive logging

Generate these files.
```

**Your Tasks:**
- [ ] Add redis[hiredis] to dependencies (should already be there)
- [ ] Test cache hit/miss behavior
- [ ] Verify X-Cache-Status header
- [ ] Check Prometheus metrics: curl localhost:8000/metrics | grep cache
- [ ] Load test: send same request 100 times, verify 1 API call

### Day 2: Semantic Caching

**Claude Code Prompt:**
```
Implement semantic caching using embedding similarity.

CONTEXT:
- Catch similar queries: "What's the weather?" ≈ "Tell me about weather"
- Use sentence-transformers for embeddings
- Store embeddings in Redis, search with cosine similarity
- Only check semantic cache if exact match misses

CREATE:

1. src/llmgateway/cache/embeddings.py:

Class: EmbeddingModel
- Use sentence-transformers library
- Model: "all-MiniLM-L6-v2" (fast, 384 dimensions)
- Method: encode(text: str) -> np.ndarray
  - Cache model instance (don't reload each time)
  - Return embedding vector
  - Add OpenTelemetry span for encoding time

Function: cosine_similarity(vec1, vec2) -> float
- Calculate similarity between embeddings
- Return score 0-1

2. Update src/llmgateway/cache/cache_manager.py:

Add semantic cache methods:
- Method: async get_semantic_match(request) -> CacheEntry | None
  - Extract user message from request
  - Generate embedding
  - Search Redis for similar embeddings (get all cached keys)
  - Calculate cosine similarity for each
  - If any score > threshold (0.95), return cached response
  - Log similarity score

- Method: async cache_with_embedding(request, response) -> None
  - Store response with exact key
  - Also store embedding in separate Redis key
  - Format: "embedding:{hash}" → base64 encoded numpy array

- Update generate_cache_key to support semantic lookups

3. Update src/llmgateway/api/completions.py:
- Check exact match first (fast path)
- If miss and semantic enabled, check semantic match
- Log which cache layer hit (exact vs semantic)
- Add X-Cache-Type header (EXACT, SEMANTIC, or MISS)

4. Add configuration:
- ENABLE_SEMANTIC_CACHE (default: true)
- SEMANTIC_CACHE_THRESHOLD (default: 0.95)
- SEMANTIC_CACHE_MAX_ENTRIES (default: 1000)

5. Add metrics:
- llm_semantic_cache_hits_total
- llm_semantic_cache_lookups_duration_seconds
- llm_embedding_generation_duration_seconds

REQUIREMENTS:
- Semantic cache is opt-in (can disable for performance)
- Handle large cache: limit to most recent N entries
- Embedding generation should be <50ms p99
- Fall back gracefully if embedding fails

Generate these files and updates.
```

**Your Tasks:**
- [ ] Add sentence-transformers to dependencies
- [ ] Test with similar queries: "hello" vs "hi there"
- [ ] Verify semantic matches when similarity >0.95
- [ ] Measure embedding latency (should be <50ms)
- [ ] Test with semantic cache disabled (should still work)

### Day 3: Rate Limiting

**Claude Code Prompt:**
```
Implement token bucket rate limiting using Redis.

CONTEXT:
- Prevent abuse and manage costs
- Use token bucket algorithm (allows bursts)
- Distributed via Redis Lua scripts
- Per-user rate limits

CREATE:

1. src/llmgateway/ratelimit/token_bucket.py:

Class: TokenBucket
- Constructor: redis_client, rate (tokens/sec), capacity (max tokens)
- Method: async consume(user_id: str, tokens: int) -> tuple[bool, float]
  - Use Redis Lua script for atomicity:
    - Get current tokens and last_refill time
    - Calculate tokens to add: (now - last_refill) * rate
    - New tokens = min(current + added, capacity)
    - If new_tokens >= requested: allow, subtract tokens
    - Else: deny, calculate retry_after
  - Return (allowed: bool, retry_after: float)
  - Add OpenTelemetry span

2. src/llmgateway/ratelimit/limiter.py:

Class: RateLimiter
- Constructor: redis_url, default limits
- Method: async check_rate_limit(user_id: str, cost: int) -> RateLimitResult
  - RateLimitResult dataclass: allowed, retry_after, remaining, reset_time
  - Check token bucket
  - Log if rate limit exceeded
  - Return result

- Method: async get_rate_limit_info(user_id: str) -> dict
  - Return current token count, capacity, refill rate
  - For debugging/admin purposes

3. Create Lua script in src/llmgateway/ratelimit/scripts/token_bucket.lua:
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

4. Update src/llmgateway/api/completions.py:
- Add rate limiting middleware
- Extract user_id from request (header or API key)
- Check rate limit before processing request
- If exceeded:
  - Return 429 Too Many Requests
  - Add Retry-After header
  - Add X-RateLimit-* headers (Limit, Remaining, Reset)
- Add rate limit info to all responses (headers)

5. Add configuration:
- RATE_LIMIT_ENABLED (default: true)
- RATE_LIMIT_DEFAULT_RATE (default: 10/minute)
- RATE_LIMIT_DEFAULT_CAPACITY (default: 20)

6. Add metrics:
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

### Day 4: Cost Tracking

**Claude Code Prompt:**
```
Implement real-time cost tracking per request.

CONTEXT:
- Track costs to prevent budget overruns
- Store usage in PostgreSQL for analytics
- Real-time cost calculation using provider pricing

CREATE:

1. src/llmgateway/cost/pricing.py:

PRICING_TABLE = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},  # per 1K tokens
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
}

Function: calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float
- Lookup pricing for model
- Calculate: (input_tokens/1000 * input_price) + (output_tokens/1000 * output_price)
- Return cost in USD

2. src/llmgateway/cost/tracker.py:

Class: CostTracker
- Constructor: db_session (SQLAlchemy async session)
- Method: async record_usage(user_id, model, input_tokens, output_tokens, cost, cached)
  - Insert into usage_records table
  - Fields: timestamp, user_id, model, input_tokens, output_tokens, cost, cached
  - Add OpenTelemetry span

- Method: async get_user_cost(user_id, start_date, end_date) -> dict
  - Query total cost for user in date range
  - Group by model
  - Return breakdown

- Method: async get_daily_cost() -> float
  - Query total cost for today
  - For monitoring/alerts

3. Database migration:
```python
# migrations/versions/001_add_usage_tracking.py
def upgrade():
    op.create_table(
        'usage_records',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('timestamp', sa.DateTime, nullable=False, index=True),
        sa.Column('user_id', sa.String(255), nullable=False, index=True),
        sa.Column('model', sa.String(100), nullable=False),
        sa.Column('input_tokens', sa.Integer, nullable=False),
        sa.Column('output_tokens', sa.Integer, nullable=False),
        sa.Column('cost_usd', sa.Numeric(10, 6), nullable=False),
        sa.Column('cached', sa.Boolean, default=False),
        sa.Column('cache_type', sa.String(20)),  # EXACT, SEMANTIC, null
    )
    op.create_index('ix_usage_records_user_timestamp', 'usage_records', ['user_id', 'timestamp'])
```

4. Update src/llmgateway/api/completions.py:
- After response, calculate cost
- Record usage in database
- Add X-Cost header with USD amount
- Update Prometheus gauge: llm_cost_usd_total

5. Create admin endpoint:
GET /admin/costs/summary:
- Query parameters: user_id, start_date, end_date
- Return: total_cost, cost_by_model, request_count
- Require admin authentication (simple for now)

6. Add budget alerts:
- Check daily cost after each request
- If > threshold (env: DAILY_COST_ALERT_THRESHOLD), log warning
- Future: webhook to Slack/email

REQUIREMENTS:
- Accurate cost calculation using current pricing
- Handle cached requests (cost = 0 for exact match)
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
- ✅ Exact match caching (5ms latency, 100% cost savings)
- ✅ Semantic caching (50ms latency, 80% cost savings)
- ✅ Token bucket rate limiting (distributed, per-user)
- ✅ Real-time cost tracking with PostgreSQL storage
- ✅ Admin API for cost reporting

**Metrics:**
- ✅ Cache hit rate: 30-40% (varies by traffic)
- ✅ Rate limiting: <10% requests blocked (tunable)
- ✅ Cost tracking: 100% accuracy vs provider bills

**Interview Talking Points:**
- "Implemented two-layer caching achieving 40% cost savings"
- "Token bucket rate limiting using Redis Lua scripts for atomicity"
- "Real-time cost tracking with PostgreSQL for analytics"
- "Trade-off analysis: semantic cache adds 50ms but catches 15% more queries"

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

1. src/llmgateway/observability/metrics.py:

Define all Prometheus metrics:
- Counter: llm_requests_total{model, status, cache_status, provider}
- Histogram: llm_request_duration_seconds{model, provider}
- Histogram: llm_token_count{model, type=input|output}
- Counter: llm_cost_usd_total{model, user_id}
- Gauge: llm_cache_hit_rate (calculated from hits/total)
- Counter: llm_cache_hits_total{cache_type=exact|semantic}
- Counter: llm_rate_limit_exceeded_total{user_id}
- Histogram: llm_provider_api_duration_seconds{provider}
- Histogram: llm_cache_lookup_duration_seconds{cache_type}
- Gauge: llm_active_requests (in-flight requests)

Class: MetricsCollector
- Singleton pattern
- Methods to record each metric
- Automatic label extraction from request context
- Thread-safe increment/observe methods

2. Update all components to use MetricsCollector:
- completions.py: record request metrics
- cache_manager.py: record cache metrics
- rate_limiter.py: record rate limit metrics
- providers: record API call duration

3. Create Grafana dashboard JSON:

.devcontainer/grafana/dashboards/llm-gateway.json:
- Row 1: Request Rate
  - Panel: Requests per second (by model)
  - Panel: Requests per second (by status code)
- Row 2: Latency
  - Panel: P50, P95, P99 latency (overall)
  - Panel: P99 latency by model
  - Panel: Provider API latency vs total latency
- Row 3: Cache Performance
  - Panel: Cache hit rate (gauge)
  - Panel: Cache lookups per second (by type)
  - Panel: Cache lookup duration
- Row 4: Cost Tracking
  - Panel: Cost per hour (by model)
  - Panel: Cost per 1K requests
  - Panel: Total daily cost
- Row 5: Rate Limiting
  - Panel: Rate limit exceeded per minute
  - Panel: Top users by request count
- Row 6: Errors
  - Panel: Error rate percentage
  - Panel: Errors by type (RateLimitError, TimeoutError, etc.)

4. Update .devcontainer/prometheus.yml:
- Add recording rules for common queries
- Add alerting rules (optional, for learning)

5. Create src/llmgateway/observability/tracing.py:

Add custom spans to key operations:
- Span: llm_gateway.request (root span)
  - Attributes: model, user_id, cached
  - Child span: cache.lookup
  - Child span: rate_limit.check
  - Child span: provider.generate
    - Child span: provider.api_call
  - Child span: cost.calculate
  - Child span: cost.record

Update all components to add spans with rich attributes.

REQUIREMENTS:
- All metrics have help text
- Dashboard is visually clear
- Alerts for: error rate >5%, p99 latency >2s, daily cost >$50
- Tracing shows complete request flow

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

**"Cache not working"**
```bash
# Check Redis
redis-cli -h redis
> KEYS *  # See cached keys
> GET "cache:abc123"  # Check cache content

# Check logs
make dev  # Watch for cache hit/miss logs

# Verify temperature=0
# Only deterministic requests are cached
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
