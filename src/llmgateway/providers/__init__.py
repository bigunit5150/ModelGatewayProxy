"""LLM provider abstraction layer.

Public surface area for the providers package.  Import from here rather than
from the individual submodules so internal structure can change freely.

Example::

    from llmgateway.providers import (
        CompletionRequest,
        CompletionChunk,
        LLMGatewayProvider,
        RateLimitError,
        AuthError,
    )

    provider = LLMGatewayProvider()
    request = CompletionRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello"}],
    )
    async for chunk in provider.generate(request):
        print(chunk.content, end="")
"""

from llmgateway.providers.errors import (
    AuthError,
    InvalidRequestError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
)
from llmgateway.providers.litellm_wrapper import LLMGatewayProvider
from llmgateway.providers.models import (
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
)

__all__ = [
    # Models
    "CompletionRequest",
    "CompletionChunk",
    "CompletionResponse",
    # Provider
    "LLMGatewayProvider",
    # Errors
    "ProviderError",
    "RateLimitError",
    "AuthError",
    "TimeoutError",
    "InvalidRequestError",
    "ProviderUnavailableError",
]
