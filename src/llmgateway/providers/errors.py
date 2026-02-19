"""Custom exception hierarchy for LLM Gateway provider errors.

All provider-level failures are mapped to one of these typed exceptions so
callers can handle them without inspecting raw LiteLLM or HTTP internals.
"""


class ProviderError(Exception):
    """Base exception for all LLM provider errors.

    Attributes:
        message: Human-readable error description.
        provider: Provider name (e.g. "openai", "anthropic").  ``None`` when
            the provider could not be determined.
        original_error: The upstream exception that caused this error, if any.
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        self.message = message
        self.provider = provider
        self.original_error = original_error
        super().__init__(message)


class RateLimitError(ProviderError):
    """Raised when the provider returns HTTP 429 (rate limit exceeded).

    Attributes:
        retry_after: Seconds to wait before retrying, when the provider
            supplies a ``Retry-After`` header.  ``None`` if unavailable.
    """

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        provider: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, provider=provider, original_error=original_error)
        self.retry_after = retry_after


class AuthError(ProviderError):
    """Raised for authentication or authorisation failures (HTTP 401 / 403)."""


class TimeoutError(ProviderError):  # noqa: A001 – intentionally shadows the built-in
    """Raised when a provider request exceeds the configured timeout."""


class InvalidRequestError(ProviderError):
    """Raised for requests rejected as malformed or unsupported (HTTP 400 / 422).

    Also raised for context-window exceeded errors so callers can catch a
    single type for all non-retryable request faults.
    """


class ProviderUnavailableError(ProviderError):
    """Raised when the provider is down or unreachable (HTTP 5xx / network error)."""
