"""Request and response dataclasses for the LLM Gateway provider layer.

These types form the public contract between gateway business logic and the
underlying LiteLLM wrapper.  All fields are immutable (``frozen=True``) and
validated at construction time so callers get a fast, explicit error rather
than a cryptic downstream failure.
"""

from dataclasses import dataclass, field

from llmgateway.providers.errors import InvalidRequestError

_VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool", "function"})


@dataclass(frozen=True)
class CompletionRequest:
    """Parameters for a single completion call.

    Args:
        model: LiteLLM model string, e.g. ``"gpt-4o"``, ``"claude-3-5-sonnet-20241022"``,
            or a provider-prefixed form such as ``"groq/llama-3.1-70b-versatile"``.
        messages: Conversation history.  Each dict must contain ``"role"`` and
            ``"content"`` keys.  Role must be one of: system, user, assistant,
            tool, function.
        temperature: Sampling temperature in ``[0.0, 2.0]``.  Defaults to ``0.7``.
        max_tokens: Maximum tokens to generate.  ``None`` defers to the provider
            default.
        stream: When ``True``, :meth:`LLMGatewayProvider.generate` yields tokens
            as they arrive.  When ``False``, a single chunk with the full response
            is yielded.
        user_id: Opaque caller identifier forwarded to the provider as the
            ``user`` field.  Useful for per-user rate-limit attribution.

    Raises:
        InvalidRequestError: If any field fails validation.
    """

    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    user_id: str | None = None

    # Sentinel to detect that __post_init__ has run; field excluded from repr.
    _validated: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise InvalidRequestError("model must be a non-empty string")

        if not self.messages:
            raise InvalidRequestError("messages must not be empty")

        for i, msg in enumerate(self.messages):
            if "role" not in msg or "content" not in msg:
                raise InvalidRequestError(
                    f"messages[{i}] must contain both 'role' and 'content' keys"
                )
            if msg["role"] not in _VALID_ROLES:
                raise InvalidRequestError(
                    f"messages[{i}] has invalid role '{msg['role']}'; "
                    f"must be one of {sorted(_VALID_ROLES)}"
                )

        if not 0.0 <= self.temperature <= 2.0:
            raise InvalidRequestError(f"temperature must be in [0.0, 2.0], got {self.temperature}")

        if self.max_tokens is not None and self.max_tokens <= 0:
            raise InvalidRequestError(
                f"max_tokens must be a positive integer, got {self.max_tokens}"
            )

        # Mark validated without triggering FrozenInstanceError
        object.__setattr__(self, "_validated", True)


@dataclass(frozen=True)
class CompletionChunk:
    """A single unit of output from a completion call.

    For streaming requests each token (or small group of tokens) arrives as its
    own chunk.  For non-streaming requests a single chunk carries the full
    response.  The final chunk in a stream always has ``finish_reason`` set.

    Attributes:
        content: Text content for this chunk (may be empty for the final chunk).
        finish_reason: Stop reason reported by the provider (e.g. ``"stop"``,
            ``"length"``, ``"tool_calls"``).  ``None`` for intermediate chunks.
        usage: Token counts ``{"input_tokens": N, "output_tokens": M}``.
            Providers that include usage in the final streaming chunk populate
            this field there; non-streaming responses always include it.
        model: Resolved model name as reported by the provider (may differ from
            the requested name due to aliasing).
    """

    content: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    model: str | None = None


@dataclass(frozen=True)
class CompletionResponse:
    """Full response for a non-streaming completion call.

    This type is provided as a convenience for callers that consume the
    :meth:`LLMGatewayProvider.generate` iterator and want a single typed
    object rather than a list of :class:`CompletionChunk`.

    Attributes:
        content: Complete generated text.
        usage: Token counts ``{"input_tokens": N, "output_tokens": M}``.
        model: Resolved model name as reported by the provider.
        finish_reason: Stop reason (e.g. ``"stop"``, ``"length"``).
    """

    content: str
    usage: dict[str, int]
    model: str
    finish_reason: str
