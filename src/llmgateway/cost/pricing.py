"""Per-model token pricing table and cost calculation.

Prices are in USD per 1 000 tokens (input and output separately).
Models not found in the exact-match table are resolved via prefix matching;
completely unknown models fall back to a conservative default.
"""

from __future__ import annotations

# (input_cost_per_1k_tokens, output_cost_per_1k_tokens) in USD
PRICING_TABLE: dict[str, tuple[float, float]] = {
    # --- OpenAI ---
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.000150, 0.000600),
    "gpt-4-turbo": (0.010, 0.030),
    "gpt-4": (0.030, 0.060),
    "gpt-3.5-turbo": (0.000500, 0.001500),
    "o1-preview": (0.015, 0.060),
    "o1-mini": (0.003, 0.012),
    # --- Anthropic ---
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-haiku-20241022": (0.001, 0.005),
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-sonnet-20240229": (0.003, 0.015),
    "claude-3-haiku-20240307": (0.000250, 0.001250),
    "claude-haiku-4-5-20251001": (0.001, 0.005),
    "claude-sonnet-4-5": (0.003, 0.015),
    "claude-opus-4-5": (0.015, 0.075),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-opus-4-6": (0.015, 0.075),
    # --- Google ---
    "gemini/gemini-1.5-pro": (0.003500, 0.010500),
    "gemini/gemini-1.5-flash": (0.000350, 0.001050),
    "gemini/gemini-pro": (0.000500, 0.001500),
    # --- Together AI ---
    "together_ai/meta-llama/Llama-3-8b-chat-hf": (0.000200, 0.000200),
    "together_ai/meta-llama/Llama-3-70b-chat-hf": (0.000900, 0.000900),
    # --- Groq ---
    "groq/llama3-8b-8192": (0.000050, 0.000080),
    "groq/llama3-70b-8192": (0.000590, 0.000790),
    "groq/mixtral-8x7b-32768": (0.000270, 0.000270),
    # --- Mistral ---
    "mistral/mistral-small-latest": (0.001, 0.003),
    "mistral/mistral-medium-latest": (0.002700, 0.008100),
    "mistral/mistral-large-latest": (0.008, 0.024),
}

# Prefix-based fallback pricing — applied when the exact model string is unknown.
# Listed from most-specific to least-specific so the first match wins.
_PREFIX_FALLBACKS: list[tuple[str, tuple[float, float]]] = [
    ("gpt-4o-mini", (0.000150, 0.000600)),
    ("gpt-4o", (0.005, 0.015)),
    ("gpt-4-turbo", (0.010, 0.030)),
    ("gpt-4", (0.030, 0.060)),
    ("gpt-3.5", (0.000500, 0.001500)),
    ("o1-", (0.015, 0.060)),
    ("o3-", (0.060, 0.240)),
    ("claude-3-5-sonnet", (0.003, 0.015)),
    ("claude-3-5-haiku", (0.001, 0.005)),
    ("claude-3-opus", (0.015, 0.075)),
    ("claude-3-sonnet", (0.003, 0.015)),
    ("claude-3-haiku", (0.000250, 0.001250)),
    ("claude-haiku", (0.001, 0.005)),
    ("claude-sonnet", (0.003, 0.015)),
    ("claude-opus", (0.015, 0.075)),
    ("claude-", (0.003, 0.015)),
    ("gemini/gemini-1.5-pro", (0.003500, 0.010500)),
    ("gemini/gemini-1.5-flash", (0.000350, 0.001050)),
    ("gemini/", (0.000500, 0.001500)),
    ("together_ai/", (0.000200, 0.000200)),
    ("groq/", (0.000050, 0.000080)),
    ("mistral/", (0.002700, 0.008100)),
]

# Conservative default for completely unknown models.
_UNKNOWN_RATE: tuple[float, float] = (0.002, 0.002)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the estimated USD cost for *input_tokens* + *output_tokens*.

    Looks up the model in :data:`PRICING_TABLE` first.  If not found, falls
    back to prefix matching via :data:`_PREFIX_FALLBACKS`.  Unknown models use
    a conservative default rate of $0.002 / 1k tokens for both input and output.

    Args:
        model:         LiteLLM model string (e.g. ``"gpt-4o"`` or ``"groq/llama3-8b-8192"``).
        input_tokens:  Number of prompt tokens consumed.
        output_tokens: Number of completion tokens generated.

    Returns:
        Estimated cost in USD, rounded to 8 decimal places.
    """
    if model in PRICING_TABLE:
        input_rate, output_rate = PRICING_TABLE[model]
    else:
        input_rate, output_rate = _UNKNOWN_RATE
        for prefix, rates in _PREFIX_FALLBACKS:
            if model.startswith(prefix):
                input_rate, output_rate = rates
                break

    cost = (input_tokens / 1000.0) * input_rate + (output_tokens / 1000.0) * output_rate
    return round(cost, 8)
