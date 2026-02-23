"""Tests for cost/pricing.py — PRICING_TABLE and calculate_cost()."""

from __future__ import annotations

import pytest

from llmgateway.cost.pricing import _UNKNOWN_RATE, PRICING_TABLE, calculate_cost

# ---------------------------------------------------------------------------
# calculate_cost — exact matches
# ---------------------------------------------------------------------------


class TestCalculateCostExactMatch:
    def test_gpt4o_basic(self) -> None:
        # 1 000 input + 1 000 output → (1*0.005) + (1*0.015) = 0.02
        cost = calculate_cost("gpt-4o", 1000, 1000)
        assert cost == pytest.approx(0.02, rel=1e-6)

    def test_gpt4o_mini(self) -> None:
        cost = calculate_cost("gpt-4o-mini", 1000, 1000)
        in_rate, out_rate = PRICING_TABLE["gpt-4o-mini"]
        expected = (1000 / 1000) * in_rate + (1000 / 1000) * out_rate
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_claude_haiku(self) -> None:
        cost = calculate_cost("claude-haiku-4-5-20251001", 500, 200)
        in_rate, out_rate = PRICING_TABLE["claude-haiku-4-5-20251001"]
        expected = (500 / 1000) * in_rate + (200 / 1000) * out_rate
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_groq_llama(self) -> None:
        cost = calculate_cost("groq/llama3-8b-8192", 2000, 500)
        in_rate, out_rate = PRICING_TABLE["groq/llama3-8b-8192"]
        expected = (2000 / 1000) * in_rate + (500 / 1000) * out_rate
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_all_table_entries_return_positive_cost(self) -> None:
        for model in PRICING_TABLE:
            cost = calculate_cost(model, 1000, 1000)
            assert cost > 0, f"{model} returned non-positive cost"

    def test_zero_tokens_returns_zero(self) -> None:
        assert calculate_cost("gpt-4o", 0, 0) == 0.0

    def test_only_input_tokens(self) -> None:
        in_rate, _ = PRICING_TABLE["gpt-4o"]
        cost = calculate_cost("gpt-4o", 1000, 0)
        assert cost == pytest.approx(in_rate, rel=1e-6)

    def test_only_output_tokens(self) -> None:
        _, out_rate = PRICING_TABLE["gpt-4o"]
        cost = calculate_cost("gpt-4o", 0, 1000)
        assert cost == pytest.approx(out_rate, rel=1e-6)

    def test_result_rounded_to_8_decimals(self) -> None:
        cost = calculate_cost("gpt-4o", 1, 1)
        # Verify result has at most 8 decimal places
        assert cost == round(cost, 8)


# ---------------------------------------------------------------------------
# calculate_cost — prefix fallback
# ---------------------------------------------------------------------------


class TestCalculateCostPrefixFallback:
    def test_unknown_gpt4_variant_uses_gpt4_prefix(self) -> None:
        cost_known = calculate_cost("gpt-4", 1000, 1000)
        cost_variant = calculate_cost("gpt-4-ultra-fictional", 1000, 1000)
        assert cost_variant == cost_known

    def test_unknown_claude_variant_uses_claude_prefix(self) -> None:
        cost = calculate_cost("claude-future-model-9000", 1000, 1000)
        assert cost > 0

    def test_groq_unknown_model_uses_groq_prefix(self) -> None:
        cost = calculate_cost("groq/unknown-model-xyz", 1000, 1000)
        assert cost > 0

    def test_mistral_unknown_model_uses_mistral_prefix(self) -> None:
        cost = calculate_cost("mistral/new-model", 1000, 1000)
        assert cost > 0

    def test_together_ai_unknown_uses_prefix(self) -> None:
        cost = calculate_cost("together_ai/some/new/model", 1000, 1000)
        assert cost > 0

    def test_completely_unknown_model_uses_default_rate(self) -> None:
        cost = calculate_cost("totally-unknown-provider/model-x", 1000, 1000)
        in_rate, out_rate = _UNKNOWN_RATE
        expected = (1000 / 1000) * in_rate + (1000 / 1000) * out_rate
        assert cost == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# PRICING_TABLE structure
# ---------------------------------------------------------------------------


class TestPricingTableStructure:
    def test_all_entries_are_two_tuples(self) -> None:
        for model, rates in PRICING_TABLE.items():
            assert len(rates) == 2, f"{model}: expected 2-tuple"

    def test_all_rates_are_positive(self) -> None:
        for model, (in_rate, out_rate) in PRICING_TABLE.items():
            assert in_rate > 0, f"{model}: input rate must be positive"
            assert out_rate > 0, f"{model}: output rate must be positive"

    def test_output_rate_gte_input_rate_for_most_models(self) -> None:
        # Output tokens are typically more expensive than input tokens.
        # A handful of models (e.g. Together AI) have equal rates.
        for model, (in_rate, out_rate) in PRICING_TABLE.items():
            assert (
                out_rate >= in_rate or in_rate == out_rate
            ), f"{model}: unexpected in_rate ({in_rate}) > out_rate ({out_rate})"
