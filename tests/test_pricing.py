from __future__ import annotations

from unittest.mock import patch

from agents.usage import Usage

from strix.config.model_catalog import catalog_entries, estimate_usage_cost, resolve_pricing_model
from strix.report.pricing import resolve_litellm_model
from strix.report.usage import LLMUsageLedger


def test_resolves_common_bare_model_names() -> None:
    resolve_pricing_model.cache_clear()
    assert resolve_litellm_model("deepseek-v4-flash") == "deepseek/deepseek-v4-flash"
    assert resolve_litellm_model("openai/deepseek-v4-flash") == "deepseek/deepseek-v4-flash"
    assert resolve_litellm_model("deepseek/deepseek-v4-flash") == "deepseek/deepseek-v4-flash"
    assert resolve_litellm_model("grok-4.5") == "github_copilot/grok-4.5"


def test_resolver_returns_none_for_unresolvable_model() -> None:
    resolve_pricing_model.cache_clear()
    assert resolve_litellm_model("provider/not-a-real-model") is None


def test_ledger_uses_estimate_when_routed_provider_reports_no_cost() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()

    with patch("strix.report.usage.estimate_usage_cost", return_value=0.42):
        ledger.record(agent_id="a", usage=usage, model="openai/deepseek-v4-flash")

    assert ledger.total_cost == 0.42


def test_ledger_prefers_observed_cost_over_estimate() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()

    with patch("strix.report.usage.estimate_usage_cost", return_value=0.42):
        ledger.record(agent_id="a", usage=usage, model="openai/deepseek-v4-flash")
    ledger.record_observed_cost(0.17)

    assert ledger.total_cost == 0.17


def test_hydrated_estimate_continues_accumulating_new_estimates() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()
    ledger.hydrate({"cost": 0.42})

    with patch("strix.report.usage.estimate_usage_cost", return_value=0.17):
        ledger.record(agent_id="a", usage=usage, model="openai/deepseek-v4-flash")

    assert ledger.total_cost == 0.59


def test_zero_cost_disables_both_observed_and_estimated_costs() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()
    ledger.zero_cost = True

    with patch("strix.report.usage.estimate_usage_cost", return_value=0.42) as estimate:
        ledger.record(agent_id="a", usage=usage, model="deepseek-v4-flash")
        ledger.record_observed_cost(1.0)

    estimate.assert_not_called()
    assert ledger.total_cost == 0.0


def test_resolver_uses_provider_when_bare_entry_has_one() -> None:
    resolve_pricing_model.cache_clear()
    assert resolve_litellm_model("example") == "example-provider/example"


def test_resolver_does_not_guess_between_differently_priced_providers() -> None:
    entries = catalog_entries()
    saved = entries.pop("example", None)
    try:
        resolve_pricing_model.cache_clear()
        assert resolve_litellm_model("example") is None
    finally:
        if saved is not None:
            entries["example"] = saved
        resolve_pricing_model.cache_clear()


def test_estimate_usage_cost_uses_catalog_rates() -> None:
    cost = estimate_usage_cost(model="deepseek/deepseek-v4-flash", prompt_tokens=1000, completion_tokens=200)
    assert cost is not None and cost > 0
