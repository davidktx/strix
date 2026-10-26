"""Tests for provider-reported LLM cost capture."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from strix.core.hooks import ReportUsageHooks
from strix.llm import request_log
from strix.report.state import (
    ReportState,
    litellm_cost_callback,
    openrouter_stream_cost,
    record_llm_request,
    set_global_report_state,
    streamed_openrouter_costs,
)
from strix.report.usage import LLMUsageLedger


@pytest.fixture(autouse=True)
def _clear_streamed_costs() -> None:
    streamed_openrouter_costs.clear()


def test_cost_callback_reads_openrouter_stream_usage_cost() -> None:
    report_state = MagicMock()
    response = SimpleNamespace(
        usage=SimpleNamespace(cost=1.2345),
        _hidden_params={},
    )

    with patch("strix.report.state.get_global_report_state", return_value=report_state):
        litellm_cost_callback({"response_cost": None}, response)

    report_state.record_observed_llm_cost.assert_called_once_with(1.2345)


def test_cost_callback_reads_usage_cost_from_mapping_response() -> None:
    report_state = MagicMock()
    response = {"usage": {"cost": 0.125}}

    with patch("strix.report.state.get_global_report_state", return_value=report_state):
        litellm_cost_callback({}, response)

    report_state.record_observed_llm_cost.assert_called_once_with(0.125)


def test_cost_callback_reads_byok_upstream_inference_cost() -> None:
    report_state = MagicMock()
    response = SimpleNamespace(
        usage=SimpleNamespace(
            cost=0,
            is_byok=True,
            cost_details=SimpleNamespace(upstream_inference_cost=6.75e-06),
        ),
        _hidden_params={},
    )

    with patch("strix.report.state.get_global_report_state", return_value=report_state):
        litellm_cost_callback({"response_cost": None}, response)

    report_state.record_observed_llm_cost.assert_called_once_with(6.75e-06)


def test_cost_callback_sums_usage_cost_and_upstream_inference_cost() -> None:
    report_state = MagicMock()
    response = {
        "usage": {
            "cost": 0.01,
            "is_byok": True,
            "cost_details": {"upstream_inference_cost": 0.2},
        }
    }

    with patch("strix.report.state.get_global_report_state", return_value=report_state):
        litellm_cost_callback({}, response)

    report_state.record_observed_llm_cost.assert_called_once_with(pytest.approx(0.21))


def test_cost_callback_ignores_upstream_cost_for_non_byok_responses() -> None:
    report_state = MagicMock()
    response = {
        "usage": {
            "cost": 0.05,
            "is_byok": False,
            "cost_details": {"upstream_inference_cost": 0.04},
        }
    }

    with patch("strix.report.state.get_global_report_state", return_value=report_state):
        litellm_cost_callback({}, response)

    report_state.record_observed_llm_cost.assert_called_once_with(0.05)


def test_cost_callback_estimates_cost_with_provider_prefixed_model() -> None:
    report_state = MagicMock()
    response = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    kwargs = {
        "response_cost": None,
        "model": "anthropic/claude-sonnet-4.5",
        "litellm_params": {"custom_llm_provider": "openrouter"},
    }

    with (
        patch("strix.report.state.get_global_report_state", return_value=report_state),
        patch("strix.report.state.resolve_litellm_model", side_effect=lambda model: model),
        patch("strix.report.state.estimate_usage_cost", return_value=0.5),
    ):
        litellm_cost_callback(kwargs, response)

    report_state.record_observed_llm_cost.assert_called_once_with(0.5)


def test_cost_callback_estimates_cost_with_bare_model_fallback() -> None:
    report_state = MagicMock()
    response = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    kwargs = {
        "response_cost": None,
        "model": "openai/gpt-4o-mini",
        "litellm_params": {"custom_llm_provider": "openrouter"},
    }

    with (
        patch("strix.report.state.get_global_report_state", return_value=report_state),
        patch("strix.report.state.resolve_litellm_model", return_value="openai/gpt-4o-mini"),
        patch("strix.report.state.estimate_usage_cost", return_value=0.025),
    ):
        litellm_cost_callback(kwargs, response)

    report_state.record_observed_llm_cost.assert_called_once_with(0.025)


def test_cost_callback_records_nothing_when_no_cost_available() -> None:
    report_state = MagicMock()
    response = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    with (
        patch("strix.report.state.get_global_report_state", return_value=report_state),
        patch("strix.report.state.estimate_usage_cost", side_effect=ValueError("unknown model")),
    ):
        litellm_cost_callback({"response_cost": None, "model": "x/y"}, response)

    report_state.record_observed_llm_cost.assert_not_called()


def test_openrouter_stream_cost_extracts_plain_and_byok_totals() -> None:
    assert openrouter_stream_cost({"cost": 0.003168}) == pytest.approx(0.003168)
    assert openrouter_stream_cost(
        {"cost": 0.01, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.2}}
    ) == pytest.approx(0.21)
    assert openrouter_stream_cost(
        {"cost": 0.05, "is_byok": False, "cost_details": {"upstream_inference_cost": 0.04}}
    ) == pytest.approx(0.05)
    assert openrouter_stream_cost({"prompt_tokens": 10}) is None
    assert openrouter_stream_cost(None) is None


def test_cost_callback_recovers_streamed_openrouter_cost_by_response_id() -> None:
    report_state = MagicMock()
    streamed_openrouter_costs.remember("gen-abc", {"cost": 0.42})
    response = SimpleNamespace(id="gen-abc", usage=SimpleNamespace(cost=None), _hidden_params={})

    with (
        patch("strix.report.state.get_global_report_state", return_value=report_state),
        patch("strix.report.state.estimate_usage_cost", side_effect=ValueError("unknown model")),
    ):
        litellm_cost_callback({"response_cost": None, "model": "moonshotai/kimi-k3"}, response)

    report_state.record_observed_llm_cost.assert_called_once_with(0.42)
    assert streamed_openrouter_costs.take(response) is None


def test_streamed_openrouter_cost_prefers_provider_report_over_estimate() -> None:
    report_state = MagicMock()
    streamed_openrouter_costs.remember("gen-xyz", {"cost": 0.9})
    response = SimpleNamespace(
        id="gen-xyz",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        _hidden_params={},
    )

    with (
        patch("strix.report.state.get_global_report_state", return_value=report_state),
        patch("strix.report.state.estimate_usage_cost", return_value=0.1) as estimate,
    ):
        litellm_cost_callback({"response_cost": None, "model": "moonshotai/kimi-k3"}, response)

    report_state.record_observed_llm_cost.assert_called_once_with(0.9)
    estimate.assert_not_called()


def test_streamed_openrouter_costs_ignores_entries_without_cost() -> None:
    streamed_openrouter_costs.remember("gen-none", {"prompt_tokens": 10})
    streamed_openrouter_costs.remember("", {"cost": 0.5})
    assert streamed_openrouter_costs.take(SimpleNamespace(id="gen-none")) is None


def test_streamed_openrouter_costs_cleared_on_new_run() -> None:
    streamed_openrouter_costs.remember("gen-stale", {"cost": 0.7})
    try:
        set_global_report_state(ReportState.__new__(ReportState))
        assert streamed_openrouter_costs.take(SimpleNamespace(id="gen-stale")) is None
    finally:
        set_global_report_state(None)


def test_provider_tally_survives_run_record_round_trip() -> None:
    ledger = LLMUsageLedger()
    for input_tokens, cached_tokens, cost in [(1000, 900, 0.002), (500, 0, 0.001)]:
        ledger.record_provider(
            "Together",
            agent_id=None,
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
            cache_block_tokens=128,
        )

    restored = LLMUsageLedger()
    restored.hydrate(ledger.to_record())

    assert restored.to_record()["providers"] == {
        "Together": {
            "requests": 2,
            "input_tokens": 1500,
            "cached_tokens": 900,
            "cost": 0.003,
            "cache_misses": 0,
            "missed_tokens": 0,
        }
    }


def test_provider_tally_counts_cache_misses_per_agent() -> None:
    ledger = LLMUsageLedger()
    calls = [
        ("Z.AI", "a1", 1000, 0),
        ("Z.AI", "a1", 1200, 960),
        ("DeepInfra", "a1", 1500, 200),
        ("Z.AI", "a2", 800, 0),
        ("Z.AI", "a1", 600, 0),
    ]
    for provider, agent_id, input_tokens, cached_tokens in calls:
        ledger.record_provider(
            provider,
            agent_id=agent_id,
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            cost=0.0,
            cache_block_tokens=128,
        )

    providers = ledger.to_record()["providers"]
    assert providers["DeepInfra"]["cache_misses"] == 1
    assert providers["DeepInfra"]["missed_tokens"] == 1000
    assert providers["Z.AI"]["cache_misses"] == 0


def test_request_timings_land_in_run_record() -> None:
    started = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
    event = request_log.LlmRequestEvent(
        call_id="call-1",
        route="openai",
        provider="openrouter",
        model="openrouter/deepseek/deepseek-v4.1-flash",
        api_host="openrouter.ai",
        streaming=True,
        outcome="success",
        status_code=200,
        provider_request_id=None,
        response_id="gen-1",
        error_type=None,
        error_message=None,
        started_at=started,
        finished_at=started + timedelta(milliseconds=4200),
        duration_ms=4200,
        input_tokens=90_000,
        output_tokens=600,
        cached_input_tokens=85_000,
        agent_id="agent-1",
        time_to_first_token_ms=900,
        details={"upstream_provider": "Together"},
    )
    state = ReportState.__new__(ReportState)
    state.run_record = {}
    try:
        set_global_report_state(state)
        record_llm_request(event)
    finally:
        set_global_report_state(None)

    assert state.run_record["llm_requests"] == [
        {
            "started_at": "2026-10-19T12:00:00+00:00",
            "agent_id": "agent-1",
            "model": "openrouter/deepseek/deepseek-v4.1-flash",
            "provider": "Together",
            "outcome": "success",
            "status": 200,
            "duration_ms": 4200,
            "ttft_ms": 900,
            "input_tokens": 90_000,
            "cached_tokens": 85_000,
            "output_tokens": 600,
            "retry_attempt": 0,
        }
    ]


async def test_tool_call_timings_land_in_run_record() -> None:
    hooks = ReportUsageHooks(model="m", record_tool_timings=True)
    context = SimpleNamespace(tool_call_id="call-1", context={"agent_id": "agent-1"})
    tool = SimpleNamespace(name="terminal_execute")
    state = ReportState.__new__(ReportState)
    state.run_record = {}
    try:
        set_global_report_state(state)
        await hooks.on_tool_start(context, MagicMock(), tool)  # type: ignore[arg-type]
        await hooks.on_tool_end(context, MagicMock(), tool, "ok")  # type: ignore[arg-type]
    finally:
        set_global_report_state(None)

    [entry] = state.run_record["tool_calls"]
    assert entry["agent_id"] == "agent-1"
    assert entry["tool"] == "terminal_execute"
    assert entry["duration_ms"] >= 0
