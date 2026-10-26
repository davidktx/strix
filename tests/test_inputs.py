"""Tests for pure input builders in strix.core.inputs."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import pytest

from strix.core.inputs import (
    build_root_task,
    build_scan_targets,
    build_scope_context,
    child_initial_input,
    make_model_settings,
)


def _child_kwargs(parent_history: list[Any]) -> dict[str, Any]:
    return {
        "name": "scout",
        "child_id": "agent-2",
        "parent_id": "agent-1",
        "task": "Audit the login flow.",
        "parent_history": parent_history,
    }


def test_child_initial_input_single_message_without_history() -> None:
    result = child_initial_input(**_child_kwargs([]))

    assert len(result) == 1
    assert result[0]["role"] == "user"
    content = result[0]["content"]
    assert "agent scout (agent-2)" in content
    assert "Audit the login flow." in content
    assert "Inherited context" not in content


def test_child_initial_input_single_message_with_history() -> None:
    history = [{"role": "assistant", "content": "previous work"}]
    result = child_initial_input(**_child_kwargs(history))

    assert len(result) == 1
    assert result[0]["role"] == "user"
    content = result[0]["content"]
    assert "Inherited context from parent" in content
    assert "previous work" in content
    assert "agent scout (agent-2)" in content
    assert "Audit the login flow." in content


@pytest.mark.parametrize(
    "parent_history",
    [[], [{"role": "assistant", "content": "previous work"}]],
)
def test_child_initial_input_no_consecutive_same_role(parent_history: list[Any]) -> None:
    result = child_initial_input(**_child_kwargs(parent_history))

    roles = [msg["role"] for msg in result]
    assert all(prev != nxt for prev, nxt in pairwise(roles))


def _cache_points(model_name: str) -> Any:
    extra = make_model_settings(None, model_name=model_name).extra_args or {}
    return extra.get("cache_control_injection_points")


@pytest.mark.parametrize(
    "model_name",
    [
        "bedrock/global.anthropic.claude-opus-4-8",
        "anthropic/claude-sonnet-4-5",
        "openrouter/anthropic/claude-3.5-sonnet",
        "vertex_ai/claude-sonnet-4-5",
        "claude-sonnet-4-5",
        "openai/claude-sonnet-4-5",
        "any-llm/anthropic/claude-sonnet-4-5",
    ],
)
def test_make_model_settings_does_not_inject_prompt_cache(model_name: str) -> None:
    assert _cache_points(model_name) is None


def test_prompt_cache_can_be_disabled() -> None:
    assert (
        make_model_settings(
            None, model_name="anthropic/claude-sonnet-4-5", prompt_cache=False
        ).extra_args
        is None
    )


@pytest.mark.parametrize("model_name", ["gpt-5", "vertex_ai/gemini-2.5-pro", "openai/o3"])
def test_make_model_settings_no_prompt_cache_for_non_claude(model_name: str) -> None:
    assert make_model_settings(None, model_name=model_name).extra_args is None


def test_max_reasoning_effort_sent_as_raw_body_field() -> None:
    # "max" is absent from the OpenAI SDK's Reasoning enum, so it rides as a raw body field.
    settings = make_model_settings(
        "max", model_name="deepseek/deepseek-v4-flash", request_timeout=30
    )
    assert settings.reasoning is None
    assert settings.extra_args == {"timeout": 30}
    assert settings.extra_body == {"reasoning_effort": "max"}


def test_build_root_task_empty_config() -> None:
    assert build_root_task({}) == ""


def test_build_root_task_repository_target() -> None:
    config = {
        "targets": [
            {
                "type": "repository",
                "details": {
                    "target_repo": "https://example.com/repo.git",
                    "cloned_repo_path": "/workspace/repo",
                    "workspace_subdir": "repo",
                },
            },
        ],
    }
    task = build_root_task(config)

    assert "Repositories:" in task
    assert "/workspace/repo" in task
    assert "https://example.com/repo.git" in task


def test_build_root_task_web_application_with_instructions() -> None:
    config = {
        "targets": [
            {"type": "web_application", "details": {"target_url": "https://app.example.com"}},
        ],
        "user_instructions": "Focus on auth.",
    }
    task = build_root_task(config)

    assert "URLs:" in task
    assert "https://app.example.com" in task
    assert "Special instructions: Focus on auth." in task


def test_build_root_task_workspace_mount_is_not_a_target() -> None:
    """A target-less run gets a working directory, not an assessment scope."""
    config = {
        "targets": [],
        "user_instructions": "Find IDOR in the checkout flow.",
        "workspace_mount": "/Users/me/code/api",
        "workspace_subdir": "api",
    }
    task = build_root_task(config)

    assert "Working Directory:" in task
    assert "/workspace/api" in task
    assert "No scan target was set" in task
    assert "Special instructions: Find IDOR in the checkout flow." in task
    # It must not be presented as an asset to test.
    for label in ("Local Codebases:", "Repositories:", "URLs:", "IP Addresses:"):
        assert label not in task


def test_build_scope_context_authorizes_nothing_without_targets() -> None:
    """A mounted workspace grants no authorized scope."""
    scope = build_scope_context(
        {"targets": [], "workspace_mount": "/Users/me/code/api", "workspace_subdir": "api"}
    )

    assert scope["authorized_targets"] == []


def test_build_root_task_diff_scope() -> None:
    config = {
        "targets": [],
        "diff_scope": {
            "active": True,
            "repos": [
                {
                    "workspace_subdir": "repo",
                    "analyzable_files_count": 3,
                    "deleted_files_count": 2,
                },
            ],
        },
    }
    task = build_root_task(config)

    assert "Scope Constraints:" in task
    assert "3 changed file(s)" in task
    assert "2 deleted file(s)" in task


@pytest.mark.parametrize("model_name", ["openai/o3", "gpt-4o"])
def test_make_model_settings_forces_required_tool_choice_for_openai_models(
    model_name: str,
) -> None:
    settings = make_model_settings(
        "none",
        model_name=model_name,
        force_required_tool_choice=True,
    )

    assert settings.tool_choice == "required"


def test_make_model_settings_skips_required_tool_choice_for_non_openai_models() -> None:
    settings = make_model_settings(
        "none",
        model_name="anthropic/claude-3-7-sonnet-latest",
        force_required_tool_choice=True,
    )

    assert settings.tool_choice is None


def test_make_model_settings_forces_required_for_routed_openai_model() -> None:
    settings = make_model_settings(
        None,
        model_name="litellm/openai/gpt-4o",
        force_required_tool_choice=True,
    )

    assert settings.tool_choice == "required"


def test_make_model_settings_forces_required_for_anyllm_routed_openai_model() -> None:
    settings = make_model_settings(
        None,
        model_name="any-llm/openai/gpt-4o",
        force_required_tool_choice=True,
    )

    assert settings.tool_choice == "required"


def test_make_model_settings_disables_parallel_tool_calls_by_default() -> None:
    assert make_model_settings("none", model_name="gpt-4o").parallel_tool_calls is False


def test_make_model_settings_omits_parallel_tool_calls_without_tools() -> None:
    settings = make_model_settings("none", model_name="gpt-4o", has_tools=False)

    assert settings.parallel_tool_calls is None


def test_make_model_settings_sets_request_timeout() -> None:
    settings = make_model_settings(
        "none",
        model_name="gpt-4o",
        request_timeout=300.0,
    )

    assert settings.extra_args is not None
    assert settings.extra_args["timeout"] == 300.0


def test_make_model_settings_omits_timeout_when_unset() -> None:
    settings = make_model_settings("none", model_name="gpt-4o")

    assert settings.extra_args is None


def test_make_model_settings_sets_extra_headers() -> None:
    settings = make_model_settings(
        "none",
        model_name="openai/some-model",
        extra_headers={"X-Feature-Key": "svc", "X-Tenant": "acme"},
    )

    assert settings.extra_headers == {"X-Feature-Key": "svc", "X-Tenant": "acme"}


def test_make_model_settings_omits_extra_headers_when_unset() -> None:
    assert make_model_settings("none", model_name="gpt-4o").extra_headers is None


def test_make_model_settings_extra_headers_survive_reasoning_resolve() -> None:
    settings = make_model_settings(
        "high",
        model_name="openai/o3",
        extra_headers={"X-Feature-Key": "svc"},
    )

    assert settings.extra_headers == {"X-Feature-Key": "svc"}


def test_make_model_settings_sets_extra_body() -> None:
    settings = make_model_settings(
        "none",
        model_name="openai/gpt-5",
        extra_body={"service_tier": "flex", "safety_identifier": "user-hash"},
    )

    assert settings.extra_body == {"service_tier": "flex", "safety_identifier": "user-hash"}


def test_make_model_settings_omits_extra_body_when_unset() -> None:
    assert make_model_settings("none", model_name="gpt-4o").extra_body is None


def test_make_model_settings_extra_body_survives_reasoning_resolve() -> None:
    settings = make_model_settings(
        "high",
        model_name="openai/o3",
        extra_body={"service_tier": "flex"},
    )

    assert settings.extra_body == {"service_tier": "flex"}


def test_make_model_settings_extra_body_merges_with_max_reasoning_field() -> None:
    settings = make_model_settings(
        "max",
        model_name="deepseek/deepseek-v4-flash",
        extra_body={"safety_identifier": "user-hash"},
    )

    assert settings.extra_body == {"safety_identifier": "user-hash", "reasoning_effort": "max"}


def test_make_model_settings_max_reasoning_overrides_extra_body_reasoning_field() -> None:
    settings = make_model_settings(
        "max",
        model_name="deepseek/deepseek-v4-flash",
        extra_body={"reasoning_effort": "low"},
    )

    assert settings.extra_body == {"reasoning_effort": "max"}


def test_make_model_settings_does_not_mutate_caller_extra_body() -> None:
    extra_body = {"safety_identifier": "user-hash"}

    make_model_settings("max", model_name="deepseek/deepseek-v4-flash", extra_body=extra_body)

    assert extra_body == {"safety_identifier": "user-hash"}


def test_make_model_settings_timeout_survives_reasoning_resolve() -> None:
    # Reasoning is resolved via ModelSettings.resolve(); the timeout in extra_args
    # must not be dropped when a reasoning override is merged in.
    settings = make_model_settings(
        "high",
        model_name="openai/o3",
        request_timeout=120.0,
    )

    assert settings.extra_args is not None
    assert settings.extra_args["timeout"] == 120.0


def test_scan_targets_prefer_the_workspace_checkout_over_the_remote_url() -> None:
    config = {
        "targets": [
            {
                "type": "repository",
                "details": {
                    "target_repo": "https://github.com/acme/billing",
                    "workspace_subdir": "billing",
                },
            },
            {"type": "web_application", "details": {"target_url": "https://app.example.com"}},
        ]
    }

    assert build_scan_targets(config) == ["/workspace/billing", "https://app.example.com"]


def test_scan_targets_drop_empty_and_duplicate_entries() -> None:
    config = {
        "targets": [
            {"type": "web_application", "details": {"target_url": "https://app.example.com"}},
            {"type": "web_application", "details": {"target_url": "https://app.example.com"}},
            {"type": "ip_address", "details": {}},
        ]
    }

    assert build_scan_targets(config) == ["https://app.example.com"]


def test_openrouter_attribution_rides_on_the_request_headers() -> None:
    # Attribution must be part of the per-request headers, not a global client default.
    headers = make_model_settings(
        None, model_name="openrouter/anthropic/claude-sonnet-4-5"
    ).extra_headers
    assert headers == {
        "HTTP-Referer": "https://strix.ai",
        "X-Title": "Strix",
        "X-OpenRouter-Categories": "cli-agent",
    }


def test_openrouter_attribution_absent_for_other_providers() -> None:
    assert make_model_settings(None, model_name="anthropic/claude-sonnet-4-5").extra_headers is None


def test_user_headers_override_openrouter_attribution() -> None:
    headers = make_model_settings(
        None,
        model_name="openrouter/anthropic/claude-sonnet-4-5",
        extra_headers={"X-Title": "Custom", "X-Tenant": "acme"},
    ).extra_headers
    assert headers is not None
    assert headers["X-Title"] == "Custom"
    assert headers["X-Tenant"] == "acme"
    assert headers["HTTP-Referer"] == "https://strix.ai"


def test_reasoning_effort_sent_as_configured_and_none_omitted() -> None:
    assert make_model_settings("none", model_name="gpt-5.6-sol").reasoning is None
    settings = make_model_settings("high", model_name="gpt-daybreak-blue-latest")
    assert settings.reasoning is not None
    assert settings.reasoning.effort == "high"
