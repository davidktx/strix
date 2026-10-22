"""SDK model configuration helpers."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from agents import (
    set_default_openai_api,
    set_default_openai_key,
    set_tracing_disabled,
)
from agents.model_settings import ModelSettings
from agents.models.fake_id import FAKE_RESPONSES_ID
from agents.models.interface import Model, ModelProvider
from agents.models.multi_provider import MultiProvider
from agents.models.openai_responses import OpenAIResponsesModel
from agents.retry import (
    ModelRetryBackoffSettings,
    ModelRetrySettings,
    RetryPolicyContext,
    retry_policies,
)
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
)
from openai.types.responses.response_usage import ResponseUsage
from openai.types.shared import Reasoning

from strix.agents.prompt import CACHE_POINT
from strix.config import codex
from strix.config.loader import load_settings
from strix.config.model_catalog import catalog_entry as _catalog_entry
from strix.config.tool_call_arguments import repair_input
from strix.config.tool_call_ids import TurnCallIdRewriter, dedupe_input
from strix.config.tool_call_limits import TurnToolCallLimiter
from strix.llm import request_log


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agents.agent_output import AgentOutputSchemaBase
    from agents.handoffs import Handoff
    from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent
    from agents.models.interface import ModelTracing
    from agents.retry import ModelRetryAdvice, ModelRetryAdviceRequest
    from agents.tool import Tool
    from agents.usage import Usage
    from openai import AsyncOpenAI
    from openai.types.responses.response_prompt_param import ResponsePromptParam

    from strix.config.settings import ApiType, LlmSettings, ReasoningEffort, Settings


logger = logging.getLogger(__name__)


def request_timeout_extra_args(timeout_s: float | None) -> dict[str, float] | None:
    """Per-request model timeout; a plain float so ``ModelSettings.to_json_dict()`` stays serializable."""  # noqa: E501
    if not timeout_s or timeout_s <= 0:
        return None
    return {"timeout": timeout_s}


def _retry_statusless_provider_errors(context: RetryPolicyContext) -> bool:
    """Retry statusless provider errors (e.g. mid-stream quota/billing), but not aborts."""
    normalized = context.normalized
    if normalized.is_abort:
        return False
    if codex.is_content_guardrail_error(context.error):
        return False
    return normalized.status_code is None


class _CodexResponsesModel(OpenAIResponsesModel):
    """Responses model for the ChatGPT subscription backend (always streamed, stateless)."""

    def __init__(
        self,
        model: str,
        openai_client: AsyncOpenAI,
        *,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        super().__init__(model, openai_client)
        self._reasoning_effort = reasoning_effort

    def _codex_settings(self, model_settings: ModelSettings) -> ModelSettings:
        overrides = ModelSettings(store=False, response_include=["reasoning.encrypted_content"])
        effort = self._reasoning_effort
        if effort and effort != "none":
            # Clamp to efforts the backend accepts.
            match effort:
                case "minimal":
                    effort = "low"
                case "xhigh" | "max":
                    effort = "high"
                case _:
                    pass
            overrides = overrides.resolve(ModelSettings(reasoning=Reasoning(effort=effort)))
        return model_settings.resolve(overrides)

    async def _fetch_response(self, *args: Any, stream: bool = False, **kwargs: Any) -> Any:
        if len(args) >= 3:  # model_settings is positional arg 2
            args = (*args[:2], self._codex_settings(args[2]), *args[3:])
        try:
            events = await super()._fetch_response(*args, stream=True, **kwargs)  # type: ignore[call-overload]
        except Exception as exc:
            guardrail = self._as_guardrail(exc)
            if guardrail is not None:
                raise guardrail from exc
            raise
        guarded = self._guarded(events)
        if stream:
            return guarded
        final_response = None
        async for event in guarded:
            if getattr(event, "type", None) == "response.completed":
                final_response = event.response
        if final_response is None:
            msg = "ChatGPT backend stream ended without a completed response"
            raise RuntimeError(msg)
        return final_response

    def _as_guardrail(self, exc: BaseException) -> codex.CodexContentGuardrailError | None:
        if isinstance(exc, codex.CodexContentGuardrailError):
            return exc
        if codex.is_content_guardrail_error(exc):
            return codex.CodexContentGuardrailError(self.model, exc)
        return None

    async def _guarded(self, events: Any) -> AsyncIterator[Any]:
        """Convert mid-stream guardrail rejections and close the stream on exit."""
        try:
            async for event in events:
                yield event
        except Exception as exc:
            guardrail = self._as_guardrail(exc)
            if guardrail is not None:
                raise guardrail from exc
            raise
        finally:
            await self._aclose(events)

    @staticmethod
    async def _aclose(events: Any) -> None:
        aclose = getattr(events, "aclose", None)
        if callable(aclose):
            with contextlib.suppress(Exception):
                await aclose()
            return
        close = getattr(events, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                result = close()
                if inspect.isawaitable(result):
                    await result


class _NonStreamingModel(Model):
    """Serve the SDK's streamed run loop from a single non-streaming request.

    Some OpenAI-compatible gateways do not support Server-Sent Events, or
    deliver them unreliably (dropping structured tool-call deltas, or stalling
    mid-stream so the whole turn waits out the read timeout). The SDK run loop
    Strix uses only issues streamed requests, so such a gateway fails every
    turn. Opt in with ``LLM_DISABLE_STREAMING=true`` to wrap the resolved model
    so each turn makes one non-streaming ``get_response`` (``stream:false`` on
    the wire) and the completed result is replayed as a single terminal stream
    event. The run loop then executes tools and emits run items from that final
    response exactly as it would for a real stream, so nothing else changes.
    """

    def __init__(self, inner: Model) -> None:
        self._inner = inner

    async def close(self) -> None:
        await self._inner.close()

    def get_retry_advice(self, request: ModelRetryAdviceRequest) -> ModelRetryAdvice | None:
        return self._inner.get_retry_advice(request)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],  # noqa: A002
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        return await self._inner.get_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],  # noqa: A002
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        response = await self._inner.get_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )
        yield _completed_stream_event(response, getattr(self._inner, "model", None))


class _TurnGuardModel(Model):
    """Keep one turn from corrupting the conversation or running away.

    Tool-call ids: providers that number calls per turn (``exec_command:0``,
    ...) restart the counter each turn, so the same id eventually appears twice
    in one conversation and strict providers reject every subsequent request.
    Ids that collide with the history are rewritten before the turn is
    recorded, and already-corrupted histories are repaired on the way out.

    Tool-call arguments: a turn whose ``arguments`` are not valid JSON fails
    that one call with a parse error, but the raw string is recorded and
    strict providers then reject every request that replays it. Such
    arguments are rewritten to a valid JSON object on the way out (see
    :mod:`strix.config.tool_call_arguments`).

    Tool-call volume: a degenerate response can queue hundreds of calls that
    the run loop then honours one by one. Only the first
    ``LLM_MAX_TOOL_CALLS_PER_TURN`` calls of a response are kept.

    Stalled streams: a turn that emits a few tokens and then goes silent is
    not covered by the request timeout, which resets on any byte (keepalives
    included). ``LLM_STREAM_IDLE_TIMEOUT`` bounds the gap between events so the
    turn fails instead of hanging, and the existing retry path replays it.
    ``LLM_STREAM_FIRST_EVENT_TIMEOUT`` and ``LLM_STREAM_TOTAL_TIMEOUT`` bound
    the first event and the whole stream.
    """

    def __init__(
        self,
        inner: Model,
        *,
        max_tool_calls_per_turn: int = 0,
        stream_idle_timeout: float | None = None,
        first_event_timeout: float | None = None,
        total_timeout: float | None = None,
    ) -> None:
        self._inner = inner
        self._max_tool_calls_per_turn = max_tool_calls_per_turn
        self._stream_idle_timeout = stream_idle_timeout
        self._first_event_timeout = first_event_timeout
        self._total_timeout = total_timeout

    def _limiter(self) -> TurnToolCallLimiter:
        return TurnToolCallLimiter(self._max_tool_calls_per_turn)

    def _log_dropped(self, limiter: TurnToolCallLimiter) -> None:
        if limiter.dropped:
            logger.warning(
                "dropped %d tool call(s) past the per-response limit of %d",
                limiter.dropped,
                self._max_tool_calls_per_turn,
            )

    async def close(self) -> None:
        await self._inner.close()

    def get_retry_advice(self, request: ModelRetryAdviceRequest) -> ModelRetryAdvice | None:
        return self._inner.get_retry_advice(request)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],  # noqa: A002
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        sanitized = _sanitize_input(input)
        rewriter = TurnCallIdRewriter(sanitized)
        system_instructions, sanitized = _split_cached_prefix(
            system_instructions, sanitized, model_settings
        )
        response = await self._inner.get_response(
            system_instructions,
            cast("str | list[TResponseInputItem]", sanitized),
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )
        limiter = self._limiter()
        response.output = limiter.filter_items(rewriter.rewrite_items(list(response.output)))
        self._log_dropped(limiter)
        return response

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],  # noqa: A002
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        sanitized = _sanitize_input(input)
        rewriter = TurnCallIdRewriter(sanitized)
        system_instructions, sanitized = _split_cached_prefix(
            system_instructions, sanitized, model_settings
        )
        limiter = self._limiter()
        stream = self._inner.stream_response(
            system_instructions,
            cast("str | list[TResponseInputItem]", sanitized),
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )
        async for event in _with_timeouts(
            stream,
            idle=self._stream_idle_timeout,
            first_event=self._first_event_timeout,
            total=self._total_timeout,
        ):
            guarded = _guard_event(event, rewriter, limiter)
            if guarded is not None:
                yield guarded
        self._log_dropped(limiter)


def _sanitize_input(model_input: str | list[TResponseInputItem]) -> str | list[Any]:
    return repair_input(dedupe_input(model_input))


def _split_cached_prefix(
    system_instructions: str | None,
    model_input: str | list[Any],
    model_settings: ModelSettings,
) -> tuple[str | None, str | list[Any]]:
    """Split the system prompt at each ``CACHE_POINT`` on cache-point routes.

    LiteLLM puts a cache point at the end of each system message, so each part
    gets its own. Other routes get the prompt with the markers removed.
    """
    if not system_instructions or CACHE_POINT not in system_instructions:
        return system_instructions, model_input
    extra_args = model_settings.extra_args or {}
    if "cache_control_injection_points" not in extra_args:
        return system_instructions.replace(CACHE_POINT, ""), model_input
    if isinstance(model_input, str):
        model_input = [{"role": "user", "content": model_input}]
    parts = [part for part in system_instructions.split(CACHE_POINT) if part.strip()]
    return None, [*({"role": "system", "content": part} for part in parts), *model_input]


async def _aclose(stream: AsyncIterator[TResponseStreamEvent]) -> None:
    if isinstance(stream, AsyncGenerator):
        with contextlib.suppress(Exception):
            await stream.aclose()


async def _with_timeouts(
    stream: AsyncIterator[TResponseStreamEvent],
    *,
    idle: float | None = None,
    first_event: float | None = None,
    total: float | None = None,
) -> AsyncIterator[TResponseStreamEvent]:
    """Bound the first event, the gap between events and the whole stream; None is no bound."""
    iterator = stream.__aiter__()
    started = time.monotonic()
    while True:
        limits: list[tuple[float, float, str]] = []
        # Wait until the soonest limit expires: first-event (else idle) until the
        # first event arrives, then idle; plus whatever is left of the total.
        if first_event is not None:
            limits.append((first_event, first_event, "stream_first_event_timeout"))
        elif idle is not None:
            limits.append((idle, idle, "stream_idle_timeout"))
        if total is not None:
            left = max(0.0, started + total - time.monotonic())
            limits.append((left, total, "stream_total_timeout"))
        try:
            if limits:
                event = await _next_event(iterator, *min(limits))
            else:
                event = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            await _aclose(stream)
            logger.warning("%s; abandoning the turn", exc)
            raise
        first_event = None
        yield event


async def _next_event(
    iterator: AsyncIterator[TResponseStreamEvent], wait: float, limit: float, reason: str
) -> TResponseStreamEvent:
    # asyncio.timeout() cancels without a message, so the request log could not
    # tell this from a shutdown; this is asyncio.timeout() with a message.
    task = asyncio.current_task()
    assert task is not None
    cancelling = task.cancelling()
    expired = False

    def expire() -> None:
        nonlocal expired
        expired = True
        task.cancel(msg=f"{request_log.CANCEL_REASON_PREFIX}{reason}")

    handle = asyncio.get_running_loop().call_later(wait, expire)
    try:
        return await iterator.__anext__()
    except BaseException as exc:
        if expired and task.uncancel() <= cancelling and isinstance(exc, asyncio.CancelledError):
            raise TimeoutError(f"model stream hit {reason} ({limit:.0f}s)") from None
        raise
    finally:
        handle.cancel()


def _guard_event(
    event: TResponseStreamEvent, rewriter: TurnCallIdRewriter, limiter: TurnToolCallLimiter
) -> TResponseStreamEvent | None:
    if isinstance(event, ResponseOutputItemAddedEvent | ResponseOutputItemDoneEvent):
        rewritten = rewriter.rewrite_item(event.item, event.output_index)
        if not limiter.allow(rewritten):
            return None
        if rewritten is not event.item:
            return event.model_copy(update={"item": rewritten})
        return event
    if isinstance(event, ResponseCompletedEvent):
        original = list(event.response.output)
        output = limiter.filter_items(rewriter.rewrite_items(original))
        if output != original:
            return event.model_copy(
                update={"response": event.response.model_copy(update={"output": output})}
            )
    return event


def _completed_stream_event(
    model_response: ModelResponse, model_name: object | None
) -> TResponseStreamEvent:
    """Wrap a non-streamed ``ModelResponse`` as the terminal event of a stream.

    The run loop builds its authoritative per-turn response solely from the
    ``response.completed`` event, so a single event carrying the full output
    and usage is all it needs.
    """
    response = Response(
        id=model_response.response_id or FAKE_RESPONSES_ID,
        created_at=time.time(),
        model=str(model_name) if model_name else "",
        object="response",
        output=list(model_response.output),
        tool_choice="auto",
        tools=[],
        parallel_tool_calls=False,
        usage=_response_usage(model_response.usage),
    )
    return ResponseCompletedEvent(
        response=response,
        sequence_number=0,
        type="response.completed",
    )


def _response_usage(usage: Usage | None) -> ResponseUsage | None:
    if usage is None:
        return None
    return ResponseUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        input_tokens_details=usage.input_tokens_details,
        output_tokens_details=usage.output_tokens_details,
    )


class StrixProvider(MultiProvider):
    """Resolve every model through the SDK's OpenAI-compatible client.

    Provider-prefixed names (``openrouter/...``, ``deepseek/...``) are passed
    through as literal model IDs when the prefix is not ``openai/``. Legacy
    ``litellm/`` and ``ollama/`` prefixes are stripped. Set ``LLM_API_BASE``
    and ``LLM_API_KEY`` for OpenAI, OpenRouter, Ollama, or any compatible API.

    ``api_key``/``base_url`` bind this provider instance to one endpoint (e.g.
    the dedupe judge).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            openai_api_key=api_key,
            openai_base_url=base_url,
            openai_use_responses=False if base_url else None,
            unknown_prefix_mode="model_id",
            **kwargs,
        )
        self._override_api_key = api_key
        self._override_base_url = base_url

    def _resolve_prefixed_model(
        self,
        *,
        original_model_name: str,
        prefix: str,
        stripped_model_name: str | None,
    ) -> tuple[ModelProvider, str | None]:
        lowered = prefix.lower()
        if lowered == "litellm" and stripped_model_name:
            return self.openai_provider, stripped_model_name
        if lowered == "any-llm" and stripped_model_name:
            return self.openai_provider, stripped_model_name
        if lowered == "ollama" and stripped_model_name:
            return self.openai_provider, stripped_model_name
        return super()._resolve_prefixed_model(
            original_model_name=original_model_name,
            prefix=prefix,
            stripped_model_name=stripped_model_name,
        )

    def get_model(self, model_name: str | None) -> Model:
        llm = load_settings().llm
        slug = codex.subscription_model(model_name)
        idle_timeout = float(llm.stream_idle_timeout) or None
        first_event_timeout = float(llm.stream_first_event_timeout) or None
        total_timeout = float(llm.stream_total_timeout) or None
        if slug:
            model: Model = request_log.RequestLoggingModel(
                _CodexResponsesModel(
                    slug,
                    codex.get_subscription_client(),
                    reasoning_effort=llm.reasoning_effort,
                ),
                model_name=slug,
                provider="openai-codex",
                base_url=None,
            )
        else:
            model = super().get_model(model_name)
            resolved_name = model_name or llm.model or "unknown"
            model = request_log.RequestLoggingModel(
                model,
                model_name=resolved_name,
                provider=_infer_provider_name(resolved_name),
                base_url=self._override_base_url or llm.api_base,
            )
            if llm.disable_streaming:
                model = _NonStreamingModel(model)
                idle_timeout = first_event_timeout = total_timeout = None
        return _TurnGuardModel(
            model,
            max_tool_calls_per_turn=llm.max_tool_calls_per_turn,
            stream_idle_timeout=idle_timeout,
            first_event_timeout=first_event_timeout,
            total_timeout=total_timeout,
        )


def _infer_provider_name(model_name: str) -> str:
    entry = _catalog_entry(model_name)
    if entry and entry.get("provider"):
        return str(entry["provider"])
    bare = _bare_openai_name(model_name)
    if "/" in bare:
        return bare.split("/", 1)[0]
    return "openai"


DEFAULT_MODEL_RETRY = ModelRetrySettings(
    max_retries=5,
    backoff=ModelRetryBackoffSettings(
        initial_delay=2.0,
        max_delay=90.0,
        multiplier=2.0,
        jitter=False,
    ),
    policy=retry_policies.any(
        retry_policies.provider_suggested(),
        retry_policies.network_error(),
        retry_policies.http_status((429, 500, 502, 503, 504)),
        _retry_statusless_provider_errors,
    ),
)


def configure_sdk_model_defaults(settings: Settings) -> None:
    """Apply Strix config to SDK-native defaults."""
    llm = settings.llm
    set_tracing_disabled(True)
    from strix.report.state import record_llm_request

    request_log.install()
    if settings.telemetry.run_record_timings:
        request_log.register_sink(record_llm_request)
    if codex.subscription_model(llm.model):
        return
    if llm.api_key:
        set_default_openai_key(llm.api_key, use_for_tracing=False)
    if llm.api_base:
        os.environ["OPENAI_BASE_URL"] = llm.api_base
    configure_sdk_api_route(llm.model or "", settings)
    _configure_extra_headers(llm)


def configure_sdk_api_route(model_name: str, settings: Settings) -> None:
    """Point SDK-native OpenAI requests for ``model_name`` at Responses or chat completions."""
    api_type = resolve_api_type(model_name, settings)
    logger.info("OpenAI API route for %s: %s", model_name, api_type)
    set_default_openai_api(api_type)


_RESPONSES_ENDPOINT = "/v1/responses"


def resolve_api_type(model_name: str, settings: Settings) -> ApiType:
    """The SDK-native OpenAI route for ``model_name``: Responses or chat completions.

    An explicit ``STRIX_API_TYPE`` wins. Otherwise the built-in catalog lists
    ``/v1/responses`` when supported; everything else uses chat completions.
    """
    if settings.llm.api_type is not None:
        return settings.llm.api_type
    if _RESPONSES_ENDPOINT in _catalog_supported_endpoints(model_name):
        return "responses"
    return "chat_completions"


def _catalog_supported_endpoints(model_name: str) -> list[str]:
    entry = _catalog_entry(model_name)
    endpoints = entry.get("supported_endpoints") if entry else None
    return [str(e) for e in endpoints] if isinstance(endpoints, list) else []


OPENROUTER_ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://strix.ai",
    "X-Title": "Strix",
    "X-OpenRouter-Categories": "cli-agent",
}


def is_openrouter_model(model_name: str | None) -> bool:
    return bool(model_name) and "openrouter/" in (model_name or "").strip().lower()


def _configure_extra_headers(llm: LlmSettings) -> None:
    """Send default HTTP headers on every OpenAI-compatible LLM request."""
    headers: dict[str, str] = {}
    if is_openrouter_model(llm.model):
        headers.update(OPENROUTER_ATTRIBUTION_HEADERS)
    if llm.extra_headers:
        headers.update(llm.extra_headers)
    if not headers:
        return
    _register_openai_client_with_headers(llm, headers)


def _register_openai_client_with_headers(llm: LlmSettings, headers: dict[str, str]) -> None:
    from agents import set_default_openai_client
    from agents.models.openai_provider import shared_http_client
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=llm.api_key or "not-needed",
        base_url=llm.api_base,
        default_headers=dict(headers),
        # The SDK's shared client is the one the request log observes for
        # reply status, headers and provider request ids.
        http_client=shared_http_client(),
    )
    set_default_openai_client(client, use_for_tracing=False)


def uses_chat_completions_tool_schema(model_name: str, settings: Settings) -> bool:
    """Return whether the resolved SDK route can only receive JSON function tools."""
    if codex.subscription_model(model_name):
        return False
    model = model_name.strip().lower()
    if "/" in model and not model.startswith("openai/"):
        return True
    return resolve_api_type(model_name, settings) == "chat_completions"


def supports_strict_tool_schemas(model_name: str) -> bool:
    """Return whether the route accepts strict tool schemas for Strix's toolset.

    Claude caps a request at 20 strict tools and 16 union-typed parameters
    across all strict schemas. Strix ships ~30 tools and the strict dialect
    turns every optional parameter into a nullable union, so both caps are
    exceeded and the request is rejected outright.
    """
    name = model_name.strip().lower()
    return not any(marker in name for marker in _ANTHROPIC_MODEL_MARKERS)


def model_supports_reasoning(model_name: str) -> bool:
    entry = _catalog_entry(model_name)
    return bool(entry and entry.get("supports_reasoning"))


def model_supports_images(model_name: str) -> bool:
    """Return whether the model accepts image input. Assume yes until proven otherwise."""
    entry = _catalog_entry(model_name)
    return entry is None or bool(entry.get("supports_vision"))


def _bare_openai_name(model_name: str) -> str:
    name = model_name.strip().lower()
    for prefix in ("litellm/", "any-llm/", "openai/"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def is_known_openai_bare_model(model_name: str) -> bool:
    name = model_name.strip().lower()
    if not name or "/" in name:
        return False
    entry = _catalog_entry(name)
    return bool(entry and entry.get("provider") == "openai")


_ANTHROPIC_MODEL_MARKERS = ("anthropic", "claude", "sonnet", "opus", "haiku")


def is_claude_model(model_name: str) -> bool:
    return "claude" in (model_name or "").strip().lower()


def routes_through_litellm(_model_name: str | None) -> bool:
    """Legacy helper; all models use the OpenAI-compatible SDK route now."""
    return False


def is_bedrock_route(model_name: str) -> bool:
    name = (model_name or "").strip().lower()
    return name.startswith("bedrock/") or "anthropic." in name


def _prompt_cache_name_candidates(model_name: str) -> list[str]:
    # LiteLLM's model map keys the same model under several names; strip the
    # route prefix, then leading dotted segments (region, provider).
    name = (model_name or "").strip().lower()
    for prefix in ("litellm/", "bedrock/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    candidates = [name]
    rest = name
    while "." in rest:
        rest = rest.split(".", 1)[1]
        candidates.append(rest)
    return candidates


def bedrock_route_supports_prompt_caching(model_name: str) -> bool:
    for cand in _prompt_cache_name_candidates(model_name):
        entry = _catalog_entry(cand)
        if entry and entry.get("supports_prompt_caching"):
            return True
    return False
