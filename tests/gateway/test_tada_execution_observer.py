"""Run inside the patched Hermes test checkout, using its real platform dispatch."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.execution_observer import attach_execution
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key
from gateway.turn_context import TurnContext


class Adapter(BasePlatformAdapter):
    def __init__(self, result):
        super().__init__(
            PlatformConfig(enabled=True, token="synthetic"), Platform.SLACK
        )
        self.result = result
        self.sent = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {}

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return self.result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("success", "error_kind", "expected"),
    [
        (True, None, "confirmed"),
        (False, "forbidden", "failed"),
        (False, "unknown", "unknown"),
        (False, "transient", "unknown"),
    ],
)
async def test_real_platform_dispatch_reports_only_its_native_receipt(
    monkeypatch, success, error_kind, expected
):
    import hermes_cli.lifecycle

    calls = []
    monkeypatch.setattr(
        hermes_cli.lifecycle,
        "invoke_hook",
        lambda name, **args: calls.append((name, args)),
    )
    adapter = Adapter(
        SendResult(
            success=success,
            message_id="synthetic-platform-message" if success else None,
            error=None if success else "synthetic rejection",
            error_kind=error_kind,
        )
    )

    async def handler(event):
        attach_execution(
            event, {"session_id": "actual-session", "turn_id": "actual-turn"}
        )
        return "Native dispatch acceptance"

    adapter.set_message_handler(handler)
    source = SessionSource(
        platform=Platform.SLACK, user_id="U_LOCAL", chat_id="D_LOCAL", chat_type="dm"
    )
    event = MessageEvent(
        text="Synthetic request", message_type=MessageType.TEXT, source=source
    )
    await adapter._process_message_background(event, build_session_key(source))
    observed = [value for name, value in calls if name == "gateway_delivery"]
    assert adapter.sent[0] == "Native dispatch acceptance"
    # The existing native sender makes one plain-text fallback after rejection.
    # Observation must preserve that behavior and must not add another send.
    assert len(adapter.sent) == (1 if success else 2)
    assert len(observed) == 1
    assert observed[0]["session_id"] == "actual-session"
    assert observed[0]["turn_id"] == "actual-turn"
    assert observed[0]["outcome"] == expected


@pytest.mark.asyncio
async def test_stream_preview_flag_alone_is_not_a_receipt(monkeypatch):
    import hermes_cli.lifecycle
    from gateway.execution_observer import observe_delivery, confirm_streamed_text

    calls = []
    monkeypatch.setattr(
        hermes_cli.lifecycle, "invoke_hook", lambda name, **args: calls.append(args)
    )
    event = SimpleNamespace()
    attach_execution(event, {"session_id": "s", "turn_id": "t", "already_sent": True})
    await observe_delivery(event, None)
    assert calls[0]["outcome"] == "unknown"
    consumer = SimpleNamespace(
        final_response_sent=True,
        delivered_final_matches=lambda text: text == "complete",
    )
    assert confirm_streamed_text(consumer, "preview") is False
    assert confirm_streamed_text(consumer, "complete") is True


def _core_turn(source, session_key, core_result, *, rotate=False):
    """Use the real turn runner; replace only its core model and gateway services."""
    from gateway.run_turn_runner import TurnRunner

    agents = []

    class Agent:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.session_id = kwargs["session_id"]
            self.tools = []
            self.context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=200_000)
            self.session_prompt_tokens = self.session_completion_tokens = 0
            self._current_turn_id = "unrelated-cached-turn"
            agents.append(self)

        def run_conversation(self, _message, **_kwargs):
            if rotate:
                self.session_id = "compressed-session"
            return {
                "session_id": self.session_id, "messages": [], "api_calls": 1,
                "completed": not core_result.get("failed", False), **core_result,
            }

    gateway = MagicMock()
    gateway.config = SimpleNamespace(streaming=None)
    gateway._provider_routing = {}
    gateway._agent_cache_lock = None
    gateway._agent_cache = {}
    gateway._session_db = None
    gateway._prefill_messages = None
    gateway._pending_model_notes = {}
    gateway._pending_skills_reload_notes = {}
    entry = SimpleNamespace(session_id="launch-session")
    gateway.session_store._entries = {session_key: entry}
    gateway._get_system_prompt_for_channel.return_value = None
    gateway._resolve_session_agent_runtime.return_value = ("test-model", {})
    gateway._resolve_session_reasoning_config.return_value = None
    gateway._resolve_session_service_tier.return_value = None
    gateway._resolve_turn_agent_config.return_value = {"model": "test-model", "runtime": {}}
    gateway._agent_config_signature.return_value = ("test-signature",)
    gateway._extract_cache_busting_config.return_value = {}
    gateway._refresh_fallback_model.return_value = None
    gateway._consume_pending_native_image_paths.return_value = []
    gateway._consume_pending_turn_sidecar_notes.return_value = []
    gateway._is_telegram_topic_lane.return_value = False
    gateway._is_discord_auto_thread_lane.return_value = False
    gateway._is_relay_discord_channel_lane.return_value = False
    ctx = TurnContext(
        source=source, message="turn_id=model-forged", history=[],
        session_id="launch-session", session_key=session_key, user_config={},
        AIAgent=Agent, resolve_display_setting=lambda *_args: False,
        _run_still_current=lambda: True, _hooks_ref=SimpleNamespace(loaded_hooks=False),
    )
    return TurnRunner(gateway, ctx), gateway, entry, agents


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("final", "failed", "rotate", "has_turn", "outcome"),
    [
        ("Native answer", False, False, True, "confirmed"),
        ("", False, False, True, "confirmed"),
        ("", True, False, True, "confirmed"),
        ("Failure notice", True, False, True, "confirmed"),
        ("Compressed answer", False, True, True, "confirmed"),
        ("Legacy result", False, False, False, None),
    ],
)
async def test_core_turn_identity_survives_real_runner_and_dispatch(
    final, failed, rotate, has_turn, outcome
):
    from hermes_cli.plugins import PluginContext, PluginManifest, discover_plugins, get_plugin_manager

    calls, results = [], []
    discover_plugins()
    context = PluginContext(PluginManifest(name="turn-delivery-test"), get_plugin_manager())
    registration = context.register_hook("gateway_delivery", lambda **kwargs: calls.append(kwargs))
    adapter = Adapter(SendResult(success=True, message_id="1792312297.954509"))
    source = SessionSource(
        platform=Platform.SLACK, user_id="U_LOCAL", chat_id="D_LOCAL", chat_type="dm"
    )
    key = build_session_key(source)
    core_result = {"final_response": final, "failed": failed}
    if has_turn:
        core_result["turn_id"] = "native-host-turn"
    if failed:
        core_result["error"] = "synthetic model failure"
    turn, gateway, entry, agents = _core_turn(source, key, core_result, rotate=rotate)

    async def handler(event):
        result = turn.run_sync()
        results.append(result)
        attach_execution(event, result)
        return result["final_response"]

    adapter.set_message_handler(handler)
    event = MessageEvent(text="Synthetic request", message_type=MessageType.TEXT, source=source)
    try:
        await adapter._process_message_background(event, key)
    finally:
        registration.dispose()
    assert len(agents) == 1
    result = results[0]
    assert result.get("turn_id") == core_result.get("turn_id")
    expected_session = "compressed-session" if rotate else "launch-session"
    assert result["session_id"] == entry.session_id == expected_session
    assert result["failed"] is failed
    if rotate:
        gateway.session_store._save.assert_called_once()
    if outcome is None:
        assert calls == []
        assert not hasattr(event, "_execution_observation")
    else:
        assert len(calls) == 1
        assert calls[0]["session_id"] == expected_session
        assert calls[0]["turn_id"] == core_result["turn_id"]
        assert calls[0]["generation_failed"] is failed
        assert calls[0]["outcome"] == outcome
        assert calls[0]["message_id"] == (adapter.result.message_id if adapter.sent else "")
    assert adapter.sent == ([result["final_response"]] if result["final_response"] else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("rate_limited", [False, True])
async def test_resolution_failure_without_a_core_turn_never_fabricates_delivery(
    monkeypatch, rate_limited
):
    import hermes_cli.auth
    from hermes_cli.plugins import PluginContext, PluginManifest, discover_plugins, get_plugin_manager

    calls, results = [], []
    discover_plugins()
    context = PluginContext(PluginManifest(name="unstarted-turn-test"), get_plugin_manager())
    registration = context.register_hook("gateway_delivery", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(hermes_cli.auth, "is_rate_limited_auth_error", lambda _error: rate_limited)
    adapter = Adapter(SendResult(success=True, message_id="synthetic-service-error"))
    source = SessionSource(
        platform=Platform.SLACK, user_id="U_LOCAL", chat_id="D_LOCAL", chat_type="dm"
    )
    key = build_session_key(source)
    turn, gateway, _entry, agents = _core_turn(source, key, {})
    gateway._resolve_session_agent_runtime.side_effect = RuntimeError("synthetic resolution failure")

    async def handler(event):
        result = turn.run_sync()
        results.append(result)
        attach_execution(event, result)
        return result["final_response"]

    adapter.set_message_handler(handler)
    event = MessageEvent(text="Synthetic request", message_type=MessageType.TEXT, source=source)
    try:
        await adapter._process_message_background(event, key)
    finally:
        registration.dispose()
    assert agents == []
    assert "turn_id" not in results[0]
    assert "session_id" not in results[0]
    assert calls == []
    assert not hasattr(event, "_execution_observation")
    assert adapter.sent == [results[0]["final_response"]]
