"""Public plugin APIs preserve request identity and native delivery boundaries."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from plugins.platforms.slack.adapter import SlackAdapter


def context(home):
    return PluginContext(PluginManifest(name="public-contract", source="user"), PluginManager(scope_key=str(home)))


def test_session_reader_never_promotes_environment_to_actor(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "foreign-cron-user")
    monkeypatch.setenv("UNREGISTERED_ACTOR", "foreign-actor")
    reset_session_vars()
    assert ctx.session_value("HERMES_SESSION_USER_ID") == ""
    assert ctx.session_value("UNREGISTERED_ACTOR", "missing") == "missing"
    tokens = set_session_vars(platform="slack", user_id="actual-user", scope_id="T_A")
    try:
        assert ctx.session_value("HERMES_SESSION_USER_ID") == "actual-user"
        assert ctx.session_value("HERMES_SESSION_SCOPE_ID") == "T_A"
    finally:
        clear_session_vars(tokens)
    assert ctx.session_value("HERMES_SESSION_USER_ID") == ""


@pytest.mark.asyncio
async def test_async_bridge_and_background_tasks_carry_only_calling_actor(tmp_path):
    ctx = context(tmp_path)

    async def read_actor():
        await asyncio.sleep(0)
        return ctx.session_value("HERMES_SESSION_USER_ID")

    async def actor(name):
        tokens = set_session_vars(user_id=name)
        try:
            # Exercise the synchronous handler path on a real worker thread.
            return await asyncio.to_thread(ctx.run_async, read_actor())
        finally:
            clear_session_vars(tokens)

    assert await asyncio.gather(actor("A"), actor("B"), actor("A")) == ["A", "B", "A"]


@pytest.mark.asyncio
async def test_sync_worker_background_task_is_supervised_by_owning_manager(tmp_path):
    ctx = context(tmp_path)
    started, canceled = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()
    observed = []

    async def work():
        assert asyncio.get_running_loop() is loop
        observed.append(ctx.session_value("HERMES_SESSION_USER_ID"))
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()

    tokens = set_session_vars(user_id="worker-actor")
    try:
        future = await asyncio.to_thread(ctx.spawn_task, work(), name="native-supervised")
        await asyncio.wait_for(started.wait(), 5)
        assert observed == ["worker-actor"]
        ctx._manager.unload()
        await asyncio.wait_for(canceled.wait(), 5)
        assert future.cancelled()
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
async def test_slack_plugin_blocks_use_exact_workspace_and_thread():
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    client_a = SimpleNamespace(chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "101.1"}))
    client_b = SimpleNamespace(chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "102.2"}))
    adapter._team_clients = {"T_A": client_a, "T_B": client_b}
    adapter._app = SimpleNamespace(client=client_a)
    blocks = [{"type": "actions", "elements": [{
        "type": "button", "text": {"type": "plain_text", "text": "Confirm"},
        "action_id": "confirm", "value": "nonce-only",
    }]}]
    result = await adapter.send_plugin_message(
        "D_SAME", "Confirm", reply_to="100.1", metadata={"workspace_id": "T_B", "blocks": blocks},
    )
    assert result.success and result.message_id == "102.2"
    client_a.chat_postMessage.assert_not_awaited()
    payload = client_b.chat_postMessage.await_args.kwargs
    assert payload["channel"] == "D_SAME" and payload["thread_ts"] == "100.1"
    assert payload["blocks"][0]["elements"][0]["value"] == "nonce-only"
    for metadata in ({}, {"workspace_id": "UNKNOWN"}, {"workspace_id": "T_B", "team_id": "T_A"}):
        assert not (await adapter.send_plugin_message("D_SAME", "Confirm", metadata=metadata)).success
    assert client_b.chat_postMessage.await_count == 1


@pytest.mark.asyncio
async def test_public_send_keeps_consent_and_gateway_loop(tmp_path, monkeypatch):
    import gateway.run

    ctx = context(tmp_path)
    adapter = SimpleNamespace(send_plugin_message=AsyncMock(return_value=SimpleNamespace(success=True, message_id="102.2")))
    actions = ctx.platform_actions
    runner = SimpleNamespace(_gateway_loop=asyncio.get_running_loop())
    monkeypatch.setattr(gateway.run, "_gateway_runner_ref", lambda: runner)
    monkeypatch.setattr(actions, "_resolve_adapter", lambda _platform: (adapter, None))
    monkeypatch.setattr(actions, "_capability_granted", lambda: False)
    assert (await ctx.send_platform_message("slack", "D_A", "Confirm"))["error"] == "capability_not_granted"
    adapter.send_plugin_message.assert_not_awaited()
    monkeypatch.setattr(actions, "_capability_granted", lambda: True)

    async def assert_host_loop(*args, **kwargs):
        assert asyncio.get_running_loop() is runner._gateway_loop
        assert ctx.session_value("HERMES_SESSION_USER_ID") == "actual-user"
        return SimpleNamespace(success=True, message_id="102.2")

    adapter.send_plugin_message.side_effect = assert_host_loop
    tokens = set_session_vars(user_id="actual-user")
    try:
        # The tool's async bridge owns a different loop. The SDK stays on the host loop.
        result = await asyncio.to_thread(ctx.run_async, ctx.send_platform_message(
            "slack", "D_A", "Confirm", metadata={"workspace_id": "T_A"},
        ))
    finally:
        clear_session_vars(tokens)
    assert result == {"ok": True, "action": "send_message", "message_id": "102.2"}


@pytest.mark.asyncio
async def test_source_reread_is_read_only_exact_workspace_and_host_loop(tmp_path, monkeypatch):
    import gateway.run

    ctx = context(tmp_path)
    host = asyncio.get_running_loop()

    async def read(method, *, http_verb, params):
        assert asyncio.get_running_loop() is host
        if method == "chat.getPermalink":
            assert params == {"channel": "D_A", "message_ts": "123.45"}
            return {"ok": True, "permalink": "https://workspace.slack.com/archives/D_A/p12345"}
        assert method == "conversations.history" and http_verb == "GET"
        assert params == {"channel": "D_A", "latest": "123.45", "inclusive": True, "limit": 1}
        return {"ok": True, "messages": [{"ts": "123.45", "text": "source"}]}

    client = SimpleNamespace(api_call=AsyncMock(side_effect=read))
    adapter = SimpleNamespace(_team_clients={"T_A": client})
    monkeypatch.setattr(gateway.run, "_gateway_runner_ref", lambda: SimpleNamespace(_gateway_loop=host))
    monkeypatch.setattr(ctx.platform_actions, "_capability_granted", lambda: True)
    monkeypatch.setattr(ctx.platform_actions, "_resolve_adapter", lambda _platform: (adapter, None))
    result = await asyncio.to_thread(ctx.run_async, ctx.call_slack_api(
        "T_A", "conversations.history", channel="D_A", latest="123.45", inclusive=True, limit=1,
    ))
    assert result["messages"][0]["text"] == "source"
    permalink = await ctx.call_slack_api("T_A", "chat.getPermalink", channel="D_A", message_ts="123.45")
    assert permalink["permalink"] == "https://workspace.slack.com/archives/D_A/p12345"
    assert (await ctx.call_slack_api("UNKNOWN", "conversations.history"))["error"] == "workspace_unavailable"
    assert (await ctx.call_slack_api("T_A", "chat.delete"))["error"] == "unsupported_platform_action"
    assert client.api_call.await_count == 2
