"""Native SDK wiring covers message observers, shortcuts and real release registrars."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from hermes_cli import plugins
from plugins.platforms.slack.adapter import SlackAdapter


class BoltTransport:
    def __init__(self):
        self.events, self.shortcuts, self.commands = {}, {}, {}

    def event(self, key):
        def register(callback):
            self.events[key] = callback
            return callback
        return register

    def action(self, _key):
        return lambda callback: callback

    def command(self, key):
        def register(callback):
            self.commands[key] = callback
            return callback
        return register

    def shortcut(self, key):
        def register(callback):
            self.shortcuts[key] = callback
            return callback
        return register


@pytest.mark.asyncio
async def test_named_message_observer_runs_before_chat_with_exact_client(tmp_path, monkeypatch):
    manager = plugins.PluginManager(scope_key=str(tmp_path))
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    ctx = plugins.PluginContext(plugins.PluginManifest(name="observer", source="user"), manager)
    seen = []
    exact = object()

    async def observer(*, event, body, client):
        assert client is exact
        seen.append("observer")

    async def chat(*_args):
        seen.append("chat")

    ctx.register_slack_event_handler("message", observer)
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    adapter._app = BoltTransport()
    adapter._team_clients = {"T_B": exact}
    adapter._handle_slack_message = chat
    adapter._wire_plugin_handlers = lambda _app: None
    adapter._register_bolt_handlers()
    await adapter._app.events["message"]({"type": "message", "user": "U_B"}, None, {"team_id": "T_B"})
    assert seen == ["observer", "chat"]
    seen.clear()
    await adapter._app.events["message"]({"type": "message", "user": "U_B"}, None, {"team_id": "UNKNOWN"})
    assert seen == ["chat"]


@pytest.mark.asyncio
async def test_shortcut_listener_is_exact_scoped_and_survives_reload(tmp_path, monkeypatch):
    manager = plugins.PluginManager(scope_key=str(tmp_path))
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    ctx = plugins.PluginContext(plugins.PluginManifest(name="shortcut", source="user"), manager)
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    adapter._app = BoltTransport()
    exact = object()
    adapter._team_clients = {"T_B": exact}
    ack, seen = AsyncMock(), []

    async def first(ack, body, *, client):
        seen.append(("first", client))
        await ack()

    async def second(ack, body, *, client):
        seen.append(("second", client))
        await ack()

    handle = ctx.register_slack_shortcut_handler("source", first)
    adapter._register_plugin_shortcut_handlers()
    native = adapter._app.shortcuts["source"]
    await native(ack, {"team": {"id": "T_B"}, "user": {"id": "U_B"}})
    await native(ack, {"team": {"id": "UNKNOWN"}})
    assert seen == [("first", exact)]
    handle.dispose()
    await native(ack, {"team": {"id": "T_B"}})
    ctx.register_slack_shortcut_handler("source", second)
    adapter._register_plugin_shortcut_handlers()
    assert adapter._app.shortcuts["source"] is native
    await native(ack, {"team": {"id": "T_B"}})
    assert seen == [("first", exact), ("second", exact)]


@pytest.mark.asyncio
async def test_reviewed_release_registers_real_native_actions_and_command(
    tmp_path, monkeypatch, tada_release_plugin_parent,
):
    monkeypatch.syspath_prepend(str(tada_release_plugin_parent))
    from company_knowledge_v1 import actions, commands, session, shortcuts

    manager = plugins.PluginManager(scope_key=str(tmp_path))
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    ctx = plugins.PluginContext(plugins.PluginManifest(name="company_knowledge_v1", source="user"), manager)
    session.configure(ctx)
    actions.register(ctx)
    commands.register(ctx)
    shortcuts.register(ctx)
    assert callable(manager._plugin_commands["tada"]["handler"])
    assert any(name == "message" for name, *_ in manager.get_slack_event_handlers())
    shortcut_names = {key for key, *_ in manager.get_slack_shortcut_handlers()}
    assert {shortcuts.SAVE_MEMORY, shortcuts.SUMMARIZE_THREAD} <= shortcut_names
    callback = next(callback for key, callback, _ in manager.get_slack_action_handlers() if key == actions.ACTION_ID)
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    client = SimpleNamespace(chat_postEphemeral=AsyncMock())
    adapter._team_clients = {"T_A": client}
    # A manually declared plugin slash remains routed when automatic manifest slots
    # have been filled by built-ins, which is the v0.21.5 production configuration.
    import hermes_cli.commands_platforms
    monkeypatch.setattr(hermes_cli.commands_platforms, "slack_native_slashes", lambda: [("hermes", "", "")])
    adapter._app = BoltTransport()
    adapter._wire_plugin_handlers = lambda _app: None
    adapter._register_bolt_handlers()
    assert any(pattern.fullmatch("/tada") for pattern in adapter._app.commands)
    ack = AsyncMock()
    await adapter._invoke_plugin_action_callback(callback, ack, {
        "team": {"id": "T_A"}, "user": {"id": "U_A"}, "channel": {"id": "D_A"},
    }, {"action_id": actions.ACTION_ID, "value": "invalid-nonce"})
    ack.assert_awaited_once()
    client.chat_postEphemeral.assert_awaited_once()
