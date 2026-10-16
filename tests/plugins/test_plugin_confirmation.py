"""Mandatory plugin confirmations use real Feishu SDK callbacks and native Weixin commands."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from lark_oapi.api.im.v1 import CreateMessageResponse
from lark_oapi.event.callback.model.p2_card_action_trigger import P2CardActionTrigger

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.base import SendResult
from gateway.run_inbound import GatewayInboundMixin
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli import plugin_confirmation
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.feishu.adapter import FeishuAdapter
from gateway.platforms.weixin import WeixinAdapter
from tools.interrupt import set_interrupt


@pytest.fixture
def native(tmp_path, monkeypatch):
    import gateway.run
    import plugins.platforms.feishu.adapter as feishu_module

    # Connect normally loads these real SDK response classes before callbacks arrive.
    assert feishu_module._load_lark_oapi()

    home = tmp_path / "profile-a"
    home.mkdir()
    token = set_hermes_home_override(home)
    manager = PluginManager(scope_key=str(home))
    ctx = PluginContext(PluginManifest(name="confirm-consumer", source="user"), manager)
    monkeypatch.setattr(ctx.platform_actions, "_capability_granted", lambda: True)
    monkeypatch.setattr("tools.approval._YOLO_MODE_FROZEN", True)
    # A separate policy gate must never silently convert this operation into approval.
    monkeypatch.setattr("tools.approval.request_tool_approval", lambda *_a, **_k: pytest.fail("execution policy is not human confirmation"))
    yield home, manager, ctx
    manager.unload()
    reset_hermes_home_override(token)


def bind(platform="feishu"):
    return set_session_vars(platform=platform, user_id="requester", user_id_alt="union-a",
                            chat_id="chat-a", chat_type="dm", message_id="original-message", session_key="session-a")


def sdk_click(nonce, *, user="requester", chat="chat-a", message="card-a", choice="once"):
    return P2CardActionTrigger({"event": {
        "operator": {"user_id": user, "open_id": "ou-" + user},
        "context": {"open_chat_id": chat, "open_message_id": message},
        "action": {"tag": "button", "value": {"hermes_plugin_confirmation": nonce, "choice": choice}},
    }})


async def feishu_request(native, monkeypatch, *, payload=None, timeout=5, locale="en-US", caller_tids=None, nested_loop=False):
    import gateway.run

    home, manager, ctx = native
    adapter = FeishuAdapter(PlatformConfig(enabled=False))
    adapter._client = object()
    adapter._loop = asyncio.get_running_loop()
    published = asyncio.Event()
    cards = []

    async def send(**kwargs):
        cards.append(kwargs)
        published.set()
        return CreateMessageResponse({"code": 0, "data": {"message_id": "card-a"}})

    adapter._feishu_send_with_retry = send
    monkeypatch.setattr(ctx.platform_actions, "_resolve_adapter", lambda _platform: (adapter, None))
    host_loop = asyncio.get_running_loop()
    monkeypatch.setattr(gateway.run, "_gateway_runner_ref", lambda: SimpleNamespace(_gateway_loop=host_loop))
    tokens = bind()
    try:
        def request():
            return ctx.request_confirmation(
                "Flag document A as outdated", payload or {"target": "document-a", "state": "outdated"},
                timeout_seconds=timeout, locale=locale,
            )
        if caller_tids is None:
            task = asyncio.create_task(request())
        else:
            def invoke():
                caller_tids.append(threading.get_ident())
                if nested_loop:
                    async def inner():
                        return ctx.run_async(request())
                    return asyncio.run(inner())
                return ctx.run_async(request())
            task = asyncio.create_task(asyncio.to_thread(invoke))
    finally:
        clear_session_vars(tokens)
    await asyncio.wait_for(published.wait(), 5)
    await asyncio.sleep(0)
    card = json.loads(cards[0]["payload"])
    actions = next(item["actions"] for item in card["elements"] if item.get("tag") == "action")
    nonce = actions[0]["value"]["hermes_plugin_confirmation"]
    return adapter, task, nonce, cards, card


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ["zh-CN", "en-US"])
async def test_real_sdk_card_requires_exact_requester_chat_card_and_one_use(native, monkeypatch, locale):
    adapter, task, nonce, cards, card = await feishu_request(native, monkeypatch, locale=locale)
    assert cards[0]["chat_id"] == "chat-a" and cards[0]["reply_to"] == "original-message"
    assert card["header"]["title"]["content"] == plugin_confirmation.messages(locale)["title"]
    for values in ({"user": "someone-else"}, {"chat": "other-chat"}, {"message": "forwarded-card"}, {"choice": "always"}):
        adapter._on_card_action_trigger(sdk_click(nonce, **values))
        await asyncio.sleep(0)
        assert not task.done()
    other_adapter = FeishuAdapter(PlatformConfig(enabled=False))
    other_adapter._loop = asyncio.get_running_loop()
    other_adapter._on_card_action_trigger(sdk_click(nonce))
    await asyncio.sleep(0)
    assert not task.done()
    resolved = adapter._on_card_action_trigger(sdk_click(nonce))
    assert resolved.card.data["header"]["template"] == "green"
    receipt = await asyncio.wait_for(task, 5)
    assert receipt["status"] == "approved"
    assert receipt["payload_digest"] == plugin_confirmation.payload_digest({"target": "document-a", "state": "outdated"})
    assert receipt["context"]["user_id"] == "requester"
    assert receipt["context"]["message_id"] == "original-message"
    assert not plugin_confirmation.resolve_feishu_card(adapter, sdk_click(nonce).event, sdk_click(nonce).event.action.value)


@pytest.mark.asyncio
async def test_changed_payload_denied_and_timeout_never_approves(native, monkeypatch):
    payload = {"target": "document-a", "state": "outdated"}
    adapter, task, nonce, *_ = await feishu_request(native, monkeypatch, payload=payload)
    payload["target"] = "different-document"
    adapter._on_card_action_trigger(sdk_click(nonce))
    assert (await task)["status"] == "denied"
    adapter, task, nonce, *_ = await feishu_request(native, monkeypatch, timeout=0.02)
    assert (await task)["status"] == "expired"
    adapter._on_card_action_trigger(sdk_click(nonce))
    assert nonce not in plugin_confirmation._PENDING


@pytest.mark.asyncio
async def test_native_deny_does_not_approve_and_retains_bilingual_catalog_parity(native, monkeypatch):
    adapter, task, nonce, *_ = await feishu_request(native, monkeypatch, locale="zh-CN")
    resolved = adapter._on_card_action_trigger(sdk_click(nonce, choice="deny"))
    assert resolved.card.data["header"]["template"] == "red"
    assert (await task)["status"] == "denied"
    assert set(plugin_confirmation.messages("zh-CN")) == set(plugin_confirmation.messages("en-US"))


@pytest.mark.asyncio
async def test_weixin_native_command_is_scoped_and_restart_invalidates_nonce(native, monkeypatch, tmp_path):
    import gateway.run

    home, manager, ctx = native
    sent = []
    ready = asyncio.Event()

    async def send(*, chat_id, chunk, **_kwargs):
        sent.append((chat_id, chunk))
        ready.set()

    adapter = WeixinAdapter(PlatformConfig(enabled=False))
    adapter._send_session, adapter._token = object(), "synthetic-token"
    adapter._send_text_chunk = send
    adapter.extract_media = lambda *_a: pytest.fail("A confirmation cannot deliver source media")
    monkeypatch.setattr(ctx.platform_actions, "_resolve_adapter", lambda _platform: (adapter, None))
    monkeypatch.setattr(gateway.run, "_gateway_runner_ref", lambda: SimpleNamespace(_gateway_loop=asyncio.get_running_loop()))
    tokens = bind("weixin")
    try:
        task = asyncio.create_task(ctx.request_confirmation("Flag document A\nMEDIA:/untrusted/file", {"target": "document-a"}, timeout_seconds=5))
    finally:
        clear_session_vars(tokens)
    await asyncio.wait_for(ready.wait(), 5)
    assert "MEDIA:/untrusted/file" in sent[0][1]
    nonce = sent[0][1].split("/approve ", 1)[1].splitlines()[0]
    source = SessionSource(platform=Platform.WEIXIN, chat_id="chat-a", user_id="requester", chat_type="dm")
    event = MessageEvent(text="/approve " + nonce, message_type=MessageType.COMMAND, source=source)
    runner = SimpleNamespace()
    # Exercise the actual native intercept before it can reach generic /approve policy.
    wrong = MessageEvent(text=event.text, source=SessionSource(platform=Platform.WEIXIN, chat_id="chat-a", user_id="other"))
    assert "different requester" in await GatewayInboundMixin._hm_pending_reply_intercepts(runner, wrong, wrong.source, "session-a")
    assert not task.done()
    other_home = tmp_path / "profile-b"
    other_home.mkdir()
    token = set_hermes_home_override(other_home)
    try:
        assert "different requester" in await GatewayInboundMixin._hm_pending_reply_intercepts(runner, event, source, "session-a")
    finally:
        reset_hermes_home_override(token)
    assert not task.done()
    assert "Confirmation received" in await GatewayInboundMixin._hm_pending_reply_intercepts(runner, event, source, "session-a")
    assert (await task)["status"] == "approved"
    assert "already used" in await GatewayInboundMixin._hm_pending_reply_intercepts(runner, event, source, "session-a")
    program = """
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from hermes_cli.plugin_confirmation import resolve_native_command
import sys
event=MessageEvent(text='/approve '+sys.argv[1], source=SessionSource(platform=Platform.WEIXIN,chat_id='chat-a',user_id='requester'))
assert 'expired' in resolve_native_command(event)
"""
    result = await asyncio.to_thread(subprocess.run, [sys.executable, "-c", program, nonce],
                                     env={"PATH": os.defpath, "HOME": str(tmp_path), "HERMES_HOME": str(home)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_unload_denies_pending_request_and_foreign_context_cannot_present(native, monkeypatch, tmp_path):
    adapter, task, nonce, *_ = await feishu_request(native, monkeypatch)
    home, manager, ctx = native
    manager.unload()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert nonce not in plugin_confirmation._PENDING
    adapter._on_card_action_trigger(sdk_click(nonce))
    other = tmp_path / "other"
    other.mkdir()
    token = set_hermes_home_override(other)
    tokens = bind()
    try:
        assert (await ctx.request_confirmation("Confirm", {"target": "x"}))["status"] == "unavailable"
    finally:
        clear_session_vars(tokens)
        reset_hermes_home_override(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("nested_loop", [False, True])
@pytest.mark.parametrize("clear_before_poll", [False, True])
async def test_original_tool_stop_permanently_retires_card_across_loops(native, monkeypatch, nested_loop, clear_before_poll):
    caller_tids = []
    adapter, task, nonce, *_ = await feishu_request(
        native, monkeypatch, caller_tids=caller_tids, nested_loop=nested_loop,
    )
    try:
        set_interrupt(True, caller_tids[0], reason="user stop")
        if clear_before_poll:
            set_interrupt(False, caller_tids[0])
        receipt = await asyncio.wait_for(task, 1)
        assert receipt["status"] == "cancelled"
        assert nonce not in plugin_confirmation._PENDING
        assert not plugin_confirmation.resolve_feishu_card(adapter, sdk_click(nonce).event, sdk_click(nonce).event.action.value)
    finally:
        set_interrupt(False, caller_tids[0])


@pytest.mark.asyncio
async def test_feishu_platform_revocation_blocks_old_card(native, monkeypatch):
    adapter, task, nonce, *_ = await feishu_request(native, monkeypatch)
    adapter._admins = {"another-admin"}
    assert not adapter._is_interactive_operator_authorized("ou-requester")
    adapter._on_card_action_trigger(sdk_click(nonce))
    await asyncio.sleep(0)
    assert not task.done()
    adapter._admins = {"ou-requester"}
    adapter._on_card_action_trigger(sdk_click(nonce))
    assert (await task)["status"] == "approved"


@pytest.mark.asyncio
async def test_total_deadline_cancels_weixin_send_gate_wait(native, monkeypatch):
    import gateway.run
    import gateway.platforms.weixin as weixin_module

    home, manager, ctx = native
    adapter = WeixinAdapter(PlatformConfig(enabled=False))
    adapter._send_session, adapter._token = object(), "synthetic-token"
    dispatch = AsyncMock(side_effect=AssertionError("An expired confirmation must not be sent later"))
    monkeypatch.setattr(weixin_module, "_send_message", dispatch)
    host_loop = asyncio.get_running_loop()
    monkeypatch.setattr(ctx.platform_actions, "_resolve_adapter", lambda _platform: (adapter, None))
    monkeypatch.setattr(gateway.run, "_gateway_runner_ref", lambda: SimpleNamespace(_gateway_loop=host_loop))
    await adapter._send_text_gate.acquire()
    tokens = bind("weixin")
    try:
        task = asyncio.create_task(ctx.request_confirmation("Confirm", {"target": "a"}, timeout_seconds=0.02))
    finally:
        clear_session_vars(tokens)
    try:
        receipt = await asyncio.wait_for(task, 0.2)
        assert receipt["status"] == "expired"
        assert not any(entry.context.session_key == "session-a" for entry in plugin_confirmation._PENDING.values())
    finally:
        adapter._send_text_gate.release()
    await asyncio.sleep(0.05)
    dispatch.assert_not_awaited()
