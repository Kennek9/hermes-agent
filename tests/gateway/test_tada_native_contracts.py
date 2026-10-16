"""Offline native contracts; run via verify.py against an isolated patched Hermes tree."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from cron import jobs, scheduler
from gateway.config import PlatformConfig
from gateway.run_adapters import GatewayAdapterLifecycleMixin
from hermes_cli import plugins
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


@pytest.fixture
def manager(isolated_home, monkeypatch):
    value = plugins.PluginManager(scope_key=str(isolated_home))
    value._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: value)
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    yield value
    value.unload()


def context(manager, name="synthetic_team"):
    return plugins.PluginContext(
        plugins.PluginManifest(name=name, source="user"), manager
    )


def test_native_modules_come_from_selected_runtime():
    runtime = Path(__file__).resolve().parents[2]
    for module in (plugins, jobs, scheduler):
        assert Path(module.__file__).resolve().is_relative_to(runtime)


def test_profile_registrations_are_disposable_and_do_not_cross_homes(tmp_path, manager):
    other_home = tmp_path / "second-home"
    other_home.mkdir(mode=0o700)
    second = plugins.PluginManager(scope_key=str(other_home))
    first_context = context(manager)
    first_runner = lambda **_kwargs: (True, "first", "", None)
    second_runner = lambda **_kwargs: (True, "second", "", None)
    handle = first_context.register_cron_job_runner("same-name", first_runner)
    second_handle = context(second).register_cron_job_runner("same-name", second_runner)
    assert manager.get_cron_job_runner("same-name") is first_runner
    assert second.get_cron_job_runner("same-name") is second_runner
    with pytest.raises(ValueError):
        first_context.register_cron_job_runner("same-name", first_runner)
    handle.dispose()
    assert manager.get_cron_job_runner("same-name") is None
    assert second.get_cron_job_runner("same-name") is second_runner
    replacement = first_context.register_cron_job_runner("same-name", first_runner)
    handle.dispose()
    assert manager.get_cron_job_runner("same-name") is first_runner
    replacement.dispose()
    second_handle.dispose()
    second.unload()


def test_native_delivery_and_handoff_hooks_still_execute(manager):
    calls = []
    ctx = context(manager)
    ctx.register_hook("gateway_delivery", lambda **data: calls.append(data["outcome"]))
    ctx.register_hook("context_handoff", lambda session_id: {"context": session_id})
    manager.invoke_hook("gateway_delivery", outcome="confirmed")
    assert calls == ["confirmed"]
    assert manager.invoke_hook("context_handoff", session_id="actual-session") == [
        {"context": "actual-session"}
    ]


@pytest.mark.asyncio
async def test_home_events_stay_on_native_loop_and_exact_workspace(manager):
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    loop = asyncio.get_running_loop()
    client_a, client_b = object(), object()
    adapter._team_clients = {"T_A": client_a, "T_B": client_b}
    seen = []

    async def callback(*, event, client):
        assert asyncio.get_running_loop() is loop
        seen.append((event["user"], client))

    handle = context(manager).register_slack_event_handler("app_home_opened", callback)
    event = {"type": "app_home_opened", "tab": "home", "user": "U_B"}
    await adapter._handle_app_home_opened(event, {"team_id": "T_B"})
    await adapter._handle_app_home_opened(event, {"team_id": "UNKNOWN"})
    await adapter._handle_app_home_opened({**event, "team": "T_A"}, {"team_id": "T_B"})
    assert seen == [("U_B", client_b)]
    handle.dispose()
    await adapter._handle_app_home_opened(event, {"team_id": "T_B"})
    assert seen == [("U_B", client_b)]


@pytest.mark.asyncio
async def test_action_keeps_legacy_positional_contract_and_optional_exact_client():
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    exact = object()
    adapter._team_clients = {"T_B": exact}
    ack, seen = AsyncMock(), []

    async def legacy(_ack, _payload, _action):
        seen.append("legacy")
        await _ack()

    async def modern(ack, body, action, *, client):
        seen.append(client)
        await ack()

    await adapter._invoke_plugin_action_callback(
        legacy, ack, {"team": {"id": "T_B"}}, {}
    )
    await adapter._invoke_plugin_action_callback(
        modern, ack, {"team": {"id": "T_B"}}, {}
    )
    await adapter._invoke_plugin_action_callback(
        modern, ack, {"team": {"id": "UNKNOWN"}}, {}
    )
    assert seen == ["legacy", exact]
    assert ack.await_count == 3


def revision(kind="message_changed"):
    value = {
        "type": "message",
        "subtype": kind,
        "channel": "C_LOCAL",
        "channel_type": "channel",
        "event_ts": "200.000001",
        "previous_message": {
            "user": "U_LOCAL",
            "ts": "123.456",
            "text": "old",
            "thread_ts": "100.001",
        },
    }
    if kind == "message_changed":
        value["message"] = {
            "user": "U_LOCAL",
            "ts": "123.456",
            "text": "new",
            "thread_ts": "100.001",
            "edited": {"ts": "199.123456"},
        }
    else:
        value["deleted_ts"] = "123.456"
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["message_changed", "message_deleted"])
@pytest.mark.parametrize("authorized", [True, False])
async def test_revision_passes_native_gateway_authorization_and_never_starts_chat(
    manager, kind, authorized
):
    seen = []
    context(manager).register_hook(
        "gateway_platform_event", lambda **event: seen.append(event)
    )
    runner = SimpleNamespace(_is_user_authorized_for_source=lambda source: authorized)
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    adapter._processed_message_ts["123.456"] = 1
    adapter._resolve_user_is_bot = AsyncMock(
        side_effect=AssertionError("no SDK probe for revisions")
    )
    adapter.handle_message = AsyncMock(
        side_effect=AssertionError("revision is not a new chat")
    )

    async def dispatch(event, source):
        assert source.scope_id == "T_LOCAL" and source.user_id == "U_LOCAL"
        await GatewayAdapterLifecycleMixin._handle_gateway_platform_event(
            runner, event, source
        )

    adapter.set_platform_event_handler(dispatch)
    await adapter._handle_slack_message(
        revision(kind), {"team_id": "T_LOCAL", "event_id": "Ev1"}
    )
    assert bool(seen) is authorized
    if authorized:
        assert seen[0]["payload"]["thread_root_ts"] == "100.001"
        assert seen[0]["payload"]["message_id"] == "123.456"
        if kind == "message_deleted":
            assert "content" not in seen[0]["payload"]
    adapter.handle_message.assert_not_awaited()
    adapter._resolve_user_is_bot.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "problem", ["author", "workspace", "previous", "time", "message"]
)
async def test_unverifiable_revision_never_reaches_observer(problem):
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    adapter._processed_message_ts["123.456"] = 1
    observer = AsyncMock()
    adapter.set_platform_event_handler(observer)
    value = revision()
    if problem == "author":
        value["message"]["user"] = "U_OTHER"
    elif problem == "workspace":
        value["team"] = "T_OTHER"
    elif problem == "previous":
        value.pop("previous_message")
    elif problem == "time":
        value["message"]["edited"]["ts"] = "invalid"
        value["event_ts"] = "invalid"
    else:
        value["previous_message"]["ts"] = "124.456"
    await adapter._handle_slack_message(value, {"team_id": "T_LOCAL"})
    observer.assert_not_awaited()


@pytest.mark.asyncio
async def test_unmentioned_message_is_observed_before_native_strict_mention_gate():
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    adapter._bot_user_id = "U_BOT"
    adapter._team_bot_user_ids = {"T_A": "U_BOT"}
    adapter._resolve_user_is_bot = AsyncMock(return_value=False)
    adapter.config.extra.update({"require_mention": True, "strict_mention": True})
    observer = AsyncMock()
    adapter.set_platform_event_handler(observer)
    adapter.handle_message = AsyncMock(
        side_effect=AssertionError("unmentioned is not a new chat")
    )
    await adapter._handle_slack_message(
        {
            "type": "message",
            "user": "U_A",
            "channel": "C_A",
            "channel_type": "channel",
            "ts": "1790570000.000001",
            "text": "original discussion",
            "client_msg_id": "synthetic",
        },
        {"team_id": "T_A"},
    )
    observer.assert_awaited_once()
    assert observer.await_args.args[0]["payload"]["event_kind"] == "created"
    adapter.handle_message.assert_not_awaited()


def test_plugin_runner_missing_invalid_failure_and_copied_job(manager, monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "_prepare_job_prompt",
        lambda *_a: pytest.fail("Plugin jobs must not enter the agent path"),
    )
    job = {
        "id": "synthetic",
        "plugin_runner": "missing",
        "plugin_data": {"revision": 1},
    }
    assert scheduler.run_job(job)[3] == "plugin_runner_unavailable"
    ctx = context(manager)

    def callback(*, job, cancel_event=None):
        job["plugin_data"]["revision"] = 2
        return True, "audit", "", None

    handle = ctx.register_cron_job_runner("team", callback)
    job["plugin_runner"] = "team"
    assert scheduler.run_job(job) == (True, "audit", "", None)
    assert job["plugin_data"]["revision"] == 1
    handle.dispose()
    bad = ctx.register_cron_job_runner("team", lambda **_kw: (True,))
    assert scheduler.run_job(job)[3] == "plugin_runner_invalid"
    bad.dispose()

    def fail(**_kwargs):
        raise ValueError("synthetic-private-value")

    ctx.register_cron_job_runner("team", fail)
    failure = scheduler.run_job(job)
    assert failure[3] == "plugin_runner_failed"
    assert "synthetic-private-value" not in str(failure)


def test_plugin_job_is_created_paused_without_changing_native_paused_contract(manager):
    job = jobs.create_job(
        prompt="read-only",
        schedule="every 5m",
        plugin_runner="team",
        plugin_data={"task_id": "bound", "revision": 1},
        enabled=False,
        deliver="local",
    )
    assert (
        not job["enabled"] and job["state"] == "paused" and job["next_run_at"] is None
    )
    assert not jobs.claim_job_for_fire(job["id"])
    assert jobs.get_job(job["id"])["plugin_data"] == {"task_id": "bound", "revision": 1}
    ordinary = jobs.create_job(prompt="ordinary", schedule="every 5m", paused=True)
    assert not ordinary["enabled"] and ordinary["next_run_at"] is None
    assert "plugin_runner" not in ordinary and "plugin_data" not in ordinary


@pytest.mark.parametrize(
    "base,expr,zone,expected",
    [
        (
            "2026-09-28T00:00:00+00:00",
            "0 9 * * *",
            "Asia/Shanghai",
            "2026-09-28T09:00:00+08:00",
        ),
        (
            "2026-03-07T02:30:00-05:00",
            "30 2 * * *",
            "America/New_York",
            "2026-03-09T02:30:00-04:00",
        ),
        (
            "2026-11-01T01:30:00-04:00",
            "30 1 * * *",
            "America/New_York",
            "2026-11-02T01:30:00-05:00",
        ),
    ],
)
def test_explicit_zone_skips_gap_and_does_not_repeat_fold(base, expr, zone, expected):
    assert (
        jobs.compute_next_run({"kind": "cron", "expr": expr, "timezone": zone}, base)
        == expected
    )


def test_native_implicit_zone_fold_behavior_is_unchanged(monkeypatch):
    monkeypatch.setattr(jobs, "get_timezone", lambda: ZoneInfo("America/New_York"))
    assert (
        jobs.compute_next_run(
            {"kind": "cron", "expr": "*/5 1 * * *"}, "2026-11-01T01:45:00-05:00"
        )
        == "2026-11-01T01:50:00-05:00"
    )


@pytest.mark.parametrize("changed", [None, "runner", "revision"])
def test_native_claim_snapshot_preserves_occurrence_and_refuses_changed_task(
    monkeypatch, changed
):
    original = {
        "id": "synthetic",
        "execution_id": "execution-1",
        "plugin_runner": "team",
        "plugin_data": {"revision": 1},
        "_scheduled_instant": "2026-10-16T01:00:00Z",
        "next_run_at": "2026-10-17T01:00:00Z",
    }
    claimed = {**original, "plugin_data": {"revision": 1}}
    if changed == "runner":
        claimed["plugin_runner"] = "other"
    elif changed == "revision":
        claimed["plugin_data"]["revision"] = 2
    seen, finished = [], []
    monkeypatch.setattr(scheduler, "claim_job_for_fire", lambda *_a, **_kw: claimed)
    monkeypatch.setattr(
        scheduler, "finish_execution", lambda *_a, **kw: finished.append(kw)
    )
    monkeypatch.setattr(
        scheduler, "run_one_job", lambda job, **_kw: seen.append(job) or True
    )
    assert scheduler._process_due_job(original, None, None, False) is True
    if changed:
        assert not seen and finished[0]["success"] is False
    else:
        assert (
            not finished and seen[0]["scheduled_for"] == original["_scheduled_instant"]
        )
        assert seen[0]["execution_id"] == "execution-1"


@pytest.mark.asyncio
async def test_sdk_maintenance_reaches_only_revocation_when_chat_actor_is_unauthorized(
    manager, monkeypatch, tada_release_plugin_parent
):
    monkeypatch.syspath_prepend(str(tada_release_plugin_parent))
    authority = importlib.import_module("company_knowledge_v1.slack_authority")
    calls = []

    def runner(module, request):
        assert module == "company_docs_mcp.transport.slack_authority_bridge"
        assert request["operation"] == "revoke"
        assert request["protocol_version"] == 2
        assert request["workspace_id"] == request["token_workspace_id"] == "T_A"
        calls.append(request)
        return {"ok": True, "data": {}}

    monkeypatch.setattr(
        authority.services,
        "delivery_config",
        lambda **_kw: {"workspaces": {"T_A": {"bot_user_id": "U_BOT"}}},
    )
    monkeypatch.setattr(authority.services, "bridge_runner", runner)
    monkeypatch.setattr(
        authority,
        "_api",
        AsyncMock(side_effect=AssertionError("Revocation must not wait for a Slack API probe")),
    )
    authority.register(context(manager))
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    adapter._team_clients = {"T_A": object()}
    chat_dispatch = AsyncMock(
        side_effect=AssertionError("maintenance must not become a chat")
    )
    adapter.set_platform_event_handler(chat_dispatch)
    # A retired or departed actor could never pass the ordinary gateway chat gate.
    denied_runner = SimpleNamespace(
        _is_user_authorized_for_source=lambda _source: False
    )
    assert denied_runner._is_user_authorized_for_source(None) is False
    await adapter._dispatch_plugin_slack_event(
        {"type": "member_left_channel", "user": "U_RETIRED", "channel": "C_A"},
        {"team_id": "T_A"},
    )
    await adapter._dispatch_plugin_slack_event(
        {
            "type": "user_change",
            "user": {"id": "U_RETIRED", "team_id": "T_A", "deleted": True},
        },
        {"team_id": "T_A"},
    )
    assert [value["input"]["kind"] for value in calls] == ["membership", "user"]
    assert all(value["workspace_id"] == "T_A" for value in calls)
    await adapter._dispatch_plugin_slack_event(
        {
            "type": "user_change",
            "user": {"id": "U_ACTIVE", "team_id": "T_A", "deleted": False},
        },
        {"team_id": "T_A"},
    )
    await adapter._dispatch_plugin_slack_event(
        {
            "type": "member_left_channel",
            "team": "T_OTHER",
            "user": "U_RETIRED",
            "channel": "C_A",
        },
        {"team_id": "T_A"},
    )
    assert len(calls) == 2
    chat_dispatch.assert_not_awaited()
