"""Public Slack source reads preserve the real asynchronous SDK response contract."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from aiohttp import web
import pytest
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.web.async_slack_response import AsyncSlackResponse


def _context(tmp_path, monkeypatch, base_url):
    import gateway.run
    from gateway.config import Platform, PlatformConfig
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from hermes_cli.profiles import get_active_profile_name
    from hermes_constants import set_hermes_home_override
    from plugins.platforms.slack.adapter import SlackAdapter

    root = tmp_path / "hermes"
    home = root / "profiles" / "served"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "plugins:\n  entries:\n    slack-response-probe:\n"
        "      granted_capabilities: [gateway.platform_actions]\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    token = set_hermes_home_override(home)
    assert get_active_profile_name() == "served"
    loop = asyncio.get_running_loop()
    manager = PluginManager(scope_key=str(home))
    ctx = PluginContext(
        PluginManifest(name="slack-response-probe", source="user"), manager
    )
    client = AsyncWebClient(
        token="synthetic-served", base_url=base_url, retry_handlers=[]
    )
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    adapter._running = True
    adapter._team_clients = {"T_SERVED": client}
    runner = gateway.run.GatewayRunner.__new__(gateway.run.GatewayRunner)
    runner._primary_profile_name = "default"
    foreign = SlackAdapter(PlatformConfig(enabled=False))
    foreign._running = True
    foreign._team_clients = {
        "T_PRIMARY": AsyncWebClient(
            token="synthetic-primary", base_url=base_url, retry_handlers=[]
        )
    }
    runner.adapters = {Platform.SLACK: foreign}
    runner._profile_adapters = {"served": {Platform.SLACK: adapter}}
    runner._gateway_loop = loop
    monkeypatch.setattr(gateway.run, "_gateway_runner_ref", lambda: runner)
    return SimpleNamespace(
        ctx=ctx, manager=manager, client=client, home=home, token=token, loop=loop
    )


@asynccontextmanager
async def _http_api(payload):
    requests = []

    async def read(request):
        requests.append((request.method, request.path, dict(request.query)))
        return web.json_response(payload)

    app = web.Application()
    app.router.add_route("*", "/{method}", read)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        port = runner.addresses[0][1]
        yield f"http://127.0.0.1:{port}/", requests
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_real_sdk_response_survives_public_profile_and_loop_dispatch(
    tmp_path, monkeypatch, caplog
):
    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    payload = {"ok": True, "messages": [{"ts": "123.45", "text": "source"}]}
    caplog.set_level(logging.INFO, logger="hermes_cli.platform_actions")
    async with _http_api(payload) as (url, requests):
        runtime = _context(tmp_path, monkeypatch, url)
        original = runtime.client.api_call
        responses = []

        async def read(method, **kwargs):
            assert asyncio.get_running_loop() is runtime.loop
            assert get_hermes_home() == runtime.home
            response = await original(method, **kwargs)
            assert isinstance(response, AsyncSlackResponse)
            assert response.validate() is response
            responses.append(response)
            return response

        monkeypatch.setattr(runtime.client, "api_call", read)
        try:
            assert runtime.manager._task_loop is runtime.loop
            result = await asyncio.to_thread(
                runtime.ctx.run_async,
                runtime.ctx.call_slack_api(
                    "T_SERVED", "conversations.history", channel="C_SERVED"
                ),
            )
            assert result == payload
            assert requests == [
                ("GET", "/conversations.history", {"channel": "C_SERVED"})
            ]
            assert len(responses) == 1

            started, cancelled = asyncio.Event(), asyncio.Event()

            async def background_read():
                try:
                    assert (
                        await runtime.ctx.call_slack_api(
                            "T_SERVED", "conversations.history", channel="C_SERVED"
                        )
                        == payload
                    )
                    started.set()
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            task = await asyncio.to_thread(
                runtime.ctx.spawn_task, background_read(), name="sdk-read-probe"
            )
            await asyncio.wait_for(started.wait(), 5)
            assert len(requests) == 2
            for workspace in ("T_UNKNOWN", "T_PRIMARY"):
                assert await runtime.ctx.call_slack_api(
                    workspace, "conversations.history"
                ) == {"ok": False, "error": "workspace_unavailable"}
            assert await runtime.ctx.call_slack_api("T_SERVED", "chat.delete") == {
                "ok": False,
                "error": "unsupported_platform_action",
            }
            foreign = runtime.home.with_name("unserved")
            foreign.mkdir()
            (foreign / "config.yaml").write_bytes(
                (runtime.home / "config.yaml").read_bytes()
            )
            token = set_hermes_home_override(foreign)
            try:
                denied = await runtime.ctx.call_slack_api(
                    "T_PRIMARY", "conversations.history"
                )
                assert denied["error"] == "adapter_not_registered"
            finally:
                reset_hermes_home_override(token)
            (runtime.home / "config.yaml").write_text("plugins: {}\n")
            denied = await runtime.ctx.call_slack_api(
                "T_SERVED", "conversations.history"
            )
            assert denied["error"] == "capability_not_granted"
            assert len(requests) == len(responses) == 2
            assert runtime.manager.unload("slack-response-probe")
            await asyncio.wait_for(cancelled.wait(), 5)
            assert task.cancelled()
            assert any(
                "verb=conversations.history platform=slack ok=True"
                in record.getMessage()
                for record in caplog.records
            )
        finally:
            runtime.manager.unload()
            reset_hermes_home_override(runtime.token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    ["slack-error", "bytes", "none", "list", "missing-ok", "false-ok", "nonbool-ok"],
)
async def test_invalid_real_sdk_payloads_fail_without_leaking_details(
    tmp_path, monkeypatch, caplog, kind
):
    from hermes_constants import get_hermes_home, reset_hermes_home_override

    caplog.set_level(logging.INFO, logger="hermes_cli.platform_actions")
    async with _http_api({"ok": False, "error": "synthetic-private-error"}) as (
        url,
        requests,
    ):
        runtime = _context(tmp_path, monkeypatch, url)
        observed = []
        if kind == "slack-error":
            original = runtime.client.api_call

            async def read(method, **kwargs):
                assert asyncio.get_running_loop() is runtime.loop
                try:
                    return await original(method, **kwargs)
                except SlackApiError as error:
                    assert isinstance(error.response, AsyncSlackResponse)
                    observed.append(error.response)
                    raise

        else:
            payload = {
                "bytes": b"private-bytes",
                "none": None,
                "list": [],
                "missing-ok": {"messages": []},
                "false-ok": {"ok": False},
                "nonbool-ok": {"ok": 1},
            }[kind]
            # Fault injection keeps the malformed value; cast affects only its static type.
            response = AsyncSlackResponse(
                client=runtime.client,
                http_verb="GET",
                api_url=url,
                req_args={},
                data=cast("dict | bytes", payload),
                headers={},
                status_code=200,
            )

            async def read(method, **kwargs):
                assert asyncio.get_running_loop() is runtime.loop
                assert get_hermes_home() == runtime.home
                observed.append(response)
                return response

        monkeypatch.setattr(runtime.client, "api_call", read)
        try:
            result = await asyncio.to_thread(
                runtime.ctx.run_async,
                runtime.ctx.call_slack_api(
                    "T_SERVED", "conversations.history", channel="C_SERVED"
                ),
            )
            assert result == {"ok": False, "error": "slack_api_failed"}
            assert len(observed) == 1
            assert len(requests) == (1 if kind == "slack-error" else 0)
            messages = [record.getMessage() for record in caplog.records]
            assert any(
                "verb=conversations.history platform=slack ok=False" in message
                for message in messages
            )
            assert not any(
                "synthetic-private-error" in message or "private-bytes" in message
                for message in messages
            )
        finally:
            runtime.manager.unload()
            reset_hermes_home_override(runtime.token)
