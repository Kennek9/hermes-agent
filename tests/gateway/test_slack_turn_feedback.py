"""Scoped routing, persisted participation and Slack acknowledgement clocks."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from gateway.execution_clock import reading, text_receipt, timings
from gateway.platforms.base import SendResult
from hermes_state import SessionDB
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.slack.turn_feedback import TurnFeedback


def test_thread_participation_survives_restart_and_expires(tmp_path):
    path = tmp_path / "state.db"
    key = ("T1", "C1", "123.45", "U1")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; from hermes_state import SessionDB; "
            "db=SessionDB(Path(sys.argv[1])); assert db.slack_thread_participation(*sys.argv[2:],refresh=True); db.close()",
            str(path),
            *key,
        ],
        check=True,
    )
    reopened = SessionDB(path)
    try:
        assert reopened.slack_thread_participation(*key)
        for changed in [
            ("T2", "C1", "123.45", "U1"),
            ("T1", "C2", "123.45", "U1"),
            ("T1", "C1", "123.46", "U1"),
            ("T1", "C1", "123.45", "U2"),
        ]:
            assert not reopened.slack_thread_participation(*changed)
        reopened._execute_write(
            lambda conn: conn.execute(
                "UPDATE slack_mentioned_threads SET last_active_at=0"
            )
        )
        assert not reopened.slack_thread_participation(*key)
    finally:
        reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("edit_ts", [None, "1004.500000"])
async def test_final_edit_does_not_reuse_progress_creation_timestamp(edit_ts):
    ingress = reading()
    ingress["tick"] -= 2.51
    posted = asyncio.Event()

    async def post(**kwargs):
        assert (kwargs["channel"], kwargs["thread_ts"]) == ("C", "900.000000")
        posted.set()
        return {"ok": True, "ts": "1002.500000"}

    client = NS(chat_postMessage=post, chat_delete=AsyncMock())
    adapter = NS(
        _react=AsyncMock(return_value=True),
        _reactions_enabled=lambda: True,
        _get_client=lambda *a, **kwargs: client,
        _outbound_blocked=lambda *a: None,
    )
    state = TurnFeedback(adapter, "T", "C", "900.000000", "1000.000000", ingress, True)
    await asyncio.wait_for(posted.wait(), 1)
    assert await state.finish_progress() == "1002.500000"
    actual = object.__new__(SlackAdapter)
    actual._outbound_blocked = lambda *a: None
    actual.format_message = lambda text: text
    actual.truncate_message = lambda text, limit: [text]
    actual._maybe_blocks = lambda content: None
    actual._clear_thread_status_quietly = AsyncMock()
    actual._call_with_block_fallback = AsyncMock(
        return_value={
            "ok": True,
            "ts": state.posted,
            "message": {"edited": {"ts": edit_ts}} if edit_ts else {},
        }
    )
    result = await actual.edit_message(
        "C", state.posted, "final", finalize=True, metadata={"slack_team_id": "T"}
    )
    assert result.success and result.slack_final_ts == edit_ts
    event = NS(
        _execution_received_clock=ingress,
        _slack_feedback=state,
        _slack_event_ts="1000.000000",
    )
    text_receipt(event, "final", result)
    clock = timings(event, confirmed=True)
    assert clock["ingress_to_ack_ms"] >= 0
    assert clock.get("slack_final_ts") == edit_ts
    assert clock["delivery_elapsed_ms"] >= 0
    state.finalized = True
    await state.close()
    client.chat_delete.assert_not_called()
    assert adapter._react.await_args.kwargs["remove"]


def test_new_final_post_uses_confirmed_slack_timestamp():
    event = NS(_execution_received_clock=reading(), _slack_event_ts="1000.000000")
    text_receipt(
        event,
        "final",
        SendResult(
            success=True, message_id="1005.000000", slack_final_ts="1005.000000"
        ),
    )
    assert timings(event, confirmed=True)["slack_final_ts"] == "1005.000000"
    assert "slack_final_ts" not in timings(event, confirmed=False)


@pytest.mark.asyncio
async def test_real_slack_sdk_edits_one_placeholder_and_clears_eyes():
    from aiohttp import web
    from slack_sdk.web.async_client import AsyncWebClient
    from gateway.config import PlatformConfig
    from gateway.platforms.event import ProcessingOutcome

    calls = []
    posted = asyncio.Event()
    loop = asyncio.get_running_loop()
    start = loop.time()

    async def api(request):
        payload = (
            await request.json()
            if request.content_type == "application/json"
            else dict(await request.post())
        )
        method = request.match_info["method"]
        calls.append((method, loop.time() - start, payload))
        if method == "chat.postMessage":
            posted.set()
        return web.json_response({
            "ok": True,
            "ts": "1002.500000",
            "message": {"edited": {"ts": "1004.000000"}},
        })

    app = web.Application()
    app.router.add_post("/api/{method}", api)
    server = web.AppRunner(app)
    await server.setup()
    try:
        site = web.TCPSite(server, "127.0.0.1", 0)
        await site.start()
        client = AsyncWebClient(
            base_url=f"http://127.0.0.1:{server.addresses[0][1]}/api/", token="loopback"
        )
        adapter = SlackAdapter(PlatformConfig(enabled=True, token="loopback"))
        adapter._app = NS(client=client)
        adapter._running = True
        ingress = reading()
        state = TurnFeedback(
            adapter, "T", "C", "999.000000", "1000.000000", ingress, True
        )
        event = NS(
            _execution_received_clock=ingress,
            _slack_feedback=state,
            _slack_event_ts="1000.000000",
        )
        adapter._turn_feedback["T", "C", "1000.000000"] = state
        await asyncio.wait_for(posted.wait(), 5)
        result = await adapter.send(
            "C",
            "Final evidence.",
            metadata={
                "slack_team_id": "T",
                "thread_id": "999.000000",
                "final_for_message_id": "1000.000000",
            },
        )
        assert result.success
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        text_receipt(event, "Final evidence.", result)
        observed = timings(event, confirmed=True)
        assert observed["slack_final_ts"] == "1004.000000"
        assert [m for m, _, _ in calls if m.startswith(("chat.", "reactions."))] == [
            "reactions.add",
            "chat.postMessage",
            "chat.update",
            "reactions.remove",
        ]
        assert calls[0][1] < 1
        assert 2.4 <= calls[1][1] <= 4
    finally:
        await server.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,history_size", [(401, 0), (400, 60), (500, 60)])
async def test_slack_exception_preserves_localized_notice_and_private_diagnostics(
    status, history_size
):
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._hmwa_stop_typing_for_turn = AsyncMock()
    source = NS(platform=Platform.SLACK)
    prepared = NS(history=[{}] * history_size, message_text=None)
    error = RuntimeError("private provider diagnostic")
    error.status_code = status
    reply = await runner._hmwa_agent_error_reply(
        error, NS(text="请核对"), source, None, "key", prepared
    )
    assert "请先核对结果" in reply
    assert "provider" not in reply and "/" not in reply
