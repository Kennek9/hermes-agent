"""Slack edits/deletes reach the generic event boundary without changing chat admission."""
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


def event(kind="message_changed"):
    value = {
        "type": "message", "subtype": kind, "channel": "D_LOCAL", "channel_type": "im",
        "event_ts": "200.000001",
        "previous_message": {"user": "U_LOCAL", "ts": "123.456", "text": "old"},
    }
    if kind == "message_changed":
        value["message"] = {"user": "U_LOCAL", "ts": "123.456", "text": "new",
                            "edited": {"user": "U_LOCAL", "ts": "199.123456"}}
    else:
        value["deleted_ts"] = "123.456"
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["message_changed", "message_deleted"])
async def test_source_lifecycle_survives_chat_early_returns(kind):
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    adapter._processed_message_ts["123.456"] = True
    handler = AsyncMock()
    adapter.set_platform_event_handler(handler)
    adapter.handle_message = AsyncMock()
    value = event(kind)
    if kind == "message_deleted":
        value["user"] = "U_LOCAL"
        adapter._resolve_user_is_bot = AsyncMock(side_effect=AssertionError("no deletion probe"))
    await adapter._handle_slack_message(value, {"team_id": "T_LOCAL", "event_id": "Ev1"})
    handler.assert_awaited_once()
    normalized, source = handler.await_args.args
    assert source.scope_id == "T_LOCAL" and source.user_id == "U_LOCAL"
    assert normalized["payload"]["message_id"] == "123.456"
    assert normalized["payload"]["event_id"] == "Ev1"
    if kind == "message_changed":
        assert normalized["payload"]["occurred_at"] == "1970-01-01T00:03:19.123456+00:00"
    else:
        assert "content" not in normalized["payload"]
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["author", "workspace", "previous", "timestamp", "message_id", "previous_workspace"])
async def test_unverifiable_source_lifecycle_is_not_dispatched(problem):
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    adapter._processed_message_ts["123.456"] = True
    handler = AsyncMock()
    adapter.set_platform_event_handler(handler)
    value = event()
    if problem == "author":
        value["message"]["user"] = "U_OTHER"
    elif problem == "workspace":
        value["team"] = "T_OTHER"
    elif problem == "previous":
        del value["previous_message"]
    elif problem == "timestamp":
        value["event_ts"] = "invalid"
        value["message"]["edited"]["ts"] = "invalid"
    elif problem == "previous_workspace":
        value["previous_message"]["team"] = "T_OTHER"
    else:
        value["previous_message"]["ts"] = "124.456"
    await adapter._handle_slack_message(value, {"team_id": "T_LOCAL", "event_id": "Ev1"})
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_observer_failure_does_not_escape_adapter():
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    adapter._processed_message_ts["123.456"] = True
    adapter.set_platform_event_handler(AsyncMock(side_effect=RuntimeError("synthetic observer failure")))
    await adapter._handle_slack_message(event(), {"team_id": "T_LOCAL", "event_id": "Ev1"})


@pytest.mark.asyncio
async def test_revision_without_mention_does_not_probe_or_dispatch_unprocessed_chat():
    adapter = SlackAdapter(PlatformConfig(enabled=False))
    handler = AsyncMock()
    adapter.set_platform_event_handler(handler)
    adapter._resolve_user_is_bot = AsyncMock(side_effect=AssertionError("no external probe"))
    adapter.handle_message = AsyncMock(side_effect=AssertionError("no new chat"))
    await adapter._handle_slack_message(event(), {"team_id": "T_LOCAL", "event_id": "Ev1"})
    handler.assert_awaited_once()
    adapter._resolve_user_is_bot.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()
