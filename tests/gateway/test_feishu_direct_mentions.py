"""Only a directed mention may start a mention-gated group conversation."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tests.gateway.feishu_helpers import (
    install_dedup_state,
    make_adapter_skeleton,
    make_message,
    make_sender,
)


def mention(open_id="ou_me", key="@_user_1", name="Tada"):
    return SimpleNamespace(
        key=key,
        id=SimpleNamespace(open_id=open_id, user_id=""),
        name=name,
    )


@pytest.mark.parametrize(
    "kind,content,mentions,admitted",
    [
        ("text", {"text": "@_all attention"}, [], False),
        ("text", {"text": "@_all attention"}, [mention("all", "@_all")], False),
        ("text", {"text": "@_user_1 question"}, [mention()], True),
        ("text", {"text": "@_user_1 question"}, [mention("ou_other")], False),
        ("text", {"text": "ordinary message"}, [], False),
        ("text", {"text": "@_all @_user_1 question"}, [mention()], True),
        (
            "post",
            {"zh_cn": {"content": [[{"tag": "at", "user_id": "all"}]]}},
            [],
            False,
        ),
        (
            "post",
            {"zh_cn": {"content": [[{"tag": "at", "user_id": "@_user_1"}]]}},
            [mention()],
            True,
        ),
    ],
)
def test_group_admission_requires_direct_mention(kind, content, mentions, admitted):
    adapter = make_adapter_skeleton(group_policy="open")
    adapter._bot_name = "Tada"
    message = make_message(chat_type="group", mentions=mentions)
    message.message_type = kind
    message.content = json.dumps(content)
    assert (adapter._admit(make_sender(), message) is None) is admitted


@pytest.mark.parametrize(
    "text,stripped,flag,admitted",
    [
        ("@_all announcement", "@_all announcement", True, False),
        ("  @_all announcement  ", "@_all announcement", True, False),
        ("@所有人 announcement", "@所有人 announcement", True, False),
        ("@Tada question", "question", True, True),
        ("@_all @Tada question", "@_all question", True, True),
        ("@Tada", None, True, True),
        ("@_all", None, True, False),
        ("@other", None, True, False),
        ("ordinary message", None, True, False),
        ("ordinary message", None, False, False),
    ],
)
def test_legacy_flag_alone_does_not_invent_a_bot_mention(
    text, stripped, flag, admitted
):
    adapter = make_adapter_skeleton(group_policy="open")
    adapter._bot_name = "Tada"
    event = {
        "open_message_id": "message",
        "open_chat_id": "chat",
        "chat_type": "group",
        "open_id": "ou_human",
        "is_mention": flag,
        "text": text,
    }
    if stripped is not None:
        event["text_without_at_bot"] = stripped
    with patch.object(adapter, "_on_message_event") as received:
        adapter._on_legacy_message_event(SimpleNamespace(event=event))
    normalized = received.call_args.args[0].event
    assert (adapter._admit(normalized.sender, normalized.message) is None) is admitted


def test_broadcast_is_stored_without_starting_a_reply():
    adapter = make_adapter_skeleton(group_policy="open")
    install_dedup_state(adapter)
    from gateway.config import Platform

    adapter.platform = Platform.FEISHU
    adapter._platform_event_handler = AsyncMock()
    adapter._process_inbound_message = AsyncMock()
    message = make_message(chat_type="group")
    message.content = json.dumps({"text": "@_all announcement"})
    with patch("hermes_cli.lifecycle.has_hook", return_value=True):
        asyncio.run(
            adapter._handle_message_event_data(
                SimpleNamespace(
                    event=SimpleNamespace(message=message, sender=make_sender())
                )
            )
        )
    adapter._platform_event_handler.assert_awaited_once()
    adapter._process_inbound_message.assert_not_awaited()


def test_direct_message_still_accepts_unmentioned_text():
    adapter = make_adapter_skeleton()
    assert adapter._admit(make_sender(), make_message()) is None


@pytest.mark.parametrize("key", ["@_all", "@_user_1"])
def test_broadcast_never_uses_bot_name_fallback_before_identity_hydration(key):
    adapter = make_adapter_skeleton(group_policy="open", bot_open_id="")
    adapter._bot_name = "Tada"
    message = make_message(chat_type="group", mentions=[mention("all", key)])
    message.content = json.dumps({"text": key + " announcement"})
    assert adapter._admit(make_sender(), message) is not None
