"""Storage contracts stay independent from reply admission and SDK resources."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from plugins.platforms.feishu.chat_evidence import (
    build_chat_evidence,
    evidence_timestamp,
)


def message(**changes):
    fields = dict(
        chat_id="chat",
        message_id="message",
        chat_type="group",
        create_time="1725000000000",
        update_time="1725000001000",
        parent_id="parent",
    )
    return SimpleNamespace(**(fields | changes))


def sender(**changes):
    return SimpleNamespace(
        sender_id=SimpleNamespace(
            **({"user_id": "employee", "open_id": "ou_employee"} | changes)
        )
    )


@pytest.mark.parametrize("kind", ["group", "p2p", "private"])
def test_document_events_preserve_order_identity_and_parent(kind):
    result = build_chat_evidence(
        message(chat_type=kind), sender(), "https://tenant.feishu.cn/file/token"
    )
    assert result is not None
    assert result.sender_id == "employee"
    assert [event["event_type"] for event in result.events] == (
        ["message_received"]
        if kind == "group"
        else ["member_added", "message_received"]
    )
    payload = result.events[-1]["payload"]
    assert payload["parent_message_id"] == "parent"
    assert payload["sender_id"] == "employee"
    assert payload["sent_at"] < payload["event_at"]


def test_group_broadcast_is_bounded_but_kept_as_evidence():
    result = build_chat_evidence(message(), sender(user_id=None), "@all " + "x" * 9000)
    assert result is not None
    assert result.sender_id == "ou_employee"
    assert result.events[0]["payload"]["text"].startswith("@all ")
    assert len(result.events[0]["payload"]["text"]) == 8192


@pytest.mark.parametrize(
    "changes,text",
    [
        ({"chat_type": "p2p"}, "ordinary private chat"),
        ({"chat_id": ""}, "group text"),
        ({"message_id": ""}, "group text"),
    ],
)
def test_non_document_dm_and_incomplete_identity_do_not_emit(changes, text):
    assert build_chat_evidence(message(**changes), sender(), text) is None


@pytest.mark.parametrize("value", [None, "", "invalid", float("inf"), float("nan")])
def test_invalid_timestamp_is_absent(value):
    assert evidence_timestamp(value) is None


def test_timestamp_preserves_milliseconds_in_utc():
    assert evidence_timestamp("1001") == datetime(
        1970, 1, 1, 0, 0, 1, 1000, timezone.utc
    )
