"""Forwarding carries content and provenance through live capture and thread rendering."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.slack.message_sources import message_files, nested_message_blocks


@pytest.mark.parametrize("flag", ["is_msg_unfurl", "is_share"])
@pytest.mark.parametrize("author", ["U_ORIGINAL", None, "", "U_BOT"])
def test_live_and_thread_projection_keep_other_and_unknown_authors(flag, author):
    attachment = {
        flag: True,
        "text": "supplier receipt is still missing",
        "files": [{"id": "F_FORWARD", "name": "receipt.pdf"}],
    }
    if author is not None:
        attachment["author_id"] = author
    message = {"text": "please check", "attachments": [attachment]}
    inbound = SlackAdapter._append_link_unfurls("please check", [attachment], "U_BOT")
    history = SlackAdapter._render_message_text(message, "U_BOT")
    assert ("supplier receipt is still missing" in inbound) == (author != "U_BOT")
    assert ("supplier receipt is still missing" in history) == (author != "U_BOT")
    assert ("receipt.pdf" in history) == (author != "U_BOT")
    assert len(message_files(message, "U_BOT")) == (0 if author == "U_BOT" else 1)
    refs = nested_message_blocks(message, "U_BOT")
    assert refs == (
        []
        if author == "U_BOT"
        else [
            {
                "kind": "forward",
                "author_id": author or None,
                "channel_id": None,
                "ts": None,
            }
        ]
    )


@pytest.mark.asyncio
async def test_created_and_edited_capture_preserve_forward_body_files_and_source_identifiers():
    adapter = SlackAdapter(PlatformConfig(enabled=True))
    adapter._team_bot_user_ids["T_SOURCE"] = "U_BOT"
    handler = AsyncMock()
    adapter._platform_event_handler = handler
    attachment = {
        "is_share": True,
        "author_id": "U_ORIGINAL",
        "channel_id": "C_ORIGINAL",
        "ts": "100.000001",
        "text": "Do not claim this is done.",
        "files": [{"id": "F_SOURCE", "name": "source.pdf"}],
    }
    message = {
        "text": "<@U_BOT> review this",
        "user": "U_FORWARDER",
        "channel": "D_SOURCE",
        "channel_type": "im",
        "team": "T_SOURCE",
        "ts": "101.000001",
        "attachments": [attachment],
    }
    await adapter._emit_message_created_event(message, {"team_id": "T_SOURCE"})
    assert handler.await_count == 1
    created = handler.await_args.args[0]["payload"]
    assert "Do not claim this is done." in created["content"]
    assert "<@U_BOT>" in created["content"]
    assert created["user_id"] == "U_FORWARDER"
    assert created["files"] == attachment["files"]
    assert created["nested_blocks"] == [
        {
            "kind": "forward",
            "author_id": "U_ORIGINAL",
            "channel_id": "C_ORIGINAL",
            "ts": "100.000001",
        }
    ]
    edited = {**message, "edited": {"ts": "102.000001"}}
    await adapter._emit_message_lifecycle_event(
        {
            "subtype": "message_changed",
            "team": "T_SOURCE",
            "channel": "D_SOURCE",
            "channel_type": "im",
            "message": edited,
            "previous_message": message,
        },
        {"team_id": "T_SOURCE", "event_id": "edited-event"},
    )
    changed = handler.await_args.args[0]["payload"]
    assert changed["event_kind"] == "edited"
    assert changed["content"] == created["content"]
    assert changed["nested_blocks"] == created["nested_blocks"]
    assert changed["files"] == created["files"]


@pytest.mark.parametrize(
    "message",
    [
        {"text": "> I approve the release"},
        {
            "text": "",
            "blocks": [
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_quote",
                            "elements": [{"type": "text", "text": "I approve"}],
                        }
                    ],
                }
            ],
        },
    ],
)
def test_quote_metadata_never_claims_the_forwarding_speaker(message):
    assert nested_message_blocks({**message, "user": "U_QUOTER"}) == [
        {"kind": "quote", "author_id": None, "channel_id": None, "ts": None}
    ]
