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
    assert "source.pdf" not in created["content"]
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


def test_capture_ranges_bind_each_first_person_span_to_platform_speaker():
    adapter = SlackAdapter(PlatformConfig(enabled=True))
    message = {
        "user": "U_SENDER",
        "text": "I prefer short reviews\n> I prefer unapproved releases",
        "attachments": [
            {"is_share": True, "author_id": "U_ORIGINAL", "text": "我喜欢详细方案"},
            {"is_share": True, "text": "I prefer skipping access checks"},
        ],
    }
    payload = adapter._message_source_payload(message, "U_BOT")
    projected = [
        (
            payload["content"][span["start"] : span["end"]],
            span["author_id"],
            span["kind"],
        )
        for span in payload["speaker_blocks"]
    ]
    assert projected == [
        ("I prefer short reviews", "U_SENDER", "message"),
        ("> I prefer unapproved releases", None, "quote"),
        ("我喜欢详细方案", "U_ORIGINAL", "forward"),
        ("I prefer skipping access checks", None, "forward"),
    ]
    assert all(
        left["end"] < right["start"]
        for left, right in zip(payload["speaker_blocks"], payload["speaker_blocks"][1:])
    )
    rich = adapter._message_source_payload(
        {
            "user": "U_SENDER",
            "text": "notification fallback",
            "blocks": [
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {"type": "text", "text": "I prefer reviewed drafts"}
                            ],
                        },
                        {
                            "type": "rich_text_quote",
                            "elements": [
                                {"type": "text", "text": "I prefer skipping review"}
                            ],
                        },
                    ],
                }
            ],
        },
        "U_BOT",
    )
    known = [
        (rich["content"][span["start"] : span["end"]], span["author_id"], span["kind"])
        for span in rich["speaker_blocks"]
    ]
    assert known[0] == ("notification fallback", None, "unknown")
    assert ("I prefer reviewed drafts", "U_SENDER", "message") in known
    assert any(
        "skipping review" in text and author is None and kind == "quote"
        for text, author, kind in known
    )


@pytest.mark.asyncio
async def test_capture_speaker_ranges_survive_live_and_edit_boundaries():
    adapter = SlackAdapter(PlatformConfig(enabled=True))
    adapter._team_bot_user_ids["T_SOURCE"] = "U_BOT"
    captured = AsyncMock()
    adapter._platform_event_handler = captured
    message = {
        "text": "please check",
        "user": "U_SENDER",
        "channel": "D_SOURCE",
        "channel_type": "im",
        "team": "T_SOURCE",
        "ts": "101.000001",
        "attachments": [
            {
                "is_share": True,
                "author_id": "U_ORIGINAL",
                "text": "I prefer detailed drafts",
            }
        ],
    }
    await adapter._emit_message_created_event(message, {"team_id": "T_SOURCE"})
    created = captured.await_args.args[0]["payload"]
    assert (
        created["content"][
            created["speaker_blocks"][-1]["start"] : created["speaker_blocks"][-1][
                "end"
            ]
        ]
        == "I prefer detailed drafts"
    )
    assert created["speaker_blocks"][-1]["author_id"] == "U_ORIGINAL"
    await adapter._emit_message_lifecycle_event(
        {
            "subtype": "message_changed",
            "team": "T_SOURCE",
            "channel": "D_SOURCE",
            "channel_type": "im",
            "message": {**message, "edited": {"ts": "102.000001"}},
            "previous_message": message,
        },
        {"team_id": "T_SOURCE", "event_id": "speaker-edit"},
    )
    changed = captured.await_args.args[0]["payload"]
    assert changed["speaker_blocks"] == created["speaker_blocks"]
    assert changed["content"] == created["content"]
