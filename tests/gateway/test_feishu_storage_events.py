"""Contract checks for Feishu chat-evidence platform events."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from gateway.config import PlatformConfig
_ADAPTER_MODULE = "plugins.platforms.feishu.adapter"
_ADAPTER_PATH = os.getenv("FEISHU_ADAPTER_PATH")
if _ADAPTER_PATH:
    _spec = importlib.util.spec_from_file_location(_ADAPTER_MODULE, _ADAPTER_PATH)
    if _spec is None or _spec.loader is None:
        raise RuntimeError("Unable to load FEISHU_ADAPTER_PATH")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_ADAPTER_MODULE] = _module
    _spec.loader.exec_module(_module)

from plugins.platforms.feishu.adapter import FeishuAdapter  # noqa: E402


def _sender() -> SimpleNamespace:
    return SimpleNamespace(
        sender_type="user",
        sender_id=SimpleNamespace(open_id="actor", user_id=None, union_id=None),
    )


def _message(*, message_id: str = "message", chat_type: str = "group", kind: str = "text") -> SimpleNamespace:
    content = {"text": "stored text"} if kind == "text" else {"image_key": "image"}
    return SimpleNamespace(
        message_id=message_id,
        chat_id="chat",
        chat_type=chat_type,
        message_type=kind,
        content=json.dumps(content),
        mentions=[],
        create_time="1725000000000",
        update_time="1725000001000",
        root_id=None,
        parent_id=None,
    )


def _envelope(message: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(event=SimpleNamespace(message=message, sender=_sender()))


def _adapter() -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig())
    adapter._seen_message_ids = {}
    adapter._seen_message_order = []
    adapter._dedup_cache_size = 100
    adapter._dedup_lock = threading.Lock()
    adapter._persist_seen_message_ids = lambda: None
    adapter._bot_open_id = "bot"
    adapter._bot_user_id = ""
    adapter._bot_name = "Hermes"
    return adapter


class FeishuStorageMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_message_emits_before_mention_admission(self) -> None:
        # Given: a visible group message that the reply path will reject.
        adapter = _adapter()
        adapter._admit = Mock(return_value="mention required")
        adapter._process_inbound_message = AsyncMock()
        emitted = AsyncMock()
        adapter._platform_event_handler = emitted

        # When: the shared p1/p2 message handler receives it.
        with patch("hermes_cli.lifecycle.has_hook", return_value=True):
            await adapter._handle_message_event_data(_envelope(_message()))

        # Then: storage sees it once while the reply path remains gated.
        self.assertEqual(emitted.await_count, 1)
        self.assertEqual(emitted.await_args.args[0]["event_type"], "message_received")
        adapter._process_inbound_message.assert_not_awaited()

    async def test_p2p_message_is_not_emitted(self) -> None:
        # Given: a direct message that remains eligible for normal reply handling.
        adapter = _adapter()
        adapter._admit = Mock(return_value=None)
        adapter._process_inbound_message = AsyncMock()
        emitted = AsyncMock()
        adapter._platform_event_handler = emitted

        # When: the shared message handler receives it.
        with patch("hermes_cli.lifecycle.has_hook", return_value=True):
            await adapter._handle_message_event_data(_envelope(_message(chat_type="p2p")))

        # Then: storage excludes it without changing the reply path.
        emitted.assert_not_awaited()
        adapter._process_inbound_message.assert_awaited_once()

    async def test_storage_normalization_does_not_download_media(self) -> None:
        # Given: a non-mentioned group image and a download seam that must stay untouched.
        adapter = _adapter()
        adapter._admit = Mock(return_value="mention required")
        adapter._process_inbound_message = AsyncMock()
        adapter._download_feishu_message_resources = AsyncMock(side_effect=AssertionError("downloaded"))
        emitted = AsyncMock()
        adapter._platform_event_handler = emitted

        # When: storage observes the inbound image.
        with patch("hermes_cli.lifecycle.has_hook", return_value=True):
            await adapter._handle_message_event_data(_envelope(_message(kind="image")))

        # Then: the envelope is emitted without fetching media.
        self.assertEqual(emitted.await_count, 1)
        adapter._download_feishu_message_resources.assert_not_awaited()

    async def test_missing_consumer_is_a_no_op(self) -> None:
        # Given: a valid group message with no gateway callback installed.
        adapter = _adapter()
        adapter._platform_event_handler = None

        # When: the producer is invoked directly.
        with patch("hermes_cli.lifecycle.has_hook", return_value=True):
            await adapter._emit_chat_evidence_message(_message(), _sender())

        # Then: it returns without dispatch or error.
        self.assertIsNone(adapter._platform_event_handler)

    async def test_storage_normalization_error_does_not_break_reply_path(self) -> None:
        # Given: storage normalization fails for an otherwise admissible message.
        adapter = _adapter()
        adapter._admit = Mock(return_value=None)
        adapter._process_inbound_message = AsyncMock()
        adapter._platform_event_handler = AsyncMock()

        # When: the shared handler receives the message.
        with (
            patch("hermes_cli.lifecycle.has_hook", return_value=True),
            patch(
                "plugins.platforms.feishu.adapter.normalize_feishu_message",
                side_effect=RuntimeError("normalization failed"),
            ),
        ):
            await adapter._handle_message_event_data(_envelope(_message()))

        # Then: the reply path still runs.
        adapter._process_inbound_message.assert_awaited_once()

    async def test_p1_and_p2_share_message_id_dedup(self) -> None:
        # Given: one logical message delivered once through each protocol.
        adapter = _adapter()
        adapter._admit = Mock(return_value=None)
        adapter._process_inbound_message = AsyncMock()
        emitted = AsyncMock()
        adapter._platform_event_handler = emitted
        captured: list[SimpleNamespace] = []
        adapter._on_message_event = captured.append
        adapter._on_legacy_message_event(
            SimpleNamespace(
                event={
                    "open_message_id": "same-message",
                    "open_chat_id": "chat",
                    "chat_type": "group",
                    "open_id": "actor",
                    "is_mention": True,
                    "text": "stored text",
                }
            )
        )

        # When: both normalized deliveries enter the shared handler.
        with patch("hermes_cli.lifecycle.has_hook", return_value=True):
            await adapter._handle_message_event_data(captured[0])
            await adapter._handle_message_event_data(
                _envelope(_message(message_id="same-message"))
            )

        # Then: storage and reply paths each process the shared identity once.
        self.assertEqual(emitted.await_count, 1)
        self.assertEqual(adapter._process_inbound_message.await_count, 1)


async def manual_scalar_qa() -> None:
    """Feed SDK-shaped p1 group/DM events and print scalar results only."""
    adapter = _adapter()
    adapter._loop = asyncio.get_running_loop()
    adapter._admit = Mock(return_value="mention required")
    adapter._process_inbound_message = AsyncMock()
    event_types: list[str] = []

    async def record(event: dict[str, object], _source: object) -> None:
        event_types.append(str(event["event_type"]))

    adapter._platform_event_handler = record
    scheduled: list[asyncio.Task[object]] = []
    adapter._submit_on_loop = (
        lambda _loop, coroutine: scheduled.append(asyncio.create_task(coroutine)) or True
    )
    with patch("hermes_cli.lifecycle.has_hook", return_value=True):
        for chat_type, message_id in (("group", "group-message"), ("private", "dm-message")):
            adapter._on_legacy_message_event(
                SimpleNamespace(
                    event={
                        "open_message_id": message_id,
                        "open_chat_id": "chat",
                        "chat_type": chat_type,
                        "open_id": "actor",
                        "is_mention": False,
                        "text": "stored text",
                    }
                )
            )
        await asyncio.gather(*scheduled)
    print(f"emitted_count={len(event_types)} event_types={','.join(sorted(event_types))}")


if __name__ == "__main__":
    unittest.main()
