"""Regression coverage for legacy events sent by Lark's Channel protocol."""

import json
from importlib.util import find_spec
from types import SimpleNamespace
from unittest import SkipTest
from unittest.mock import Mock, patch

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


def test_event_handler_registers_legacy_channel_message() -> None:
    adapter = FeishuAdapter(PlatformConfig())
    builder = Mock()
    for method_name in (
        "register_p1_customized_event",
        "register_p2_im_message_message_read_v1",
        "register_p2_im_message_receive_v1",
        "register_p2_im_message_reaction_created_v1",
        "register_p2_im_message_reaction_deleted_v1",
        "register_p2_card_action_trigger",
        "register_p2_im_chat_member_bot_added_v1",
        "register_p2_im_chat_member_bot_deleted_v1",
        "register_p2_im_chat_access_event_bot_p2p_chat_entered_v1",
        "register_p2_im_message_recalled_v1",
        "register_p2_customized_event",
    ):
        getattr(builder, method_name).return_value = builder
    builder.build.return_value = SimpleNamespace()
    dispatcher = Mock()
    dispatcher.builder.return_value = builder

    with patch("plugins.platforms.feishu.adapter.EventDispatcherHandler", dispatcher):
        adapter._build_event_handler()

    builder.register_p1_customized_event.assert_any_call(
        "message",
        adapter._on_legacy_message_event,
    )


def test_legacy_channel_text_message_reaches_normal_handler() -> None:
    adapter = FeishuAdapter(PlatformConfig())
    adapter._bot_open_id = "ou_bot"
    adapter._bot_user_id = ""
    adapter._bot_name = "Hermes"
    legacy = SimpleNamespace(
        event={
            "type": "message",
            "open_message_id": "om_legacy",
            "open_chat_id": "oc_group",
            "chat_type": "group",
            "msg_type": "text",
            "open_id": "ou_user",
            "is_mention": True,
            "text": "@Hermes 介绍一下 Glorify Ring",
            "text_without_at_bot": "介绍一下 Glorify Ring",
        }
    )

    with patch.object(adapter, "_on_message_event") as on_message:
        adapter._on_legacy_message_event(legacy)

    normalized = on_message.call_args.args[0]
    assert normalized.event.message.message_id == "om_legacy"
    assert normalized.event.message.chat_id == "oc_group"
    assert normalized.event.message.chat_type == "group"
    assert normalized.event.message.content == '{"text": "介绍一下 Glorify Ring"}'
    assert normalized.event.message.mentions[0].id.open_id == "ou_bot"
    assert normalized.event.sender.sender_id.open_id == "ou_user"


def test_sdk_dispatcher_accepts_legacy_channel_message() -> None:
    if find_spec("lark_oapi") is None:
        raise SkipTest("lark-oapi is optional")

    from plugins.platforms.feishu.adapter import _load_lark_oapi

    assert _load_lark_oapi()
    adapter = FeishuAdapter(PlatformConfig())
    payload = json.dumps({
        "uuid": "legacy-event",
        "event": {
            "type": "message",
            "open_message_id": "om_legacy",
            "open_chat_id": "oc_group",
            "chat_type": "group",
            "msg_type": "text",
            "open_id": "ou_user",
            "is_mention": True,
            "text_without_at_bot": "介绍一下 Glorify Ring",
        },
    }).encode()

    with patch.object(adapter, "_on_message_event") as on_message:
        adapter._build_event_handler()._do_without_validation(payload)

    on_message.assert_called_once()
