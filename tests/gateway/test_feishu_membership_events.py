from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


class FeishuMembershipEventTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _adapter() -> FeishuAdapter:
        adapter = FeishuAdapter(PlatformConfig())
        adapter._bot_open_id = "bot"
        adapter._bot_user_id = ""
        adapter._bot_name = "Hermes"
        return adapter

    async def test_removal_emits_tenant_user_identity(self) -> None:
        adapter = self._adapter()
        adapter._loop = asyncio.get_running_loop()
        emitted = AsyncMock()
        adapter._platform_event_handler = emitted
        scheduled: list[asyncio.Task[object]] = []
        adapter._submit_on_loop = (
            lambda _loop, coroutine: scheduled.append(asyncio.create_task(coroutine)) or True
        )
        data = SimpleNamespace(
            event=SimpleNamespace(
                chat_id="chat",
                event_time="1725000000000",
                users=(
                    SimpleNamespace(
                        user_id=SimpleNamespace(
                            user_id="tenant-user",
                            open_id="open-user",
                        )
                    ),
                ),
            )
        )

        with patch("hermes_cli.lifecycle.has_hook", return_value=True):
            adapter._on_member_changed("member_removed", data)
            await asyncio.gather(*scheduled)

        emitted.assert_awaited_once()
        event = emitted.await_args.args[0]
        self.assertEqual(event["event_type"], "member_removed")
        self.assertEqual(event["payload"]["actor_id"], "tenant-user")

    def test_builder_registers_membership_events(self) -> None:
        adapter = self._adapter()
        builder = Mock()
        builder.register_p1_customized_event.return_value = builder
        for name in (
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
            getattr(builder, name).return_value = builder
        builder.build.return_value = object()
        dispatcher = Mock()
        dispatcher.builder.return_value = builder

        with patch.dict(
            FeishuAdapter._build_event_handler.__globals__,
            {"EventDispatcherHandler": dispatcher},
        ):
            adapter._build_event_handler()

        self.assertEqual(dispatcher.builder.call_count, 1)
        builder.register_p2_customized_event.assert_has_calls(
            [
                call("im.chat.member.user.added_v1", unittest.mock.ANY),
                call("im.chat.member.user.deleted_v1", unittest.mock.ANY),
                call("im.chat.member.user.withdrawn_v1", unittest.mock.ANY),
            ]
        )


if __name__ == "__main__":
    unittest.main()
