"""Regression coverage for Feishu reply target selection."""

from types import SimpleNamespace

from gateway.platforms.base import _reply_anchor_for_event


def test_feishu_reply_targets_the_users_question() -> None:
    event = SimpleNamespace(
        message_id="om_user_question",
        reply_to_message_id="om_bot_previous_answer",
        source=SimpleNamespace(
            platform="feishu",
            thread_id="omt_existing_topic",
            chat_type="group",
        ),
        raw_message=None,
    )

    assert _reply_anchor_for_event(event) == "om_user_question"
