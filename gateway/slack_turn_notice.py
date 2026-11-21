"""Employee-facing notices contain a local correlation ID, never operator diagnostics."""

import logging
import re
from uuid import uuid4

_MESSAGES = {
    "unavailable": (
        "这次回复未完成。若涉及操作，请先核对结果。",
        "This reply was interrupted. Check the outcome before repeating an action.",
    ),
    "empty": (
        "这次没有可发送的回答，请换个问法。",
        "I couldn't produce an answer. Please rephrase the question.",
    ),
    "history": (
        "暂时无法读取这段对话，请稍后重试。",
        "I couldn't read this conversation. Please try again later.",
    ),
    "timeout": (
        "这次处理超时，请缩小问题范围后重试。",
        "This took too long. Please narrow the question and try again.",
    ),
}


def notice(reason, authored=""):
    key = uuid4().hex[:8]
    logging.getLogger(__name__).warning(
        "Slack turn notice reason=%s ref=%s", reason, key
    )
    zh, en = _MESSAGES[reason]
    return (
        zh if re.search(r"[\u3400-\u9fff]", str(authored)[:1000]) else en
    ) + f" ({key})"
