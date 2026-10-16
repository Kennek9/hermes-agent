"""Build storage events without admitting replies or downloading message resources.

The adapter owns SDK normalization and dispatch. This module owns the knowledge
event contract, including identity precedence, retention bounds and event order.
It has no dependency on the gateway, model, network or knowledge database.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def evidence_timestamp(value: Any) -> datetime | None:
    """Convert a provider millisecond timestamp; invalid values have no timestamp."""
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


@dataclass(frozen=True)
class ChatEvidence:
    """Routing identity plus ordered events for one normalized message."""

    chat_id: str
    chat_type: str
    sender_id: str
    message_id: str
    events: tuple[dict[str, Any], ...]


def build_chat_evidence(message: Any, sender: Any, text: str) -> ChatEvidence | None:
    """Observe groups and direct document shares, independently of mention policy."""
    chat_type = str(getattr(message, "chat_type", "p2p") or "p2p").lower()
    chat_id = str(getattr(message, "chat_id", "") or "")
    message_id = str(getattr(message, "message_id", "") or "")
    sender_ref = getattr(sender, "sender_id", None)
    sender_id = getattr(sender_ref, "user_id", None) or getattr(
        sender_ref, "open_id", None
    )
    if not chat_id or not message_id or not sender_id:
        return None

    lowered = text.lower()
    direct_document = chat_type in {"p2p", "private"} and (
        any(host in lowered for host in ("feishu.cn/", "larksuite.com/"))
        and any(path in lowered for path in ("/docx/", "/docs/", "/wiki/", "/file/"))
    )
    if chat_type != "group" and not direct_document:
        return None
    event_at = evidence_timestamp(
        getattr(message, "update_time", None) or getattr(message, "create_time", None)
    ) or datetime.now(timezone.utc)
    sent_at = evidence_timestamp(getattr(message, "create_time", None)) or event_at
    events = []
    if direct_document:
        # Membership must reach the consumer before the document's message.
        events.append({
            "platform": "feishu",
            "event_type": "member_added",
            "payload": {
                "chat_id": chat_id[:128],
                "actor_id": str(sender_id)[:128],
                "event_at": event_at.isoformat(),
            },
        })
    events.append({
        "platform": "feishu",
        "event_type": "message_received",
        "payload": {
            "chat_id": chat_id[:128],
            "message_id": message_id[:128],
            "chat_type": chat_type,
            "sender_type": str(getattr(sender, "sender_type", "") or "") or None,
            "sender_id": str(sender_id)[:128],
            "sent_at": sent_at.isoformat(),
            "event_at": event_at.isoformat(),
            "text": text[:8192],
            "parent_message_id": getattr(message, "parent_id", None),
        },
    })
    return ChatEvidence(chat_id, chat_type, str(sender_id), message_id, tuple(events))
