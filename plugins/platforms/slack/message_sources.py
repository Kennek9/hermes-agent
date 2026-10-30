"""Preserve forwarded Slack sources without attributing their words to the sender."""

from __future__ import annotations


def retained_attachments(attachments: object, bot_uid: str = "") -> list[dict]:
    """Only an attachment positively attributed to this bot is its own message echo."""
    return (
        [
            item
            for item in attachments
            if isinstance(item, dict)
            and not (
                bot_uid
                and item.get("author_id") == bot_uid
                and (item.get("is_msg_unfurl") or item.get("is_share"))
            )
        ]
        if isinstance(attachments, list)
        else []
    )


def message_files(message: dict, bot_uid: str = "") -> list[dict]:
    """Forwarded attachments carry files outside the message's top-level files array."""
    groups = [message.get("files", [])]
    groups.extend(
        item.get("files", [])
        for item in retained_attachments(message.get("attachments"), bot_uid)
    )
    files, seen = [], set()
    for group in groups:
        for item in group if isinstance(group, list) else []:
            if not isinstance(item, dict):
                continue
            file_id = item.get("id")
            if isinstance(file_id, str) and file_id in seen:
                continue
            if isinstance(file_id, str):
                seen.add(file_id)
            files.append(item)
    return files


def nested_message_blocks(message: dict, bot_uid: str = "") -> list[dict]:
    """Keep source identifiers only; Slack quote blocks do not identify their original author."""
    refs = []
    for item in retained_attachments(message.get("attachments"), bot_uid):
        if not (item.get("is_msg_unfurl") or item.get("is_share")):
            continue
        refs.append({
            "kind": "forward",
            "author_id": item.get("author_id")
            if isinstance(item.get("author_id"), str) and item.get("author_id")
            else None,
            "channel_id": item.get("channel_id")
            if isinstance(item.get("channel_id"), str) and item.get("channel_id")
            else None,
            "ts": (item.get("ts") or item.get("message_ts"))
            if isinstance(item.get("ts") or item.get("message_ts"), str)
            else None,
        })

    def has_quote(value: object) -> bool:
        if isinstance(value, dict):
            return value.get("type") == "rich_text_quote" or any(
                has_quote(child) for child in value.values()
            )
        return isinstance(value, list) and any(has_quote(child) for child in value)

    if has_quote(message.get("blocks")) or any(
        line.lstrip().startswith(">")
        for line in str(message.get("text") or "").splitlines()
    ):
        refs.append({
            "kind": "quote",
            "author_id": None,
            "channel_id": None,
            "ts": None,
        })
    return refs
