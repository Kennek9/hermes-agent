"""Preserve forwarded Slack sources without attributing their words to the sender."""

from __future__ import annotations

from collections.abc import Callable


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


def speaker_projection(
    message: dict,
    bot_uid: str,
    render_text: Callable[[dict], str],
    render_attachment: Callable[[list, str], str],
) -> tuple[str, list[dict]]:
    """Bind ranges while assembling source parts, rather than matching repeated text later."""
    sender = message.get("user") if isinstance(message.get("user"), str) else None
    pieces: list[tuple[str, str, str | None, list[str | None]]] = []

    def append(
        text: str, kind: str, author: str | None, chain: list[str | None]
    ) -> None:
        lines: list[str] = []
        previous = None
        for line in text.strip().splitlines():
            quoted = kind == "quote" or line.lstrip().startswith(">")
            if previous is not None and quoted != previous:
                pieces.append((
                    "\n".join(lines),
                    "quote" if previous else kind,
                    None if previous else author,
                    chain + [None] if previous else chain,
                ))
                lines = []
            lines.append(line)
            previous = quoted
        if lines:
            pieces.append((
                "\n".join(lines),
                "quote" if previous else kind,
                None if previous else author,
                chain + [None] if previous else chain,
            ))

    blocks = message.get("blocks") or []
    has_rich_quote = any(
        element.get("type") == "rich_text_quote"
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "rich_text"
        for element in block.get("elements", [])
        if isinstance(element, dict)
    )
    if has_rich_quote:
        # Slack's notification fallback may flatten author boundaries; keep it as unknown
        # while the original structured parts retain their own source identities.
        append(str(message.get("text") or ""), "unknown", None, [sender, None])
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "rich_text":
                for element in block.get("elements", []):
                    text = render_text({
                        "text": "",
                        "blocks": [{"type": "rich_text", "elements": [element]}],
                    })
                    quote = element.get("type") == "rich_text_quote"
                    append(
                        text,
                        "quote" if quote else "message",
                        None if quote else sender,
                        [sender],
                    )
            else:
                append(
                    render_text({"text": "", "blocks": [block]}),
                    "message",
                    sender,
                    [sender],
                )
    else:
        append(
            render_text({**message, "attachments": [], "files": []}),
            "message",
            sender,
            [sender],
        )
    for item in retained_attachments(message.get("attachments"), bot_uid):
        author = (
            item.get("author_id")
            if isinstance(item.get("author_id"), str) and item["author_id"]
            else None
        )
        kind = (
            "forward"
            if item.get("is_share") or item.get("is_msg_unfurl")
            else "unknown"
        )
        append(render_attachment([item], bot_uid), kind, author, [sender, author])
    content, spans = "", []
    for text, kind, author, chain in pieces:
        if not text:
            continue
        if content:
            content += "\n"
        start = len(content)
        content += text
        spans.append({
            "start": start,
            "end": len(content),
            "kind": kind,
            "sender_id": sender,
            "author_id": author,
            "author_chain": chain,
        })
    return content, spans
