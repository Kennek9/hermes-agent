"""Explicitly scoped erasure of JSON/JSONL archive copies; no content matching or path discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from hermes_state_messages import SessionMessagesMixin, _ERASED_CONTENT

_IDENTIFIERS = {"id", "session_id", "role", "from", "timestamp", "tool_call_id", "tool_name",
                "platform_message_id", "active", "compacted"}
_PAYLOAD_KEYS = {"content", "value", "api_content", "reasoning", "reasoning_content", "tool_calls"}


def _erase_tree(value, origin):
    if isinstance(value, list):
        children = [_erase_tree(item, origin) for item in value]
        return [item[0] for item in children], sum(item[1] for item in children), any(item[2] for item in children)
    if not isinstance(value, dict):
        return value, 0, False
    metadata = SessionMessagesMixin._decode_display_metadata(value.get("display_metadata")) or {}
    origins = SessionMessagesMixin._message_origins(metadata)
    if origins:
        if origin not in origins:
            return value, 0, False
        calls, kept = SessionMessagesMixin._cleared_message_values(value.get("tool_calls"), metadata)
        cleared = {key: item for key, item in value.items() if key in _IDENTIFIERS}
        cleared["display_metadata"] = json.loads(kept)
        if "content" in value:
            cleared["content"] = _ERASED_CONTENT
        if "value" in value:
            cleared["value"] = _ERASED_CONTENT
        if calls is not None:
            cleared["tool_calls"] = json.loads(calls)
        return cleared, 1, False
    children = {key: _erase_tree(item, origin) for key, item in value.items()}
    return ({key: item[0] for key, item in children.items()},
            sum(item[1] for item in children.values()),
            bool(_PAYLOAD_KEYS & value.keys()) or any(item[2] for item in children.values()))


def erase_archive_files(paths: list[Path], expected_origin: dict, *, whole_file_scope: bool = False) -> dict:
    """Erase only explicit caller-owned files; legacy payloads require whole-file confirmation.

    Validate every file before writing any. Each replacement is atomic. A filesystem failure raises,
    so the caller must keep a multi-store operation pending rather than claiming complete cleanup.
    """
    prepared = []
    for path in dict.fromkeys(Path(value) for value in paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError("archive_scope_changed")
        original = path.read_bytes()
        lines = original.splitlines(keepends=True) if path.suffix == ".jsonl" else [original]
        transformed = []
        changed = 0
        legacy = False
        for line in lines:
            if not line.strip():
                transformed.append(line)
                continue
            value, count, unmapped = _erase_tree(json.loads(line), expected_origin)
            changed += count
            legacy = legacy or unmapped
            transformed.append((json.dumps(value, ensure_ascii=False) + "\n").encode() if count else line)
        if legacy:
            if not whole_file_scope:
                raise ValueError("archive_scope_required")
            transformed = [b'{"contents_cleared":true}\n']
            changed += 1
        prepared.append((path, original, b"".join(transformed), changed, path.stat().st_mode & 0o777))
    for path, original, output, changed, mode in prepared:
        if not changed:
            continue
        if path.read_bytes() != original:
            raise ValueError("archive_scope_changed")
        descriptor, temporary = tempfile.mkstemp(prefix=".archive-erasure-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                os.chmod(temporary, mode)
                handle.write(output)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {"scope": "explicit_archive_files", "files_checked": len(prepared),
            "files_changed": sum(bool(item[3]) for item in prepared), "contents_cleared": True}
