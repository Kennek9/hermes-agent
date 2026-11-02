"""Explicitly scoped erasure of JSON/JSONL archive copies; no content matching or path discovery."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
import tempfile

from hermes_state_messages import SessionMessagesMixin, _ERASED_CONTENT

_IDENTIFIERS = {"id", "session_id", "role", "from", "timestamp", "tool_call_id", "tool_name",
                "platform_message_id", "active", "compacted"}
_PAYLOAD_KEYS = {"content", "value", "api_content", "reasoning", "reasoning_content", "tool_calls"}


def archive_profile_key(state_path: Path) -> str:
    """A file's namespace is the owning database path, without persisting that path."""
    return hashlib.sha256(json.dumps([
        str(state_path.parent.resolve()), str(state_path.resolve())]).encode()).hexdigest()


def _cleared_archive_value(value, metadata):
    calls, kept = SessionMessagesMixin._cleared_message_values(value.get("tool_calls"), metadata)
    cleared = {key: item for key, item in value.items() if key in _IDENTIFIERS}
    cleared["display_metadata"] = json.loads(kept)
    for key in ("_hermes_archive_profile", "_hermes_session_id"):
        if metadata.get(key) is not None:
            cleared["display_metadata"][key] = metadata[key]
    if "content" in value:
        cleared["content"] = _ERASED_CONTENT
    if "value" in value:
        cleared["value"] = _ERASED_CONTENT
    if calls is not None:
        cleared["tool_calls"] = json.loads(calls)
    return cleared


def prepare_archive_rows(messages, state_path: Path, *, session_id=None):
    """Apply known origin retirements before a fallback write, without mutating caller rows.

    A corrupt or unavailable current database cannot attest retirements. Preserve the fallback
    and its provenance for recovery; that condition is never a successful erasure receipt.
    """
    try:
        profile_key = archive_profile_key(state_path)
        with closing(sqlite3.connect(state_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("BEGIN")
            scoped = bool(session_id and connection.execute(
                "SELECT 1 FROM erased_message_origins WHERE origin_key=?",
                (SessionMessagesMixin._session_scope_key(session_id),)).fetchone())
            prepared = []
            for message in messages:
                if not isinstance(message, dict):
                    prepared.append(_ERASED_CONTENT if scoped else message)
                    continue
                metadata = SessionMessagesMixin._decode_display_metadata(
                    message.get("display_metadata")) or {}
                host_session = message.get("session_id") or metadata.get("_hermes_session_id")
                if (scoped or host_session and connection.execute(
                        "SELECT 1 FROM erased_message_origins WHERE origin_key=?",
                        (SessionMessagesMixin._session_scope_key(host_session),)).fetchone()) and metadata.get(
                            "_hermes_archive_profile", profile_key) == profile_key:
                    prepared.append(_cleared_archive_value(message, metadata))
                    continue
                retired = next((origin for origin in SessionMessagesMixin._message_origins(metadata)
                    if connection.execute("SELECT 1 FROM erased_message_origins WHERE origin_key=?",
                        (SessionMessagesMixin._origin_key(origin),)).fetchone()), None)
                prepared.append(_erase_tree(message, retired, profile_key)[0]
                                if retired else message)
            return prepared
    except (OSError, sqlite3.Error):
        return messages


def _erase_tree(value, origin, profile_key=None, strict_profile=False):
    if isinstance(value, list):
        children = [_erase_tree(item, origin, profile_key, strict_profile) for item in value]
        return [item[0] for item in children], sum(item[1] for item in children), any(item[2] for item in children)
    if not isinstance(value, dict):
        return value, 0, False
    metadata = SessionMessagesMixin._decode_display_metadata(value.get("display_metadata")) or {}
    origins = SessionMessagesMixin._message_origins(metadata)
    if origins:
        if origin not in origins:
            return value, 0, False
        owner = metadata.get("_hermes_archive_profile")
        if owner is not None and profile_key is not None and owner != profile_key:
            return value, 0, False
        if strict_profile and owner is None:
            return value, 0, True
        return _cleared_archive_value(value, metadata), 1, False
    children = {key: _erase_tree(item, origin, profile_key, strict_profile) for key, item in value.items()}
    return ({key: item[0] for key, item in children.items()},
            sum(item[1] for item in children.values()),
            bool(_PAYLOAD_KEYS & value.keys()) or any(item[2] for item in children.values()))


def erase_archive_files(paths: list[Path], expected_origin: dict, *, whole_file_scope: bool = False,
                        expected_profile_key: str | None = None) -> dict:
    """Erase only explicit caller-owned files; legacy payloads require whole-file confirmation.

    Validate every file before writing any. Each replacement is atomic. A filesystem failure raises,
    so the caller must keep a multi-store operation pending rather than claiming complete cleanup.
    """
    from hermes_constants import get_hermes_home
    profile_key = expected_profile_key or archive_profile_key(get_hermes_home() / "state.db")
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
            value, count, unmapped = _erase_tree(json.loads(line), expected_origin,
                                                profile_key, expected_profile_key is not None)
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
