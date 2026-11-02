"""Explicit archive files preserve unrelated bytes and never guess legacy ownership."""

import json

import pytest

from hermes_archive_erasure import erase_archive_files


def test_archive_file_erasure_clears_exact_origin_and_preserves_other_jsonl_bytes(tmp_path):
    origin = {"session_id": "owned", "turn_id": "turn-one", "message_id": "input-one"}
    selected = {"role": "assistant", "content": "FILE_PAYLOAD_717", "api_content": "FILE_PAYLOAD_717",
                "display_metadata": {"_hermes_origin": origin}}
    unselected = b'{"role":"user", "content":"Keep formatting", "display_metadata":{"_hermes_origin":{"session_id":"owned","turn_id":"turn-two","message_id":"input-two"}}}\n'
    path = tmp_path / "owned.jsonl"
    path.write_bytes(json.dumps(selected).encode() + b"\n" + unselected)
    untouched = tmp_path / "other.jsonl"
    untouched.write_bytes(unselected)
    receipt = erase_archive_files([path, untouched], origin)
    assert receipt["files_changed"] == 1
    assert b"FILE_PAYLOAD_717" not in path.read_bytes()
    assert path.read_bytes().splitlines(keepends=True)[-1] == unselected
    assert untouched.read_bytes() == unselected


def test_unmapped_file_requires_explicit_scope_and_no_earlier_file_is_changed(tmp_path):
    origin = {"session_id": "owned", "turn_id": "turn-one", "message_id": "input-one"}
    mapped = tmp_path / "mapped.json"
    mapped.write_text(json.dumps({"role": "assistant", "content": "Mapped value", "display_metadata": {"_hermes_origin": origin}}))
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"conversations": [{"from": "human", "value": "Unmapped value"}]}))
    before = mapped.read_bytes(), legacy.read_bytes()
    with pytest.raises(ValueError, match="archive_scope_required"):
        erase_archive_files([mapped, legacy], origin)
    assert (mapped.read_bytes(), legacy.read_bytes()) == before
    erase_archive_files([legacy], origin, whole_file_scope=True)
    assert json.loads(legacy.read_text()) == {"contents_cleared": True}


def test_explicit_file_scope_does_not_expand_to_another_file(tmp_path):
    origin = {"session_id": "owned", "turn_id": "turn-one", "message_id": "input-one"}
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps({"role": "assistant", "content": "Legacy selected content"}))
    other = tmp_path / "unselected.json"
    other.write_text(json.dumps({"role": "assistant", "content": "Keep another file"}))
    before = other.read_bytes()
    erase_archive_files([selected], origin, whole_file_scope=True)
    assert other.read_bytes() == before
