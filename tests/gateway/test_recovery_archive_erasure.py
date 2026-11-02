"""Recovery writers preserve retired origins and independent profiles across late flushes."""

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("writer", ["spool", "history"])
def test_late_recovery_flush_keeps_retired_payloads_cleared_without_mutating_other_inputs(
    tmp_path, monkeypatch, writer,
):
    from gateway.shutdown_flush import flush_agent_history_to_file, spool_dropped_transcript_message
    from hermes_state import SessionDB

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    origin = {"session_id": "recovery-owned", "turn_id": "old-turn", "message_id": "old-input"}
    messages = [
        {"role": "assistant", "content": "RETIRED_RECOVERY_721",
         "display_metadata": {"_hermes_origin": origin}},
        {"role": "user", "content": "Preserve the new recovery input",
         "display_metadata": {"_hermes_origin": {**origin, "turn_id": "new-turn"}}},
    ]
    homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
    for index, home in enumerate(homes):
        home.mkdir()
        with SessionDB(home / "state.db") as database:
            database.create_session(origin["session_id"], "cli")
            database.append_messages_batch(origin["session_id"], messages)
            if index == 0:
                first = database.get_messages(origin["session_id"])[0]["id"]
                database.erase_message_payloads(origin["session_id"], [first], expected_origin=origin)
    original = json.dumps(messages, sort_keys=True)
    for home in (homes[0], homes[1], homes[0]):
        monkeypatch.setenv("HERMES_HOME", str(home))
        if writer == "spool":
            for message in messages:
                assert spool_dropped_transcript_message(origin["session_id"], message) is not None
        else:
            flush_agent_history_to_file(origin["session_id"], [*messages, "Opaque recovery entry"])
        files = list((home / "pending_messages").glob("*.json"))
        assert files
        contents = "".join(path.read_text() for path in files)
        assert ("RETIRED_RECOVERY_721" in contents) is (home == homes[1])
        assert "Preserve the new recovery input" in contents
        if writer == "history":
            assert "Opaque recovery entry" in contents
        assert json.dumps(messages, sort_keys=True) == original
