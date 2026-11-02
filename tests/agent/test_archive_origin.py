"""Archived message copies carry host provenance without mutating the API transcript."""

import copy
from types import SimpleNamespace

from agent.session_persistence import _db_flush_row
from hermes_state import SessionDB


def test_archive_rows_use_host_origin_and_preserve_unrelated_display_metadata(tmp_path):
    agent = SimpleNamespace(
        session_id="archive-origin", _current_turn_id="turn-one",
        _persist_user_message_platform_id="input-message",
    )
    messages = [
        {"role": "user", "content": "Review the project"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-one", "type": "function", "function": {"name": "read_file", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call-one", "content": "Project evidence"},
        {"role": "assistant", "content": "Project conclusion", "api_content": "Project conclusion (wire)",
         "display_metadata": {"badge": "review", "_hermes_origin": {"session_id": "forged"}}},
    ]
    original = copy.deepcopy(messages)
    rows = [_db_flush_row(agent, message, index == 0) for index, message in enumerate(messages)]
    assert messages == original
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session(agent.session_id, "cli")
        assert db.append_messages_batch(agent.session_id, rows) == len(messages)
        stored = db.get_messages(agent.session_id)
        assert [row["role"] for row in stored] == [message["role"] for message in messages]
        assert all(row["display_metadata"]["_hermes_origin"] == {
            "session_id": agent.session_id, "turn_id": agent._current_turn_id,
            "message_id": agent._persist_user_message_platform_id,
        } for row in stored)
        assert stored[-1]["display_metadata"]["badge"] == "review"
        assert stored[-1]["api_content"] == messages[-1]["api_content"]


def test_unattributed_archive_does_not_adopt_a_message_claimed_origin():
    row = _db_flush_row(SimpleNamespace(), {
        "role": "assistant", "content": "Imported note",
        "display_metadata": {"_hermes_origin": {"session_id": "claimed"}},
    }, False)
    assert row["display_metadata"] is None
