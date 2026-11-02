"""Archived message copies carry host provenance without mutating the API transcript."""

import copy
import json
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


def test_real_debug_and_trajectory_exports_can_erase_their_recorded_input_origins(tmp_path, monkeypatch):
    from hermes_archive_erasure import erase_archive_files
    from run_agent import AIAgent

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    agent = object.__new__(AIAgent)
    agent.session_id = "export-owned"
    agent._current_turn_id = "turn-one"
    agent._persist_user_message_platform_id = "input-one"
    agent.tools = []
    agent.model = "synthetic-model"
    agent.save_trajectories = True
    agent.api_mode = "chat_completions"
    agent.client = SimpleNamespace(api_key=None)
    agent.base_url = "https://example.invalid"
    agent.suppress_status_output = True
    agent.log_prefix = ""
    agent.logs_dir = tmp_path / "home" / "sessions"
    agent.logs_dir.mkdir(parents=True)
    origin = {"session_id": agent.session_id, "turn_id": "turn-one", "message_id": "input-one"}
    messages = [{"role": "user", "content": "EXPORT_INPUT_719"},
                {"role": "assistant", "content": "EXPORT_INPUT_719 answer"}]
    original = copy.deepcopy(messages)
    with SessionDB() as db:
        agent._session_db = db
        db.create_session(agent.session_id, "cli")
        db.append_messages_batch(agent.session_id,
            [_db_flush_row(agent, row, index == 0) for index, row in enumerate(messages)])
        agent._current_turn_id = "turn-two"
        agent._persist_user_message_platform_id = "input-two"
        dump = agent._dump_api_request_debug({"messages": messages}, reason="synthetic failure")
        agent._save_trajectory(messages, "Current question", completed=True)
        trajectory = tmp_path / "trajectory_samples.jsonl"
        assert "EXPORT_INPUT_719" in dump.read_text()
        assert "EXPORT_INPUT_719" in trajectory.read_text()
        assert json.loads(dump.read_text())["display_metadata"]["_hermes_source_origins"] == [origin]
        assert json.loads(trajectory.read_text())["display_metadata"]["_hermes_source_origins"] == [origin]
        assert messages == original
        erase_archive_files([dump, trajectory], origin)
        assert "EXPORT_INPUT_719" not in dump.read_text() + trajectory.read_text()
        selected = [row['id'] for row in db.get_messages(agent.session_id)]
        db.erase_message_payloads(agent.session_id, selected, expected_origin=origin)
        later = agent._dump_api_request_debug({"messages": messages}, reason="late synthetic failure")
        agent._save_trajectory(messages, "Current question", completed=True)
        assert "EXPORT_INPUT_719" not in later.read_text() + trajectory.read_text()
        assert messages == original
