"""Owner-directed payload erasure keeps unrelated transcript rows and protocol structure intact."""

import json
import os

import pytest

from hermes_state import SessionDB


def test_selected_payload_erasure_clears_values_and_fts_but_keeps_other_rows(tmp_path):
    origin = {"session_id": "owned", "turn_id": "turn-one", "message_id": "input-one"}
    value = "ERASURE_PAYLOAD_714"
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("owned", "cli")
        db.create_session("other", "cli")
        selected = [db.append_message("owned", role, value, **fields,
            display_metadata={"_hermes_origin": origin, "private_note": value}) for role, fields in (
            ("user", {"api_content": value + " wire"}),
            ("assistant", {"tool_calls": [{"id": "call-one", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": value}),
            }}], "reasoning_content": value, "reasoning_details": {"text": value}}),
            ("tool", {"tool_call_id": "call-one", "tool_name": "read_file"}),
            ("assistant", {"codex_reasoning_items": [{"text": value}], "codex_message_items": [{"text": value}]}),
        )]
        db.append_message("owned", "user", "Keep unrelated history")
        db.append_message("other", "user", "Keep another session")
        before = db.get_messages("owned")
        other = db.get_messages("other")
        assert db.search_messages(value)
        receipt = db.erase_message_payloads("owned", selected, expected_origin=origin)
        assert receipt["contents_cleared"] is True
        assert db.erase_message_payloads("owned", selected, expected_origin=origin) == receipt
        after = db.get_messages("owned")
        assert [(row["id"], row["role"]) for row in after] == [(row["id"], row["role"]) for row in before]
        assert after[-1] == before[-1]
        assert db.get_messages("other") == other
        assert after[2]["tool_call_id"] == after[1]["tool_calls"][0]["id"]
        assert after[1]["tool_calls"][0]["function"]["arguments"] == "{}"
        assert value not in json.dumps(after)
        assert not db.search_messages(value)


def test_old_transcript_replacement_cannot_restore_an_erased_origin(tmp_path):
    origin = {"session_id": "owned", "turn_id": "turn-one", "message_id": "input-one"}
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("owned", "cli")
        erased = db.append_message("owned", "user", "ERASED_REPLAY_715", display_metadata={"_hermes_origin": origin})
        db.append_message("owned", "assistant", "Keep a later origin", display_metadata={
            "_hermes_origin": {**origin, "turn_id": "turn-two", "message_id": "input-two"},
        })
        old = db.get_messages("owned")
        db.erase_message_payloads("owned", [erased], expected_origin=origin)
        db.replace_messages("owned", old)
        assert "ERASED_REPLAY_715" not in json.dumps(db.get_messages("owned", include_inactive=True))
        assert db.get_messages("owned")[-1]["content"] == "Keep a later origin"


def test_compacted_summary_retains_origins_and_is_cleared_with_its_sources(tmp_path):
    origin = {"session_id": "owned", "turn_id": "turn-one", "message_id": "input-one"}
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("owned", "cli")
        db.append_message("owned", "user", "SOURCE_FRAGMENT_716", display_metadata={"_hermes_origin": origin})
        db.archive_and_compact("owned", [{"role": "assistant", "content": "DERIVED_FRAGMENT_716", "_compressed_summary": True}])
        rows = db.get_messages("owned", include_inactive=True)
        summary = next(row for row in rows if row["content"] == "DERIVED_FRAGMENT_716")
        assert summary["display_metadata"]["_hermes_source_origins"] == [origin]
        selected = [row["id"] for row in rows]
        db.erase_message_payloads("owned", selected, expected_origin=origin)
        assert "FRAGMENT_716" not in json.dumps(db.get_messages("owned", include_inactive=True))


def test_partial_origin_selection_cannot_retire_an_unselected_sibling(tmp_path):
    origin = {"session_id": "owned", "turn_id": "turn-one", "message_id": "input-one"}
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("owned", "cli")
        first = db.append_message("owned", "user", "Shared input", display_metadata={"_hermes_origin": origin})
        db.append_message("owned", "assistant", "Unselected sibling", display_metadata={"_hermes_origin": origin})
        before = db.get_messages("owned")
        with pytest.raises(ValueError, match="message_selection_incomplete"):
            db.erase_message_payloads("owned", [first], expected_origin=origin)
        assert db.get_messages("owned") == before
        db.append_message("owned", "assistant", "Later valid append", display_metadata={"_hermes_origin": origin})
        assert db.get_messages("owned")[-1]["content"] == "Later valid append"


def test_active_turn_keeps_erasure_pending_without_changing_payloads(tmp_path):
    from hermes_state_errors import SessionTurnLeaseLostError

    origin = {"session_id": "owned", "turn_id": "turn-one", "message_id": "input-one"}
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("owned", "cli")
        message = db.append_message("owned", "user", "Keep while turn is active", display_metadata={"_hermes_origin": origin})
        holder = f"pid={os.getpid()}:turn=active"
        assert db.try_acquire_session_turn_lease("owned", holder, ttl_seconds=30)
        before = db.get_messages("owned")
        with pytest.raises(SessionTurnLeaseLostError):
            db.erase_message_payloads("owned", [message], expected_origin=origin)
        assert db.get_messages("owned") == before
        db.release_session_turn_lease("owned", holder)
        db.erase_message_payloads("owned", [message], expected_origin=origin)


@pytest.mark.parametrize("failure", ["wrong_session", "wrong_origin", "legacy"])
def test_payload_scope_rejection_is_atomic(tmp_path, failure):
    origin = {"session_id": "owned", "turn_id": "turn-one", "message_id": "input-one"}
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("owned", "cli")
        db.create_session("other", "cli")
        first = db.append_message("owned", "user", "Keep until all targets match", display_metadata={"_hermes_origin": origin})
        target = db.append_message("other" if failure == "wrong_session" else "owned", "assistant", "Keep selected data",
            display_metadata=None if failure == "legacy" else {"_hermes_origin": origin})
        before = db.get_messages("owned")
        expected = {**origin, "turn_id": "changed"} if failure == "wrong_origin" else origin
        with pytest.raises(ValueError, match="message_(?:scope_changed|origin_required)"):
            db.erase_message_payloads("owned", [first, target], expected_origin=expected)
        assert db.get_messages("owned") == before
