"""Owner-directed payload erasure keeps unrelated transcript rows and protocol structure intact."""

import json
import os

import pytest

from hermes_state import SessionDB


def test_selected_legacy_payload_cannot_return_from_cached_transcript(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("legacy-retired", "cli")
        first = db.append_message("legacy-retired", "user", "LEGACY_REPLAY_722")
        second = db.append_message("legacy-retired", "assistant", "LEGACY_REPLAY_722 answer")
        cached = db.get_messages("legacy-retired")
        later = db.append_message("legacy-retired", "user", "Keep the later confirmation",
            display_metadata={"_hermes_origin": {"session_id":"legacy-retired",
                "turn_id":"later-turn","message_id":"later-input"}})
        db.create_session("unselected-session", "cli")
        db.append_message("unselected-session", "user", "Keep the other session")
        untouched = db.get_messages("legacy-retired")
        with pytest.raises(ValueError, match="session_selection_incomplete"):
            db.erase_message_payloads("legacy-retired", [second], allow_legacy=True,
                retire_session_scope=True)
        assert db.get_messages("legacy-retired") == untouched
        receipt = db.erase_message_payloads("legacy-retired", [first, second], allow_legacy=True,
            retire_session_scope=True)
        assert db.erase_message_payloads("legacy-retired", [first, second], allow_legacy=True,
            retire_session_scope=True) == receipt
        assert "LEGACY_REPLAY_722" not in json.dumps(db.get_messages("legacy-retired"))
        before = db.get_messages("legacy-retired")
        for write in (
            lambda: db.append_messages_batch("legacy-retired", cached),
            lambda: db.replace_messages("legacy-retired", cached),
            lambda: db.replace_messages("legacy-retired", []),
            lambda: db.archive_and_compact("legacy-retired", cached),
            lambda: db.reopen_session("legacy-retired"),
        ):
            with pytest.raises(ValueError, match="session_scope_retired"):
                write()
            assert db.get_messages("legacy-retired") == before
        assert "LEGACY_REPLAY_722" not in json.dumps(db.get_messages("legacy-retired"))
        assert next(row for row in before if row['id']==later)['content']=='Keep the later confirmation'
        assert db.get_messages("unselected-session")[0]['content']=='Keep the other session'
        db.create_session("already-compressed", "cli")
        old = db.append_message("already-compressed", "user", "Old compressed source")
        db.end_session("already-compressed", "compression")
        db.erase_message_payloads("already-compressed", [old], allow_legacy=True,
            retire_session_scope=True)
        assert db.get_session("already-compressed")["end_reason"] == "compression"
        with pytest.raises(ValueError, match="session_scope_retired"):
            db.append_message("already-compressed", "assistant", "Refuse a stale compressed writer")


def test_retired_legacy_session_routes_new_input_to_fresh_history_and_neutralizes_late_files(
    tmp_path, monkeypatch,
):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.shutdown_flush import flush_agent_history_to_file, spool_dropped_transcript_message
    from hermes_state import divert_session_transcript_jsonl
    import hermes_state

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    source = SessionSource(platform=Platform.SLACK, scope_id="test-workspace",
        user_id="test-user", chat_id="test-dm", chat_type="dm")
    store = SessionStore(home / "sessions", GatewayConfig())
    first = store.get_or_create_session(source)
    store.close_all_db_handles()
    with SessionDB(home / "state.db") as db:
        assert db.get_session(first.session_id) is not None
        identity = db.append_message(first.session_id, "user", "LEGACY_RECOVERY_723")
        cached = db.get_messages(first.session_id)
        db.erase_message_payloads(first.session_id, [identity], allow_legacy=True,
            retire_session_scope=True)
    flush_agent_history_to_file(first.session_id, cached)
    assert spool_dropped_transcript_message(first.session_id, cached[0]) is not None
    diverted = divert_session_transcript_jsonl(first.session_id, cached)
    assert "LEGACY_RECOVERY_723" not in diverted.read_text()
    for path in (home / "pending_messages").glob("*.json"):
        assert "LEGACY_RECOVERY_723" not in path.read_text()
    new = store.get_or_create_session(source)
    assert new.session_id != first.session_id
    store.close_all_db_handles()
    with SessionDB(home / "state.db") as db:
        assert db.get_session(first.session_id)["end_reason"] == "owner_erasure"
        with pytest.raises(ValueError, match="session_scope_retired"):
            db.append_messages_batch(first.session_id, cached)
        db.append_message(new.session_id, "user", "Keep the fresh user input")
        assert db.get_messages(new.session_id)[0]["content"] == "Keep the fresh user input"


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
