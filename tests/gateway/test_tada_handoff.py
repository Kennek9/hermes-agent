"""The real Hermes compression path forwards bounded plugin context without a model call."""

from unittest.mock import MagicMock
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from tests.agent.test_pre_compress_memory_context_handoff import (
    _make_agent,
    _messages,
    _configure_engine_state,
)


def test_native_compaction_keeps_existing_memory_and_sanitized_handoff(monkeypatch):
    import hermes_cli.lifecycle as lifecycle

    original = lifecycle.invoke_hook

    def hook(name, **kwargs):
        if name == "context_handoff":
            assert kwargs["session_id"] == "test-session"
            return [
                {
                    "context": "UNRESOLVED: verify owner isolation; token=synthetic-private-value"
                }
            ]
        return original(name, **kwargs)

    monkeypatch.setattr(lifecycle, "invoke_hook", hook)
    manager = MagicMock()
    manager.on_pre_compress.return_value = "EXISTING MEMORY"
    compressor = MagicMock()
    seen = []

    def compress(messages, current_tokens=None, memory_context="", **kwargs):
        seen.append(memory_context)
        return [messages[0], messages[-1]]

    compressor.compress.side_effect = compress
    _configure_engine_state(compressor)
    agent = _make_agent(manager, compressor)
    agent._compress_context(_messages(), "sys", approx_tokens=100000, force=True)
    assert len(seen) == 1
    assert (
        "EXISTING MEMORY" in seen[0] and "UNRESOLVED: verify owner isolation" in seen[0]
    )


def test_handoff_failure_does_not_block_native_compression(monkeypatch):
    import hermes_cli.lifecycle as lifecycle

    original = lifecycle.invoke_hook

    def hook(name, **kwargs):
        if name == "context_handoff":
            raise RuntimeError("synthetic handoff outage")
        return original(name, **kwargs)

    monkeypatch.setattr(lifecycle, "invoke_hook", hook)
    compressor = MagicMock()
    compressor.compress.side_effect = lambda messages, **kwargs: [
        messages[0],
        messages[-1],
    ]
    _configure_engine_state(compressor)
    agent = _make_agent(None, compressor)
    result, _ = agent._compress_context(
        _messages(), "sys", approx_tokens=100000, force=True
    )
    assert len(result) == 2


def test_concurrent_same_session_handoffs_receive_the_exact_host_turn(monkeypatch):
    import hermes_cli.lifecycle as lifecycle

    original = lifecycle.invoke_hook
    rendezvous = Barrier(2)
    received = []

    def hook(name, **kwargs):
        if name != "context_handoff":
            return original(name, **kwargs)
        received.append(kwargs)
        rendezvous.wait(timeout=5)
        return [{"context": "Checkpoint: " + str(kwargs.get("turn_id"))}]

    monkeypatch.setattr(lifecycle, "invoke_hook", hook)

    def compress_turn(turn_id):
        context_seen = []
        compressor = MagicMock()

        def compress(messages, memory_context="", **kwargs):
            context_seen.append(memory_context)
            return [messages[0], messages[-1]]

        compressor.compress.side_effect = compress
        _configure_engine_state(compressor)
        agent = _make_agent(None, compressor)
        agent._current_turn_id = turn_id
        agent._parent_session_id = "parent-session"
        agent._user_id = "requesting-actor"
        agent._compress_context(_messages(), "sys", approx_tokens=100000, force=True)
        return context_seen

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(compress_turn, ("turn-a", "turn-b")))

    assert {values.get("turn_id") for values in received} == {"turn-a", "turn-b"}
    assert all(values["session_id"] == "test-session" for values in received)
    assert all(values["parent_session_id"] == "parent-session" for values in received)
    assert all(values["sender_id"] == "requesting-actor" for values in received)
    assert results == [["Checkpoint: turn-a"], ["Checkpoint: turn-b"]]
