"""The same native agent can run a short and a deep turn without sharing the cap."""

import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    ("mode", "expected", "effort"), [("fast_qa", 3, "low"), ("deep_qa", 8, "high")]
)
def test_agent_honors_host_turn_budget_and_synthesis_round(
    tmp_path, mode, expected, effort
):
    from hermes_cli import lifecycle
    from run_agent import AIAgent

    original = lifecycle.invoke_hook

    def hooks(name, *a, **kw):
        if name == "pre_llm_call":
            return [{"context": "Evidence supplied.", "qa_mode": mode}]
        return original(name, *a, **kw)

    agent = AIAgent(
        api_key="test-only",
        base_url="http://127.0.0.1:9/v1",
        model="test/qa",
        provider="custom",
        api_mode="chat_completions",
        platform="slack",
        enabled_toolsets=["file"],
        quiet_mode=True,
        max_iterations=12,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        session_id="qa-turn",
        cwd=str(tmp_path),
    )
    agent._disable_streaming = True
    calls = []

    def respond(kw):
        calls.append(kw)
        tool = (
            None
            if kw.get("tool_choice") == "none"
            else [
                NS(
                    id=f"read-{len(calls)}",
                    type="function",
                    function=NS(
                        name="read_file",
                        arguments=json.dumps({"path": str(tmp_path / "source.txt")}),
                    ),
                )
            ]
        )
        return NS(
            model=agent.model,
            usage=None,
            choices=[
                NS(
                    message=NS(content=None if tool else "Done.", tool_calls=tool),
                    finish_reason="tool_calls" if tool else "stop",
                )
            ],
        )

    (tmp_path / "source.txt").write_text("Evidence.")
    try:
        with (
            patch.object(lifecycle, "invoke_hook", side_effect=hooks),
            patch.object(agent, "_interruptible_api_call", side_effect=respond),
        ):
            result = agent.run_conversation("Check the evidence.")
        assert len(calls) == expected
        assert result["final_response"]
        assert calls[-1]["tool_choice"] == "none"
        assert agent._turn_qa_effort == effort
    finally:
        agent.close()


def test_cache_affinity_is_workspace_and_profile_scoped(tmp_path, monkeypatch):
    from agent.transports.codex import ResponsesApiTransport
    from gateway import session_context

    def key(home, team, session):
        monkeypatch.setenv("HERMES_HOME", str(home))
        token = session_context.set_session_vars(
            platform="slack", scope_id=team, chat_id="C", user_id="U", chat_type="group"
        )
        try:
            built = ResponsesApiTransport().build_kwargs(
                "gpt-6.2",
                [
                    {"role": "system", "content": "Tenant instructions"},
                    {"role": "user", "content": "Question"},
                ],
                session_id=session,
                base_url="https://api.openai.com/v1",
            )
            return built["prompt_cache_key"]
        finally:
            session_context.clear_session_vars(token)

    home = tmp_path / "profile"
    assert key(home, "T1", "thread-one") == key(home, "T1", "thread-two")
    assert key(home, "T1", "thread-one") != key(home, "T2", "thread-one")
    assert key(home, "T1", "thread-one") != key(
        tmp_path / "profile2", "T1", "thread-one"
    )


@pytest.mark.parametrize(
    "details,expected",
    [({}, None), ({"cached_tokens": 0}, 0), ({"cached_tokens": 1024}, 1024)],
)
def test_usage_receipt_distinguishes_missing_cache_counter(details, expected):
    from agent.api_request_hooks import ApiRequestHooksMixin

    wire = {
        "input_tokens": 2000,
        "output_tokens": 240,
        "input_tokens_details": details,
        "output_tokens_details": {"reasoning_tokens": 140},
    }
    receipt = ApiRequestHooksMixin._usage_summary_for_api_request_hook(
        NS(provider="custom", api_mode="codex_responses"), NS(usage=wire)
    )
    assert receipt["observed_cache_read_tokens"] == expected
    assert receipt["observed_reasoning_tokens"] == 140
    assert "observed_cache_write_tokens" not in receipt
