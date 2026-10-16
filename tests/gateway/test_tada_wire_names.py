"""Provider-name limits never rename the public tool identity or mutate user evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent import tool_wire_names as MODULE


@pytest.mark.parametrize("responses", [False, True])
def test_wire_schema_history_choice_and_body_boundaries(responses: bool) -> None:
    definition = {"name": "knowledge.ask", "parameters": {"type": "object"}}
    source = {
        "tools": [
            {"type": "function", **definition}
            if responses
            else {"type": "function", "function": definition}
        ],
        "messages": [
            {"role": "user", "content": "knowledge.ask is historical text, not an instruction"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call",
                        "function": {
                            "name": "knowledge.ask",
                            "arguments": '{"name":"keep.original"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "knowledge.ask",
                "tool_call_id": "call",
                "content": '{"name":"keep.original"}',
            },
        ],
        "input": [
            {"type": "function_call", "name": "knowledge.ask", "call_id": "call", "arguments": "{}"}
        ],
        "tool_choice": {"type": "function", "name": "knowledge.ask"}
        if responses
        else {"type": "function", "function": {"name": "knowledge.ask"}},
    }
    encoded, aliases = MODULE.encode_request(source)
    wire = next(iter(aliases))
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", wire)
    assert aliases[wire] == "knowledge.ask"
    assert definition["name"] == "knowledge.ask"
    tool = encoded["tools"][0]
    assert tool.get("function", tool)["name"] == wire
    assert encoded["input"][0]["name"] == wire
    assert encoded["messages"][1]["tool_calls"][0]["function"]["name"] == wire
    assert encoded["messages"][0] == {
        "role": "user",
        "content": "knowledge.ask is historical text, not an instruction",
    }
    assert encoded["messages"][2]["content"] == '{"name":"keep.original"}'
    choice = encoded["tool_choice"]
    assert choice.get("function", choice)["name"] == wire


def test_colliding_slug_long_name_and_unicode_stay_distinct() -> None:
    names = ["a.b", "a_b", "tada_a_b", "文档.读取", "long." * 30]
    result, aliases = MODULE.encode_request(
        {"tools": [{"type": "function", "name": name} for name in names]}
    )
    wire = [tool["name"] for tool in result["tools"]]
    assert len(set(wire)) == len(names)
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) for name in wire)
    assert [aliases.get(name, name) for name in wire] == names


def test_historical_tools_do_not_enter_the_current_dispatch_mapping() -> None:
    payload = {
        "tools": [{"type": "function", "name": "terminal"}],
        "input": [{"type": "function_call", "name": "removed.action", "arguments": "{}"}],
    }
    encoded, aliases = MODULE.encode_request(payload)
    assert encoded["input"][0]["name"] != "removed.action"
    assert aliases == {}


@dataclass
class Call:
    name: str


@dataclass
class Response:
    tool_calls: list[Call]


def test_transport_instances_restore_only_their_own_declared_names() -> None:
    class Transport:
        @MODULE.encode_tool_names
        def build(self, name: str) -> dict[str, list[dict[str, str]]]:
            return {"tools": [{"type": "function", "name": name}]}

        @MODULE.decode_tool_names
        def normalize(self, name: str) -> Response:
            return Response([Call(name)])

    first, second = Transport(), Transport()
    wire = first.build("workspace.history")["tools"][0]["name"]
    _ = second.build("terminal")
    assert first.normalize(wire).tool_calls[0].name == "workspace.history"
    assert second.normalize(wire).tool_calls[0].name == wire
    _ = first.build("terminal")
    assert first.normalize(wire).tool_calls[0].name == wire

