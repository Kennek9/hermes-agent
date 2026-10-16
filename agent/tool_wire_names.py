"""Keep canonical tool identities behind provider-compatible wire names."""

import copy
import hashlib
import re
from dataclasses import replace
from functools import wraps

_PREFIX = "tada_"


def _wire_name(name):
    if re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) and not name.startswith(_PREFIX):
        return name
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:40]
    return _PREFIX + slug + "__" + hashlib.sha256(name.encode()).hexdigest()[:16]


def encode_request(payload):
    """Copy only protocol fields; never interpret user text or serialized tool arguments."""
    result = copy.deepcopy(payload)
    aliases = {}
    seen = {}

    def rename(value, *, declared=False):
        name = value.get("name")
        if not isinstance(name, str) or not name:
            return
        wire = _wire_name(name)
        if wire in seen and seen[wire] != name:
            raise ValueError("tool_wire_name_collision")
        seen[wire] = name
        if wire != name:
            if declared:
                aliases[wire] = name
            value["name"] = wire

    for tool in result.get("tools") or []:
        if tool.get("type") == "function":
            rename(tool.get("function", tool), declared=True)
    for message in result.get("messages") or []:
        for call in message.get("tool_calls") or []:
            if call.get("type") == "function":
                rename(call.get("function", {}))
        if message.get("role") == "tool":
            rename(message)
    inputs = result.get("input")
    if isinstance(inputs, list):
        for item in inputs:
            if isinstance(item, dict) and item.get("type") == "function_call":
                rename(item)
    choice = result.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "function":
        rename(choice.get("function", choice))
    return result, aliases


def encode_tool_names(build):
    """Keep each transport instance's aliases with its matching request."""

    @wraps(build)
    def wrapped(self, *args, **kwargs):
        result, aliases = encode_request(build(self, *args, **kwargs))
        self._tada_tool_wire_names = aliases
        return result

    return wrapped


def decode_tool_names(normalize):
    """Restore normalized calls before dispatch, history recording and subsequent replay."""

    @wraps(normalize)
    def wrapped(self, *args, **kwargs):
        response = normalize(self, *args, **kwargs)
        aliases = getattr(self, "_tada_tool_wire_names", {})
        if not aliases or not response.tool_calls:
            return response
        return replace(
            response,
            tool_calls=[
                replace(call, name=aliases.get(call.name, call.name))
                for call in response.tool_calls
            ],
        )

    return wrapped
