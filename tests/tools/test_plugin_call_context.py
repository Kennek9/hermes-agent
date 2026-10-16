"""Host tool-call identity stays separate from model arguments across plugin dispatch."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest


_ID_KEYS = ("task_id", "session_id", "turn_id", "tool_call_id", "api_request_id")


@pytest.fixture
def plugin_runtime(monkeypatch):
    import hermes_cli.plugins as plugins_mod
    import model_tools
    import tools.registry as registry_mod
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    manager = PluginManager()
    manager._discovered = True
    context = PluginContext(PluginManifest(name="call_context_probe"), manager)
    monkeypatch.setattr(registry_mod, "registry", registry)
    monkeypatch.setattr(model_tools, "registry", registry)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    try:
        yield context, registry, model_tools.handle_function_call
    finally:
        manager.unload("call_context_probe")


def _register(context, handler, *, name="call_context_probe", **options):
    return context.register_tool(
        name=name,
        toolset="call_context_probe",
        schema={
            "name": name,
            "description": "Return a tool-call context probe",
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
        },
        handler=handler,
        **options,
    )


@pytest.mark.parametrize("async_handler", [False, True], ids=["sync", "async"])
def test_host_ids_reach_plugin_handlers(plugin_runtime, async_handler):
    context, registry, call = plugin_runtime
    both_handlers_entered = Barrier(2)

    def handler(args, **kwargs):
        both_handlers_entered.wait(timeout=10)
        return json.dumps({"ids": {key: kwargs.get(key) for key in _ID_KEYS}, "args": args})

    async def async_handler_fn(args, **kwargs):
        await asyncio.sleep(0)
        return handler(args, **kwargs)

    assert _register(context, async_handler_fn if async_handler else handler, is_async=async_handler)
    original_schema = deepcopy(registry.get_schema("call_context_probe"))
    calls = []
    for index in range(2):
        host_ids = {key: f"host-{key}-{index}" for key in _ID_KEYS}
        host_ids["session_id"] = "shared-session"
        args = {"value": f"probe-{index}", **{key: f"forged-{key}-{index}" for key in _ID_KEYS}}
        calls.append((host_ids, args, deepcopy(args)))

    # Both handlers must overlap: a session-wide "latest call" cannot satisfy these pairings.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call, "call_context_probe", args, **ids) for ids, args, _ in calls]
        results = [json.loads(future.result(timeout=15)) for future in futures]

    for result, (ids, args, original_args) in zip(results, calls):
        assert result["ids"] == ids
        assert result["args"] == original_args
        assert args == original_args
    assert registry.get_schema("call_context_probe") == original_schema
    definitions = registry.get_definitions({"call_context_probe"})
    assert definitions[0]["function"] == original_schema
    assert not set(_ID_KEYS) & set(original_schema["parameters"]["properties"])


def test_hidden_compatibility_and_narrow_handlers_keep_their_call_contract(plugin_runtime):
    context, registry, call = plugin_runtime
    hidden_name = "call_context_legacy"

    def hidden_handler(args, **kwargs):
        return json.dumps({"ids": {key: kwargs[key] for key in _ID_KEYS}, "args": args})

    assert _register(context, hidden_handler, name=hidden_name, model_visible=False, deprecated=True)
    assert registry.get_definitions({hidden_name}) == []
    assert hidden_name not in registry.get_all_tool_names()
    # Missing host values stay None; infrastructure does not add them to model-generated arguments.
    args = {"value": "legacy replay"}
    result = json.loads(call(hidden_name, args, enabled_tools=[]))
    assert result == {"ids": dict.fromkeys(_ID_KEYS), "args": args}
    assert args == {"value": "legacy replay"}

    narrow_calls = []

    def narrow_handler(args):
        narrow_calls.append(args)
        return json.dumps({"received": args})

    assert _register(context, narrow_handler)
    host_ids = {key: f"host-{key}" for key in _ID_KEYS}
    assert json.loads(call("call_context_probe", args, **host_ids)) == {"received": args}
    assert narrow_calls == [args]


def test_pre_tool_rejection_never_executes_the_registered_handler(plugin_runtime):
    context, registry, call = plugin_runtime
    handler_calls = []
    guard_calls = []

    def handler(args, **kwargs):
        handler_calls.append((args, kwargs))
        return json.dumps({"executed": True})

    def reject(tool_name, args, **kwargs):
        guard_calls.append((tool_name, args, {key: kwargs.get(key) for key in _ID_KEYS}))
        return {"action": "block", "message": "rejected by context probe"}

    assert _register(context, handler)
    assert registry.get_entry("call_context_probe") is not None
    assert context.register_hook("pre_tool_call", reject)
    host_ids = {key: f"host-{key}" for key in _ID_KEYS}
    args = {"value": "must not execute"}
    result = json.loads(call("call_context_probe", args, **host_ids))
    assert result == {"error": "rejected by context probe"}
    assert guard_calls == [("call_context_probe", args, host_ids)]
    assert handler_calls == []
