"""Hidden compatibility dispatch retains each profile, toolset and schema boundary."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from agent.secret_scope import (
    build_profile_secret_scope,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


@contextmanager
def profile(home):
    home_token = set_hermes_home_override(str(home))
    secret_token = set_secret_scope(build_profile_secret_scope(home))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.fixture
def registered_profiles(tmp_path, monkeypatch):
    from tools.registry import registry
    from toolsets import TOOLSETS

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    homes = [tmp_path / name for name in ("A", "B")]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    set_multiplex_active(True)
    group, alias, bundle = (
        "compat_scope_group",
        "compat_scope_alias",
        "compat_scope_bundle",
    )
    names = {
        "visible": "compat_current_read",
        "legacy": "compat_legacy_read",
        "blocked": "compat_unavailable_read",
        "only_a": "compat_only_a_read",
        "profile_gate": "compat_profile_gated_read",
    }
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    calls, registrations = [], []
    for home in homes:
        with profile(home):
            scope = registry.current_scope_key()
            for kind, name in names.items():
                if kind == "only_a" and home != homes[0]:
                    continue

                def handler(args, *, owner=home.name, **kwargs):
                    calls.append((owner, dict(args)))
                    return json.dumps({
                        "owner": owner,
                        "home": str(get_hermes_home()),
                        **args,
                    })

                check = None
                if kind == "blocked":
                    check = lambda: False
                elif kind == "profile_gate":
                    check = lambda: get_hermes_home() == homes[0]
                registry.register(
                    name=name,
                    toolset=group,
                    schema={"name": name, "parameters": schema},
                    handler=handler,
                    scope=scope,
                    model_visible=kind == "visible",
                    deprecated=kind != "visible",
                    check_fn=check,
                )
                registrations.append((scope, name))
    registry.register_toolset_alias(alias, group)
    TOOLSETS[bundle] = {"tools": [], "includes": [alias]}
    try:
        yield SimpleNamespace(
            homes=homes,
            group=group,
            alias=alias,
            bundle=bundle,
            names=names,
            calls=calls,
            agent=SimpleNamespace(),
        )
    finally:
        del TOOLSETS[bundle]
        for scope, name in registrations:
            registry.deregister(name, scope=scope)
        set_multiplex_active(False)


def invoke(path, name, args, enabled, disabled=None, agent=None):
    import model_tools
    from agent.tool_executor import _unwrap_tool_search_call

    envelope = {"calls": [{"name": name, "arguments": args}]}
    if path == "agent":
        agent = agent or SimpleNamespace()
        agent.enabled_toolsets, agent.disabled_toolsets = enabled, disabled
        target, fields, blocked = _unwrap_tool_search_call(agent, "tool_call", envelope)
        if blocked is not None:
            return {"error": blocked}
        return json.loads(
            model_tools.handle_function_call(
                target, fields, enabled_toolsets=enabled, disabled_toolsets=disabled
            )
        )
    return json.loads(
        model_tools.handle_function_call(
            "tool_call", envelope, enabled_toolsets=enabled, disabled_toolsets=disabled
        )
    )


@pytest.mark.parametrize("path", ["bridge", "agent"])
@pytest.mark.parametrize("selection", ["group", "alias", "bundle", "all", "default"])
def test_hidden_dispatch_stays_profile_scoped_without_entering_model_catalog(
    registered_profiles,
    path,
    selection,
):
    import model_tools

    case = registered_profiles
    enabled = (
        None
        if selection == "default"
        else ["all" if selection == "all" else getattr(case, selection)]
    )
    for home in (case.homes[0], case.homes[1], case.homes[0]):
        with profile(home):
            before = model_tools.get_tool_definitions(
                enabled_toolsets=enabled,
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            assert case.names["legacy"] not in {
                row["function"]["name"] for row in before
            }
            result = invoke(
                path, case.names["legacy"], {"value": 7}, enabled, agent=case.agent
            )
            assert result == {"owner": home.name, "home": str(home), "value": 7}
            after = model_tools.get_tool_definitions(
                enabled_toolsets=enabled,
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            assert after == before
    assert [owner for owner, _ in case.calls] == ["A", "B", "A"]


@pytest.mark.parametrize("path", ["bridge", "agent"])
@pytest.mark.parametrize(
    "failure",
    [
        "other_group",
        "disabled",
        "own_check",
        "bad_type",
        "extra_field",
        "unknown",
        "other_profile",
        "warm_profile_gate",
    ],
)
def test_hidden_dispatch_rejects_unavailable_or_invalid_requests_before_handler(
    registered_profiles,
    path,
    failure,
):
    case = registered_profiles
    home = (
        case.homes[1]
        if failure in {"other_profile", "warm_profile_gate"}
        else case.homes[0]
    )
    if failure == "warm_profile_gate":
        with profile(case.homes[0]):
            warm = invoke(
                path,
                case.names["profile_gate"],
                {"value": 7},
                [case.bundle],
                agent=case.agent,
            )
            assert warm["owner"] == "A"
    name = {
        "own_check": case.names["blocked"],
        "unknown": "compat_unknown_read",
        "other_profile": case.names["only_a"],
        "warm_profile_gate": case.names["profile_gate"],
    }.get(failure, case.names["legacy"])
    args = {"value": "bad"} if failure == "bad_type" else {"value": 7}
    if failure == "extra_field":
        args["actor_id"] = "forged"
    enabled = ["cronjob"] if failure == "other_group" else [case.bundle]
    disabled = [case.group] if failure == "disabled" else None
    before = list(case.calls)
    with profile(home):
        result = invoke(path, name, args, enabled, disabled, agent=case.agent)
    assert "error" in result
    assert case.calls == before
