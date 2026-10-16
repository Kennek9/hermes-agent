"""Compatibility tools remain callable without entering the model's tool surface."""

import json

from tools.registry import ToolRegistry


def _schema(name):
    return {
        "name": name,
        "description": f"Tool {name}",
        "parameters": {"type": "object", "properties": {}},
    }


def test_hidden_tools_are_excluded_from_discovery_but_keep_legacy_dispatch(monkeypatch):
    import model_tools

    registry = ToolRegistry()

    def legacy_check():
        raise AssertionError("Hidden compatibility tools must not be probed for model exposure")

    for name, toolset, options in (
        ("modern", "compat", {}),
        ("deprecated_visible", "compat", {"deprecated": True}),
        ("legacy", "compat", {"model_visible": False, "deprecated": True}),
        ("legacy_only", "legacy_only", {"model_visible": False}),
    ):
        registry.register(
            name=name,
            toolset=toolset,
            schema=_schema(name),
            handler=lambda args: json.dumps({"received": args}),
            check_fn=legacy_check if options.get("model_visible") is False else None,
            **options,
        )

    visible = {"modern", "deprecated_visible"}
    all_names = {"modern", "deprecated_visible", "legacy", "legacy_only"}
    assert {tool["function"]["name"] for tool in registry.get_definitions(all_names)} == visible
    assert set(registry.get_all_tool_names()) == visible
    assert {entry.name for entry in registry.get_all_entries()} == visible
    assert {entry.name for entry in registry.get_all_entries(include_hidden=True)} == all_names
    assert set(registry.get_tool_names_for_toolset("compat")) == visible
    assert registry.get_tool_names_for_toolset("legacy_only") == []
    assert registry.get_registered_toolset_names() == ["compat"]
    assert set(registry.get_tool_to_toolset_map()) == visible
    assert registry.check_toolset_requirements() == {"compat": True}
    assert registry.check_tool_availability() == (["compat"], [])
    assert not registry.is_toolset_available("legacy_only")
    for metadata in (registry.get_available_toolsets(), registry.get_toolset_requirements()):
        assert set(metadata) == {"compat"}
        assert set(metadata["compat"]["tools"]) == visible
    assert registry.get_toolset_requirements()["compat"]["check_fn"] is None

    entry = registry.get_entry("legacy")
    assert entry.model_visible is False
    assert entry.deprecated is True
    assert registry.get_schema("legacy") == _schema("legacy")
    assert registry.get_toolset_for_tool("legacy") == "compat"
    assert registry.get_entry("modern").model_visible is True
    assert registry.get_entry("modern").deprecated is False

    # A session may already have resolved only modern schemas when replaying an older call.
    monkeypatch.setattr(model_tools, "registry", registry)
    result = model_tools.handle_function_call(
        "legacy", {"value": "replayed"}, enabled_tools=sorted(visible),
        enabled_toolsets=["compat"],
    )
    assert json.loads(result) == {"received": {"value": "replayed"}}


def test_plugin_visibility_is_scoped_and_unload_restores_previous_registration(tmp_path, monkeypatch):
    import hermes_cli.plugins as plugins_mod
    import tools.registry as registry_mod
    from hermes_cli.plugins import PluginManager
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    registry = ToolRegistry()
    monkeypatch.setattr(registry_mod, "registry", registry)
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "empty-bundled")
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    name = "legacy_compat_probe"
    registry.register(
        name=name, toolset="base", schema=_schema(name), handler=lambda args: "base",
    )
    baseline = registry.snapshot_registration(name)
    managers = []
    homes = []
    try:
        for marker, visible in (("a", False), ("b", True)):
            home = tmp_path / marker
            plugin_dir = home / "plugins" / "compat_probe"
            plugin_dir.mkdir(parents=True)
            (plugin_dir / "plugin.yaml").write_text(
                "name: compat_probe\nversion: 0.1.0\ndescription: compatibility probe\n",
            )
            (plugin_dir / "__init__.py").write_text(
                "def register(ctx):\n"
                "    ctx.register_tool(\n"
                f"        name={name!r}, toolset='compat_probe', schema={_schema(name)!r},\n"
                f"        handler=lambda args: {marker!r}, override=True,\n"
                f"        model_visible={visible!r}, deprecated=True,\n"
                "    )\n",
            )
            (home / "config.yaml").write_text(
                "plugins:\n  enabled: [compat_probe]\n  entries:\n"
                "    compat_probe:\n      allow_tool_override: true\n",
            )
            token = set_hermes_home_override(str(home))
            try:
                manager = PluginManager()
                managers.append(manager)
                homes.append(home)
                manager.discover_and_load()
                entry = registry.snapshot_registration(name, scope=manager.scope_key)
                assert entry is not None
                assert entry.model_visible is visible
                assert entry.deprecated is True
            finally:
                reset_hermes_home_override(token)

        for index in (0, 1, 0):
            token = set_hermes_home_override(str(homes[index]))
            try:
                assert registry.dispatch(name, {}) == ("a", "b")[index]
                assert bool(registry.get_definitions({name})) is (index == 1)
                assert (name in registry.get_all_tool_names()) is (index == 1)
            finally:
                reset_hermes_home_override(token)

        assert managers[0].unload("compat_probe") is True
        assert registry.snapshot_registration(name, scope=managers[0].scope_key) is None
        token = set_hermes_home_override(str(homes[0]))
        try:
            assert registry.get_entry(name) is baseline
            assert registry.dispatch(name, {}) == "base"
            assert registry.get_definitions({name})
        finally:
            reset_hermes_home_override(token)
        assert registry.dispatch(name, {}, scope=managers[1].scope_key) == "b"
        assert registry.snapshot_registration(name) is baseline
    finally:
        for manager in managers:
            manager.unload("compat_probe")
