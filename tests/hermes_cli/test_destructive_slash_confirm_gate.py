"""Tests for the approvals.destructive_slash_confirm config gate.

Destructive session slash commands (/clear, /new, /reset, /undo) discard
conversation state. This config key defaults to False so an explicit reset
command runs immediately. Operators can opt back into confirmation prompts.

See gateway/run.py::_maybe_confirm_destructive_slash and
cli.py::_confirm_destructive_slash for the runtime gate.
"""

from __future__ import annotations

from hermes_cli.config import DEFAULT_CONFIG


class TestDestructiveSlashConfirmDefault:

    def test_default_allows_explicit_reset_without_extra_confirmation(self):
        assert DEFAULT_CONFIG["approvals"]["destructive_slash_confirm"] is False


class TestUserConfigMerge:
    """If a user has a pre-existing config without this key, load_config
    should fill it in from DEFAULT_CONFIG (deep merge preserves keys the
    user didn't override)."""

    def test_existing_user_config_without_key_gets_default(self, tmp_path, monkeypatch):
        import yaml

        home = tmp_path / ".hermes"
        home.mkdir()
        cfg_path = home / "config.yaml"
        legacy = {
            "approvals": {"mode": "manual", "timeout": 60, "cron_mode": "deny"},
        }
        cfg_path.write_text(yaml.safe_dump(legacy))

        monkeypatch.setenv("HERMES_HOME", str(home))
        import importlib
        import hermes_cli.config as cfg_mod
        importlib.reload(cfg_mod)

        cfg = cfg_mod.load_config()
        assert cfg["approvals"]["destructive_slash_confirm"] is False
