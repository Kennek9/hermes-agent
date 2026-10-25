"""Public execution scopes must not initialize account state or leak caller bindings."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_public_embedded_execution_in_readonly_homes(tmp_path):
    script = Path(__file__).parent / "fixtures/embedded_runtime_probe.py"
    result = subprocess.run(
        [sys.executable, "-I", str(script)],
        cwd=tmp_path,
        env={
            "PATH": os.defpath,
            "HOME": str(tmp_path),
            "HERMES_HOME": str(tmp_path / "process"),
            "HERMES_DISABLE_LAZY_INSTALLS": "1",
            "OPENAI_API_KEY": "synthetic-ambient-key",
        },
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr + result.stdout[-2000:]
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "homes": ["a", "b", "a"],
        "turns": 6,
        "environment_unchanged": True,
    }


def test_embedded_scope_restores_bindings_after_exception(tmp_path):
    from hermes_runtime import embedded_runtime, get_embedded_runtime
    from hermes_constants import get_hermes_home
    from agent.secret_scope import current_secret_scope
    from tools.terminal_scope import get_terminal_scope
    from gateway.session_context import get_session_env, set_current_session_id

    original = (
        get_hermes_home(),
        current_secret_scope(),
        get_terminal_scope(),
        get_session_env("HERMES_SESSION_ID"),
    )
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with embedded_runtime(
            config={},
            config_home=tmp_path,
            scratch_dir=tmp_path.parent / (tmp_path.name + "-scratch"),
        ):
            set_current_session_id("synthetic-session")
            raise RuntimeError("synthetic failure")
    assert get_embedded_runtime() is None
    assert (
        get_hermes_home(),
        current_secret_scope(),
        get_terminal_scope(),
        get_session_env("HERMES_SESSION_ID"),
    ) == original
    assert dict(os.environ) == before
