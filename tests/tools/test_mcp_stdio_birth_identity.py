"""Real child ownership does not depend on clock-adjusted epoch timestamps."""

from __future__ import annotations

import itertools
import os
import secrets
import subprocess
import sys
from contextlib import contextmanager

import psutil

from tools.mcp_tool_transport import _STDIO_SPAWN_TOKEN_ENV, _stdio_spawn_roots


@contextmanager
def _live_child(spawn_token: str):
    environment = dict(os.environ)
    environment[_STDIO_SPAWN_TOKEN_ENV] = spawn_token
    child = subprocess.Popen(
        [sys.executable, "-I", "-u", "-c", "import sys; print('ready', flush=True); sys.stdin.read()"],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert child.stdout.readline() == "ready\n"
        yield child
    finally:
        if child.stdin is not None:
            child.stdin.close()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        if child.stdout is not None:
            child.stdout.close()


def test_live_child_is_bound_without_clock_adjustment():
    token = secrets.token_hex(16)
    with _live_child(token) as child:
        assert _stdio_spawn_roots({child.pid}, token) == {child.pid}
        assert psutil.Process(child.pid).is_running()


def test_live_child_birth_identity_survives_epoch_clock_drift(monkeypatch):
    token = secrets.token_hex(16)
    with _live_child(token) as child, monkeypatch.context() as fault:
        offsets = itertools.count()
        if psutil.LINUX:
            # Exercise the real Linux epoch conversion; its monotonic process
            # start time and the kernel process itself remain unchanged.
            from psutil import _pslinux

            original_boot_time = _pslinux.boot_time
            fault.setattr(_pslinux, "boot_time", lambda: original_boot_time() + next(offsets))
        else:
            original_process = psutil.Process

            class DriftingEpochProcess(original_process):
                def __init__(self, *args, **kwargs):
                    self.reading_birth_identity = True
                    try:
                        super().__init__(*args, **kwargs)
                    finally:
                        self.reading_birth_identity = False

                def create_time(self):
                    actual = super().create_time()
                    if self.reading_birth_identity:
                        return actual
                    return actual + next(offsets)

            fault.setattr(psutil, "Process", DriftingEpochProcess)

        before = psutil.Process(child.pid).create_time()
        after = psutil.Process(child.pid).create_time()
        assert after != before, "The public epoch accessor must observe the controlled clock drift"
        assert _stdio_spawn_roots({child.pid}, token) == {child.pid}
        assert psutil.Process(child.pid).is_running()


def test_public_reused_pid_signal_rejects_an_otherwise_matching_live_child(monkeypatch):
    token = secrets.token_hex(16)
    original_process = psutil.Process
    identity_checks = []

    class ReusedIdentityProcess(original_process):
        def is_running(self):
            # A real live process supplies the ppid, environment and group.
            # Inject only the public API's documented reused-PID result.
            assert super().is_running()
            identity_checks.append(self.pid)
            return False

    with _live_child(token) as child, monkeypatch.context() as fault:
        fault.setattr(psutil, "Process", ReusedIdentityProcess)
        assert _stdio_spawn_roots({child.pid}, token) == set()
        assert set(identity_checks) == {child.pid}
        assert child.poll() is None


def test_foreign_child_is_rejected_while_matching_child_stays_owned():
    token = secrets.token_hex(16)
    with _live_child(token) as owned, _live_child(secrets.token_hex(16)) as foreign:
        assert _stdio_spawn_roots({owned.pid, foreign.pid}, token) == {owned.pid}
        assert psutil.Process(owned.pid).is_running()
        assert psutil.Process(foreign.pid).is_running()
