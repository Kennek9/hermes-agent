"""Public context-local execution for hosts that own configuration and durable state."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EmbeddedRuntime:
    """Configuration and scratch paths owned by one embedding invocation."""

    config: dict[str, Any]
    user_config: dict[str, Any]
    config_home: Path
    scratch_dir: Path


_RUNTIME: ContextVar[EmbeddedRuntime | None] = ContextVar(
    "hermes_embedded_runtime", default=None
)


def get_embedded_runtime() -> EmbeddedRuntime | None:
    """Return the active embedded invocation, or None for normal CLI/gateway execution."""
    return _RUNTIME.get()


@contextmanager
def embedded_runtime(
    *,
    config: Mapping[str, Any],
    config_home: str | Path,
    scratch_dir: str | Path,
    secrets: Mapping[str, str] | None = None,
) -> Iterator[EmbeddedRuntime]:
    """Bind caller-supplied configuration without loading accounts or changing process env.

    The configuration directory is read-only input. Temporary output belongs in scratch_dir;
    the host owns its lifetime. Existing thread-context propagation carries this scope.
    """
    if not isinstance(config, Mapping) or any(
        not isinstance(key, str) for key in config
    ):
        raise TypeError("config must be a mapping with string keys")
    home = Path(config_home).resolve(strict=True)
    if not home.is_dir():
        raise ValueError("config_home must be an existing directory")
    scratch = Path(scratch_dir).resolve()
    if scratch == home or scratch.is_relative_to(home):
        raise ValueError(
            "scratch_dir must be outside the read-only configuration directory"
        )
    scratch.mkdir(parents=True, exist_ok=True)

    from hermes_cli.config import (
        DEFAULT_CONFIG,
        _deep_merge,
        _normalize_root_model_keys,
    )
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from gateway.session_context import scoped_current_session_id
    from tools.terminal_scope import reset_terminal_scope, set_terminal_scope

    user_config = _normalize_root_model_keys(copy.deepcopy(dict(config)))
    runtime = EmbeddedRuntime(
        _deep_merge(copy.deepcopy(DEFAULT_CONFIG), user_config),
        user_config,
        home,
        scratch,
    )
    with ExitStack() as stack:
        token = _RUNTIME.set(runtime)
        stack.callback(_RUNTIME.reset, token)
        home_token = set_hermes_home_override(home)
        stack.callback(reset_hermes_home_override, home_token)
        secret_token = set_secret_scope(dict(secrets or {}), profile_home=str(home))
        stack.callback(reset_secret_scope, secret_token)
        stack.enter_context(scoped_current_session_id(""))
        from hermes_cli.config import apply_terminal_config_to_env

        terminal = apply_terminal_config_to_env(
            config=runtime.config, env={}, override=True, read_only_files=True
        )
        terminal_token = set_terminal_scope(terminal)
        stack.callback(reset_terminal_scope, terminal_token)
        yield runtime
