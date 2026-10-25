---
title: Embedded execution library
---

Hermes can run inside a host that owns configuration, credentials, conversation
history and durable storage. The CLI and messaging gateway keep their normal
initialization behavior outside the embedded scope.

## Install a pinned library build

Request the execution-library wheel explicitly. Ordinary wheel/sdist builds retain
the standalone distribution guard; the library option does not publish a package
or change the shell/Docker/Nix installation paths.

```sh
uv pip install --config-setting=--build-option=--library \
  'hermes-agent @ git+https://github.com/OWNER/hermes-agent.git@FULL_40_CHARACTER_SHA'
```

For a uv project, declare the Git dependency and build option in `pyproject.toml`,
then lock and sync it normally:

```toml
[tool.uv.sources]
hermes-agent = { git = "https://github.com/OWNER/hermes-agent.git", rev = "FULL_40_CHARACTER_SHA" }

[tool.uv.config-settings-package]
hermes-agent = { "--build-option" = "--library" }
```

The library wheel includes Python modules, platform plugin manifests, locale
catalogs, bundled skills, optional skills, MCP catalog and execution assets.
It does not provide built desktop/web applications or the interactive installers.
Applications should use the public execution interfaces from their own UI.

## Bind an invocation

```python
from hermes_runtime import embedded_runtime
from run_agent import AIAgent

with embedded_runtime(
    config=config_snapshot,
    config_home=readonly_configuration_directory,
    scratch_dir=per_invocation_scratch_directory,
    secrets=invocation_tool_credentials,
):
    agent = AIAgent(
        model=model,
        provider=provider,
        api_key=model_api_key,
        enabled_toolsets=allowed_toolsets,
        session_id=conversation_id,
        session_db=None,
        skip_context_files=True,
        load_soul_identity=False,
    )
    try:
        result = agent.run_conversation(
            user_message,
            system_message=frozen_instructions,
            conversation_history=history,
        )
    finally:
        agent.close()
```

The supplied snapshot is merged with Hermes defaults. Config loaders use that
snapshot and do not initialize the read-only directory. Credential files and
ambient provider credentials are not loaded; pass tool credentials through
`secrets` and the model credential through the existing constructor arguments.
Built-in persistent memory and automatic background review are disabled in this
scope. Hosts provide their own data tools and exclude native memory, session
search, cron and kanban toolsets when those stores belong to the host.

Inside this scope, a non-None `run_conversation(system_message=...)` is the complete
host-owned system prompt, preserved byte for byte. Hermes does not append its
default identity, timestamp, memory, skills or environment blocks. The host must
include the required tool and operating instructions in that frozen prompt. Omitting
`system_message` keeps the native assembly behavior. Standalone calls are unchanged.

With `session_db=None`, execution does not create or lazily open a default
SessionDB. Embedded execution does not install process-wide file logging or
export session/terminal settings into the process environment. Temporary files
belong in `scratch_dir`; the host creates and removes this invocation directory.
The configuration directory and scratch directory must be separate.

The scope restores its home, credential, terminal and session bindings on normal
exit and on exceptions. Hermes thread-context helpers propagate the scope to
tool threads. Background work must finish before its owning scope is disposed;
the host remains responsible for durable scheduling and completion delivery.
