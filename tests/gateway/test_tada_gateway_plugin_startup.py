"""Cold gateway startup owns plugin tasks before the first inbound message."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


_PLUGIN = """
import asyncio
import json
import os
from pathlib import Path
import threading

from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home
from hermes_cli.plugins_loader import in_plugin_load_worker


def record(event, **values):
    with Path(os.environ['STARTUP_OBSERVATIONS']).open('a') as stream:
        stream.write(json.dumps({'event': event, 'home': str(get_hermes_home()),
                                 'marker': get_secret('STARTUP_FIXTURE_MARKER'),
                                 'thread': threading.get_ident(), **values}) + '\\n')


def drain(limit):
    record('drain', limit=limit)


def register(ctx):
    record('register', deadline_worker=in_plugin_load_worker(),
           owner=ctx._manager.scope_key)
    async def worker():
        while True:
            await asyncio.to_thread(drain, limit=3)
            await asyncio.sleep(1)
    task = ctx.spawn_task(worker(), name='gateway-startup-probe')
    ctx.on_unload(task.cancel)
    task.add_done_callback(lambda value: record('cancelled', cancelled=value.cancelled()))
"""


_COLD_ENTRY = """
import asyncio
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

entry, served = sys.argv[1:]
sys.argv = (['hermes', 'gateway', 'run', '--replace'] if entry in {'cli', 'cli-main'}
            else ['python', '-m', 'gateway.run'])
network_attempts = []


def deny_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        network_attempts.append(event)
        raise AssertionError('Cold startup must not contact models, APIs, or databases')


sys.addaudithook(deny_network)
from hermes_constants import set_hermes_home_override
set_hermes_home_override(served)

if entry in {'cli', 'cli-main'}:
    import hermes_cli.main as cli_main
    if entry == 'cli':
        cli_main._prepare_agent_startup(SimpleNamespace(command='gateway', gateway_command='run',
                                                        replace=True, toolsets=None))
else:
    if entry == 'legacy-import':
        import cli
    import gateway.run

from hermes_cli import plugins
plugins._join_background_discovery(timeout=10)
manager = plugins.get_plugin_manager()
observations = Path(os.environ['STARTUP_OBSERVATIONS'])


def events():
    return [json.loads(line) for line in observations.read_text().splitlines()] if observations.exists() else []


def assert_no_premature_registration():
    plugins._join_background_discovery(timeout=10)
    loaded = manager._plugins.get('gateway_startup_probe')
    assert events() == [], json.dumps({'premature_events': events(),
                                      'plugin_error': loaded.error if loaded else None})


assert_no_premature_registration()


async def scenario():
    from gateway.run_startup import GatewayStartupMixin
    loop = asyncio.get_running_loop()
    host_thread = threading.get_ident()
    GatewayStartupMixin._start_register_plugins_relay_hooks()
    loaded = manager._plugins.get('gateway_startup_probe')
    assert loaded is not None and loaded.enabled and not loaded.error, loaded
    assert plugins.get_plugin_manager() is manager
    assert manager._task_loop is loop

    async def until(predicate):
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(until(lambda: any(row['event'] == 'drain' for row in events())), 5)
    # A second startup registration pass must reuse the first supervised worker.
    GatewayStartupMixin._start_register_plugins_relay_hooks()
    def worker_tasks():
        # Cross-thread spawn returns a Future; its ledger name is not an asyncio Task name.
        files = {str((Path(served) / 'plugins/gateway_startup_probe/__init__.py').resolve())}
        if entry == 'secondary':
            files.add(str((Path(served).with_name('secondary') / 'plugins/gateway_startup_probe/__init__.py').resolve()))
        tasks = []
        for task in asyncio.all_tasks():
            code = getattr(task.get_coro(), 'cr_code', None)
            if (code is not None and str(Path(code.co_filename).resolve()) in files
                    and code.co_qualname == 'register.<locals>.worker'):
                tasks.append(task)
        return tasks

    tasks = worker_tasks()
    assert len(tasks) == 1, tasks
    registered = [row for row in events() if row['event'] == 'register']
    assert len(registered) == 1, registered
    assert registered[0]['deadline_worker'] is True
    assert registered[0]['thread'] != host_thread
    assert registered[0]['owner'] == str(Path(served).resolve())
    assert all(row['home'] == str(Path(served).resolve()) for row in events())
    assert all(row['limit'] == 3 for row in events() if row['event'] == 'drain')

    secondary_manager = None
    secondary_tasks = []
    if entry == 'secondary':
        from gateway.run import GatewayRunner, _profile_runtime_scope
        secondary = Path(served).with_name('secondary')
        runner = SimpleNamespace(
            _subscribe_plugin_rewire=lambda *args: None,
            _register_config_hooks=GatewayStartupMixin._register_config_hooks,
            _snapshot_profile_busy_modes=lambda *args: None,
        )
        await GatewayRunner._load_secondary_profile_config(runner, 'secondary', secondary)
        with _profile_runtime_scope(secondary, hydrate_secrets=False):
            secondary_manager = plugins.get_plugin_manager()
        assert secondary_manager is not manager
        assert secondary_manager._task_loop is loop
        loaded = secondary_manager._plugins.get('gateway_startup_probe')
        assert loaded is not None and loaded.enabled and not loaded.error, loaded
        await asyncio.wait_for(until(lambda: any(
            row['event'] == 'drain' and row['home'] == str(secondary.resolve())
            for row in events()
        )), 5)
        secondary_tasks = [task for task in worker_tasks() if task not in tasks]
        assert len(secondary_tasks) == 1 and len(worker_tasks()) == 2
        registered = [row for row in events() if row['event'] == 'register']
        assert len(registered) == 2
        assert all(row['deadline_worker'] and row['thread'] != host_thread for row in registered)
        assert {row['owner'] for row in registered} == {str(Path(served).resolve()), str(secondary.resolve())}
        assert all(row['owner'] == row['home'] for row in registered)
        assert all(row['marker'] == 'secondary-fixture' for row in events()
                   if row['home'] == str(secondary.resolve()))

    assert manager.unload('gateway_startup_probe') is True
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
    assert all(task.cancelled() for task in tasks)
    if secondary_manager is not None:
        assert worker_tasks() == secondary_tasks
        drain_count = sum(row['event'] == 'drain' and row['home'] == str(secondary.resolve())
                          for row in events())
        await asyncio.wait_for(until(lambda: sum(
            row['event'] == 'drain' and row['home'] == str(secondary.resolve())
            for row in events()
        ) > drain_count), 5)
        with _profile_runtime_scope(secondary, hydrate_secrets=False):
            assert secondary_manager.unload('gateway_startup_probe') is True
        await asyncio.wait_for(asyncio.gather(*secondary_tasks, return_exceptions=True), 5)
        assert all(task.cancelled() for task in secondary_tasks)
    assert not worker_tasks()
    assert [row['cancelled'] for row in events() if row['event'] == 'cancelled'] == [True] * (2 if secondary_manager else 1)


if entry == 'cli-main':
    import hermes_cli.gateway as gateway_command
    service_calls = []

    def local_service(verbose=0, quiet=False, replace=False, force=False):
        assert replace is True
        service_calls.append(replace)
        assert_no_premature_registration()
        asyncio.run(scenario())

    # The service boundary is local; parser, prepare, command dispatch, and plugin loading are real.
    gateway_command.run_gateway = local_service
    cli_main.main()
    assert service_calls == [True]
else:
    asyncio.run(scenario())
assert network_attempts == [], network_attempts
print(json.dumps({'entry': entry, 'events': events(), 'task_count_after_unload': 0}))
"""


@pytest.mark.parametrize(
    "entry", ["cli", "direct-import", "cli-main", "legacy-import", "secondary"]
)
def test_cold_gateway_starts_one_profile_owned_task_without_messages(tmp_path, entry):
    launch = tmp_path / "launch"
    served = tmp_path / "served"
    launch.mkdir()
    plugin = served / "plugins" / "gateway_startup_probe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: gateway_startup_probe\nversion: 0.1.0\ndescription: gateway host startup probe\n"
    )
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN))
    (served / "config.yaml").write_text(
        "plugins:\n  enabled: [gateway_startup_probe]\n  load_timeout_seconds: 5\n"
        "platforms:\n  relay:\n    enabled: false\n"
    )
    if entry == "secondary":
        secondary = tmp_path / "secondary"
        secondary_plugin = secondary / "plugins" / "gateway_startup_probe"
        secondary_plugin.mkdir(parents=True)
        (secondary_plugin / "plugin.yaml").write_text(
            (plugin / "plugin.yaml").read_text()
        )
        (secondary_plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN))
        (secondary / "config.yaml").write_text((served / "config.yaml").read_text())
        (secondary / ".env").write_text("STARTUP_FIXTURE_MARKER=secondary-fixture\n")
    observations = tmp_path / "startup-observations.jsonl"
    environment = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "HERMES_HOME": str(launch),
        "HERMES_DISABLE_LAZY_INSTALLS": "1",
        "HERMES_PLUGIN_ENTRYPOINTS": "0",
        "STARTUP_OBSERVATIONS": str(observations),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(_COLD_ENTRY),
            entry,
            str(served),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=40,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    receipt = json.loads(completed.stdout.splitlines()[-1])
    assert receipt["entry"] == entry
    assert receipt["task_count_after_unload"] == 0
