"""Cold gateway imports leave activation to the host loop, before auxiliary use."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_cold_auxiliary_bridge_waits_for_plugin_discovery_and_host_loop(tmp_path):
    home = tmp_path / "home"
    plugin = home / "plugins" / "startup-probe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: startup-probe\nversion: 1.0.0\nkind: standalone\n"
    )
    (plugin / "__init__.py").write_text(
        textwrap.dedent("""\
        import asyncio
        import os
        from pathlib import Path
        from hermes_constants import get_hermes_home

        def register(ctx):
            ctx.register_auxiliary_task(
                "startup_probe", display_name="Startup probe",
                description="Local gateway readiness regression"
            )
            async def worker():
                Path(get_hermes_home(), "worker-started").write_text(
                    os.environ.get("AUXILIARY_STARTUP_PROBE_MODEL", "missing")
                )
                await asyncio.Event().wait()
            task = ctx.spawn_task(worker(), name="startup-probe")
            ctx.on_unload(task.cancel)
    """)
    )
    (home / "config.yaml").write_text(
        textwrap.dedent("""\
        plugins:
          enabled: [startup-probe]
        auxiliary:
          vision:
            provider: configured-vision
            model: configured-vision-model
          approval:
            provider: configured-approval
          startup_probe:
            model: configured-plugin-model
          unregistered_probe:
            model: must-not-be-exported
    """)
    )
    script = textwrap.dedent("""\
        import asyncio
        import os
        from pathlib import Path
        import sys

        network_attempts = []
        def deny_network(event, args):
            if event in {'socket.connect', 'socket.getaddrinfo'}:
                network_attempts.append(event)
                raise AssertionError('native startup fixture must not use the network')
        sys.addaudithook(deny_network)

        import gateway.run
        from gateway.run_startup import GatewayStartupMixin
        from hermes_cli import plugins

        manager = plugins.get_plugin_manager()
        assert not manager._discovered, "gateway import activated plugins without a host loop"
        assert not Path(os.environ['HERMES_HOME'], 'worker-started').exists()

        async def run():
            GatewayStartupMixin._start_register_plugins_relay_hooks()
            manager = plugins.get_plugin_manager()
            loaded = manager._plugins['startup-probe']
            assert loaded.enabled and loaded.error is None, loaded.error
            assert os.environ['AUXILIARY_VISION_PROVIDER'] == 'configured-vision'
            assert os.environ['AUXILIARY_VISION_MODEL'] == 'configured-vision-model'
            assert os.environ['AUXILIARY_APPROVAL_PROVIDER'] == 'configured-approval'
            assert os.environ['AUXILIARY_STARTUP_PROBE_MODEL'] == 'configured-plugin-model'
            assert 'AUXILIARY_UNREGISTERED_PROBE_MODEL' not in os.environ
            marker = Path(os.environ['HERMES_HOME'], 'worker-started')
            async with asyncio.timeout(3):
                while not marker.exists():
                    await asyncio.sleep(0.01)
            assert marker.read_text() == 'configured-plugin-model'
            assert manager.unload('startup-probe')
            await asyncio.sleep(0)

        asyncio.run(run())
        assert network_attempts == [], network_attempts
    """)
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(tmp_path),
        "HERMES_HOME": str(home),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "HERMES_SKIP_DOTENV": "1",
        "HERMES_PLUGIN_ENTRYPOINTS": "0",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
