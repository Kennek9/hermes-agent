"""Real stdio connections retain their own children during concurrent startup.

The SDK pipes, subprocesses, death supervisor and orphan sweep all run normally.
A short pause after the SDK spawn widens the historical snapshot overlap without
changing process creation or supplying the PID bookkeeping under test.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
import pytest

from tools import mcp_tool
from tools import mcp_tool_transport
from tools.mcp_isolated import IsolatedMCPClient
from tools.mcp_tool_handlers import _call_tool_racing_stdio_death


_SERVER = r'''
import json
import os
import sys
from pathlib import Path

options = json.loads(sys.argv[1])
Path(options["pid_file"]).write_text(str(os.getpid()))
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "initialize":
        if options.get("hang_init"):
            Path(options["init_file"]).write_text("received")
            continue
        value = {"protocolVersion": request["params"]["protocolVersion"],
                 "capabilities": {"tools": {}},
                 "serverInfo": {"name": "concurrent-fixture", "version": "1"}}
    elif method == "tools/list":
        value = {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}
    elif method == "tools/call":
        payload = {"pid": os.getpid(), "arguments": request["params"]["arguments"]}
        value = {"content": [{"type": "text", "text": json.dumps(payload)}]}
    elif method == "ping":
        value = {}
    else:
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": value}), flush=True)
'''


def _config(directory: Path, **options) -> dict:
    directory.mkdir()
    return {
        "command": sys.executable,
        "args": ["-u", "-c", _SERVER, json.dumps({
            "pid_file": str(directory / "pid"), **options})],
        "connect_timeout": 10,
        "timeout": 5,
    }


async def _call(client, nonce: str) -> dict:
    if isinstance(client, IsolatedMCPClient):
        result = await client.call_tool("echo", {"nonce": nonce})
    else:
        async with client._rpc_lock:
            result = await _call_tool_racing_stdio_death(client, client.name, "echo", {"nonce": nonce})
        result = result.model_dump(by_alias=True, exclude_none=True)
    return json.loads(result["content"][0]["text"])


async def _close(client) -> None:
    await (client.close() if isinstance(client, IsolatedMCPClient) else client.shutdown())


async def _wait_for_file(path: Path) -> None:
    while not path.exists():
        await asyncio.sleep(0.01)


def _server(client):
    return client._server if isinstance(client, IsolatedMCPClient) else client


class _LoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.started = threading.Event()
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        assert self.started.wait(5)

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.call_soon(self.started.set)
        self.loop.run_forever()
        self.loop.close()

    async def run(self, coroutine):
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coroutine, self.loop))

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        assert not self.thread.is_alive()


@pytest.mark.parametrize("same_name", [True, False], ids=["same-name", "different-names"])
@pytest.mark.parametrize("native_second", [False, True], ids=["isolated", "native-and-isolated"])
@pytest.mark.parametrize("other_loop", [False, True], ids=["one-loop", "two-loops"])
def test_concurrent_stdio_close_and_orphan_sweep_preserve_the_other_connection(
        tmp_path, monkeypatch, same_name, native_second, other_loop):
    mcp_tool._ensure_mcp_sdk()
    real_stdio = mcp_tool.stdio_client
    foreign = None
    foreign_lock = threading.Lock()

    @asynccontextmanager
    async def slow_spawn(*args, **kwargs):
        nonlocal foreign
        async with real_stdio(*args, **kwargs) as streams:
            # A terminal/CUA child can appear in the same snapshot delta while
            # MCP spawns are serialized. It belongs to neither connection.
            with foreign_lock:
                if foreign is None:
                    foreign = subprocess.Popen(
                        [sys.executable, "-c", "import time; time.sleep(60)"],
                        start_new_session=True)
            await asyncio.sleep(0.1)
            yield streams

    monkeypatch.setattr(mcp_tool, "stdio_client", slow_spawn)
    loop_thread = _LoopThread() if other_loop else None

    async def run():
        first = IsolatedMCPClient("concurrent")
        second_name = "concurrent" if same_name else "different"
        second = mcp_tool.MCPServerTask(second_name) if native_second else IsolatedMCPClient(second_name)
        replacement = IsolatedMCPClient("replacement")

        async def on_second(coroutine):
            return await loop_thread.run(coroutine) if loop_thread else await coroutine

        try:
            await asyncio.wait_for(asyncio.gather(
                first.start(_config(tmp_path / "first")),
                on_second(second.start(_config(tmp_path / "second")))), 10)
            first_result, before = await asyncio.gather(
                _call(first, "first"), on_second(_call(second, "before")))
            assert _server(first)._stdio_child_pids == {first_result["pid"]}
            assert _server(second)._stdio_child_pids == {before["pid"]}
            await first.close()
            await replacement.start(_config(tmp_path / "replacement"))
            after = await on_second(_call(second, "after"))
            assert after == {"pid": before["pid"], "arguments": {"nonce": "after"}}
            assert foreign.poll() is None, "orphan cleanup killed a subprocess owned by another tool"
        finally:
            await asyncio.gather(first.close(), on_second(_close(second)), replacement.close())

    try:
        asyncio.run(run())
    finally:
        if loop_thread:
            loop_thread.close()
        if foreign is not None:
            if foreign.poll() is None:
                foreign.terminate()
            foreign.wait(timeout=5)


def test_waiting_start_cancellation_and_hanging_handshake_do_not_hold_spawn_window(tmp_path, monkeypatch):
    mcp_tool._ensure_mcp_sdk()
    real_stdio = mcp_tool.stdio_client

    async def run():
        spawned = asyncio.Event()
        waiting_for_spawn = asyncio.Event()
        release_spawn = asyncio.Event()
        first = IsolatedMCPClient("hanging")
        cancelled = IsolatedMCPClient("cancelled")
        survivor = IsolatedMCPClient("survivor")
        first_config = _config(tmp_path / "hanging", hang_init=True, init_file=str(tmp_path / "initialize"))
        cancelled_config = _config(tmp_path / "cancelled")
        survivor_config = _config(tmp_path / "survivor")
        real_window = mcp_tool_transport._stdio_spawn_window

        @asynccontextmanager
        async def observed_window():
            if asyncio.current_task() is cancelled._server._task:
                waiting_for_spawn.set()
            async with real_window():
                yield

        @asynccontextmanager
        async def controlled_spawn(parameters, **kwargs):
            async with real_stdio(parameters, **kwargs) as streams:
                if parameters.args == first_config["args"]:
                    spawned.set()
                    await release_spawn.wait()
                yield streams

        monkeypatch.setattr(mcp_tool, "stdio_client", controlled_spawn)
        monkeypatch.setattr(mcp_tool_transport, "_stdio_spawn_window", observed_window)
        first_start = asyncio.create_task(first.start(first_config))
        cancelled_start = None
        try:
            await asyncio.wait_for(spawned.wait(), 5)
            cancelled_start = asyncio.create_task(cancelled.start(cancelled_config))
            await asyncio.wait_for(waiting_for_spawn.wait(), 5)
            cancelled_start.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled_start
            assert not (tmp_path / "cancelled" / "pid").exists()
            release_spawn.set()
            await asyncio.wait_for(survivor.start(survivor_config), 5)
            await asyncio.wait_for(_wait_for_file(tmp_path / "initialize"), 5)
            assert (tmp_path / "initialize").read_text() == "received"
            assert not first_start.done(), "initialize must still be waiting while another server connects"
            assert (await _call(survivor, "alive"))["arguments"] == {"nonce": "alive"}
        finally:
            release_spawn.set()
            first_start.cancel()
            if cancelled_start:
                cancelled_start.cancel()
            await asyncio.gather(first_start, *([cancelled_start] if cancelled_start else []), return_exceptions=True)
            await asyncio.gather(first.close(), cancelled.close(), survivor.close())

    asyncio.run(run())


def test_stdio_argument_matching_another_tool_name_is_still_owned(tmp_path):
    async def run():
        client = IsolatedMCPClient("ordinary-argument")
        config = _config(tmp_path / "server")
        config["args"].append("tui_gateway.slash_worker")
        try:
            await client.start(config)
            result = await _call(client, "own-child")
            assert client._server._stdio_child_pids == {result["pid"]}
        finally:
            await client.close()
        assert not psutil.pid_exists(result["pid"])

    asyncio.run(run())
