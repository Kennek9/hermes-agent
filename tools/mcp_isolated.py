"""Explicit MCP connections for callers that own their authorization policy.

These clients share Hermes' transport, OAuth and child-process lifecycle, while
keeping personal catalogs and diagnostics out of the model-tool registry. They
use only the supplied server definition and never replay a tool call.
"""

from __future__ import annotations

import asyncio
import contextvars
from copy import deepcopy
from typing import Any, TextIO

from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_handlers import _call_tool_racing_stdio_death, _track_inflight_rpc


class IsolatedMCPClient:
    """One caller-owned connection; a separate client is required after closing.

    ``stderr=None`` discards provider stderr. An explicit file is borrowed and
    stays open after ``close()``. The caller controls tool allowlists, argument
    validation and permission checks before invoking ``call_tool``.
    """

    def __init__(self, name: str, *, stderr: TextIO | None = None,
                 follow_redirects: bool = False):
        self._server = MCPServerTask(
            name, isolated=True, stderr=stderr, follow_redirects=follow_redirects)
        self._started = False
        self._closed = False
        self._close_task: asyncio.Task | None = None

    async def start(self, config: dict[str, Any]) -> None:
        """Connect and discover using a private copy of the supplied definition."""
        if self._started or self._closed:
            raise RuntimeError("An isolated MCP client can only be started once")
        self._started = True
        definition = deepcopy(config)
        # The initialize exchange is authoritative; a separate probe adds requests
        # and can disclose caller headers to endpoints unrelated to that exchange.
        definition["skip_preflight"] = True
        if definition.get("transport") == "streamable_http":
            definition["transport"] = "http"
        timeout = float(definition.get("connect_timeout", 60.0))
        try:
            await asyncio.wait_for(self._server.start(definition), timeout=timeout)
        except BaseException:
            await self.close()
            raise

    def discovered_tools(self) -> list[dict[str, Any]]:
        """Return independent MCP tool definitions, including annotations."""
        return deepcopy([
            tool.model_dump(by_alias=True, exclude_none=True)
            for tool in self._server._tools
        ])

    async def call_tool(self, name: str, arguments: dict[str, Any], *,
                        timeout: float | None = None) -> dict[str, Any]:
        """Call exactly once and return the native MCP result without rendering.

        Transport failures, timeouts and cancellation propagate to the caller.
        A failure after dispatch may have changed external state; this API never
        invokes the registry handler's authentication or session retry ladder.
        """
        if not self._started or self._closed:
            raise RuntimeError("The isolated MCP client is not connected")

        async def invoke():
            server = self._server
            async with server._rpc_lock:
                if self._closed or server.session is None:
                    raise ConnectionError("The isolated MCP transport is unavailable")
                server.mark_tool_call()
                async with _track_inflight_rpc(
                        server, server.name, f"tools/call {name}", retry_safe=False):
                    server._pending_call_context = contextvars.copy_context()
                    try:
                        result = await _call_tool_racing_stdio_death(
                            server, server.name, name, deepcopy(arguments))
                    finally:
                        server._pending_call_context = None
                server._mark_session_proven()
                return result.model_dump(by_alias=True, exclude_none=True)

        return await asyncio.wait_for(
            invoke(), timeout=self._server.tool_timeout if timeout is None else timeout)

    async def close(self) -> None:
        """Stop the transport and reap its stdio children; borrowed stderr stays open."""
        if self._close_task is None:
            self._closed = True
            # The transport's task owns its anyio cancel scopes. A caller that
            # cancels close() must not cancel or detach that task's cleanup.
            self._close_task = asyncio.create_task(self._server.shutdown())
        await asyncio.shield(self._close_task)
