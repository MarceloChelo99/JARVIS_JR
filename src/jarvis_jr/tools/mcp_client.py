"""MCP (Model Context Protocol) client.

Connects to MCP servers (stdio subprocesses), discovers their tools, and
exposes them in the same Anthropic-style schema format the rest of jarvis_jr
uses. The MCP SDK is asyncio-based; jarvis_jr's loop is synchronous, so all
sessions live on a background thread running one event loop, and `call()` is
a blocking bridge into it.

Tool names are exposed as "<server>_<tool>" to avoid collisions and to make
confirmation config unambiguous (e.g. require_for: ["files_write_file"]).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any


class MCPError(RuntimeError):
    pass


class MCPManager:
    """Owns connections to all configured MCP servers.

    `servers` config shape (from configs/default.yaml):
        - name: "files"
          command: "npx"
          args: ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/notes"]
          env: {}            # optional extra environment variables
    """

    def __init__(self, servers: list[dict[str, Any]], call_timeout_s: float = 30.0):
        self._configs = servers
        self.call_timeout_s = call_timeout_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, Any] = {}  # server name -> ClientSession
        self._tools: dict[str, tuple[str, str]] = {}  # exposed name -> (server, tool)
        self._schemas: list[dict[str, Any]] = []
        self._shutdown: asyncio.Event | None = None
        self._stopped = threading.Event()

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Connect to all servers and discover tools. Blocks until ready."""
        if not self._configs:
            return
        ready = threading.Event()
        errors: list[str] = []

        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._main(ready, errors))
            finally:
                self._loop.close()
                self._stopped.set()

        self._thread = threading.Thread(target=run, daemon=True, name="mcp-manager")
        self._thread.start()
        if not ready.wait(timeout=60):
            raise MCPError("Timed out starting MCP servers.")
        for err in errors:
            print(f"[mcp] {err}")

    async def _main(self, ready: threading.Event, errors: list[str]) -> None:
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import get_default_environment, stdio_client

        self._shutdown = asyncio.Event()
        async with AsyncExitStack() as stack:
            for cfg in self._configs:
                name = cfg["name"]
                try:
                    # Merge per-server env on top of the defaults (PATH etc.) —
                    # passing only the extras would break launchers like uvx/npx.
                    env = None
                    if cfg.get("env"):
                        env = {**get_default_environment(), **cfg["env"]}
                    params = StdioServerParameters(
                        command=cfg["command"],
                        args=cfg.get("args", []),
                        env=env,
                    )
                    read, write = await stack.enter_async_context(stdio_client(params))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    listed = await session.list_tools()
                    self._sessions[name] = session
                    for tool in listed.tools:
                        exposed = f"{name}_{tool.name}"
                        self._tools[exposed] = (name, tool.name)
                        self._schemas.append(
                            {
                                "name": exposed,
                                "description": tool.description or f"{tool.name} ({name})",
                                "input_schema": tool.inputSchema
                                or {"type": "object", "properties": {}},
                            }
                        )
                    print(f"[mcp] {name}: {len(listed.tools)} tool(s) connected.")
                except Exception as e:
                    errors.append(f"server '{name}' failed to start: {type(e).__name__}: {e}")
            ready.set()
            await self._shutdown.wait()

    def stop(self) -> None:
        if self._loop is None or self._shutdown is None:
            return
        self._loop.call_soon_threadsafe(self._shutdown.set)
        self._stopped.wait(timeout=10)

    # ---- tool surface ------------------------------------------------------

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return self._schemas

    def owns(self, exposed_name: str) -> bool:
        return exposed_name in self._tools

    def call(self, exposed_name: str, args: dict[str, Any]) -> str:
        """Synchronously call an MCP tool; returns its text content."""
        if self._loop is None:
            raise MCPError("MCPManager not started.")
        server, tool = self._tools[exposed_name]
        session = self._sessions[server]
        future = asyncio.run_coroutine_threadsafe(
            session.call_tool(tool, args or {}), self._loop
        )
        result = future.result(timeout=self.call_timeout_s)
        parts: list[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(f"[{getattr(item, 'type', 'non-text')} content]")
        out = "\n".join(parts) if parts else "(empty result)"
        if getattr(result, "isError", False):
            return f"ERROR from {exposed_name}: {out}"
        return out
