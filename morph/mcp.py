"""MCP client (R-401 … R-404).

Speaks JSON-RPC 2.0 over stdio and HTTP, performs the spec handshake
(``initialize`` → ``notifications/initialized`` → ``tools/list``) and merges
discovered tools into the registry under ``mcp__<server>__<tool>``.

A server that fails to start, times out, or dies mid-session is isolated: its
tools are dropped and the rest of the system keeps running (R-403).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import MCPServerConfig
from .tools import Tool, ToolError, ToolRegistry

log = logging.getLogger("morph.mcp")

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "morph", "version": "0.1.0"}
NAMESPACE = "mcp__{server}__{tool}"
MAX_LINE = 8 * 1024 * 1024


class MCPError(ToolError):
    """An MCP server misbehaved."""


@dataclass
class MCPTool:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return NAMESPACE.format(server=self.server, tool=self.name)


class MCPConnection:
    """One live connection to an MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.name = config.name
        self.tools: list[MCPTool] = []
        self.server_info: dict[str, Any] = {}
        self.alive = False
        self._process: asyncio.subprocess.Process | None = None
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._next_id = 0
        self._lock = asyncio.Lock()

    # -- lifecycle ------------------------------------------------------
    async def start(self) -> None:
        if self.config.transport == "stdio":
            await self._start_stdio()
        elif self.config.transport == "http":
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        else:
            raise MCPError(
                f"Unsupported MCP transport {self.config.transport!r} for server "
                f"{self.name!r}. Use 'stdio' or 'http'."
            )

        await self._handshake()
        self.tools = await self._list_tools()
        self.alive = True

    async def _start_stdio(self) -> None:
        command = self.config.command
        if not command:
            raise MCPError(f"MCP server {self.name!r} is stdio but has no 'command'")
        if shutil.which(command) is None and not os.path.exists(command):
            raise MCPError(f"MCP server {self.name!r}: command {command!r} not found on PATH")

        env = {**os.environ, **self.config.env}
        try:
            self._process = await asyncio.create_subprocess_exec(
                command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                limit=MAX_LINE,
            )
        except OSError as exc:
            raise MCPError(f"MCP server {self.name!r} failed to start: {exc}") from exc

    async def close(self) -> None:
        self.alive = False
        if self._process is not None and self._process.returncode is None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):  # pragma: no cover
                self._process.kill()
        if self._client is not None:
            await self._client.aclose()

    # -- protocol -------------------------------------------------------
    async def _handshake(self) -> None:
        """``initialize`` then ``notifications/initialized`` (R-404)."""
        result = await self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": CLIENT_INFO,
            },
        )
        self.server_info = result.get("serverInfo", {})
        await self.notify("notifications/initialized")

    async def _list_tools(self) -> list[MCPTool]:
        result = await self.request("tools/list", {})
        return [
            MCPTool(
                server=self.name,
                name=item.get("name", ""),
                description=item.get("description", ""),
                input_schema=item.get("inputSchema") or item.get("input_schema") or {},
            )
            for item in result.get("tools", [])
            if item.get("name")
        ]

    async def call_tool(self, tool: str, arguments: dict[str, Any]) -> str:
        result = await self.request("tools/call", {"name": tool, "arguments": arguments})
        chunks: list[str] = []
        for block in result.get("content", []):
            if block.get("type") == "text":
                chunks.append(block.get("text", ""))
            elif block.get("type") == "image":
                chunks.append(f"[image {block.get('mimeType', 'image/png')}]")
            elif block.get("type") == "resource":
                resource = block.get("resource", {})
                chunks.append(resource.get("text") or f"[resource {resource.get('uri', '')}]")
        body = "\n".join(c for c in chunks if c)
        if result.get("isError"):
            raise MCPError(body or f"{self.name}/{tool} reported an error")
        return body or json.dumps(result.get("structuredContent") or result, ensure_ascii=False)

    # -- transport ------------------------------------------------------
    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            self._next_id += 1
            message = {
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": method,
                "params": params,
            }
            if self.config.transport == "stdio":
                response = await self._stdio_roundtrip(message)
            else:
                response = await self._http_roundtrip(message)

        if "error" in response:
            error = response["error"]
            raise MCPError(
                f"{self.name}: {error.get('message', 'unknown error')} "
                f"(code {error.get('code', '?')})"
            )
        return response.get("result") or {}

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if self.config.transport == "stdio":
            self._write(message)
        elif self._client is not None and self.config.url:
            try:
                await self._client.post(self.config.url, json=message, headers=self._headers())
            except httpx.HTTPError as exc:  # notifications are best-effort
                log.debug("MCP notify to %s failed: %s", self.name, exc)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _write(self, message: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise MCPError(f"MCP server {self.name!r} is not running")
        payload = json.dumps(message, ensure_ascii=False) + "\n"
        self._process.stdin.write(payload.encode("utf-8"))

    async def _stdio_roundtrip(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise MCPError(f"MCP server {self.name!r} is not running")
        if self._process.returncode is not None:
            raise MCPError(f"MCP server {self.name!r} exited with {self._process.returncode}")

        self._write(message)
        try:
            await self._process.stdin.drain()  # type: ignore[union-attr]
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise MCPError(f"MCP server {self.name!r} closed its input: {exc}") from exc

        deadline = self.config.timeout
        while True:
            try:
                line = await asyncio.wait_for(self._process.stdout.readline(), timeout=deadline)
            except asyncio.TimeoutError:
                raise MCPError(
                    f"MCP server {self.name!r} did not respond to {message['method']} "
                    f"within {deadline:g}s"
                ) from None
            if not line:
                raise MCPError(f"MCP server {self.name!r} closed the connection")
            try:
                payload = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue  # servers log to stdout sometimes; skip non-JSON noise
            if payload.get("id") == message["id"]:
                return payload
            # Otherwise it is a notification or a response we are not waiting on.

    async def _http_roundtrip(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._client is None or not self.config.url:
            raise MCPError(f"MCP server {self.name!r} is http but has no 'url'")
        try:
            response = await self._client.post(
                self.config.url, json=message, headers=self._headers()
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MCPError(f"MCP server {self.name!r} HTTP request failed: {exc}") from exc

        session = response.headers.get("Mcp-Session-Id")
        if session:
            self._session_id = session

        if "text/event-stream" in response.headers.get("content-type", ""):
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    try:
                        payload = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if payload.get("id") == message["id"]:
                        return payload
            raise MCPError(f"MCP server {self.name!r} sent no response for {message['method']}")
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise MCPError(f"MCP server {self.name!r} returned non-JSON: {exc}") from exc


class MCPManager:
    """Owns every MCP connection and merges their tools into the registry (R-402)."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.connections: dict[str, MCPConnection] = {}
        self.failures: dict[str, str] = {}

    async def connect_all(self, servers: list[MCPServerConfig]) -> None:
        await asyncio.gather(
            *(self.connect(s) for s in servers if s.enabled), return_exceptions=True
        )

    async def connect(self, config: MCPServerConfig) -> MCPConnection | None:
        connection = MCPConnection(config)
        try:
            await asyncio.wait_for(connection.start(), timeout=config.timeout)
        except (Exception, asyncio.TimeoutError) as exc:  # noqa: BLE001 - isolation (R-403)
            reason = str(exc) or type(exc).__name__
            log.warning("MCP server %r unavailable: %s", config.name, reason)
            self.failures[config.name] = reason
            await connection.close()
            return None

        self.connections[config.name] = connection
        for tool in connection.tools:
            self._register(connection, tool)
        log.info("MCP server %r ready with %d tool(s)", config.name, len(connection.tools))
        return connection

    def _register(self, connection: MCPConnection, tool: MCPTool) -> None:
        source = f"mcp:{connection.name}"

        async def handler(**arguments: Any) -> str:
            if not connection.alive:
                raise MCPError(f"MCP server {connection.name!r} is no longer connected")
            try:
                return await connection.call_tool(tool.name, arguments)
            except MCPError:
                raise
            except Exception as exc:  # noqa: BLE001 - a dead server must not kill the run
                await self.disable(connection.name, f"{type(exc).__name__}: {exc}")
                raise MCPError(
                    f"MCP server {connection.name!r} failed and was disconnected: {exc}"
                ) from exc

        schema = tool.input_schema or {"type": "object", "properties": {}}
        self.registry.add(
            Tool(
                name=tool.qualified_name,
                description=tool.description or f"{tool.name} (via {connection.name})",
                input_schema=schema,
                handler=handler,
                source=source,
            )
        )

    async def disable(self, name: str, reason: str) -> list[str]:
        """Drop a server's tools and record why (R-403)."""
        connection = self.connections.pop(name, None)
        if connection is not None:
            await connection.close()
        self.failures[name] = reason
        return self.registry.remove_source(f"mcp:{name}")

    async def close_all(self) -> None:
        for name in list(self.connections):
            connection = self.connections.pop(name)
            await connection.close()

    def status(self) -> dict[str, Any]:
        return {
            "connected": {
                name: {
                    "tools": [t.name for t in conn.tools],
                    "server_info": conn.server_info,
                }
                for name, conn in self.connections.items()
            },
            "failed": dict(self.failures),
        }
