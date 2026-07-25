#!/usr/bin/env python3
"""A real (if small) MCP server over stdio, used to test the client end to end.

Speaks JSON-RPC 2.0 and implements the handshake plus ``tools/list`` and
``tools/call``. Every message it receives is appended to the file named by
``FAKE_MCP_LOG`` so tests can assert on the handshake order (R-404).

Env knobs:
  FAKE_MCP_TOOLS   comma-separated tool names to expose (default: echo)
  FAKE_MCP_LOG     path to write the received-message log
  FAKE_MCP_CRASH   if set, exit abruptly after this many requests
"""

from __future__ import annotations

import json
import os
import sys

TOOL_SCHEMAS = {
    "echo": {
        "description": "Echo the text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    "add": {
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    "boom": {
        "description": "Always returns an error result.",
        "inputSchema": {"type": "object", "properties": {}},
    },
}


def main() -> int:
    exposed = [t.strip() for t in os.environ.get("FAKE_MCP_TOOLS", "echo").split(",") if t.strip()]
    log_path = os.environ.get("FAKE_MCP_LOG")
    crash_after = int(os.environ.get("FAKE_MCP_CRASH") or 0)
    received: list[dict] = []
    handled = 0

    def record(message: dict) -> None:
        received.append({"method": message.get("method"), "jsonrpc": message.get("jsonrpc")})
        if log_path:
            with open(log_path, "w", encoding="utf-8") as handle:
                json.dump(received, handle)

    def reply(payload: dict) -> None:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue

        record(message)
        method = message.get("method")
        request_id = message.get("id")

        if request_id is None:  # notification
            continue

        handled += 1
        if crash_after and handled > crash_after:
            return 1

        if method == "initialize":
            reply(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake", "version": "1.0.0"},
                    },
                }
            )
        elif method == "tools/list":
            reply(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {"name": name, **TOOL_SCHEMAS[name]}
                            for name in exposed
                            if name in TOOL_SCHEMAS
                        ]
                    },
                }
            )
        elif method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name == "echo":
                result = {"content": [{"type": "text", "text": str(arguments.get("text", ""))}]}
            elif name == "add":
                total = float(arguments.get("a", 0)) + float(arguments.get("b", 0))
                result = {"content": [{"type": "text", "text": str(total)}]}
            elif name == "boom":
                result = {"content": [{"type": "text", "text": "tool exploded"}], "isError": True}
            else:
                reply(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"Unknown tool {name!r}"},
                    }
                )
                continue
            reply({"jsonrpc": "2.0", "id": request_id, "result": result})
        else:
            reply(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
