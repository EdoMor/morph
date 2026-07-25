#!/usr/bin/env python3
"""A benchmark MCP server: real JSON-RPC over stdio, with useful misbehaviour.

Richer than the one in ``tests/`` because the benchmark needs servers that do
more than echo — a stateful store to chain calls through, a strict schema to get
arguments wrong against, a tool that fails the first time, and one that is slow
enough to time out.

Env knobs:
  MCP_TOOLS      comma-separated subset of tools to expose (default: all)
  MCP_STATE      path to a JSON file the kv store persists to
  MCP_FLAKY_N    how many times ``flaky_fetch`` fails before succeeding (default 1)
  MCP_DIE_AFTER  exit abruptly after this many tools/call requests
"""

from __future__ import annotations

import json
import os
import sys
import time

PROTOCOL_VERSION = "2025-06-18"

TOOLS: dict[str, dict] = {
    "kv_set": {
        "description": "Store a value under a key. Returns the key that was written.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    },
    "kv_get": {
        "description": "Read the value stored under a key. Errors if the key is absent.",
        "inputSchema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    "kv_keys": {
        "description": "List every key currently in the store.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "convert_units": {
        "description": (
            "Convert a distance between units. 'unit_from' and 'unit_to' must each be "
            "one of: km, mi, m, ft. Returns the converted number only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "unit_from": {"type": "string", "enum": ["km", "mi", "m", "ft"]},
                "unit_to": {"type": "string", "enum": ["km", "mi", "m", "ft"]},
            },
            "required": ["value", "unit_from", "unit_to"],
        },
    },
    "flaky_fetch": {
        "description": "Fetch a record. Unreliable — retry if it fails.",
        "inputSchema": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
        },
    },
    "slow_scan": {
        "description": "Scan a large corpus. Takes a long time.",
        "inputSchema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
}

TO_METRES = {"km": 1000.0, "mi": 1609.344, "m": 1.0, "ft": 0.3048}


def main() -> int:
    exposed = [t.strip() for t in os.environ.get("MCP_TOOLS", ",".join(TOOLS)).split(",") if t.strip()]
    state_path = os.environ.get("MCP_STATE")
    flaky_budget = int(os.environ.get("MCP_FLAKY_N") or 1)
    die_after = int(os.environ.get("MCP_DIE_AFTER") or 0)

    store: dict[str, str] = {}
    flaky_seen = 0
    calls = 0

    def persist() -> None:
        if state_path:
            with open(state_path, "w", encoding="utf-8") as handle:
                json.dump(store, handle)

    def reply(payload: dict) -> None:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    def ok(request_id, text: str) -> None:
        reply({"jsonrpc": "2.0", "id": request_id, "result": {"content": [{"type": "text", "text": text}]}})

    def tool_error(request_id, text: str) -> None:
        reply(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": text}], "isError": True},
            }
        )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = message.get("method")
        request_id = message.get("id")
        if request_id is None:
            continue  # notification

        if method == "initialize":
            reply(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "bench-mcp", "version": "1.0.0"},
                    },
                }
            )
            continue

        if method == "tools/list":
            reply(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [{"name": n, **TOOLS[n]} for n in exposed if n in TOOLS]
                    },
                }
            )
            continue

        if method != "tools/call":
            reply(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )
            continue

        calls += 1
        if die_after and calls > die_after:
            return 1

        params = message.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}

        if name == "kv_set":
            if not isinstance(args.get("key"), str) or not isinstance(args.get("value"), str):
                tool_error(request_id, "kv_set requires string 'key' and string 'value'")
                continue
            store[args["key"]] = args["value"]
            persist()
            ok(request_id, args["key"])

        elif name == "kv_get":
            key = args.get("key")
            if key not in store:
                tool_error(
                    request_id,
                    f"no such key {key!r}. Known keys: {', '.join(sorted(store)) or '(none)'}",
                )
                continue
            ok(request_id, store[key])

        elif name == "kv_keys":
            ok(request_id, ", ".join(sorted(store)) or "(empty)")

        elif name == "convert_units":
            unit_from, unit_to = args.get("unit_from"), args.get("unit_to")
            if unit_from not in TO_METRES or unit_to not in TO_METRES:
                tool_error(
                    request_id,
                    f"units must be one of km, mi, m, ft — got {unit_from!r} and {unit_to!r}",
                )
                continue
            try:
                value = float(args.get("value"))
            except (TypeError, ValueError):
                tool_error(request_id, "'value' must be a number")
                continue
            converted = value * TO_METRES[unit_from] / TO_METRES[unit_to]
            ok(request_id, f"{converted:.4f}")

        elif name == "flaky_fetch":
            flaky_seen += 1
            if flaky_seen <= flaky_budget:
                tool_error(request_id, "upstream unavailable (503) — this is transient, retry")
                continue
            ok(request_id, f"record {args.get('record_id')}: status=active, owner=morph")

        elif name == "slow_scan":
            time.sleep(120)
            ok(request_id, "done")

        else:
            reply(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unknown tool {name!r}"},
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
