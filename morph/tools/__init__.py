"""Tool registry and the tool-call contract (R-201, R-207)."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

Handler = Callable[..., Any | Awaitable[Any]]


class ToolError(Exception):
    """A tool failed in a way the model should see and can recover from (R-109)."""


class PathEscapeError(ToolError):
    """Attempted access outside the workspace root (R-205)."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    source: str = "native"  # "native" | "mcp:<server>" | "skill"

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolResult:
    """Outcome of one tool invocation. Always recorded (R-108)."""

    tool: str
    ok: bool
    content: str
    duration_ms: float = 0.0
    arguments: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "content": self.content,
            "duration_ms": round(self.duration_ms, 2),
            "arguments": self.arguments,
            "meta": self.meta,
        }


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Minimal JSON-Schema check: required keys and primitive types (R-207).

    Deliberately dependency-free and permissive about what it does not
    understand — the goal is to turn obvious model mistakes into a readable tool
    error, not to be a conformant validator.
    """
    errors: list[str] = []
    for key in schema.get("required", []):
        if key not in arguments or arguments[key] is None:
            errors.append(f"missing required argument {key!r}")

    types: dict[str, tuple[type, ...]] = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "array": (list, tuple),
        "object": (dict,),
    }
    for key, spec in (schema.get("properties") or {}).items():
        if key not in arguments or arguments[key] is None:
            continue
        expected = spec.get("type")
        if isinstance(expected, str) and expected in types:
            if expected == "integer" and isinstance(arguments[key], bool):
                errors.append(f"argument {key!r} must be integer, got boolean")
            elif not isinstance(arguments[key], types[expected]):
                got = type(arguments[key]).__name__
                errors.append(f"argument {key!r} must be {expected}, got {got}")
        allowed = spec.get("enum")
        if allowed and arguments[key] not in allowed:
            errors.append(f"argument {key!r} must be one of {allowed}")
    return errors


class ToolRegistry:
    """Holds every tool the model can call, native or MCP-provided (R-201, R-402)."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Handler,
        source: str = "native",
    ) -> Tool:
        tool = Tool(name, description, input_schema, handler, source)
        self._tools[name] = tool
        return tool

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def remove_source(self, source: str) -> list[str]:
        """Drop every tool from one source. Used to isolate a dead MCP server (R-403)."""
        dropped = [n for n, t in self._tools.items() if t.source == source]
        for name in dropped:
            del self._tools[name]
        return dropped

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [self._tools[n].spec() for n in self.names()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Invoke a tool. Never raises — failures come back as ``ok=False`` (R-109)."""
        started = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool=name,
                ok=False,
                content=(
                    f"No tool named {name!r}. Available tools: {', '.join(self.names())}"
                ),
                duration_ms=(time.perf_counter() - started) * 1000,
                arguments=arguments,
            )

        errors = validate_arguments(tool.input_schema, arguments)
        if errors:
            return ToolResult(
                tool=name,
                ok=False,
                content="Invalid arguments: " + "; ".join(errors),
                duration_ms=(time.perf_counter() - started) * 1000,
                arguments=arguments,
            )

        try:
            result = tool.handler(**arguments)
            if inspect.isawaitable(result):
                result = await result
            ok, content, meta = _normalise(result)
        except ToolError as exc:
            ok, content, meta = False, str(exc), {}
        except TypeError as exc:
            ok, content, meta = False, f"Invalid arguments: {exc}", {}
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            ok, content, meta = False, f"{type(exc).__name__}: {exc}", {}

        return ToolResult(
            tool=name,
            ok=ok,
            content=content,
            duration_ms=(time.perf_counter() - started) * 1000,
            arguments=arguments,
            meta=meta,
        )


def _normalise(result: Any) -> tuple[bool, str, dict[str, Any]]:
    if isinstance(result, ToolResult):
        return result.ok, result.content, result.meta
    if isinstance(result, tuple) and len(result) == 2:
        content, meta = result
        return True, str(content), dict(meta)
    if result is None:
        return True, "", {}
    return True, result if isinstance(result, str) else repr(result), {}


def build_default_registry(config: Any) -> ToolRegistry:
    """Assemble the standard tool set for a workspace."""
    from .files import register_file_tools
    from .image import register_image_tools
    from .shell import register_shell_tools
    from .web import register_web_tools

    registry = ToolRegistry()
    register_file_tools(registry, config)
    if getattr(config, "allow_shell", True):
        register_shell_tools(registry, config)
    register_web_tools(registry, config)
    register_image_tools(registry, config)
    return registry


__all__ = [
    "Handler",
    "PathEscapeError",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "validate_arguments",
]
