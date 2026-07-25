"""Model-provider interface and the text tool-call protocol.

Providers are pluggable (R-103). Providers without native function calling still
support tools through a documented text protocol (R-105).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

Message = dict[str, Any]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class ProviderError(RuntimeError):
    """Raised when a provider cannot serve a request.

    Message must be actionable: name the env var or the service to start,
    never leak a bare traceback to the user (R-505).
    """


@runtime_checkable
class Provider(Protocol):
    name: str
    supports_native_tools: bool

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse: ...


# ----------------------------------------------------------------------------
# Text tool-call protocol (R-105)
# ----------------------------------------------------------------------------

TOOL_BLOCK_RE = re.compile(
    r"```tool_call\s*\n(?P<body>.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)

TEXT_PROTOCOL_INSTRUCTIONS = """\
## Calling tools

You do not have native function calling. To call a tool, emit a fenced block
tagged `tool_call` containing a single JSON object with `name` and `arguments`:

```tool_call
{"name": "read_file", "arguments": {"path": "morph/agent.py"}}
```

Rules:
- One JSON object per block. Emit several blocks to call several tools.
- `arguments` must match the tool's schema exactly. No comments, no trailing commas.
- Stop after emitting your tool calls; the results come back in the next turn.
- When you are done and need no more tools, reply with plain prose and no
  `tool_call` block.
"""


def parse_text_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Split model output into prose and tool calls.

    Returns ``(prose, calls)``. Malformed JSON inside a block is left in the
    prose rather than raising, so a sloppy model degrades to a normal answer
    instead of crashing the run (R-109).
    """
    calls: list[ToolCall] = []
    leftovers: list[str] = []
    cursor = 0

    for match in TOOL_BLOCK_RE.finditer(text):
        leftovers.append(text[cursor : match.start()])
        cursor = match.end()
        body = match.group("body").strip()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            leftovers.append(match.group(0))
            continue
        if not isinstance(payload, dict) or "name" not in payload:
            leftovers.append(match.group(0))
            continue
        args = payload.get("arguments") or payload.get("args") or {}
        if not isinstance(args, dict):
            args = {"value": args}
        calls.append(ToolCall(name=str(payload["name"]), arguments=args))

    leftovers.append(text[cursor:])
    return "".join(leftovers).strip(), calls


def render_tools_for_text_protocol(tools: list[dict[str, Any]]) -> str:
    """Render tool schemas into the system prompt for text-protocol providers."""
    if not tools:
        return ""
    lines = [TEXT_PROTOCOL_INSTRUCTIONS, "\n## Available tools\n"]
    for tool in tools:
        schema = json.dumps(tool.get("input_schema", {}), separators=(",", ":"))
        lines.append(f"### {tool['name']}\n{tool.get('description', '').strip()}\n")
        lines.append(f"Schema: `{schema}`\n")
    return "\n".join(lines)


def messages_to_prompt(messages: list[Message]) -> str:
    """Flatten a conversation into a single prompt for completion-style APIs."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):  # content blocks
            content = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        if role == "tool":
            name = msg.get("name", "tool")
            parts.append(f"<tool_result name=\"{name}\">\n{content}\n</tool_result>")
        else:
            parts.append(f"<{role}>\n{content}\n</{role}>")
    parts.append("<assistant>")
    return "\n\n".join(parts)
