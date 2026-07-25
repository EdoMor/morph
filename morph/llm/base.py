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
    #: Tool-call blocks that could not be parsed. Never silently discarded —
    #: the agent hands these back to the model so it can correct itself.
    malformed_calls: list[str] = field(default_factory=list)


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
- **Backslashes must be doubled**, because this is JSON. A regex for `average(`
  is written `"average\\\\("`, not `"average\\("`. Getting this wrong is the most
  common way a tool call fails.
- Stop after emitting your tool calls; the results come back in the next turn.
- When you are done and need no more tools, reply with plain prose and no
  `tool_call` block.
"""


# JSON accepts a backslash only before one of `"\/bfnrtu`. A model writing a
# regex emits `\)` or `\d` constantly — invalid JSON, but unambiguously meant as
# a literal backslash.
#
# The alternation matters: valid escapes are consumed as a unit so that the pair
# `\\` is recognised as one escaped backslash. Scanning character by character
# instead would see the second backslash of `\\(` as a stray one and "repair" a
# string that was already correct.
ESCAPE_RE = re.compile(r'\\(?:(["\\/bfnrtu]|u[0-9a-fA-F]{4})|(.))', re.DOTALL)


def repair_json(body: str) -> str:
    """Double the stray backslashes that make a model's regex invalid JSON."""

    def fix(match: re.Match[str]) -> str:
        if match.group(1) is not None:
            return match.group(0)  # already a valid escape
        return "\\\\" + match.group(2)

    return ESCAPE_RE.sub(fix, body)


def _load_tool_payload(body: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a tool-call body, repairing invalid escapes before giving up."""
    try:
        return json.loads(body), None
    except json.JSONDecodeError as first:
        try:
            return json.loads(repair_json(body)), None
        except json.JSONDecodeError:
            return None, f"{first.msg} at position {first.pos}"


@dataclass
class ParsedToolCalls:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_tool_calls(text: str) -> ParsedToolCalls:
    """Split model output into prose, tool calls, and parse failures.

    A block that cannot be parsed is reported rather than dropped. Silently
    treating it as prose ends the run at that step with no explanation to the
    model and no changes made — which is precisely what a small model emitting
    an unescaped regex used to trigger.
    """
    calls: list[ToolCall] = []
    errors: list[str] = []
    leftovers: list[str] = []
    cursor = 0

    for match in TOOL_BLOCK_RE.finditer(text):
        leftovers.append(text[cursor : match.start()])
        cursor = match.end()
        body = match.group("body").strip()

        payload, error = _load_tool_payload(body)
        if payload is None:
            errors.append(f"{error}. Block was: {body[:300]}")
            continue
        if not isinstance(payload, dict) or "name" not in payload:
            errors.append(
                f"a tool_call block must be a JSON object with a \"name\". Block was: {body[:300]}"
            )
            continue

        args = payload.get("arguments") or payload.get("args") or {}
        if not isinstance(args, dict):
            args = {"value": args}
        calls.append(ToolCall(name=str(payload["name"]), arguments=args))

    leftovers.append(text[cursor:])
    return ParsedToolCalls(text="".join(leftovers).strip(), calls=calls, errors=errors)


def parse_text_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Backwards-compatible view of :func:`parse_tool_calls`."""
    parsed = parse_tool_calls(text)
    return parsed.text, parsed.calls


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
