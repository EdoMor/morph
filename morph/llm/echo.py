"""Deterministic offline provider (R-103, R-802).

``EchoProvider`` is how the test suite and the benchmark exercise the full agent
loop with no network and no GPU. It has two modes:

* **scripted** — replay a fixed list of :class:`ModelResponse` objects, so a test
  can drive an exact sequence of tool calls;
* **reflex** — a tiny rule engine that reacts to the conversation, used by
  benchmark tasks that need "a model that basically works" without an LLM.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from .base import Message, ModelResponse, ToolCall

Reflex = Callable[[list[Message], list[dict[str, Any]]], ModelResponse | None]


class EchoProvider:
    name = "echo"
    supports_native_tools = True

    def __init__(
        self,
        script: Sequence[ModelResponse | str] | None = None,
        reflexes: Sequence[Reflex] | None = None,
        final_text: str = "Done.",
    ) -> None:
        self.script: list[ModelResponse] = [
            item if isinstance(item, ModelResponse) else ModelResponse(text=item)
            for item in (script or [])
        ]
        self.reflexes = list(reflexes or [])
        self.final_text = final_text
        self.calls: list[list[Message]] = []
        self.cursor = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(list(messages))

        if self.cursor < len(self.script):
            response = self.script[self.cursor]
            self.cursor += 1
            return response

        for reflex in self.reflexes:
            result = reflex(messages, tools or [])
            if result is not None:
                return result

        return ModelResponse(text=self._summarise(messages), stop_reason="end_turn")

    # ------------------------------------------------------------------
    def _summarise(self, messages: list[Message]) -> str:
        last_tool = next(
            (m for m in reversed(messages) if m.get("role") == "tool"),
            None,
        )
        if last_tool is not None:
            body = str(last_tool.get("content", ""))[:500]
            return f"{self.final_text}\n\n{body}".strip()
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if last_user is not None:
            return f"{self.final_text} (echo: {str(last_user.get('content', ''))[:200]})"
        return self.final_text

    # ------------------------------------------------------------------
    @staticmethod
    def call(name: str, /, **arguments: Any) -> ModelResponse:
        """Convenience constructor: one scripted tool call."""
        return ModelResponse(
            text="",
            tool_calls=[ToolCall(name=name, arguments=arguments)],
            stop_reason="tool_use",
        )

    @staticmethod
    def text_response(text: str) -> ModelResponse:
        return ModelResponse(text=text, stop_reason="end_turn")


def json_reflex(pattern: str, tool: str, **arguments: Any) -> Reflex:
    """Reflex that fires a tool call when ``pattern`` appears in the last user turn."""

    def _reflex(messages: list[Message], _tools: list[dict[str, Any]]) -> ModelResponse | None:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                if pattern.lower() in json.dumps(msg.get("content", "")).lower():
                    return EchoProvider.call(tool, **arguments)
                return None
        return None

    return _reflex
