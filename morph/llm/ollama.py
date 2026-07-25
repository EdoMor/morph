"""Ollama provider — this is how Gemma runs locally (R-103, R-104).

Ollama's ``/api/chat`` endpoint supports native tool calling for some models but
not reliably for Gemma, so this provider drives the text protocol from
:mod:`morph.llm.base` and parses tool calls out of the reply (R-105).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .base import (
    Message,
    ModelResponse,
    ProviderError,
    messages_to_prompt,
    parse_tool_calls,
    render_tools_for_text_protocol,
)

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class OllamaProvider:
    name = "ollama"
    supports_native_tools = False

    def __init__(
        self,
        model: str = "gemma3:12b",
        base_url: str | None = None,
        temperature: float = 0.2,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        payload_messages = self._prepare(messages, tools)
        body = {
            "model": kwargs.get("model", self.model),
            "messages": payload_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_ctx": kwargs.get("context_tokens", 32_768),
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=body)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve` "
                f"and pull the model with `ollama pull {self.model}`."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400]
            if exc.response.status_code == 404:
                raise ProviderError(
                    f"Ollama has no model named {self.model!r}. "
                    f"Run `ollama pull {self.model}`."
                ) from exc
            raise ProviderError(f"Ollama returned {exc.response.status_code}: {detail}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        raw_text = (data.get("message") or {}).get("content", "")
        parsed = parse_tool_calls(raw_text)
        return ModelResponse(
            text=parsed.text,
            tool_calls=parsed.calls,
            malformed_calls=parsed.errors,
            stop_reason="tool_use" if parsed.calls else "end_turn",
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
            raw=data,
        )

    # ------------------------------------------------------------------
    def _prepare(
        self, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        """Fold tool schemas into the system prompt and flatten tool results.

        Ollama's chat API has no ``tool`` role for Gemma, so tool results are
        delivered as user turns wrapped in ``<tool_result>`` markers.
        """
        prepared: list[dict[str, Any]] = []
        tool_prompt = render_tools_for_text_protocol(tools or [])
        injected = not tool_prompt

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False)

            if role == "system" and not injected:
                content = f"{content}\n\n{tool_prompt}"
                injected = True
            elif role == "tool":
                role = "user"
                name = msg.get("name", "tool")
                content = f'<tool_result name="{name}">\n{content}\n</tool_result>'

            prepared.append({"role": role, "content": content})

        if not injected:
            prepared.insert(0, {"role": "system", "content": tool_prompt})
        return prepared

    async def generate(self, prompt_messages: list[Message], **kwargs: Any) -> str:
        """Raw completion helper, used by the benchmark's efficiency probes."""
        response = await self.complete(
            [{"role": "user", "content": messages_to_prompt(prompt_messages)}], **kwargs
        )
        return response.text
