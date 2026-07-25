"""Google AI provider — hosted Gemma / Gemini via the generativelanguage API.

Used when the phone or Codespace cannot host a local Gemma. Native function
calling is available for Gemini models; Gemma models on this endpoint fall back
to the text protocol (R-105).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .base import (
    Message,
    ModelResponse,
    ProviderError,
    ToolCall,
    parse_tool_calls,
    render_tools_for_text_protocol,
)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
API_KEY_ENV = "GOOGLE_API_KEY"


class GoogleProvider:
    name = "google"

    def __init__(
        self,
        model: str = "gemma-3-27b-it",
        base_url: str | None = None,
        temperature: float = 0.2,
        api_key: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self._api_key = api_key

    @property
    def supports_native_tools(self) -> bool:
        # Gemma endpoints reject `tools`; Gemini accepts them.
        return self.model.lower().startswith("gemini")

    @property
    def api_key(self) -> str:
        key = self._api_key or os.environ.get(API_KEY_ENV)
        if not key:
            raise ProviderError(
                f"Missing API key. Set {API_KEY_ENV} in the environment "
                "(never in a file — see R-803)."
            )
        return key

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        model = kwargs.get("model", self.model)
        contents, system_instruction = self._to_contents(messages, tools)

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.temperature),
            },
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if tools and self.supports_native_tools:
            body["tools"] = [{"functionDeclarations": [self._declare(t) for t in tools]}]

        url = f"{self.base_url}/models/{model}:generateContent"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url, json=body, headers={"x-goog-api-key": self.api_key}
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Google API returned {exc.response.status_code}: {exc.response.text[:400]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Google API request failed: {exc}") from exc

        return self._parse(data)

    # ------------------------------------------------------------------
    @staticmethod
    def _declare(tool: dict[str, Any]) -> dict[str, Any]:
        schema = dict(tool.get("input_schema") or {})
        schema.pop("$schema", None)
        schema.pop("additionalProperties", None)
        return {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": schema or {"type": "object", "properties": {}},
        }

    def _to_contents(
        self, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> tuple[list[dict[str, Any]], str]:
        system_bits: list[str] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False)

            if role == "system":
                system_bits.append(str(content))
                continue
            if role == "tool":
                name = msg.get("name", "tool")
                if self.supports_native_tools:
                    contents.append(
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "functionResponse": {
                                        "name": name,
                                        "response": {"result": str(content)},
                                    }
                                }
                            ],
                        }
                    )
                else:
                    contents.append(
                        {
                            "role": "user",
                            "parts": [
                                {"text": f'<tool_result name="{name}">\n{content}\n</tool_result>'}
                            ],
                        }
                    )
                continue

            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": str(content)}],
                }
            )

        if tools and not self.supports_native_tools:
            system_bits.append(render_tools_for_text_protocol(tools))
        return contents, "\n\n".join(b for b in system_bits if b)

    @staticmethod
    def _parse(data: dict[str, Any]) -> ModelResponse:
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            return ModelResponse(text="", stop_reason=f"blocked:{reason}", raw=data)

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text_bits: list[str] = []
        calls: list[ToolCall] = []
        for part in parts:
            if "text" in part:
                text_bits.append(part["text"])
            if "functionCall" in part:
                fc = part["functionCall"]
                calls.append(ToolCall(name=fc.get("name", ""), arguments=fc.get("args") or {}))

        text = "".join(text_bits)
        errors: list[str] = []
        if not calls:  # text-protocol fallback
            parsed = parse_tool_calls(text)
            text, calls, errors = parsed.text, parsed.calls, parsed.errors

        usage = data.get("usageMetadata") or {}
        return ModelResponse(
            text=text.strip(),
            tool_calls=calls,
            malformed_calls=errors,
            stop_reason="tool_use" if calls else "end_turn",
            usage={
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
            },
            raw=data,
        )
