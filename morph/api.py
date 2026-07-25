"""Framework-free API core (R-601, R-605).

Every endpoint is a plain async method returning either a JSON-able object or an
async iterator of SSE events. :mod:`morph.server` only does HTTP framing, which
keeps the API testable with no sockets and no web framework installed.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from .agent import Agent
from .tools import PathEscapeError, ToolError
from .tools.image import ImageRequest, run_image_flow


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class MorphAPI:
    """The agent, exposed as an API. One instance per server process."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.started_at = time.time()

    async def start(self) -> None:
        await self.agent.start()

    async def close(self) -> None:
        await self.agent.close()

    # -- read-only endpoints -------------------------------------------
    async def health(self) -> dict[str, Any]:
        from . import __version__

        return {
            "status": "ok",
            "version": __version__,
            "provider": self.agent.provider.name,
            "model": getattr(self.agent.provider, "model", self.agent.config.model),
            "image_backend": self.agent.config.image_backend,
            "tools": len(self.agent.tools),
            "skills": len(self.agent.skills),
            "mcp": self.agent.mcp.status(),
            "uptime_s": round(time.time() - self.started_at, 1),
        }

    async def tools(self) -> dict[str, Any]:
        registry = self.agent.tools
        return {
            "tools": [
                {**registry.get(name).spec(), "source": registry.get(name).source}  # type: ignore[union-attr]
                for name in registry.names()
            ]
        }

    async def skills(self) -> dict[str, Any]:
        return {"skills": [s.to_dict() for s in self.agent.skills.all()]}

    async def sessions(self) -> dict[str, Any]:
        return {"sessions": self.agent.sessions.list()}

    async def session(self, session_id: str) -> dict[str, Any]:
        try:
            session = self.agent.sessions.load(session_id)
        except (FileNotFoundError, ValueError) as exc:
            raise ApiError(str(exc), status=404) from exc
        return {**session.to_dict(), "history": session.messages}

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        try:
            deleted = self.agent.sessions.delete(session_id)
        except ValueError as exc:
            raise ApiError(str(exc), status=400) from exc
        if not deleted:
            raise ApiError(f"No session {session_id!r}", status=404)
        return {"deleted": session_id}

    # -- chat ------------------------------------------------------------
    async def chat(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream agent events for one user message (R-601)."""
        message = (payload or {}).get("message") or (payload or {}).get("prompt")
        if not isinstance(message, str) or not message.strip():
            raise ApiError("Field 'message' is required and must be a non-empty string")

        session_id = payload.get("session_id")
        if session_id is not None and not isinstance(session_id, str):
            raise ApiError("Field 'session_id' must be a string")

        max_steps = payload.get("max_steps")
        if max_steps is not None:
            try:
                max_steps = int(max_steps)
            except (TypeError, ValueError) as exc:
                raise ApiError("Field 'max_steps' must be an integer") from exc
            if not 1 <= max_steps <= 200:
                raise ApiError("Field 'max_steps' must be between 1 and 200")

        try:
            async for event in self.agent.stream(
                message, session=session_id, max_steps=max_steps
            ):
                yield event.to_dict()
        except PathEscapeError as exc:  # defence in depth (R-605)
            yield {"type": "error", "message": str(exc)}

    async def chat_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming chat, for clients that cannot consume SSE."""
        result: dict[str, Any] = {}
        async for event in self.chat(payload):
            if event.get("type") == "done":
                result = event.get("result", {})
        return result

    # -- images ----------------------------------------------------------
    async def image(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = payload or {}
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ApiError("Field 'prompt' is required")

        def _int(name: str, default: int) -> int:
            value = payload.get(name, default)
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ApiError(f"Field {name!r} must be an integer") from exc

        seed = payload.get("seed")
        request = ImageRequest(
            prompt=prompt,
            negative_prompt=str(payload.get("negative_prompt") or ""),
            width=_int("width", 512),
            height=_int("height", 512),
            seed=int(seed) if seed is not None else None,
            count=_int("count", 1),
        )
        try:
            result = await run_image_flow(request, self.agent.config)
        except ToolError as exc:
            raise ApiError(str(exc), status=422) from exc

        return {
            "paths": result.paths,
            "previews": result.previews,
            "backend": result.backend,
            "meta": result.meta,
        }


def sse(event: dict[str, Any]) -> bytes:
    """Encode one event as a Server-Sent Event frame."""
    body = json.dumps(event, ensure_ascii=False, default=str)
    return f"event: {event.get('type', 'message')}\ndata: {body}\n\n".encode("utf-8")
