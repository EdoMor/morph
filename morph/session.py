"""Append-only JSONL session persistence (R-106)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Session:
    """A conversation. Every turn is appended; nothing is ever rewritten."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    _path: Path | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def append(self, message: dict[str, Any]) -> dict[str, Any]:
        record = {**message, "ts": message.get("ts", time.time())}
        self.messages.append(record)
        self.updated_at = record["ts"]
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    def user(self, content: str) -> dict[str, Any]:
        return self.append({"role": "user", "content": content})

    def assistant(self, content: str, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return self.append(message)

    def tool(self, name: str, content: str, call_id: str = "", ok: bool = True) -> dict[str, Any]:
        return self.append(
            {"role": "tool", "name": name, "content": content, "tool_call_id": call_id, "ok": ok}
        )

    def for_model(self) -> list[dict[str, Any]]:
        """Strip bookkeeping fields the providers do not need."""
        drop = {"ts", "ok"}
        return [{k: v for k, v in m.items() if k not in drop} for m in self.messages]

    @property
    def title(self) -> str:
        for message in self.messages:
            if message.get("role") == "user":
                return str(message.get("content", ""))[:80]
        return "(empty session)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": len(self.messages),
            "metadata": self.metadata,
        }


class SessionStore:
    """Filesystem-backed session storage: one JSONL file per session."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError(f"Invalid session id: {session_id!r}")
        return self.directory / f"{safe}.jsonl"

    def create(self, session_id: str | None = None, **metadata: Any) -> Session:
        session = Session(id=session_id or uuid.uuid4().hex[:16], metadata=metadata)
        session._path = self._path(session.id)
        session._path.touch()
        return session

    def load(self, session_id: str) -> Session:
        """Resume a session with every tool call and result intact (R-106)."""
        path = self._path(session_id)
        if not path.is_file():
            raise FileNotFoundError(f"No session {session_id!r} in {self.directory}")

        session = Session(id=session_id)
        session._path = path
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                session.messages.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn final write must not make the session unloadable
        if session.messages:
            session.created_at = session.messages[0].get("ts", session.created_at)
            session.updated_at = session.messages[-1].get("ts", session.updated_at)
        return session

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id:
            try:
                return self.load(session_id)
            except FileNotFoundError:
                return self.create(session_id)
        return self.create()

    def list(self) -> list[dict[str, Any]]:
        sessions = []
        for path in self.directory.glob("*.jsonl"):
            try:
                sessions.append(self.load(path.stem).to_dict())
            except (OSError, ValueError):
                continue
        return sorted(sessions, key=lambda s: s["updated_at"], reverse=True)

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.is_file():
            path.unlink()
            return True
        return False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.list())
