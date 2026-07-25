"""Live trace rendering (R-718).

The agent already emits a structured event per step (R-107); until now the
self-improvement loop consumed that stream silently and printed a summary an
hour later. Everything interesting — which tool, which arguments, what came
back, where it got stuck — was happening invisibly.

This renders that stream as it happens, so a run can be watched rather than
waited on.

Everything here writes to **stderr** on purpose. The loop and the benchmark both
emit machine-readable JSON on stdout that CI pipes into a file; a trace line in
the middle of that would corrupt it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

MAX_ARG_CHARS = 150
MAX_RESULT_CHARS = 320
MAX_TEXT_CHARS = 600


def _elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:4.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes:2d}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _squash(text: str, limit: int) -> str:
    """One line, bounded — a trace is for scanning, not for reading in full."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _preview_args(arguments: dict[str, Any]) -> str:
    parts = []
    for key, value in (arguments or {}).items():
        rendered = value if isinstance(value, (int, float, bool)) else repr(str(value))
        parts.append(f"{key}={_squash(str(rendered), 60)}")
    return _squash(", ".join(parts), MAX_ARG_CHARS)


@dataclass
class TraceRenderer:
    """Turns agent events into scannable console lines."""

    stream: TextIO = field(default_factory=lambda: sys.stderr)
    prefix: str = ""
    started: float = field(default_factory=time.perf_counter)
    colour: bool = field(default_factory=lambda: os.environ.get("NO_COLOR") is None)

    def _write(self, body: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        since = _elapsed(time.perf_counter() - self.started)
        self.stream.write(f"  {stamp} {since:>7}  {self.prefix}{body}\n")
        self.stream.flush()  # unbuffered: a live log that lags is not live

    def header(self, title: str, rule: str = "─") -> None:
        self.stream.write(f"\n{rule * 4} {title} {rule * max(4, 66 - len(title))}\n")
        self.stream.flush()

    def note(self, text: str) -> None:
        self._write(text)

    def event(self, event: Any) -> None:
        """Render one :class:`morph.agent.AgentEvent`."""
        kind = getattr(event, "type", None) or event.get("type")
        data = getattr(event, "data", None) or event

        if kind == "tool_use":
            step = data.get("step", "?")
            self._write(f"step {step:>2}  → {data.get('name')}({_preview_args(data.get('arguments'))})")

        elif kind == "tool_result":
            mark = "ok  " if data.get("ok") else "FAIL"
            took = f"{data.get('duration_ms', 0) / 1000:.1f}s"
            body = _squash(data.get("content", ""), MAX_RESULT_CHARS)
            self._write(f"         ← {mark} {took:>6}  {body}")

        elif kind == "text":
            text = _squash(data.get("text", ""), MAX_TEXT_CHARS)
            if text:
                self._write(f"         · {text}")

        elif kind == "error":
            recoverable = " (retrying)" if data.get("recoverable") else ""
            self._write(f"         ! {_squash(data.get('message', ''), MAX_RESULT_CHARS)}{recoverable}")

        elif kind == "done":
            result = data.get("result", {})
            self._write(
                f"         ⤷ {result.get('stop_reason')} after {result.get('steps')} step(s), "
                f"{len(result.get('tool_calls') or [])} tool call(s)"
            )


@dataclass
class ProgressFile:
    """A heartbeat anything can poll: CI, a dashboard, or a human with `cat`.

    Written on every event, so a stalled run is distinguishable from a slow one
    — the difference being whether ``updated_at`` is still moving.
    """

    path: Path
    state: dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.state.setdefault("phase", "starting")
        self.state.setdefault("started_at", self.started)

    def update(self, **fields: Any) -> None:
        self.state.update(fields)
        self.state["updated_at"] = time.time()
        self.state["elapsed_s"] = round(time.time() - self.started, 1)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic-ish: a reader must never catch a half-written file.
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.state, indent=2, default=str), "utf-8")
            temporary.replace(self.path)
        except OSError:
            pass  # progress reporting must never break the run it reports on

    def observe(self, event: Any) -> None:
        kind = getattr(event, "type", None) or event.get("type")
        data = getattr(event, "data", None) or event

        if kind == "tool_use":
            self.update(
                step=data.get("step"),
                activity=f"{data.get('name')}({_preview_args(data.get('arguments'))})",
            )
        elif kind == "tool_result":
            self.update(last_tool_ok=bool(data.get("ok")))
        elif kind == "text":
            text = _squash(data.get("text", ""), 240)
            if text:
                self.update(activity=f"thinking: {text}")
        elif kind == "done":
            self.update(activity="finished", step=(data.get("result") or {}).get("steps"))
