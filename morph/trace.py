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

#: The live log is read on a screen rather than scanned in a terminal, so it can
#: afford whole paragraphs — the model's reasoning is the most interesting thing
#: in the stream and truncating it to one line hides why it did what it did.
LIVE_TEXT_CHARS = 1600
LIVE_RESULT_CHARS = 800
#: Bounded so an eight-hour run cannot grow a file the page has to download.
LIVE_MAX_EVENTS = 400


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
class EventLog:
    """The same event stream as :class:`TraceRenderer`, as JSONL for a reader.

    One flat record per line — ``{t, kind, text, …}`` — so a page can render it
    without knowing anything about Morph's internals. Rewritten whole on every
    append rather than appended to: it keeps the file bounded, and a reader that
    polls mid-write gets the previous complete version instead of half a line.
    """

    path: Path
    limit: int = LIVE_MAX_EVENTS
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def append(self, kind: str, text: str = "", **fields: Any) -> None:
        self.records.append({"t": round(time.time(), 3), "kind": kind, "text": text, **fields})
        del self.records[: max(0, len(self.records) - self.limit)]
        self._flush()

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            body = "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in self.records)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(body + "\n", "utf-8")
            temporary.replace(self.path)
        except OSError:
            pass  # a log of the work must never be able to stop the work

    def observe(self, event: Any) -> None:
        """Record one :class:`morph.agent.AgentEvent`."""
        kind = getattr(event, "type", None) or event.get("type")
        data = getattr(event, "data", None) or event

        if kind == "tool_use":
            self.append(
                "tool_use",
                _preview_args(data.get("arguments")),
                step=data.get("step"),
                name=data.get("name"),
            )
        elif kind == "tool_result":
            self.append(
                "tool_result",
                _squash(data.get("content", ""), LIVE_RESULT_CHARS),
                step=data.get("step"),
                name=data.get("name"),
                ok=bool(data.get("ok")),
                ms=round(data.get("duration_ms", 0)),
            )
        elif kind == "text":
            text = _squash(data.get("text", ""), LIVE_TEXT_CHARS)
            if text:
                self.append("text", text, step=data.get("step"))
        elif kind == "error":
            self.append(
                "error",
                _squash(data.get("message", ""), LIVE_RESULT_CHARS),
                step=data.get("step"),
                recoverable=bool(data.get("recoverable")),
            )
        elif kind == "done":
            result = data.get("result") or {}
            self.append(
                "done",
                f"{result.get('stop_reason')} after {result.get('steps')} step(s), "
                f"{len(result.get('tool_calls') or [])} tool call(s)",
                steps=result.get("steps"),
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
    #: Optional event log fed from the same calls, so the loop wires the live
    #: view up once rather than passing two objects through every function.
    events: "EventLog | None" = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.state.setdefault("phase", "starting")
        self.state.setdefault("started_at", self.started)

    def update(self, **fields: Any) -> None:
        # A phase change is the structure of the run — mark it in the log so the
        # stream reads as "iteration 2 … then these steps", not one flat list.
        phase = fields.get("phase")
        if self.events is not None and phase is not None and phase != self.state.get("phase"):
            self.events.append(
                "phase",
                str(fields.get("activity") or phase),
                phase=phase,
                iteration=fields.get("iteration", self.state.get("iteration")),
            )
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

        if self.events is not None:
            self.events.observe(event)

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
