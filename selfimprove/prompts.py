"""Prompt construction for the improvement loop.

The prompt is assembled from four things:

1. the requirements — the contract, verbatim;
2. the scorecard — where the system currently stands;
3. the failure digest — what is broken, with detail;
4. the history — what has already been tried, and what happened (R-705).

(4) matters more than it looks. Without it the model rediscovers the same dead
end every iteration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .guard import guard_prompt_section

SYSTEM_PROMPT = """\
You are Gemma, working on Morph — a self-hosted coding agent platform. You are
editing Morph's own source code, using Morph's own tools. The agent loop you are
running inside is the same one you are improving.

You have file, search and shell tools. Use them. Never guess at file contents.

How to work:
1. Read the failing checks and pick ONE concrete problem. Depth beats breadth —
   a single correct fix that raises the score is worth more than five
   speculative edits that break the build.
2. Find the relevant code with `grep` and `read_file` before editing anything.
3. Make the smallest change that fixes the problem.
4. Run the affected tests with `shell` and confirm they pass.
5. Finish with a short summary: what you changed, why, and which requirement IDs
   it addresses.

Rules:
- Every change must keep `python -m pytest tests -q` green. A higher score with a
  failing suite is scored as zero.
- Do not add new third-party dependencies. Morph has to run on a phone.
- Do not delete or weaken tests to make them pass.
- Prefer fixing the cause over widening an exception handler.
"""


def build_improvement_prompt(
    requirements: str,
    scorecard: dict[str, Any],
    feedback: str,
    history: list[dict[str, Any]],
    focus: str | None = None,
) -> str:
    sections: list[str] = []

    sections.append(
        "# Your task\n\n"
        "Raise Morph's benchmark score by fixing something that is genuinely broken.\n"
        "You get one attempt. It is measured, then kept or reverted automatically."
    )

    composite = scorecard.get("composite", 0.0)
    categories = scorecard.get("categories") or {}
    lines = [f"## Current score: {composite:.1f} / 100", ""]
    if scorecard.get("gated"):
        lines.append(
            "**The conformance suite is failing.** Until `tests/` passes, the composite "
            "score is clamped to 0. Fix that first — nothing else counts.\n"
        )
    lines.append("| category | points | of | passing |")
    lines.append("| --- | --- | --- | --- |")
    for name, data in categories.items():
        lines.append(
            f"| {name} | {data.get('points', 0):.1f} | {data.get('weight', 0):.0f} | "
            f"{data.get('passed', 0)}/{data.get('total', 0)} |"
        )
    sections.append("\n".join(lines))

    sections.append(f"## What is failing\n\n{feedback}")

    if history:
        sections.append(_render_history(history))

    sections.append(f"## The requirements (the contract)\n\n{requirements}")
    sections.append(guard_prompt_section())

    if focus:
        sections.append(f"## Focus for this iteration\n\n{focus}")

    sections.append(
        "## Now\n\n"
        "Investigate, make one focused improvement, verify it with the test suite, "
        "and summarise what you did."
    )
    return "\n\n---\n\n".join(sections)


def _render_history(history: list[dict[str, Any]], limit: int = 8) -> str:
    """Recent attempts, newest first — so the model stops repeating itself."""
    lines = [
        "## Previous attempts",
        "",
        "Do not repeat a rejected approach. If an approach was rejected twice, the",
        "problem is upstream of where you were looking.",
        "",
    ]
    for entry in list(reversed(history))[:limit]:
        verdict = "ACCEPTED" if entry.get("accepted") else "REJECTED"
        delta = entry.get("score_after", 0) - entry.get("score_before", 0)
        lines.append(
            f"- **{verdict}** ({entry.get('score_before', 0):.1f} → "
            f"{entry.get('score_after', 0):.1f}, {delta:+.1f}) — "
            f"{(entry.get('summary') or '(no summary)').strip()[:300]}"
        )
        if entry.get("rejection_reason"):
            lines.append(f"  - rejected because: {entry['rejection_reason']}")
        if entry.get("files_changed"):
            lines.append(f"  - touched: {', '.join(entry['files_changed'][:8])}")
    return "\n".join(lines)


def load_history(path: Path, limit: int = 50) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries[-limit:]


def append_history(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
