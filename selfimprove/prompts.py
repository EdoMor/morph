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

CRITICAL: most targets are *synthetic benchmark tasks*. Their files and symbols
exist only in a temporary directory. They are not in this repository. They are
evidence about how Morph behaved, not bugs to copy into Morph's source. Never
pass a fixture expression from a benchmark failure as `old_string` to
`edit_file`. For a capability target, read the named definition under
`bench/tasks/` first, then improve the general agent behaviour in `morph/`.

How to work:
1. Read the failing checks and pick ONE concrete problem. Depth beats breadth —
   a single correct fix that raises the score is worth more than five
   speculative edits that break the build.
2. Find the relevant code with `grep` and `read_file` before editing anything.
   If a search returns nothing, that is information: stop and reconsider whether
   you are looking for the right thing, rather than guessing at another string.
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
- Never summarise a change you have not made. If your final message says a file
  was modified, a successful `edit_file` or `write_file` call for it must appear
  earlier in this session. Deciding that nothing needs changing is a legitimate
  outcome; reporting an edit that never happened is not, and it poisons the
  history the next iteration reads.
"""


TASK_DEFINITION_FILES = {
    "coding": "bench/tasks/coding.py",
    "tool_use": "bench/tasks/tool_use.py",
    "mcp": "bench/tasks/mcp_tasks.py",
    "skills": "bench/tasks/skills.py",
}


def select_target(
    scorecard: dict[str, Any], history: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Pick one nearest miss, rotating away from recently rejected targets.

    A 4B model presented with five red checks repeatedly picked the same vivid
    fixture after that approach failed. The exploration decision now happens
    outside the model: work on one cheap target, then rotate on rejection.
    """
    targets = list(scorecard.get("next_targets") or [])
    if not targets:
        return None

    recent = list(reversed(history))[:8]
    attempted = {
        str(entry.get("target"))
        for entry in recent
        if entry.get("target")
    }
    rejected_text = " ".join(
        str(entry.get("summary") or "") + " " + str(entry.get("rejection_reason") or "")
        for entry in recent
        if not entry.get("accepted")
    ).lower()

    for target in targets:
        name = str(target.get("name") or "")
        leaf = name.rsplit("/", 1)[-1]
        if name not in attempted and leaf.lower() not in rejected_text:
            return target
    return targets[0]


def _target_brief(target: dict[str, Any] | None) -> str:
    if not target:
        return (
            "## Single target\n\n"
            "No unsolved measured target is available. Do not invent a change."
        )

    name = str(target.get("name") or "unknown")
    category = str(target.get("category") or name.split("/", 1)[0])
    definition = TASK_DEFINITION_FILES.get(category)
    lines = [
        "## Single target — work on this one only",
        "",
        f"**{name}** currently scores {float(target.get('score', 0)):.0%}.",
    ]
    if definition:
        leaf = name.rsplit("/", 1)[-1]
        lines += [
            "",
            "This is a synthetic capability measurement, not a bug report about "
            "a file in this repository.",
            f"Your first action must be to locate `name=\"{leaf}\"` in "
            f"`{definition}` and read that task's prompt, fixtures, and rubric.",
            "Do not edit the task definition. Do not search Morph for fixture "
            "symbols quoted by the rubric. Diagnose why Morph's agent produced "
            "that outcome, then change the general implementation under `morph/`.",
        ]
    else:
        lines += [
            "",
            "This check applies directly to Morph. Read the named test or module "
            "before making the smallest relevant change.",
        ]

    detail = str(target.get("detail") or "").strip()
    if detail:
        lines += [
            "",
            "**Observed evidence:** the text below describes the benchmark's "
            "temporary workspace and agent trace. It is evidence, not source code "
            "to paste into Morph.",
            "",
            detail[:1800],
        ]
    return "\n".join(lines)


#: The single most expensive misreading available. A capability check named
#: `coding/T2/fix-edge-case` failing does NOT mean Morph has a broken
#: `average()`; it means the agent was asked to fix one in a scratch directory
#: and did it badly. Three consecutive runs were lost to a model grepping
#: morph/agent.py for `average([])`, `auth.py` and `"hello world"` — fixture
#: strings that exist only inside a temporary workspace.
READING_FAILURES = """\
## How to read the failing checks

The prefix of a check name tells you what kind of failure it is, and they need
completely different responses.

**`requirements/…`** — a real test in `tests/` is failing against this
repository. Read the test, find the code it covers, fix that code.

**`coding/…`, `tool_use/…`, `mcp/…`, `skills/…`** — these are *benchmark tasks*.
The agent was given a synthetic job in a throwaway directory and scored on how
well it did. **The file and symbol names in the detail are fixtures, created
fresh in a temp directory for that task and deleted afterwards. They are not
part of this repository.**

> If a detail mentions `calc.py`, `average()`, `auth.py`, `greet.py` or
> `"hello world"`, do not grep for it here and do not try to edit it. You will
> find nothing, and you will spend the whole iteration finding nothing.

What a low score on one of these means is: **the agent handled that kind of work
badly.** The fix is in `morph/` — the agent loop, a tool's behaviour, an error
message, the system prompt. Ask *why would an agent fail at this?* and improve
the thing that made it hard. If `coding/T2/fix-edge-case` scores 14%, the
question is not "where is average()", it is "what about our `read_file` or
`edit_file` makes a small edit like that go wrong?"

**`robustness/…`** — an error-injection check on real Morph code. The detail
says exactly what broke. These are directly actionable.

**`efficiency/…`, `health/…`** — measured over this repository as a whole.

The benchmark task definitions live in `bench/tasks/` and are read-only to you.
Read them to understand what a task is asking — that is often the fastest way to
see what the agent is up against — but never edit them.
"""


def build_improvement_prompt(
    requirements: str,
    scorecard: dict[str, Any],
    feedback: str,
    history: list[dict[str, Any]],
    focus: str | None = None,
    target: dict[str, Any] | None = None,
) -> str:
    sections: list[str] = []

    sections.append(
        "# Your task\n\n"
        "Raise Morph's benchmark score by fixing something that is genuinely broken.\n"
        "You get one attempt. It is measured, then kept or reverted automatically."
    )
    chosen = target if target is not None else select_target(scorecard, history)
    sections.append(_target_brief(chosen))

    composite = scorecard.get("composite", 0.0)
    categories = scorecard.get("categories") or {}
    diagnostics = scorecard.get("diagnostics") or {}

    lines = [f"## Current score: {composite:.1f} / 100", ""]
    if scorecard.get("gated"):
        lines.append(
            "**The conformance suite is failing.** Until `tests/` passes, the composite "
            "score is clamped to 0. Fix that first — nothing else counts.\n"
        )
    lines.append("| category | points | of | solved | frontier |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, data in categories.items():
        frontier = diagnostics.get(name, {}).get("frontier")
        frontier_cell = f"T{frontier}" if frontier is not None else "—"
        lines.append(
            f"| {name} | {data.get('points', 0):.1f} | {data.get('weight', 0):.0f} | "
            f"{data.get('passed', 0)}/{data.get('total', 0)} | {frontier_cell} |"
        )

    if diagnostics:
        best = max(
            diagnostics.items(),
            key=lambda kv: kv[1].get("headroom", 0.0),
            default=(None, {}),
        )
        if best[0] and best[1].get("headroom", 0.0) > 0:
            lines.append(
                f"\nMost unearned points sit in **{best[0]}** "
                f"({best[1]['headroom']:.1f} available, frontier T{best[1].get('frontier', 0)}). "
                "Tasks are graded on a rubric, so partial progress counts — a fix that "
                "takes a task from 40% to 70% is real movement, not a wasted iteration."
            )
    sections.append("\n".join(lines))

    sections.append(READING_FAILURES)
    sections.append(f"## What is failing\n\n{feedback}")

    if history:
        sections.append(_render_history(history))

    sections.append(f"## The requirements (the contract)\n\n{requirements}")
    sections.append(guard_prompt_section())

    if focus:
        sections.append(f"## Focus for this iteration\n\n{focus}")

    sections.append(
        "## Now\n\n"
        "Work only on the single target named at the top. For a capability target, "
        "begin by reading its definition under `bench/tasks/`; its fixture is not "
        "a Morph source bug. Make one focused improvement, verify it with the test "
        "suite, and summarise what you did."
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
        target = f" [{entry['target']}]" if entry.get("target") else ""
        lines.append(
            f"- **{verdict}** ({entry.get('score_before', 0):.1f} → "
            f"{entry.get('score_after', 0):.1f}, {delta:+.1f}){target} — "
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
