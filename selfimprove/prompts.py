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
from .memory import enrich_entry, render_experience_memory, target_stats

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

DIAGNOSIS_SYSTEM_PROMPT = """\
You are the diagnosis stage of Morph's self-improvement loop. You cannot edit
files in this stage. Inspect the benchmark dossier and Morph's real
implementation with `read_file` and `grep`, then hand a small, concrete plan to
the implementation stage.

Benchmark fixtures are evidence from a temporary workspace. They are never
Morph source files. Do not propose creating a fixture such as `calc.py` under
`morph/`, and do not propose special-casing fixture symbols such as `average`.

Finish with exactly these headings:
CAUSE:
FILES:
CHANGE:
TEST:
Each section should be short and specific. FILES must name existing repository
files you inspected. If the evidence does not justify a code change, say so
under CHANGE.
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

    # Once the gate is green, prefer an individually re-runnable capability
    # task. Whole-run metrics such as efficiency/no-timeouts are consequences,
    # not actionable units of work; run #11 chased one, copied its synthetic
    # `calc.py` fixture into morph/, and appeared to improve only because the
    # full model sweep is noisy.
    if not scorecard.get("gated"):
        capability = [
            target
            for target in targets
            if str(target.get("category") or "").strip() in TASK_DEFINITION_FILES
        ]
        if capability:
            targets = capability

    # Explore every target once before revisiting any of them. Unlike the old
    # eight-row window, this remains true across scheduled runs and a long
    # archive.
    for target in targets:
        name = str(target.get("name") or "")
        if target_stats(history, name)["attempts"] == 0:
            return target

    # A changed exact-target baseline is new evidence and justifies an earlier
    # retry. Otherwise prefer the target with the fewest consecutive failures,
    # then the one left alone longest. The original target order breaks ties,
    # preserving the scorecard's nearest-miss preference.
    for target in targets:
        name = str(target.get("name") or "")
        stats = target_stats(history, name)
        previous = stats.get("last_target_score")
        if isinstance(previous, (int, float)) and abs(
            float(target.get("score") or 0) - float(previous)
        ) > 0.005:
            return target

    ranked = sorted(
        enumerate(targets),
        key=lambda pair: (
            target_stats(history, str(pair[1].get("name") or ""))[
                "consecutive_failures"
            ],
            target_stats(history, str(pair[1].get("name") or ""))["last_index"],
            pair[0],
        ),
    )
    return ranked[0][1]


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


def _task_dossier(target: dict[str, Any] | None) -> str:
    """Render the exact synthetic task into the prompt.

    Telling a 4B model to find a task inside a 900-line definition file was not
    enough: the warning was separated from the vivid fixture failure by too
    much context. The loop now extracts the task itself, clearly labels every
    temporary file, and repeats the boundary beside the evidence.
    """
    if not target:
        return _target_brief(target)

    name = str(target.get("name") or "")
    try:
        from bench.tasks import ALL_TASKS

        task = next(item for item in ALL_TASKS if item.label == name)
    except (ImportError, StopIteration):
        return _target_brief(target)

    lines = [
        f"# Benchmark dossier: {task.label}",
        "",
        f"Current target score: {float(target.get('score', 0.0)):.0%}",
        "",
        "## Prompt given to Morph in the temporary benchmark workspace",
        "",
        task.prompt.strip(),
        "",
        "## Temporary fixture files",
        "",
        "These paths and their symbols are deleted after the benchmark. They are",
        "**not repository files and must never be created under `morph/`**.",
    ]
    for relative, content in task.files.items():
        lines += [
            "",
            f"### fixture `{relative}`",
            "```text",
            content[:2400].rstrip(),
            "```",
        ]

    lines += ["", "## Rubric"]
    for criterion in task.rubric.criteria:
        gate = " (required gate)" if criterion.critical else ""
        lines.append(f"- {criterion.name}{gate}")

    detail = str(target.get("detail") or "").strip()
    if detail:
        lines += [
            "",
            "## Observed result from the temporary workspace",
            "",
            detail[:2400],
        ]
    lines += [
        "",
        "## Boundary",
        "",
        "Improve the general agent/tool implementation that caused this behavior.",
        "Do not implement the fixture's requested function inside Morph itself.",
    ]
    return "\n".join(lines)


def build_diagnosis_prompt(
    target: dict[str, Any] | None,
    history: list[dict[str, Any]],
    focus: str | None = None,
) -> str:
    """Small read-only prompt for the archive/diagnosis stage."""
    sections = [_task_dossier(target)]
    if history:
        sections.append(render_experience_memory(history, target))
    if focus:
        sections.append(f"# Human focus\n\n{focus}")
    sections.append(
        "# Diagnose\n\n"
        "Use `grep` and `read_file` to inspect the real implementation under "
        "`morph/`. Identify one general cause of the measured failure and hand "
        "off one minimal change. Do not edit in this stage."
    )
    return "\n\n---\n\n".join(sections)


def build_implementation_prompt(
    target: dict[str, Any] | None,
    diagnosis: str,
    history: list[dict[str, Any]] | None = None,
    focus: str | None = None,
) -> str:
    """Focused fresh-context prompt for the code-writing stage."""
    sections = [
        "# Implement one measured improvement",
        _task_dossier(target),
        (
            "# Diagnosis handoff\n\n"
            f"{diagnosis.strip()[:5000] or 'The diagnosis stage produced no usable handoff.'}"
        ),
        render_experience_memory(history or [], target),
        guard_prompt_section(),
        (
            "# Working rules\n\n"
            "- Inspect every real file before editing it.\n"
            "- Change the general implementation, never a benchmark fixture.\n"
            "- Do not create under `morph/` a file whose name appears in the "
            "temporary fixture list.\n"
            "- Make one small change. Use `write_file` when creating a file; "
            "`edit_file` requires all of `path`, `old_string`, and `new_string`.\n"
            "- Run the narrow affected tests, then `python -m pytest tests -q`.\n"
            "- Finish with what actually changed and what you ran."
        ),
    ]
    if focus:
        sections.append(f"# Human focus\n\n{focus}")
    return "\n\n---\n\n".join(sections)


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
        sections.append(render_experience_memory(history, chosen))

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


def load_history(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(enrich_entry(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return entries[-limit:] if limit is not None else entries


def append_history(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = enrich_entry(entry)
    entry["experience"] = enriched["experience"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(enriched, ensure_ascii=False, default=str) + "\n")
