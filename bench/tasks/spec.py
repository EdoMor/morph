"""Task types: difficulty tiers and graded rubrics.

Two properties matter more than the task list itself, because without them a
self-improving loop has nothing to climb:

**Gradient within a task.** A task is graded on a rubric of weighted criteria
and scores continuously in ``[0, 1]``. Binary pass/fail throws away exactly the
information the loop needs — "the fix parses and preserves old behaviour but
misses the edge case" has to score above "deleted the file" (R-709).

**Gradient across tasks.** Every category spans five difficulty tiers, so at any
level of competence some tasks are solved, some are borderline, and some are out
of reach. That band of borderline tasks is the learning signal (R-708).
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from morph.config import MCPServerConfig
from morph.llm import ModelResponse


class Tier(IntEnum):
    """How hard a task is. Every capability suite must span the range (R-708)."""

    TRIVIAL = 1  # one tool call, no reasoning
    BASIC = 2  # two or three calls, obvious sequence
    INTERMEDIATE = 3  # must read and understand before acting
    HARD = 4  # ambiguity, debugging, or recovery from a wrong turn
    EXPERT = 5  # multi-file, adversarial, or requires a plan


# Harder tasks are worth more, but not so much that the easy ones stop counting:
# a system that solves every T1-T2 and nothing else still earns a visible score,
# which is what makes early progress legible.
TIER_WEIGHT: dict[Tier, float] = {
    Tier.TRIVIAL: 1.0,
    Tier.BASIC: 2.0,
    Tier.INTERMEDIATE: 3.0,
    Tier.HARD: 5.0,
    Tier.EXPERT: 8.0,
}


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


@dataclass
class TaskContext:
    """What a criterion gets to inspect: the workspace, and what the agent did."""

    root: Path
    result: Any  # morph.agent.RunResult

    # -- workspace helpers ---------------------------------------------
    def exists(self, relative: str) -> bool:
        return (self.root / relative).exists()

    def read(self, relative: str) -> str:
        path = self.root / relative
        return path.read_text("utf-8") if path.is_file() else ""

    def parses(self, relative: str) -> bool:
        source = self.read(relative)
        if not source:
            return False
        try:
            ast.parse(source)
        except SyntaxError:
            return False
        return True

    def module(self, relative: str) -> dict[str, Any]:
        """Execute a workspace Python file and return its namespace.

        Raises if the file does not parse or blows up at import time — criteria
        that need a working module should be marked ``critical``.
        """
        namespace: dict[str, Any] = {"__name__": "__bench__"}
        exec(compile(self.read(relative), relative, "exec"), namespace)  # noqa: S102
        return namespace

    def defines(self, relative: str, symbol: str) -> bool:
        try:
            tree = ast.parse(self.read(relative))
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == symbol:
                    return True
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == symbol:
                        return True
        return False

    # -- trace helpers -------------------------------------------------
    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return list(getattr(self.result, "tool_calls", []) or [])

    @property
    def tools_used(self) -> list[str]:
        return [c["tool"] for c in self.tool_calls]

    def used(self, *tools: str) -> bool:
        return any(tool in self.tools_used for tool in tools)

    def call_count(self, tool: str) -> int:
        return self.tools_used.count(tool)

    def calls_to(self, tool: str) -> list[dict[str, Any]]:
        return [c for c in self.tool_calls if c["tool"] == tool]

    @property
    def failed_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.tool_calls if not c.get("ok")]

    @property
    def answer(self) -> str:
        return (getattr(self.result, "text", "") or "").lower()

    @property
    def steps(self) -> int:
        return int(getattr(self.result, "steps", 0))

    def mentions(self, *needles: str) -> bool:
        return any(needle.lower() in self.answer for needle in needles)


CheckFn = Callable[[TaskContext], "bool | float"]


@dataclass
class Criterion:
    """One graded dimension of a task.

    ``check`` returns a bool or a float in ``[0, 1]``.

    ``critical`` marks a **necessary condition**, not an achievement: "the code
    still parses", "existing behaviour is intact", "the test file was not
    edited". Critical criteria gate the task — failing one zeroes it — but earn
    no points, because an agent that does nothing at all satisfies most of them.
    Scoring them would hand a do-nothing run a free floor and flatten exactly the
    part of the range where the loop starts out.
    """

    name: str
    weight: float
    check: CheckFn
    critical: bool = False


@dataclass
class Grade:
    score: float  # [0, 1]
    detail: str
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        """A task counts as solved at 80% — enough room for style differences."""
        return self.score >= 0.8


@dataclass
class Rubric:
    criteria: list[Criterion]

    def grade(self, ctx: TaskContext) -> Grade:
        # Critical criteria gate but do not score — see :class:`Criterion`.
        scored = [c for c in self.criteria if not c.critical]
        total = sum(c.weight for c in scored)

        earned = 0.0
        breakdown: dict[str, float] = {}
        notes: list[str] = []
        zeroed: str | None = None

        for criterion in self.criteria:
            try:
                raw = criterion.check(ctx)
            except Exception as exc:  # noqa: BLE001 - a criterion that throws scores 0
                raw = 0.0
                notes.append(f"{criterion.name}: raised {type(exc).__name__}: {exc}")

            value = float(raw) if not isinstance(raw, bool) else (1.0 if raw else 0.0)
            value = max(0.0, min(1.0, value))
            breakdown[criterion.name] = value

            if criterion.critical:
                if value < 1.0 and zeroed is None:
                    zeroed = criterion.name
                continue

            earned += value * criterion.weight
            if value < 1.0 and not any(n.startswith(criterion.name) for n in notes):
                notes.append(f"{criterion.name}: {value:.0%}")

        if zeroed is not None:
            return Grade(
                score=0.0,
                detail=f"critical criterion failed ({zeroed}); "
                + ("; ".join(notes) if notes else "no further detail"),
                breakdown=breakdown,
            )

        # A rubric of nothing but gates is fully satisfied once they all pass.
        score = 1.0 if total == 0 else earned / total
        detail = (
            "all criteria met"
            if score >= 1.0
            else f"{score:.0%} — " + "; ".join(notes[:6])
        )
        return Grade(score=score, detail=detail, breakdown=breakdown)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """One end-to-end agent task.

    ``reference_script`` replays a competent solution for the deterministic
    ``echo`` provider, which lets CI verify the harness can execute the work.
    Tasks with no script are skipped in that mode rather than scored zero —
    a plumbing check must not masquerade as a capability measurement.
    """

    name: str
    category: str  # coding | tool_use | mcp | skills
    tier: Tier
    prompt: str
    rubric: Rubric
    files: dict[str, str] = field(default_factory=dict)
    skills: dict[str, str] = field(default_factory=dict)  # skill name -> SKILL.md body
    mcp_servers: Callable[[Path], list[MCPServerConfig]] | None = None
    budget_steps: int = 14
    budget_seconds: float = 180.0
    requirement_ids: list[str] = field(default_factory=list)
    reference_script: Sequence[ModelResponse] = field(default_factory=list)

    @property
    def weight(self) -> float:
        return TIER_WEIGHT[self.tier]

    @property
    def label(self) -> str:
        return f"{self.category}/T{int(self.tier)}/{self.name}"

    def materialise(self, root: Path) -> None:
        """Write the fixture files and skills into a fresh workspace."""
        for relative, content in self.files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, "utf-8")

        for name, body in self.skills.items():
            directory = root / "skills" / name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "SKILL.md").write_text(body, "utf-8")


# ---------------------------------------------------------------------------
# Small criterion builders, so task definitions stay readable
# ---------------------------------------------------------------------------


def file_exists(relative: str, weight: float = 1.0, critical: bool = False) -> Criterion:
    return Criterion(f"{relative} exists", weight, lambda c: c.exists(relative), critical)


def still_parses(relative: str, weight: float = 1.0) -> Criterion:
    return Criterion(f"{relative} still parses", weight, lambda c: c.parses(relative), True)


def used_tool(*tools: str, weight: float = 1.0) -> Criterion:
    label = " or ".join(tools)
    return Criterion(f"used {label}", weight, lambda c: c.used(*tools))


def answer_mentions(*needles: str, weight: float = 1.0) -> Criterion:
    label = " / ".join(needles)
    return Criterion(f"answer mentions {label}", weight, lambda c: c.mentions(*needles))


def no_failed_calls(weight: float = 0.5) -> Criterion:
    return Criterion("no failed tool calls", weight, lambda c: not c.failed_calls)


def within_steps(limit: int, weight: float = 1.0) -> Criterion:
    """Graded, not binary: overshooting the budget by a little costs a little."""

    def check(ctx: TaskContext) -> float:
        if ctx.steps <= limit:
            return 1.0
        return max(0.0, 1.0 - (ctx.steps - limit) / max(limit, 1))

    return Criterion(f"solved within {limit} steps", weight, check)


def behaviour(description: str, fn: CheckFn, weight: float = 1.0, critical: bool = False) -> Criterion:
    return Criterion(description, weight, fn, critical)


@dataclass
class TaskOutcome:
    """Result of a robustness check.

    Robustness is genuinely binary — the failure was contained or it was not —
    so these checks keep a pass/fail shape rather than a rubric.
    """

    passed: bool
    detail: str = ""

    @classmethod
    def ok(cls, detail: str = "") -> "TaskOutcome":
        return cls(True, detail)

    @classmethod
    def fail(cls, detail: str) -> "TaskOutcome":
        return cls(False, detail)
