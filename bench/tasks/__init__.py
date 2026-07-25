"""Benchmark task definitions.

PROTECTED PACKAGE — the self-improvement loop may not modify anything under
``bench/tasks/`` (R-707). A loop that can author its own tasks will author easy
ones. Growing the benchmark is a human's job; the scorecard tells the human when
it needs growing (R-711).

Four capability suites, each spanning difficulty tiers T1-T5, plus the
error-injection checks:

``coding``     change software correctly
``tool_use``   drive the tools well — selection, precision, restraint, recovery
``mcp``        use tools discovered at runtime from MCP servers
``skills``     find, load and actually follow packaged instructions
``robustness`` survive things going wrong
"""

from __future__ import annotations

from .coding import CODING_TASKS
from .mcp_tasks import MCP_TASKS
from .robustness import ROBUSTNESS_CHECKS
from .skills import SKILLS_TASKS
from .tool_use import TOOL_USE_TASKS
from .spec import Criterion, Grade, Rubric, Task, TaskContext, TaskOutcome, Tier

#: Every capability task, in one list.
ALL_TASKS: list[Task] = [*CODING_TASKS, *TOOL_USE_TASKS, *MCP_TASKS, *SKILLS_TASKS]

SUITES: dict[str, list[Task]] = {
    "coding": CODING_TASKS,
    "tool_use": TOOL_USE_TASKS,
    "mcp": MCP_TASKS,
    "skills": SKILLS_TASKS,
}

__all__ = [
    "ALL_TASKS",
    "CODING_TASKS",
    "Criterion",
    "Grade",
    "MCP_TASKS",
    "ROBUSTNESS_CHECKS",
    "Rubric",
    "SKILLS_TASKS",
    "SUITES",
    "TOOL_USE_TASKS",
    "Task",
    "TaskContext",
    "TaskOutcome",
    "Tier",
]
