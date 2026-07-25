"""Task types shared by the benchmark suites."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from morph.llm import ModelResponse

Verifier = Callable[[Path, Any], "TaskOutcome"]


@dataclass
class TaskOutcome:
    passed: bool
    detail: str = ""

    @classmethod
    def ok(cls, detail: str = "") -> "TaskOutcome":
        return cls(True, detail)

    @classmethod
    def fail(cls, detail: str) -> "TaskOutcome":
        return cls(False, detail)


@dataclass
class AgentTask:
    """One end-to-end agent task.

    ``reference_script`` lets the deterministic ``echo`` provider replay a known
    good trace, so the harness itself is exercised in CI with no model. With a
    real provider configured the script is ignored and the model must solve the
    task on its own.
    """

    name: str
    prompt: str
    verify: Verifier
    files: dict[str, str] = field(default_factory=dict)
    budget_steps: int = 12
    budget_seconds: float = 120.0
    requirement_ids: list[str] = field(default_factory=list)
    reference_script: Sequence[ModelResponse] = field(default_factory=list)
    weight: float = 1.0

    def materialise(self, root: Path) -> None:
        """Write the task's fixture files into a fresh workspace."""
        for relative, content in self.files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, "utf-8")
