"""Benchmark task definitions.

A task is a workspace fixture + a prompt + a verifier. Tasks are
model-agnostic: the same task measures a real Gemma when one is configured, and
measures harness integrity when the deterministic ``echo`` provider is used.
"""

from __future__ import annotations

from .capability import CAPABILITY_TASKS
from .robustness import ROBUSTNESS_CHECKS
from .types import AgentTask, TaskOutcome

__all__ = ["AgentTask", "CAPABILITY_TASKS", "ROBUSTNESS_CHECKS", "TaskOutcome"]
