"""Benchmark harness: turns the requirements into a number the loop can optimise."""

from __future__ import annotations

from .scorecard import WEIGHTS, CheckResult, Scorecard, compare

__all__ = ["WEIGHTS", "CheckResult", "Scorecard", "compare"]
