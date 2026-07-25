"""Scoring model for the benchmark.

PROTECTED FILE — the self-improvement loop may not modify this (R-707). It
defines what "better" means; a model allowed to edit it would optimise by
redefining success instead of improving the system.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Category weights sum to 100. Mirrors the table in REQUIREMENTS.md.
WEIGHTS: dict[str, float] = {
    "requirements": 40.0,
    "capability": 30.0,
    "robustness": 15.0,
    "efficiency": 10.0,
    "health": 5.0,
}

# `requirements` gates everything: a failing conformance suite means score 0.
GATE_CATEGORY = "requirements"


@dataclass
class CheckResult:
    """One measured thing."""

    name: str
    category: str
    passed: bool
    detail: str = ""
    duration_ms: float = 0.0
    weight: float = 1.0
    requirement_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "detail": self.detail[:2000],
            "duration_ms": round(self.duration_ms, 2),
            "weight": self.weight,
            "requirement_ids": self.requirement_ids,
        }


@dataclass
class Scorecard:
    results: list[CheckResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    def add(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        return result

    def by_category(self, category: str) -> list[CheckResult]:
        return [r for r in self.results if r.category == category]

    def category_score(self, category: str) -> float:
        """Weighted pass fraction in ``[0, 1]``. Absent category scores 0."""
        results = self.by_category(category)
        if not results:
            return 0.0
        total = sum(r.weight for r in results) or 1.0
        earned = sum(r.weight for r in results if r.passed)
        return earned / total

    @property
    def gated(self) -> bool:
        """True when the conformance gate is failing."""
        gate = self.by_category(GATE_CATEGORY)
        return bool(gate) and any(not r.passed for r in gate)

    @property
    def composite(self) -> float:
        """Composite score in ``[0, 100]``, clamped to 0 when gated."""
        if self.gated:
            return 0.0
        return round(
            sum(self.category_score(cat) * weight for cat, weight in WEIGHTS.items()), 2
        )

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "composite": self.composite,
            "gated": self.gated,
            "categories": {
                cat: {
                    "score": round(self.category_score(cat), 4),
                    "weight": weight,
                    "points": round(self.category_score(cat) * weight, 2),
                    "passed": sum(1 for r in self.by_category(cat) if r.passed),
                    "total": len(self.by_category(cat)),
                }
                for cat, weight in WEIGHTS.items()
            },
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), "utf-8")
        return target

    @classmethod
    def read(cls, path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text("utf-8"))

    # ------------------------------------------------------------------
    def render(self) -> str:
        """Human-readable scorecard for terminals and CI logs."""
        lines = [
            "",
            f"  MORPH SCORECARD — {self.composite:.1f} / 100"
            + ("   [GATED: conformance suite failing]" if self.gated else ""),
            "  " + "─" * 58,
        ]
        for category, weight in WEIGHTS.items():
            results = self.by_category(category)
            score = self.category_score(category)
            passed = sum(1 for r in results if r.passed)
            bar = "█" * int(round(score * 20)) + "·" * (20 - int(round(score * 20)))
            lines.append(
                f"  {category:<13} {bar}  {score * weight:5.1f}/{weight:<4.0f} "
                f"({passed}/{len(results)})"
            )

        note = self.metadata.get("note")
        if note:
            lines += ["", f"  note: {note}"]

        if self.failures:
            lines += ["", "  Failing:"]
            for failure in self.failures[:25]:
                ids = f" [{', '.join(failure.requirement_ids)}]" if failure.requirement_ids else ""
                lines.append(f"    ✗ {failure.name}{ids}")
                if failure.detail:
                    first = failure.detail.strip().splitlines()[0][:100]
                    lines.append(f"        {first}")
            if len(self.failures) > 25:
                lines.append(f"    … and {len(self.failures) - 25} more")

        lines.append("")
        return "\n".join(lines)

    def feedback(self, limit: int = 20) -> str:
        """The failure digest handed to the model for the next iteration (R-705)."""
        if not self.failures:
            return (
                "Everything passes. Look for improvements that raise the efficiency "
                "and health scores, or add capability without breaking anything."
            )
        lines = [f"Composite score: {self.composite:.1f}/100.", "", "Failing checks:"]
        for failure in self.failures[:limit]:
            ids = f" (requirements: {', '.join(failure.requirement_ids)})" if failure.requirement_ids else ""
            lines.append(f"\n### {failure.name} [{failure.category}]{ids}")
            lines.append(failure.detail.strip()[:1500] or "(no detail captured)")
        if len(self.failures) > limit:
            lines.append(f"\n… and {len(self.failures) - limit} further failures.")
        return "\n".join(lines)


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Per-category delta between two scorecards (R-705)."""
    deltas = {}
    for category in WEIGHTS:
        old = (before.get("categories") or {}).get(category, {}).get("points", 0.0)
        new = (after.get("categories") or {}).get(category, {}).get("points", 0.0)
        deltas[category] = round(new - old, 2)
    return {
        "composite_before": before.get("composite", 0.0),
        "composite_after": after.get("composite", 0.0),
        "delta": round(after.get("composite", 0.0) - before.get("composite", 0.0), 2),
        "categories": deltas,
    }
