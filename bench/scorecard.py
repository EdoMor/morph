"""Scoring model for the benchmark.

PROTECTED FILE — the self-improvement loop may not modify this (R-707). It
defines what "better" means; a model allowed to edit it would optimise by
redefining success instead of improving the system.

Three properties keep this usable as a loop objective:

**Graded, not binary.** Every check carries a score in ``[0, 1]``. Partial
progress moves the number, so an iteration that gets halfway is distinguishable
from one that achieved nothing (R-709).

**Tier-weighted.** Checks declare a difficulty tier and are weighted by it, so
the score reflects *what* was solved and not just how many (R-708).

**Self-diagnosing.** The scorecard reports its own calibration. A suite where
everything passes, or where nothing does, has stopped being an instrument — it
says so rather than quietly reporting a number nobody can act on (R-711).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Category weights sum to 100. Mirrors the table in REQUIREMENTS.md.
WEIGHTS: dict[str, float] = {
    "requirements": 25.0,
    "coding": 20.0,
    "tool_use": 15.0,
    "mcp": 12.0,
    "skills": 12.0,
    "robustness": 10.0,
    "efficiency": 4.0,
    "health": 2.0,
}

# The four suites that measure agent capability, as opposed to hygiene.
CAPABILITY_CATEGORIES: tuple[str, ...] = ("coding", "tool_use", "mcp", "skills")

# `requirements` gates everything: a failing conformance suite means score 0.
GATE_CATEGORY = "requirements"

# A task is "solved" at 80% of its rubric — enough room for stylistic variation.
SOLVED_AT = 0.8

# Calibration thresholds. A suite outside this band gives the loop no gradient.
SATURATED_AT = 0.95
FLOORED_AT = 0.10


@dataclass
class CheckResult:
    """One measured thing, scored continuously in ``[0, 1]``."""

    name: str
    category: str
    score: float = 0.0
    detail: str = ""
    duration_ms: float = 0.0
    weight: float = 1.0
    tier: int | None = None
    requirement_ids: list[str] = field(default_factory=list)
    skipped: bool = False

    @property
    def passed(self) -> bool:
        return self.score >= SOLVED_AT

    @classmethod
    def binary(cls, name: str, category: str, passed: bool, **kwargs: Any) -> "CheckResult":
        """Convenience for checks that genuinely have no middle ground."""
        return cls(name=name, category=category, score=1.0 if passed else 0.0, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "score": round(self.score, 4),
            "passed": self.passed,
            "detail": self.detail[:2000],
            "duration_ms": round(self.duration_ms, 2),
            "weight": self.weight,
            "tier": self.tier,
            "requirement_ids": self.requirement_ids,
            "skipped": self.skipped,
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
        return [r for r in self.results if r.category == category and not r.skipped]

    def by_tier(self, category: str, tier: int) -> list[CheckResult]:
        return [r for r in self.by_category(category) if r.tier == tier]

    def category_score(self, category: str) -> float:
        """Weighted mean score in ``[0, 1]``. Absent category scores 0."""
        results = self.by_category(category)
        if not results:
            return 0.0
        total = sum(r.weight for r in results) or 1.0
        return sum(r.score * r.weight for r in results) / total

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
        return [r for r in self.results if not r.passed and not r.skipped]

    # ------------------------------------------------------------------
    # Gradient diagnostics (R-710, R-711)
    # ------------------------------------------------------------------

    def tier_profile(self, category: str) -> dict[int, float]:
        """Mean score per difficulty tier — the shape of the learning curve."""
        profile: dict[int, float] = {}
        for tier in range(1, 6):
            results = self.by_tier(category, tier)
            if results:
                profile[tier] = sum(r.score for r in results) / len(results)
        return profile

    def frontier(self, category: str) -> int:
        """The hardest tier the system reliably handles.

        Defined as the highest tier scoring at least 50% with every easier tier
        also at 50%. Tier ``0`` means it cannot yet do the trivial ones — the
        loop should be working on fundamentals, not on the expert tasks.
        """
        profile = self.tier_profile(category)
        frontier = 0
        for tier in sorted(profile):
            if profile[tier] >= 0.5:
                frontier = tier
            else:
                break
        return frontier

    def skipped_in(self, category: str) -> list[CheckResult]:
        return [r for r in self.results if r.category == category and r.skipped]

    def calibration(self, category: str) -> str:
        """``"floored"``, ``"saturated"``, ``"partial"``, ``"empty"`` or ``"healthy"``.

        A suite is only useful while some tasks are solved and some are not.
        Outside that band the loop is flying blind, and the fix is new tasks —
        which is a human's job, not the loop's (R-711).

        ``partial`` means some checks did not run, so calibration cannot be
        judged: a full sweep of the reference traces would otherwise look
        identical to genuine saturation, and they mean opposite things.
        """
        results = self.by_category(category)
        if not results:
            return "empty"
        if self.skipped_in(category):
            return "partial"
        mean = self.category_score(category)
        if mean >= SATURATED_AT:
            return "saturated"
        if mean <= FLOORED_AT:
            return "floored"
        return "healthy"

    def headroom(self, category: str) -> float:
        """Fraction of this category's weight still unearned."""
        return round((1.0 - self.category_score(category)) * WEIGHTS.get(category, 0.0), 2)

    def next_targets(self, limit: int = 5) -> list[CheckResult]:
        """The nearest misses: unsolved checks, easiest and closest first.

        This is what the loop should attack. Chasing the hardest failure is how
        a loop stalls; chasing the *nearest* one is how it climbs.
        """
        candidates = [r for r in self.results if not r.passed and not r.skipped]
        candidates.sort(key=lambda r: (r.tier or 0, -r.score))
        return candidates[:limit]

    def diagnostics(self) -> dict[str, Any]:
        return {
            category: {
                "calibration": self.calibration(category),
                "frontier": self.frontier(category),
                "tier_profile": {str(k): round(v, 3) for k, v in self.tier_profile(category).items()},
                "headroom": self.headroom(category),
            }
            for category in CAPABILITY_CATEGORIES
        }

    @property
    def instrument_warnings(self) -> list[str]:
        """Problems with the benchmark itself, not with the system under test."""
        warnings: list[str] = []
        for category in CAPABILITY_CATEGORIES:
            state = self.calibration(category)
            if state == "saturated":
                warnings.append(
                    f"{category}: saturated ({self.category_score(category):.0%}) — "
                    "every task is solved, so this suite can no longer show progress. "
                    "Add harder tasks."
                )
            elif state == "floored":
                warnings.append(
                    f"{category}: floored ({self.category_score(category):.0%}) — "
                    "almost nothing passes, so improvements will not register. "
                    "Add easier tasks, or fix the fundamentals first."
                )
            elif state == "empty":
                warnings.append(f"{category}: no checks ran.")
            elif state == "partial":
                ran, skipped = len(self.by_category(category)), len(self.skipped_in(category))
                warnings.append(
                    f"{category}: partial run — {skipped} of {ran + skipped} checks skipped, "
                    "so this score is not comparable to a full sweep and calibration "
                    "cannot be judged."
                )
        return warnings

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
            "diagnostics": self.diagnostics(),
            "instrument_warnings": self.instrument_warnings,
            "next_targets": [r.to_dict() for r in self.next_targets()],
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
            "  " + "─" * 66,
        ]
        for category, weight in WEIGHTS.items():
            results = self.by_category(category)
            score = self.category_score(category)
            solved = sum(1 for r in results if r.passed)
            filled = int(round(score * 18))
            bar = "█" * filled + "·" * (18 - filled)
            suffix = ""
            if category in CAPABILITY_CATEGORIES and results:
                suffix = f"  T{self.frontier(category)}"
            lines.append(
                f"  {category:<13} {bar} {score * weight:5.1f}/{weight:<4.0f} "
                f"({solved}/{len(results)} solved){suffix}"
            )

        # Tier profile: where the curve breaks down is where the work is.
        lines += ["", "  difficulty profile (mean score per tier)"]
        header = "                  " + "".join(f"  T{t}   " for t in range(1, 6))
        lines.append(header)
        for category in CAPABILITY_CATEGORIES:
            profile = self.tier_profile(category)
            if not profile:
                continue
            cells = "".join(
                f"  {profile[t]:.2f} " if t in profile else "   --  " for t in range(1, 6)
            )
            lines.append(f"  {category:<15} {cells}  [{self.calibration(category)}]")

        if self.instrument_warnings:
            lines += ["", "  ⚠ benchmark calibration"]
            for warning in self.instrument_warnings:
                lines.append(f"    {warning}")

        targets = self.next_targets()
        if targets:
            lines += ["", "  nearest misses (attack these first)"]
            for target in targets:
                tier = f"T{target.tier}" if target.tier else "  "
                lines.append(f"    {tier} {target.score:.2f}  {target.name}")
                if target.detail:
                    lines.append(f"           {target.detail.strip().splitlines()[0][:88]}")

        note = self.metadata.get("note")
        if note:
            lines += ["", f"  note: {note}"]

        skipped = [r for r in self.results if r.skipped]
        if skipped:
            lines.append(f"  {len(skipped)} check(s) skipped in this mode")

        lines.append("")
        return "\n".join(lines)

    def feedback(self, limit: int = 12) -> str:
        """The failure digest handed to the model for the next iteration (R-705)."""
        lines = [f"Composite score: {self.composite:.1f}/100.", ""]

        if self.gated:
            lines.append(
                "**The conformance suite is failing.** The composite is clamped to 0 "
                "until it is green. Fix that before anything else.\n"
            )

        lines.append("Difficulty frontier per suite (the tier where you stop being reliable):")
        for category in CAPABILITY_CATEGORIES:
            profile = self.tier_profile(category)
            if not profile:
                continue
            shape = " ".join(f"T{t}:{v:.2f}" for t, v in sorted(profile.items()))
            lines.append(
                f"- **{category}** — frontier T{self.frontier(category)} "
                f"({self.headroom(category):.1f} points unearned) — {shape}"
            )

        targets = self.next_targets(limit=6)
        if targets:
            lines += ["", "Nearest misses — these are the cheapest points on the board:"]
            for target in targets:
                ids = f" (requirements: {', '.join(target.requirement_ids)})" if target.requirement_ids else ""
                lines.append(
                    f"\n### {target.name} — scored {target.score:.0%}"
                    f"{f' [tier {target.tier}]' if target.tier else ''}{ids}"
                )
                lines.append((target.detail or "(no detail captured)").strip()[:1200])

        others = [r for r in self.failures if r not in targets]
        if others:
            lines += ["", f"Also failing ({len(others)}):"]
            for failure in others[:limit]:
                lines.append(f"- {failure.name} — {failure.score:.0%} — {failure.detail[:180]}")

        if self.instrument_warnings:
            lines += ["", "Benchmark calibration warnings (report these, do not try to fix them):"]
            lines += [f"- {w}" for w in self.instrument_warnings]

        if not self.failures:
            lines.append(
                "\nEverything currently measured passes. Say so in your summary rather "
                "than inventing a change — the benchmark needs harder tasks, which is "
                "a human's call."
            )
        return "\n".join(lines)


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Per-category delta between two scorecards (R-705)."""
    deltas = {}
    for category in WEIGHTS:
        old = (before.get("categories") or {}).get(category, {}).get("points", 0.0)
        new = (after.get("categories") or {}).get(category, {}).get("points", 0.0)
        deltas[category] = round(new - old, 2)

    frontier_moves = {}
    for category in CAPABILITY_CATEGORIES:
        old_f = (before.get("diagnostics") or {}).get(category, {}).get("frontier", 0)
        new_f = (after.get("diagnostics") or {}).get(category, {}).get("frontier", 0)
        if old_f != new_f:
            frontier_moves[category] = f"T{old_f} -> T{new_f}"

    return {
        "composite_before": before.get("composite", 0.0),
        "composite_after": after.get("composite", 0.0),
        "delta": round(after.get("composite", 0.0) - before.get("composite", 0.0), 2),
        "categories": deltas,
        "frontier_moves": frontier_moves,
    }
