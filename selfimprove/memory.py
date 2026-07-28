"""Persistent experience memory for Morph's self-improvement loop.

The raw JSONL history is an audit log. This module turns each audit record into
the smaller unit a planner actually needs:

    target -> hypothesis -> approach -> evidence -> outcome -> lesson

Old history rows are enriched when read, while new rows persist the structured
experience directly. No model call is needed to remember a lesson, so memory
quality does not depend on the same small model that made the mistake.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SCORE_CHANGE_EPSILON = 0.005


def _compact(value: Any, limit: int = 600) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _diagnosis_section(text: str, heading: str) -> str:
    match = re.search(
        rf"(?ims)^\s*{re.escape(heading)}:\s*(.*?)(?=^\s*[A-Z][A-Z _-]*:\s*|\Z)",
        text or "",
    )
    return _compact(match.group(1) if match else "")


def _synthetic_fixture_copy(entry: dict[str, Any]) -> bool:
    """Recognise false historical wins that copied benchmark fixtures.

    Run 11 predates the explicit fixture-leak gate, so its copied ``calc.py``
    was recorded as accepted. Reinterpreting that row from the durable evidence
    keeps a known false positive from becoming advice.
    """
    changed = [str(path).replace("\\", "/") for path in entry.get("files_changed") or []]
    morph_names = {
        Path(path).name for path in changed if path.startswith("morph/")
    }
    if not morph_names:
        return False
    try:
        from bench.tasks import ALL_TASKS

        fixture_names = {
            Path(path).name
            for task in ALL_TASKS
            for path in task.files
        }
    except ImportError:
        fixture_names = {"calc.py", "greet.py", "auth.py"}
    return bool(morph_names & fixture_names)


def _failure_kind(
    entry: dict[str, Any], *, invalidated: bool, target_gain: float | None
) -> str:
    if invalidated:
        return "invalidated_false_positive"
    if entry.get("accepted"):
        return "accepted"

    reason = _compact(entry.get("rejection_reason"), 500).lower()
    failures = " ".join(
        _compact(item, 500).lower() for item in entry.get("tool_failures") or []
    )
    if "old_string not found" in failures or "identical" in failures:
        return "edit_context_mismatch"
    if "copied synthetic benchmark fixture" in reason:
        return "fixture_copy"
    if "made no changes" in reason:
        return "no_change"
    if "target did not improve" in reason:
        return "target_no_gain"
    if "repeated rejected change" in reason:
        return "repeated_change"
    if "modified protected files" in reason:
        return "protected_goalpost"
    if "conformance suite is failing" in reason:
        return "conformance_failure"
    if "score regressed" in reason:
        return "local_gain_global_regression" if target_gain and target_gain > 0 else "regression"
    if "exceeded" in reason or "timeout" in reason:
        return "timeout"
    if "diagnosis failed" in reason:
        return "diagnosis_failure"
    if "agent failed" in reason:
        return "agent_failure"
    return "other_rejection"


LESSONS: dict[str, tuple[str, str]] = {
    "invalidated_false_positive": (
        "This apparent win was invalidated: it copied a temporary benchmark fixture "
        "into Morph and the score movement was not trustworthy.",
        "Never retry this approach. Fix the general agent behavior under morph/ instead.",
    ),
    "accepted": (
        "This change passed the gate and improved or preserved the measured system.",
        "Build on the successful mechanism; do not silently undo its verified behavior.",
    ),
    "edit_context_mismatch": (
        "The plan did not become a valid edit because edit_file used guessed or stale context.",
        "Retry only after reading the exact real file and choosing a verbatim, unique edit anchor.",
    ),
    "fixture_copy": (
        "The candidate confused benchmark evidence with repository source.",
        "Retry only with a general agent/tool change; never create the fixture under morph/.",
    ),
    "no_change": (
        "Reasoning ended without a repository change, so there was nothing to evaluate.",
        "Retry only with a concrete existing file, a specific mechanism, and an executable edit.",
    ),
    "target_no_gain": (
        "The code changed but the exact target did not improve.",
        "Do not repeat the same mechanism unless there is a new causal hypothesis or the target baseline changed.",
    ),
    "repeated_change": (
        "The candidate reproduced an already rejected final code state.",
        "Retry only after the relevant benchmark baseline changes or with a materially different patch.",
    ),
    "protected_goalpost": (
        "The candidate changed the measurement contract instead of the agent.",
        "Retry only in unprotected implementation files; goalposts remain human-owned.",
    ),
    "conformance_failure": (
        "The candidate damaged required behavior despite its intended local benefit.",
        "Retry with a narrower change that preserves the failing conformance requirement.",
    ),
    "local_gain_global_regression": (
        "The target improved locally but the full benchmark regressed.",
        "Preserve the useful local mechanism only if its cross-category side effects can be removed.",
    ),
    "regression": (
        "The full benchmark regressed after the change.",
        "Retry only with a different mechanism or substantially narrower scope.",
    ),
    "timeout": (
        "The attempt exhausted its time budget before producing a validated improvement.",
        "Retry only with a smaller plan and an early narrow test.",
    ),
    "diagnosis_failure": (
        "The archive-analysis stage failed before it produced a usable plan.",
        "Retry only after simplifying the dossier or using clearer repository evidence.",
    ),
    "agent_failure": (
        "The implementation agent failed before evaluation.",
        "Retry only after addressing the recorded runtime/tool failure.",
    ),
    "other_rejection": (
        "The candidate did not satisfy the acceptance gate.",
        "Use the recorded evidence to form a different, testable hypothesis before retrying.",
    ),
}


def experience_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Derive a compact, deterministic experience record from one history row."""
    diagnosis = str(entry.get("diagnosis") or "")
    hypothesis = _diagnosis_section(diagnosis, "CAUSE")
    planned_change = _diagnosis_section(diagnosis, "CHANGE")
    summary = _compact(entry.get("summary"), 700)
    if planned_change.lower() in {"none", "no change", "no code change is warranted"}:
        planned_change = ""
    approach = planned_change or summary or "(no approach recorded)"

    before = entry.get("target_score_before")
    after = entry.get("target_score_after")
    target_gain = (
        float(after) - float(before)
        if isinstance(before, (int, float)) and isinstance(after, (int, float))
        else None
    )
    composite_gain = float(entry.get("score_after") or 0) - float(
        entry.get("score_before") or 0
    )

    invalidated = bool(entry.get("invalidated")) or _synthetic_fixture_copy(entry)
    kind = _failure_kind(entry, invalidated=invalidated, target_gain=target_gain)
    lesson, retry = LESSONS[kind]

    evidence: list[str] = []
    reason = _compact(entry.get("invalidated_reason") or entry.get("rejection_reason"))
    if reason:
        evidence.append(reason)
    elif invalidated:
        evidence.append(
            "historical audit: copied a temporary benchmark fixture into morph/"
        )
    if target_gain is not None:
        evidence.append(f"exact target {float(before):.1%} -> {float(after):.1%}")
    evidence.append(
        f"composite {float(entry.get('score_before') or 0):.2f} -> "
        f"{float(entry.get('score_after') or 0):.2f}"
    )
    evidence.extend(
        _compact(item, 360) for item in (entry.get("tool_failures") or [])[:3]
    )

    files = sorted(str(path) for path in entry.get("files_changed") or [])
    fingerprint_material = {
        "target": str(entry.get("target") or ""),
        "files": files,
        "approach": re.sub(r"[^a-z0-9]+", " ", approach.lower())[:500],
    }
    approach_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_material, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    identity_material = {
        "ts": entry.get("ts"),
        "base_commit": entry.get("base_commit"),
        "iteration": entry.get("iteration"),
        "target": entry.get("target"),
    }
    experience_id = hashlib.sha256(
        json.dumps(identity_material, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    return {
        "schema": SCHEMA_VERSION,
        "id": experience_id,
        "target": str(entry.get("target") or ""),
        "outcome": "invalidated" if invalidated else ("accepted" if entry.get("accepted") else "rejected"),
        "failure_kind": kind,
        "hypothesis": hypothesis or "(no causal hypothesis recorded)",
        "approach": approach,
        "files": files,
        "evidence": evidence,
        "lesson": lesson,
        "retry_condition": retry,
        "composite_delta": round(composite_gain, 3),
        "target_delta": round(target_gain, 4) if target_gain is not None else None,
        "approach_fingerprint": approach_fingerprint,
        "change_fingerprint": str(entry.get("change_fingerprint") or ""),
    }


def enrich_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Return a history row with a stable structured experience attached."""
    enriched = dict(entry)
    derived = experience_from_entry(enriched)
    persisted = enriched.get("experience")
    if isinstance(persisted, dict):
        # Persisted fields are the historical record. Newly derived fields fill
        # additions introduced by later schema versions without rewriting it.
        derived.update(persisted)
    enriched["experience"] = derived
    return enriched


def target_stats(history: list[dict[str, Any]], target: str) -> dict[str, Any]:
    """All-time statistics for one target, used for non-repeating selection."""
    attempts: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for index, raw in enumerate(history):
        entry = enrich_entry(raw)
        experience = entry["experience"]
        if experience.get("target") == target:
            attempts.append((index, entry, experience))

    consecutive_failures = 0
    for _, _, experience in reversed(attempts):
        if experience.get("outcome") == "accepted":
            break
        consecutive_failures += 1

    last_entry = attempts[-1][1] if attempts else {}
    last_experience = attempts[-1][2] if attempts else {}
    if (
        last_experience.get("outcome") == "accepted"
        and last_entry.get("target_score_after") is not None
    ):
        last_target_score = last_entry.get("target_score_after")
    else:
        # A rejected candidate never became the new baseline, so its measured
        # target_score_after describes throwaway code. Compare future runs with
        # the baseline that candidate actually started from.
        last_target_score = last_entry.get("target_score_before")
    return {
        "attempts": len(attempts),
        "accepted": sum(
            1 for _, _, exp in attempts if exp.get("outcome") == "accepted"
        ),
        "rejected": sum(
            1 for _, _, exp in attempts if exp.get("outcome") != "accepted"
        ),
        "consecutive_failures": consecutive_failures,
        "last_index": attempts[-1][0] if attempts else -1,
        "last_target_score": last_target_score,
        "last_experience": attempts[-1][2] if attempts else None,
    }


def _deduplicated_cards(
    experiences: list[dict[str, Any]], limit: int
) -> list[tuple[dict[str, Any], int]]:
    cards: list[tuple[dict[str, Any], int]] = []
    positions: dict[str, int] = {}
    for experience in reversed(experiences):
        key = str(experience.get("approach_fingerprint") or experience.get("id"))
        if key in positions:
            card, count = cards[positions[key]]
            cards[positions[key]] = (card, count + 1)
            continue
        positions[key] = len(cards)
        cards.append((experience, 1))
        if len(cards) >= limit:
            break
    return cards


def render_experience_memory(
    history: list[dict[str, Any]],
    target: dict[str, Any] | str | None,
    *,
    max_successes: int = 3,
    max_failures: int = 6,
) -> str:
    """Render target-relevant archive evidence for Gemma's small context."""
    target_name = (
        str(target.get("name") or "") if isinstance(target, dict) else str(target or "")
    )
    all_experiences = [enrich_entry(entry)["experience"] for entry in history]
    relevant = (
        [exp for exp in all_experiences if exp.get("target") == target_name]
        if target_name
        else all_experiences
    )
    successes = [exp for exp in relevant if exp.get("outcome") == "accepted"]
    failures = [exp for exp in relevant if exp.get("outcome") != "accepted"]

    lines = [
        "# Experience memory",
        "",
        "This is the durable archive of what Morph actually tried. Treat outcomes",
        "and evaluator evidence as facts; the old model's prose is only a hypothesis.",
        f"Target: `{target_name or 'all targets'}` — {len(relevant)} prior attempt(s), "
        f"{len(successes)} verified success(es), {len(failures)} failure(s).",
    ]

    if successes:
        lines += ["", "## What worked"]
        for experience, count in _deduplicated_cards(successes, max_successes):
            repeat = f" (observed {count} times)" if count > 1 else ""
            lines.append(
                f"- `{experience['id']}` [ACCEPTED]{repeat}: "
                f"{experience['approach']}"
            )
            lines.append(f"  - evidence: {'; '.join(experience['evidence'][:2])}")
            lines.append(f"  - retain: {experience['lesson']}")
    else:
        lines += ["", "## What worked", "", "- No verified successful approach is recorded for this target."]

    if failures:
        lines += ["", "## What failed"]
        for experience, count in _deduplicated_cards(failures, max_failures):
            repeat = f" (repeated {count} times)" if count > 1 else ""
            lines.append(
                f"- `{experience['id']}` [REJECTED:{experience['failure_kind']}]"
                f"{repeat}: {experience['approach']}"
            )
            lines.append(f"  - evidence: {'; '.join(experience['evidence'][:3])}")
            lines.append(f"  - lesson: {experience['lesson']}")
            lines.append(f"  - retry only if: {experience['retry_condition']}")
    else:
        lines += ["", "## What failed", "", "- No rejected approach is recorded for this target."]

    pattern_counts = Counter(
        exp["failure_kind"]
        for exp in all_experiences
        if exp.get("outcome") != "accepted"
    )
    if pattern_counts:
        lines += ["", "## Cross-target failure patterns"]
        for kind, count in pattern_counts.most_common(4):
            lines.append(f"- {kind}: {count} occurrence(s). {LESSONS[kind][0]}")

    lines += [
        "",
        "## Planning constraint",
        "",
        "Before proposing a change, identify which archived lesson it uses. If it",
        "resembles a rejected approach, state the new evidence or changed condition",
        "that makes retrying rational. Otherwise choose a materially different mechanism.",
    ]
    return "\n".join(lines)


def repeated_rejected_change(
    history: list[dict[str, Any]],
    *,
    target: str,
    change_fingerprint: str,
    current_metric: float | None,
) -> dict[str, Any] | None:
    """Return a matching failed experience when its retry condition is unmet."""
    if not change_fingerprint:
        return None
    for raw in reversed(history):
        entry = enrich_entry(raw)
        experience = entry["experience"]
        if experience.get("target") != target:
            continue
        if experience.get("outcome") == "accepted":
            continue
        if experience.get("change_fingerprint") != change_fingerprint:
            continue

        previous_metric = (
            entry.get("target_score_before")
            if entry.get("target_score_before") is not None
            else entry.get("score_before")
        )
        if (
            current_metric is not None
            and isinstance(previous_metric, (int, float))
            and abs(float(current_metric) - float(previous_metric)) > SCORE_CHANGE_EPSILON
        ):
            continue
        return experience
    return None
