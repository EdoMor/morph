"""Protected paths (R-707).

The loop optimises a score. Anything that lets it edit the definition of the
score, the requirements, or the acceptance criteria turns self-improvement into
self-congratulation. These paths are off limits, and an iteration that touches
one is rejected outright — no partial credit, no merge.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

PROTECTED: tuple[str, ...] = (
    "REQUIREMENTS.md",
    "tests/test_requirements.py",
    "bench/scorecard.py",
    "bench/runner.py",
    # The whole task suite, not just the scorer. A loop that can author its own
    # benchmark tasks will author easy ones, which is the same failure as editing
    # the scorer, one level down. Growing the benchmark stays a human's job; the
    # scorecard's calibration warnings tell the human when it needs growing.
    "bench/tasks",
    "selfimprove/guard.py",
    "selfimprove/loop.py",
    "selfimprove/memory.py",
    "selfimprove/proposals.py",
    "selfimprove/strategies.json",
    "selfimprove/history.jsonl",
    ".github/workflows/self-improve.yml",
)


def is_protected(path: str) -> bool:
    # removeprefix, not lstrip: lstrip takes a character set and would turn
    # ".github/workflows/..." into "github/workflows/...".
    normalised = path.replace("\\", "/").removeprefix("./")
    return any(normalised == p or normalised.startswith(f"{p}/") for p in PROTECTED)


def changed_files(repo: Path, base_ref: str = "HEAD") -> list[str]:
    """Every path that differs from ``base_ref``, including untracked files."""
    tracked = subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    names = {
        line.strip()
        for line in (tracked.stdout + "\n" + untracked.stdout).splitlines()
        if line.strip()
    }
    return sorted(names)


def violations(repo: Path, base_ref: str = "HEAD") -> list[str]:
    """Protected paths the iteration modified. Empty means the iteration is eligible."""
    return [name for name in changed_files(repo, base_ref) if is_protected(name)]


def guard_prompt_section() -> str:
    """The rule as the model sees it. Stated plainly, and enforced regardless."""
    listing = "\n".join(f"- `{p}`" for p in PROTECTED)
    return (
        "## Files you must not modify\n\n"
        f"{listing}\n\n"
        "These define the goal and how it is measured. An iteration that changes any of "
        "them is rejected in full, however good the rest of the work is. If you believe a "
        "requirement or a test is wrong, say so in your final message — a human will "
        "decide. Do not route around it.\n"
    )
