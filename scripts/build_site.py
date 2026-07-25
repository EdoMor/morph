#!/usr/bin/env python3
"""Build the GitHub Pages dashboard data (R-717).

Reads what the loop already commits — ``selfimprove/history.jsonl`` and the
latest scorecard — and flattens it into one JSON file the static page consumes.
No server, no build step, no dependencies: the page is plain HTML and the data
is a file next to it.

    python scripts/build_site.py [--out site/data/dashboard.json]

Runs happily with no data at all, which is the state of a fresh fork: the page
then says so instead of rendering an empty chart.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY = Path("selfimprove/history.jsonl")
SCORECARD = Path("selfimprove/scorecard.json")
CHANGELOG = Path("CHANGELOG.md")
VERSION_RE = re.compile(r'__version__ = "(\d+\.\d+\.\d+)"')


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def read_history(repo: Path) -> list[dict[str, Any]]:
    path = repo / HISTORY
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def read_scorecard(repo: Path) -> dict[str, Any]:
    path = repo / SCORECARD
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError:
        return {}


def current_version(repo: Path) -> str:
    match = VERSION_RE.search((repo / "morph" / "__init__.py").read_text("utf-8"))
    return match.group(1) if match else "0.0.0"


def releases(repo: Path) -> list[dict[str, str]]:
    """Version tags, newest first, with their changelog blurb."""
    tags = [t for t in _git(repo, "tag", "-l", "v*", "--sort=-v:refname").splitlines() if t]

    blurbs: dict[str, str] = {}
    changelog = repo / CHANGELOG
    if changelog.is_file():
        for block in re.split(r"^## ", changelog.read_text("utf-8"), flags=re.M)[1:]:
            heading, _, body = block.partition("\n")
            tag = heading.split("—")[0].strip()
            blurbs[tag] = body.strip()

    return [
        {
            "tag": tag,
            "date": _git(repo, "log", "-1", "--format=%ad", "--date=short", tag),
            "notes": blurbs.get(tag, ""),
        }
        for tag in tags
    ]


def score_series(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Composite score over time, one point per iteration attempt."""
    series = []
    for entry in history:
        series.append(
            {
                "ts": entry.get("ts", 0),
                "iteration": entry.get("iteration", 0),
                "before": entry.get("score_before", 0.0),
                "after": entry.get("score_after", 0.0),
                "accepted": bool(entry.get("accepted")),
                "version": entry.get("version") or "",
            }
        )
    return series


def summarise(history: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [e for e in history if e.get("accepted")]
    rejected = [e for e in history if not e.get("accepted")]

    reasons: dict[str, int] = {}
    for entry in rejected:
        reason = (entry.get("rejection_reason") or "unknown").strip()
        # Collapse the parameterised ones so the tally means something.
        reason = re.sub(r"\(.*?\)", "", reason).strip() or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1

    total_seconds = sum(e.get("duration_s", 0) for e in history)
    return {
        "attempts": len(history),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": round(len(accepted) / len(history), 3) if history else 0.0,
        "rejection_reasons": sorted(
            ({"reason": k, "count": v} for k, v in reasons.items()),
            key=lambda r: -r["count"],
        ),
        "model_hours": round(total_seconds / 3600, 1),
        "first_run": min((e.get("ts", 0) for e in history), default=0),
        "last_run": max((e.get("ts", 0) for e in history), default=0),
    }


def build(repo: Path = REPO_ROOT) -> dict[str, Any]:
    history = read_history(repo)
    scorecard = read_scorecard(repo)

    return {
        "generated_at": time.time(),
        "repo": _git(repo, "config", "--get", "remote.origin.url")
        .removesuffix(".git")
        .replace("git@github.com:", "https://github.com/"),
        "commit": _git(repo, "rev-parse", "--short", "HEAD"),
        "version": current_version(repo),
        "scorecard": {
            "composite": scorecard.get("composite"),
            "gated": scorecard.get("gated", False),
            "categories": scorecard.get("categories", {}),
            "diagnostics": scorecard.get("diagnostics", {}),
            "instrument_warnings": scorecard.get("instrument_warnings", []),
            "next_targets": scorecard.get("next_targets", [])[:6],
            "metadata": scorecard.get("metadata", {}),
        }
        if scorecard
        else None,
        "summary": summarise(history),
        "series": score_series(history),
        "history": list(reversed(history))[:60],  # newest first
        "releases": releases(repo)[:20],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_site", description="Build the Pages data")
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--out", default="site/data/dashboard.json")
    args = parser.parse_args(argv)

    repo = Path(args.repo)
    data = build(repo)

    out = Path(args.out)
    if not out.is_absolute():
        out = repo / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), "utf-8")

    summary = data["summary"]
    score = data["scorecard"]["composite"] if data["scorecard"] else "—"
    print(
        f"wrote {out} — version {data['version']}, composite {score}, "
        f"{summary['accepted']}/{summary['attempts']} accepted, "
        f"{len(data['releases'])} release(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
