"""The self-improvement loop (R-701 … R-707).

    benchmark  →  feedback  →  Gemma edits the code  →  benchmark  →  keep or revert

Each iteration runs in its own git worktree, so the working branch is never left
broken (R-703). The edits are made by **Morph's own agent** (R-702): if a change
degrades the agent, the next iteration feels it immediately.

PROTECTED FILE — the loop may not modify this (R-707).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.runner import run_benchmark
from bench.scorecard import compare
from morph.agent import Agent
from morph.config import Config
from morph.llm import get_provider

from .guard import violations
from .prompts import SYSTEM_PROMPT, append_history, build_improvement_prompt, load_history

log = logging.getLogger("selfimprove")

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = REPO_ROOT / "selfimprove" / "history.jsonl"
WORKTREE_ROOT = Path(".morph") / "worktrees"
AGENT_MAX_STEPS = 60
AGENT_TIMEOUT = 3600.0


class GitError(RuntimeError):
    pass


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


@dataclass
class Iteration:
    index: int
    base_commit: str
    branch: str
    worktree: Path
    score_before: float = 0.0
    score_after: float = 0.0
    accepted: bool = False
    rejection_reason: str = ""
    summary: str = ""
    files_changed: list[str] = field(default_factory=list)
    agent_steps: int = 0
    duration_s: float = 0.0
    deltas: dict[str, Any] = field(default_factory=dict)

    def to_entry(self) -> dict[str, Any]:
        return {
            "ts": time.time(),
            "iteration": self.index,
            "base_commit": self.base_commit,
            "branch": self.branch,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "summary": self.summary,
            "files_changed": self.files_changed,
            "agent_steps": self.agent_steps,
            "duration_s": round(self.duration_s, 1),
            "deltas": self.deltas,
        }


# ---------------------------------------------------------------------------


def _bench_config(root: Path, config: Config) -> Config:
    """Config for measuring.

    Capability is measured against the *same* model that does the editing, so the
    score is a real signal the loop can move. Measuring with ``echo`` instead
    replays reference traces, which pins capability at full marks and leaves the
    loop able to detect regressions but not improvements — useful as a fast
    smoke test, not as an objective. Set ``MORPH_BENCH_PROVIDER=echo`` to opt
    into that cheaper, fully deterministic mode.

    The image backend stays on ``stub`` either way: determinism there costs no
    signal and no GPU.
    """
    provider = os.environ.get("MORPH_BENCH_PROVIDER") or config.provider
    return Config(
        workspace=root,
        provider=provider,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        image_backend="stub",
    )


def _agent_config(root: Path, config: Config) -> Config:
    """Config for the editing agent. This is where Gemma actually runs."""
    return Config(
        workspace=root,
        provider=config.provider,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        max_steps=AGENT_MAX_STEPS,
        image_backend=config.image_backend,
        skill_paths=[root / "skills"],
        allow_shell=True,
        shell_timeout=900.0,
    )


async def measure(root: Path, config: Config) -> dict[str, Any]:
    scorecard = await run_benchmark(_bench_config(root, config), repo=root)
    return scorecard.to_dict()


def _create_worktree(repo: Path, branch: str, base: str) -> Path:
    path = repo / WORKTREE_ROOT / branch.replace("/", "-")
    if path.exists():
        _remove_worktree(repo, path, branch)
    path.parent.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "add", "-f", "-b", branch, str(path), base)
    return path


def _remove_worktree(repo: Path, path: Path, branch: str) -> None:
    git(repo, "worktree", "remove", "--force", str(path), check=False)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    git(repo, "worktree", "prune", check=False)
    git(repo, "branch", "-D", branch, check=False)


# ---------------------------------------------------------------------------


async def run_iteration(
    index: int,
    repo: Path,
    config: Config,
    baseline: dict[str, Any],
    history: list[dict[str, Any]],
    dry_run: bool = False,
    focus: str | None = None,
) -> Iteration:
    """One full cycle: edit in isolation, measure, keep or revert."""
    started = time.perf_counter()
    base_commit = git(repo, "rev-parse", "HEAD")
    branch = f"selfimprove/iter-{index}-{int(time.time())}"
    worktree = _create_worktree(repo, branch, base_commit)

    iteration = Iteration(
        index=index,
        base_commit=base_commit,
        branch=branch,
        worktree=worktree,
        score_before=baseline.get("composite", 0.0),
    )

    try:
        requirements = (repo / "REQUIREMENTS.md").read_text("utf-8")
        prompt = build_improvement_prompt(
            requirements=requirements,
            scorecard=baseline,
            feedback=baseline.get("feedback") or _feedback_from(baseline),
            history=history,
            focus=focus,
        )

        # Morph improves Morph: the editing agent is this project's own agent (R-702).
        agent_config = _agent_config(worktree, config)
        agent = Agent(
            config=agent_config,
            provider=get_provider(
                config.provider,
                model=config.model,
                base_url=config.base_url,
                temperature=config.temperature,
            ),
            system_prompt=SYSTEM_PROMPT,
        )
        try:
            result = await asyncio.wait_for(
                agent.run(prompt, max_steps=AGENT_MAX_STEPS), timeout=AGENT_TIMEOUT
            )
        finally:
            await agent.close()

        iteration.summary = (result.text or "").strip()[:2000]
        iteration.agent_steps = result.steps

        if result.error:
            iteration.rejection_reason = f"agent failed: {result.error}"
            iteration.score_after = iteration.score_before
            return iteration

        iteration.files_changed = [
            name
            for name in _changed(worktree, base_commit)
            if not name.startswith(".morph/")
        ]
        if not iteration.files_changed:
            iteration.rejection_reason = "the agent made no changes"
            iteration.score_after = iteration.score_before
            return iteration

        # R-707: touching the goalposts rejects the whole iteration.
        breached = violations(worktree, base_commit)
        if breached:
            iteration.rejection_reason = f"modified protected files: {', '.join(breached)}"
            iteration.score_after = 0.0
            return iteration

        after = await measure(worktree, config)
        iteration.score_after = after.get("composite", 0.0)
        iteration.deltas = compare(baseline, after)

        if after.get("gated"):
            iteration.rejection_reason = "conformance suite is failing after the change"
            return iteration
        if iteration.score_after < iteration.score_before:
            iteration.rejection_reason = (
                f"score regressed ({iteration.score_before:.1f} → {iteration.score_after:.1f})"
            )
            return iteration

        # R-704: accepted.
        if dry_run:
            iteration.accepted = True
            iteration.rejection_reason = "(dry run: not merged)"
            return iteration

        _commit_and_merge(repo, worktree, branch, iteration)
        iteration.accepted = True
        return iteration

    except asyncio.TimeoutError:
        iteration.rejection_reason = f"the agent exceeded {AGENT_TIMEOUT:g}s"
        iteration.score_after = iteration.score_before
        return iteration
    except Exception as exc:  # noqa: BLE001 - one bad iteration must not end the loop
        log.exception("Iteration %d failed", index)
        iteration.rejection_reason = f"{type(exc).__name__}: {exc}"
        iteration.score_after = iteration.score_before
        return iteration
    finally:
        iteration.duration_s = time.perf_counter() - started
        if not dry_run or not iteration.accepted:
            _remove_worktree(repo, worktree, branch)


def _changed(worktree: Path, base: str) -> list[str]:
    tracked = git(worktree, "diff", "--name-only", base, check=False)
    untracked = git(worktree, "ls-files", "--others", "--exclude-standard", check=False)
    return sorted(
        {line.strip() for line in f"{tracked}\n{untracked}".splitlines() if line.strip()}
    )


def _commit_and_merge(repo: Path, worktree: Path, branch: str, iteration: Iteration) -> None:
    """Commit the accepted work and fast-forward the main working branch."""
    git(worktree, "add", "-A")
    message = (
        f"selfimprove: iteration {iteration.index} "
        f"({iteration.score_before:.1f} -> {iteration.score_after:.1f})\n\n"
        f"{iteration.summary[:1200]}\n\n"
        f"Files: {', '.join(iteration.files_changed[:20])}"
    )
    git(worktree, "commit", "-m", message)
    git(repo, "merge", "--ff-only", branch)


def _feedback_from(scorecard: dict[str, Any]) -> str:
    """Rebuild a failure digest from a serialised scorecard."""
    failures = [r for r in scorecard.get("results", []) if not r.get("passed")]
    if not failures:
        return (
            "Everything passes. Look for improvements to efficiency and health, or add "
            "capability without breaking anything."
        )
    lines = [f"Composite score: {scorecard.get('composite', 0):.1f}/100.", "", "Failing checks:"]
    for failure in failures[:20]:
        ids = failure.get("requirement_ids") or []
        suffix = f" (requirements: {', '.join(ids)})" if ids else ""
        lines.append(f"\n### {failure['name']} [{failure['category']}]{suffix}")
        lines.append((failure.get("detail") or "(no detail)")[:1500])
    if len(failures) > 20:
        lines.append(f"\n… and {len(failures) - 20} further failures.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------


async def run_loop(
    config: Config | None = None,
    iterations: int = 1,
    repo: Path | None = None,
    dry_run: bool = False,
    focus: str | None = None,
    history_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run ``iterations`` improvement cycles. Returns one history entry each."""
    repo = Path(repo or REPO_ROOT)
    cfg = config or Config(workspace=repo)
    history_file = Path(history_path or HISTORY_PATH)

    if git(repo, "rev-parse", "--is-inside-work-tree", check=False) != "true":
        raise GitError(f"{repo} is not a git repository; the loop needs git for isolation")

    baseline = await measure(repo, cfg)
    log.info("Baseline score: %.1f", baseline.get("composite", 0.0))

    entries: list[dict[str, Any]] = []
    for index in range(1, iterations + 1):
        history = load_history(history_file)
        iteration = await run_iteration(
            index=index,
            repo=repo,
            config=cfg,
            baseline=baseline,
            history=history,
            dry_run=dry_run,
            focus=focus,
        )
        entry = iteration.to_entry()
        append_history(history_file, entry)  # R-705
        entries.append(entry)

        log.info(
            "Iteration %d: %s (%.1f -> %.1f) %s",
            index,
            "accepted" if iteration.accepted else "rejected",
            iteration.score_before,
            iteration.score_after,
            iteration.rejection_reason,
        )

        if iteration.accepted and not dry_run:
            baseline = await measure(repo, cfg)

    return entries


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="selfimprove", description="Gemma improves Morph")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--base-url")
    parser.add_argument("--focus", help="Steer this run at a specific area")
    parser.add_argument("--dry-run", action="store_true", help="Measure but never merge")
    parser.add_argument("--repo", default=str(REPO_ROOT))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = Config(
        workspace=Path(args.repo),
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )
    entries = asyncio.run(
        run_loop(
            config,
            iterations=args.iterations,
            repo=Path(args.repo),
            dry_run=args.dry_run,
            focus=args.focus,
        )
    )

    print(json.dumps(entries, indent=2, default=str))
    return 0 if any(e["accepted"] for e in entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
