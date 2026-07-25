"""Publish accepted iterations to the default branch (R-713).

The loop commits accepted work onto whatever branch it is running on. This
module is what gets those commits onto `origin/main` safely and unattended.

Unattended pushing to a shared branch has exactly two failure modes worth
engineering against:

**The branch moved under us.** A run takes hours. Someone pushes to main in the
meantime. Publishing rebases onto the new head and **re-runs the gate** before
pushing — a rebase is a merge, and a merge can break tests that both sides
passed independently.

**The push races.** Between fetch and push, main moves again. The push is
rejected, and we start over rather than reaching for `--force`. Nothing here
ever force-pushes; a self-improving loop that can overwrite history is one bad
iteration away from deleting the project.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("selfimprove.publish")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ATTEMPTS = 5

#: Returns ``(ok, detail)``. Run after any rebase, before any push.
Verifier = Callable[[Path], tuple[bool, str]]


class PublishError(RuntimeError):
    pass


@dataclass
class PublishResult:
    published: bool
    reason: str
    commits: list[str] = field(default_factory=list)
    attempts: int = 0
    rebased: bool = False
    head: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "published": self.published,
            "reason": self.reason,
            "commits": self.commits,
            "attempts": self.attempts,
            "rebased": self.rebased,
            "head": self.head,
        }


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise PublishError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _try_git(repo: Path, *args: str) -> tuple[bool, str]:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    return result.returncode == 0, (result.stderr or result.stdout).strip()


def pytest_verifier(repo: Path) -> tuple[bool, str]:
    """The default gate: the conformance suite must be green post-rebase."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = (result.stdout + result.stderr).strip().splitlines()[-15:]
    return result.returncode == 0, "\n".join(tail)


def commits_ahead(repo: Path, upstream: str) -> list[str]:
    output = git(repo, "log", "--oneline", f"{upstream}..HEAD", check=False)
    return [line for line in output.splitlines() if line.strip()]


def commits_behind(repo: Path, upstream: str) -> list[str]:
    output = git(repo, "log", "--oneline", f"HEAD..{upstream}", check=False)
    return [line for line in output.splitlines() if line.strip()]


def record_history(repo: Path, history: Path | None = None) -> bool:
    """Stage and commit the run history so it survives the runner.

    Without this the loop forgets every previous attempt the moment the job
    ends, and R-705's "do not repeat a rejected approach" only holds within a
    single run — which on a scheduled loop is almost never the interesting case.
    """
    target = history or (repo / "selfimprove" / "history.jsonl")
    if not target.is_file():
        return False

    relative = str(target.relative_to(repo))
    git(repo, "add", "--", relative, check=False)
    if not git(repo, "diff", "--cached", "--name-only", check=False):
        return False

    entries = sum(1 for line in target.read_text("utf-8").splitlines() if line.strip())
    git(repo, "commit", "-m", f"selfimprove: record run history ({entries} attempts)")
    return True


def publish(
    repo: Path = REPO_ROOT,
    branch: str = "main",
    remote: str = "origin",
    verify: Verifier | None = pytest_verifier,
    attempts: int = DEFAULT_ATTEMPTS,
    dry_run: bool = False,
) -> PublishResult:
    """Get local commits onto ``remote/branch``, re-verifying after any rebase."""
    upstream = f"{remote}/{branch}"
    result = PublishResult(published=False, reason="not attempted")

    for attempt in range(1, attempts + 1):
        result.attempts = attempt
        git(repo, "fetch", remote, branch)

        ahead = commits_ahead(repo, upstream)
        if not ahead:
            result.reason = "nothing to publish — no accepted iterations this run"
            return result
        result.commits = ahead

        behind = commits_behind(repo, upstream)
        if behind:
            log.info("%s moved by %d commit(s); rebasing", upstream, len(behind))
            ok, detail = _try_git(repo, "rebase", upstream)
            if not ok:
                _try_git(repo, "rebase", "--abort")
                result.reason = (
                    f"{upstream} moved and the rebase conflicted; leaving it for a human: {detail[:400]}"
                )
                return result
            result.rebased = True

        # A rebase is a merge. Both sides being green separately proves nothing
        # about the combination, so the gate runs again before anything is pushed.
        if verify is not None and (result.rebased or attempt > 1):
            ok, detail = verify(repo)
            if not ok:
                result.reason = f"conformance suite failed after rebasing onto {upstream}:\n{detail}"
                return result

        result.head = git(repo, "rev-parse", "HEAD")
        if dry_run:
            result.reason = f"dry run: would push {len(ahead)} commit(s) to {upstream}"
            return result

        # Never --force. A loop that can overwrite history is one bad iteration
        # away from deleting the project.
        pushed, detail = _try_git(repo, "push", remote, f"HEAD:refs/heads/{branch}")
        if pushed:
            result.published = True
            result.reason = f"pushed {len(ahead)} commit(s) to {upstream}"
            return result

        log.warning("push rejected (attempt %d/%d): %s", attempt, attempts, detail[:200])
        result.reason = f"push rejected: {detail[:400]}"

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="selfimprove.publish", description="Publish accepted iterations to the default branch"
    )
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--branch", default="main")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Do not re-run the suite after a rebase (not recommended)",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not commit selfimprove/history.jsonl",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo = Path(args.repo)

    if not args.no_history:
        if record_history(repo):
            log.info("Recorded run history")

    outcome = publish(
        repo=repo,
        branch=args.branch,
        remote=args.remote,
        verify=None if args.skip_verify else pytest_verifier,
        attempts=args.attempts,
        dry_run=args.dry_run,
    )

    print(json.dumps(outcome.to_dict(), indent=2))
    log.info("%s", outcome.reason)

    # "Nothing to publish" is a normal, successful outcome for a run where no
    # iteration was good enough — it must not fail the workflow.
    if outcome.published or outcome.reason.startswith(("nothing to publish", "dry run")):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
