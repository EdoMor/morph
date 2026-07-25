#!/usr/bin/env python3
"""Publish the running loop's trace so the dashboard can show it live (R-721).

GitHub Pages only redeploys when a workflow finishes, which is exactly when a
live view stops being interesting. The trick is not to deploy at all: push the
trace to a dedicated branch, and let the static page fetch it from
``raw.githubusercontent.com``, which is public, CORS-enabled, and needs no
token.

    python scripts/publish_live.py --every 15        # until killed
    python scripts/publish_live.py --once            # one push, for tests

Two properties matter and are worth stating plainly:

**It never touches the working tree.** The loop is concurrently creating
worktrees, committing, and rebasing in the same repository. This writes its
commit with plumbing — ``hash-object``, ``update-index`` against a *temporary*
index, ``commit-tree`` — so the real index, HEAD and working tree are untouched
and the two cannot interfere.

**It force-pushes, and that is not a contradiction.** The no-force-push rule
(R-714) is about the publish branch, where a force would destroy someone's work.
This branch holds only the current run's telemetry, has no ancestry worth
keeping, and is rewritten from a fresh parentless commit every few seconds
precisely so it never accumulates history. It must never be given the publish
branch's name — asserted below.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BRANCH = "live"
#: Refused as targets: pushing telemetry over real code would be unrecoverable.
NEVER = frozenset({"main", "master", "gh-pages", "HEAD"})
#: What gets published, as {name in the branch: path in the repo}.
FILES = {
    "trace.jsonl": Path("selfimprove/live-trace.jsonl"),
    "status.json": Path("selfimprove/progress.json"),
}


class PublishError(RuntimeError):
    pass


def _git(repo: Path, *args: str, env: dict[str, str] | None = None, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
    )
    if check and result.returncode != 0:
        raise PublishError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def publish_once(repo: Path, branch: str = DEFAULT_BRANCH, remote: str = "origin") -> bool:
    """Push the current trace as a single parentless commit. False if nothing to send."""
    if branch in NEVER:
        raise PublishError(f"refusing to publish live telemetry to {branch!r}")

    present = {name: repo / path for name, path in FILES.items() if (repo / path).is_file()}
    if not present:
        return False

    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        _git(repo, "read-tree", "--empty", env=env)
        for name, path in present.items():
            blob = _git(repo, "hash-object", "-w", "--", str(path))
            _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{name}", env=env)
        tree = _git(repo, "write-tree", env=env)

    # Parentless: the branch is a snapshot, not a history.
    commit = _git(
        repo,
        "-c",
        "user.name=morph-selfimprove[bot]",
        "-c",
        "user.email=morph-selfimprove@users.noreply.github.com",
        "commit-tree",
        tree,
        "-m",
        f"live trace {time.strftime('%H:%M:%SZ', time.gmtime())}",
    )
    _git(repo, "push", "--force", remote, f"{commit}:refs/heads/{branch}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the live trace branch")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--every", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    while True:
        try:
            publish_once(args.repo, args.branch, args.remote)
        except Exception as exc:  # noqa: BLE001
            # Telemetry is never worth failing a run over: a transient push
            # rejection, a network blip, a race with the loop's own git — all of
            # them mean "try again in fifteen seconds", not "stop".
            print(f"live publish skipped: {exc}", file=sys.stderr)
        if args.once:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
