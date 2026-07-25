"""Versioning and release preparation (R-715).

Every accepted iteration produces a new version of the agent. The version bump
is made by the loop, **not** by the model — a system that picks its own version
numbers as part of the change it is being graded on has one more thing to get
wrong, and nothing to gain.

Tagging is deliberately *not* done here. Publishing may have to rebase onto a
branch that moved during the run, which rewrites every commit; a tag created
before that would point at an orphan. :mod:`selfimprove.publish` creates the
tags after the push succeeds, finding them by the release commit's subject.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = Path("morph") / "__init__.py"
CHANGELOG = Path("CHANGELOG.md")

VERSION_RE = re.compile(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"', re.M)
RELEASE_SUBJECT_RE = re.compile(r"^release: v(\d+\.\d+\.\d+)")

#: Patch rolls into minor here so ``version_code`` stays strictly increasing.
PATCH_CEILING = 100
MINOR_CEILING = 100


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def tag(self) -> str:
        return f"v{self}"

    @property
    def code(self) -> int:
        """Android ``versionCode``: monotonic, and never reused."""
        return self.major * 10_000 + self.minor * 100 + self.patch

    def bump(self, part: str = "patch") -> "Version":
        if part == "major":
            return Version(self.major + 1, 0, 0)
        if part == "minor":
            return Version(self.major, self.minor + 1, 0)
        if part != "patch":
            raise ValueError(f"unknown version part {part!r}; use major, minor or patch")
        if self.patch + 1 >= PATCH_CEILING:
            if self.minor + 1 >= MINOR_CEILING:
                return Version(self.major + 1, 0, 0)
            return Version(self.major, self.minor + 1, 0)
        return Version(self.major, self.minor, self.patch + 1)

    @classmethod
    def parse(cls, text: str) -> "Version":
        match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", text.strip())
        if not match:
            raise ValueError(f"not a version: {text!r}")
        return cls(*(int(g) for g in match.groups()))


@dataclass
class ReleaseInfo:
    previous: Version
    version: Version
    summary: str
    score_before: float
    score_after: float

    @property
    def tag(self) -> str:
        return self.version.tag

    def to_dict(self) -> dict[str, object]:
        return {
            "version": str(self.version),
            "previous": str(self.previous),
            "tag": self.tag,
            "version_code": self.version.code,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------


def read_version(repo: Path = REPO_ROOT) -> Version:
    source = (Path(repo) / VERSION_FILE).read_text("utf-8")
    match = VERSION_RE.search(source)
    if not match:
        raise ValueError(f"no __version__ found in {VERSION_FILE}")
    return Version(*(int(g) for g in match.groups()))


def write_version(version: Version, repo: Path = REPO_ROOT) -> None:
    path = Path(repo) / VERSION_FILE
    source = path.read_text("utf-8")
    updated, count = VERSION_RE.subn(f'__version__ = "{version}"', source, count=1)
    if count != 1:
        raise ValueError(f"could not rewrite __version__ in {VERSION_FILE}")
    path.write_text(updated, "utf-8")


def changelog_entry(info: ReleaseInfo, when: str | None = None) -> str:
    """One release section, in the house style the changelog skill describes."""
    date = when or time.strftime("%Y-%m-%d")
    summary = (info.summary or "").strip() or "Improvements from an automated iteration."

    lines = [f"## {info.tag} — {date}", ""]
    for raw in summary.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = line.lstrip("-*").strip()
        if not line.endswith("."):
            line += "."
        lines.append(f"- {line[0].upper()}{line[1:]}")
    lines += [
        "",
        f"Composite score: {info.score_after:.1f} (was {info.score_before:.1f}). "
        "Produced by Gemma editing Morph through Morph's own agent, accepted only "
        "after the conformance suite passed and the score held.",
        "",
    ]
    return "\n".join(lines)


def update_changelog(info: ReleaseInfo, repo: Path = REPO_ROOT) -> Path:
    path = Path(repo) / CHANGELOG
    header = "# Changelog\n\nEvery version here was produced by the self-improvement loop.\n"
    body = path.read_text("utf-8") if path.is_file() else header + "\n"

    entry = changelog_entry(info)
    if "# Changelog" in body:
        head, _, rest = body.partition("\n\n")
        # Newest release first, under the file's own heading.
        path.write_text(f"{head}\n\n{entry}\n{rest.lstrip()}", "utf-8")
    else:  # pragma: no cover - only when the file was hand-mangled
        path.write_text(f"{header}\n{entry}\n{body}", "utf-8")
    return path


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def cut_release(
    summary: str,
    score_before: float,
    score_after: float,
    repo: Path = REPO_ROOT,
    part: str = "patch",
    commit: bool = True,
) -> ReleaseInfo:
    """Bump the version, record it in the changelog, and commit.

    The commit subject is ``release: vX.Y.Z`` — that string is the contract
    :mod:`selfimprove.publish` uses to find which commits to tag, after any
    rebase has settled their final SHAs.
    """
    repo = Path(repo)
    previous = read_version(repo)
    version = previous.bump(part)
    info = ReleaseInfo(
        previous=previous,
        version=version,
        summary=summary,
        score_before=score_before,
        score_after=score_after,
    )

    write_version(version, repo)
    update_changelog(info, repo)

    if commit:
        _git(repo, "add", "--", str(VERSION_FILE), str(CHANGELOG))
        _git(
            repo,
            "commit",
            "-m",
            f"release: {info.tag} (score {score_before:.1f} -> {score_after:.1f})",
        )
    return info


def release_commits(repo: Path, rev_range: str) -> list[tuple[str, str]]:
    """``(sha, tag)`` for every release commit in ``rev_range``, oldest first."""
    output = _git(repo, "log", "--reverse", "--format=%H%x1f%s", rev_range, check=False)
    found: list[tuple[str, str]] = []
    for line in output.splitlines():
        if "\x1f" not in line:
            continue
        sha, _, subject = line.partition("\x1f")
        match = RELEASE_SUBJECT_RE.match(subject)
        if match:
            found.append((sha, f"v{match.group(1)}"))
    return found
