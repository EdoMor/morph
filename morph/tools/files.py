"""Filesystem tools with workspace confinement (R-202, R-203, R-205)."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from . import PathEscapeError, ToolError, ToolRegistry

MAX_READ_BYTES = 400_000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}


def resolve_in_root(root: Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` and prove it stays inside ``root`` (R-205).

    The check happens **before** any I/O: symlinks are resolved first, so a
    symlink pointing out of the workspace is rejected rather than followed.
    """
    root = Path(root).resolve()
    path = Path(candidate)
    if not path.is_absolute():
        path = root / path

    resolved = Path(os.path.normpath(str(path)))
    try:
        real = resolved.resolve()
    except OSError as exc:  # pragma: no cover - exotic filesystem states
        raise ToolError(f"Cannot resolve path {candidate!r}: {exc}") from exc

    if real != root and root not in real.parents:
        raise PathEscapeError(
            f"Path {candidate!r} resolves outside the workspace root ({root}). Refused."
        )
    return real


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover - guarded by resolve_in_root
        return str(path)


def register_file_tools(registry: ToolRegistry, config: Any) -> None:
    root = Path(config.root)

    # -- read -----------------------------------------------------------
    def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
        target = resolve_in_root(root, path)
        if not target.exists():
            raise ToolError(f"File not found: {path}")
        if target.is_dir():
            raise ToolError(f"{path} is a directory — use list_dir")
        if target.stat().st_size > MAX_READ_BYTES:
            raise ToolError(
                f"{path} is {target.stat().st_size} bytes, over the {MAX_READ_BYTES} limit. "
                "Use grep or read it in slices with offset/limit."
            )
        try:
            lines = target.read_text("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ToolError(f"{path} is not UTF-8 text: {exc}") from exc
        window = lines[offset : offset + limit]
        width = len(str(offset + len(window)))
        return "\n".join(f"{offset + i + 1:>{width}}\t{line}" for i, line in enumerate(window))

    registry.register(
        "read_file",
        "Read a UTF-8 text file from the workspace. Returns line-numbered content.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path"},
                "offset": {"type": "integer", "description": "First line (0-indexed)"},
                "limit": {"type": "integer", "description": "Max lines to return"},
            },
            "required": ["path"],
        },
        read_file,
    )

    # -- write ----------------------------------------------------------
    def write_file(path: str, content: str) -> str:
        target = resolve_in_root(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(content, "utf-8")
        verb = "Overwrote" if existed else "Created"
        return f"{verb} {_rel(root, target)} ({len(content)} bytes)"

    registry.register(
        "write_file",
        "Write a file, creating parent directories. Overwrites existing content.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        write_file,
    )

    # -- edit -----------------------------------------------------------
    def edit_file(
        path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        target = resolve_in_root(root, path)
        if not target.is_file():
            raise ToolError(f"File not found: {path}")
        original = target.read_text("utf-8")

        occurrences = original.count(old_string)
        # Fail loudly rather than silently no-op (R-203).
        if occurrences == 0:
            raise ToolError(
                f"old_string not found in {path}. The file was not modified. "
                "Read the file and match the exact text, including indentation."
            )
        if occurrences > 1 and not replace_all:
            raise ToolError(
                f"old_string appears {occurrences} times in {path} — ambiguous. "
                "Include more surrounding context, or pass replace_all=true."
            )
        if old_string == new_string:
            raise ToolError("old_string and new_string are identical; nothing to do.")

        updated = original.replace(old_string, new_string) if replace_all else original.replace(
            old_string, new_string, 1
        )
        target.write_text(updated, "utf-8")
        count = occurrences if replace_all else 1
        return f"Edited {_rel(root, target)} ({count} replacement{'s' if count != 1 else ''})"

    registry.register(
        "edit_file",
        (
            "Replace an exact string in a file. Fails if the string is absent, or if it "
            "appears more than once and replace_all is false."
        ),
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"],
        },
        edit_file,
    )

    # -- list -----------------------------------------------------------
    def list_dir(path: str = ".") -> str:
        target = resolve_in_root(root, path)
        if not target.is_dir():
            raise ToolError(f"Not a directory: {path}")
        entries = []
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if child.name in SKIP_DIRS:
                continue
            entries.append(f"{child.name}/" if child.is_dir() else child.name)
        return "\n".join(entries) or "(empty)"

    registry.register(
        "list_dir",
        "List the entries of a workspace directory.",
        {"type": "object", "properties": {"path": {"type": "string"}}},
        list_dir,
    )

    # -- glob -----------------------------------------------------------
    def glob_files(pattern: str, path: str = ".") -> str:
        base = resolve_in_root(root, path)
        matches = [
            p
            for p in base.rglob("*")
            if p.is_file()
            and not SKIP_DIRS.intersection(p.parts)
            and fnmatch.fnmatch(str(p.relative_to(base)), pattern)
        ]
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            return f"No files match {pattern!r} under {path}"
        return "\n".join(_rel(root, p) for p in matches[:500])

    registry.register(
        "glob",
        "Find files by glob pattern (e.g. '**/*.py'), newest first.",
        {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
        glob_files,
    )

    # -- grep -----------------------------------------------------------
    def grep(
        pattern: str, path: str = ".", glob: str = "*", max_results: int = 100
    ) -> str:
        base = resolve_in_root(root, path)
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"Invalid regex {pattern!r}: {exc}") from exc

        hits: list[str] = []
        files = [base] if base.is_file() else sorted(base.rglob("*"))
        for file in files:
            if not file.is_file() or SKIP_DIRS.intersection(file.parts):
                continue
            if not fnmatch.fnmatch(file.name, glob):
                continue
            try:
                text = file.read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{_rel(root, file)}:{lineno}: {line.strip()[:300]}")
                    if len(hits) >= max_results:
                        return "\n".join(hits) + f"\n(truncated at {max_results})"
        return "\n".join(hits) or f"No matches for {pattern!r}"

    registry.register(
        "grep",
        "Search file contents with a regular expression. Returns path:line: match.",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string", "description": "Filename filter, e.g. '*.py'"},
                "max_results": {"type": "integer"},
            },
            "required": ["pattern"],
        },
        grep,
    )
