"""Shell tool: timeout-bounded execution inside the workspace (R-204, R-205)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from . import ToolError, ToolRegistry
from .files import resolve_in_root

MAX_OUTPUT = 60_000

# Commands that are never worth running from an agent loop: they either destroy
# the workspace or hang forever waiting on a human.
BLOCKED_PATTERNS = (
    "rm -rf /",
    "mkfs",
    ":(){:|:&};:",
    "shutdown",
    "reboot",
)


def register_shell_tools(registry: ToolRegistry, config: Any) -> None:
    root = Path(config.root)
    default_timeout = float(getattr(config, "shell_timeout", 120.0))

    async def run_shell(
        command: str, cwd: str = ".", timeout: float | None = None
    ) -> tuple[str, dict[str, Any]]:
        lowered = command.lower()
        for blocked in BLOCKED_PATTERNS:
            if blocked in lowered:
                raise ToolError(f"Refused: command matches blocked pattern {blocked!r}")

        workdir = resolve_in_root(root, cwd)
        if not workdir.is_dir():
            raise ToolError(f"cwd is not a directory: {cwd}")

        limit = float(timeout or default_timeout)
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(workdir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise ToolError(f"Could not start shell: {exc}") from exc

        try:
            stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=limit)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise ToolError(f"Command timed out after {limit:g}s: {command}") from None

        stdout = stdout_b.decode("utf-8", "replace")
        stderr = stderr_b.decode("utf-8", "replace")
        code = process.returncode or 0

        body = stdout
        if stderr.strip():
            body = f"{body}\n[stderr]\n{stderr}" if body.strip() else f"[stderr]\n{stderr}"
        if len(body) > MAX_OUTPUT:
            body = body[:MAX_OUTPUT] + f"\n... (truncated, {len(body)} bytes total)"
        if code != 0:
            body = f"[exit {code}]\n{body}"

        return body.strip() or f"[exit {code}] (no output)", {
            "exit_code": code,
            "timed_out": False,
        }

    registry.register(
        "shell",
        (
            "Run a shell command inside the workspace. Returns combined stdout/stderr "
            "prefixed with the exit code when it is non-zero."
        ),
        {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string", "description": "Workspace-relative working directory"},
                "timeout": {"type": "number", "description": "Seconds before the command is killed"},
            },
            "required": ["command"],
        },
        run_shell,
    )
