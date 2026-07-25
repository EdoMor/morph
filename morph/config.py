"""Configuration for Morph.

Everything is overridable by environment variable or by ``morph.json`` in the
workspace root. Secrets are read from the environment only and are never
persisted (R-803).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "morph.json"

# The default model is a Gemma model: Gemma writes Morph, so Morph must run on
# Gemma (R-104).
DEFAULT_MODEL = "gemma3:12b"
DEFAULT_PROVIDER = "ollama"


@dataclass
class MCPServerConfig:
    """One MCP server Morph should connect to as a client (R-401)."""

    name: str
    transport: str = "stdio"  # "stdio" | "http"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    timeout: float = 30.0
    enabled: bool = True


@dataclass
class Config:
    workspace: Path = field(default_factory=lambda: Path.cwd())
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    base_url: str | None = None
    max_steps: int = 24  # R-102
    temperature: float = 0.2
    context_tokens: int = 32_768

    skill_paths: list[Path] = field(default_factory=list)
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)

    image_backend: str = "stub"
    image_output_dir: str = ".morph/images"

    sessions_dir: str = ".morph/sessions"
    shell_timeout: float = 120.0
    allow_shell: bool = True

    host: str = "127.0.0.1"
    port: int = 8787

    # ------------------------------------------------------------------
    @property
    def root(self) -> Path:
        """Absolute, resolved workspace root. All I/O is confined here (R-205)."""
        return self.workspace.resolve()

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["workspace"] = str(self.workspace)
        data["skill_paths"] = [str(p) for p in self.skill_paths]
        return data


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_config(workspace: str | Path | None = None) -> Config:
    """Build a :class:`Config` from defaults, ``morph.json`` and the environment.

    Precedence: environment > ``morph.json`` > defaults.
    """
    root = Path(workspace or os.environ.get("MORPH_WORKSPACE") or Path.cwd())
    cfg = Config(workspace=root)

    file_data: dict[str, Any] = {}
    config_file = root / CONFIG_FILENAME
    if config_file.is_file():
        try:
            file_data = json.loads(config_file.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            file_data = {}

    for key in (
        "provider",
        "model",
        "base_url",
        "temperature",
        "image_backend",
        "image_output_dir",
        "sessions_dir",
        "host",
    ):
        if key in file_data:
            setattr(cfg, key, file_data[key])
    for key in ("max_steps", "context_tokens", "port"):
        if key in file_data:
            setattr(cfg, key, int(file_data[key]))
    if "allow_shell" in file_data:
        cfg.allow_shell = bool(file_data["allow_shell"])

    cfg.skill_paths = [root / p for p in file_data.get("skill_paths", ["skills", ".morph/skills"])]
    cfg.mcp_servers = [
        MCPServerConfig(name=name, **spec)
        for name, spec in (file_data.get("mcp_servers") or {}).items()
    ]

    # Environment overrides win.
    cfg.provider = os.environ.get("MORPH_PROVIDER", cfg.provider)
    cfg.model = os.environ.get("MORPH_MODEL", cfg.model)
    cfg.base_url = os.environ.get("MORPH_BASE_URL", cfg.base_url)
    cfg.image_backend = os.environ.get("MORPH_IMAGE_BACKEND", cfg.image_backend)
    cfg.host = os.environ.get("MORPH_HOST", cfg.host)
    cfg.allow_shell = _env_bool("MORPH_ALLOW_SHELL", cfg.allow_shell)
    if os.environ.get("MORPH_MAX_STEPS"):
        cfg.max_steps = int(os.environ["MORPH_MAX_STEPS"])
    if os.environ.get("MORPH_PORT"):
        cfg.port = int(os.environ["MORPH_PORT"])

    extra_skills = os.environ.get("MORPH_SKILL_PATH", "")
    for chunk in extra_skills.split(os.pathsep):
        if chunk.strip():
            cfg.skill_paths.append(Path(chunk.strip()))

    return cfg
