"""Claude-compatible skills (R-301 … R-304).

A skill is a directory containing ``SKILL.md`` with YAML frontmatter::

    ---
    name: pdf-extract
    description: Extract text and tables from PDF files.
    allowed-tools: read_file, shell
    ---

    # Instructions the model reads *only when the skill is invoked*.

Only ``name`` and ``description`` enter the system prompt. The body loads on
invocation, so context cost scales with use rather than with the size of the
installed skill library (R-303).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools import ToolError, ToolRegistry

log = logging.getLogger("morph.skills")

SKILL_FILE = "SKILL.md"
FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n(?P<body>.*)\Z", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    allowed_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _body: str | None = field(default=None, repr=False)

    @property
    def body(self) -> str:
        """Lazily read the instruction body (R-303)."""
        if self._body is None:
            try:
                text = (self.path / SKILL_FILE).read_text("utf-8")
            except OSError as exc:  # pragma: no cover - file vanished mid-session
                raise ToolError(f"Cannot read skill {self.name!r}: {exc}") from exc
            match = FRONTMATTER_RE.match(text)
            self._body = (match.group("body") if match else text).strip()
        return self._body

    @property
    def loaded(self) -> bool:
        return self._body is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "allowed_tools": self.allowed_tools,
        }


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter, falling back to a tiny scalar parser without PyYAML."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text.strip()

    raw, body = match.group("yaml"), match.group("body").strip()
    try:
        import yaml

        data = yaml.safe_load(raw) or {}
        if isinstance(data, dict):
            return data, body
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 - malformed YAML is a skill defect, not fatal
        log.warning("Malformed skill frontmatter: %s", exc)
        return {}, body

    data = {}
    for line in raw.splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition(":")
            data[key.strip()] = value.strip().strip("'\"")
    return data, body


def _coerce_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    return []


def load_skill(directory: Path) -> Skill | None:
    """Load one skill directory. Returns ``None`` if malformed (R-304)."""
    skill_file = directory / SKILL_FILE
    if not skill_file.is_file():
        return None
    try:
        text = skill_file.read_text("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("Skipping skill at %s: %s", directory, exc)
        return None

    meta, _body = parse_frontmatter(text)
    name = str(meta.get("name") or directory.name).strip()
    description = str(meta.get("description") or "").strip()
    if not name or not description:
        log.warning(
            "Skipping skill at %s: SKILL.md needs both 'name' and 'description' in frontmatter",
            directory,
        )
        return None

    return Skill(
        name=name,
        description=description,
        path=directory,
        allowed_tools=_coerce_list(meta.get("allowed-tools") or meta.get("allowed_tools")),
        metadata={k: v for k, v in meta.items() if k not in {"name", "description"}},
    )


class SkillRegistry:
    """Discovers skills across a search path and exposes them to the model (R-302)."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def discover(self, search_paths: list[Path]) -> list[Skill]:
        found: list[Skill] = []
        for base in search_paths:
            base = Path(base)
            if not base.is_dir():
                continue
            for child in sorted(base.iterdir()):
                if not child.is_dir():
                    continue
                skill = load_skill(child)
                if skill is not None:
                    self._skills[skill.name] = skill
                    found.append(skill)
        return found

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        return [self._skills[n] for n in sorted(self._skills)]

    def names(self) -> list[str]:
        return sorted(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    # ------------------------------------------------------------------
    def prompt_section(self) -> str:
        """The *only* skill text that goes into every system prompt (R-303)."""
        if not self._skills:
            return ""
        lines = [
            "## Skills",
            "",
            "Specialised instruction sets. Call `load_skill` with a name to read the full",
            "instructions before doing that kind of work.",
            "",
        ]
        for skill in self.all():
            lines.append(f"- **{skill.name}** — {skill.description}")
        return "\n".join(lines)

    def register_tool(self, registry: ToolRegistry) -> None:
        """Expose skill invocation as a tool."""

        def load_skill_tool(name: str) -> str:
            skill = self.get(name)
            if skill is None:
                raise ToolError(
                    f"No skill named {name!r}. Available: {', '.join(self.names()) or '(none)'}"
                )
            header = f"# Skill: {skill.name}\n\n{skill.description}\n"
            if skill.allowed_tools:
                header += f"\nAllowed tools: {', '.join(skill.allowed_tools)}\n"
            return f"{header}\n---\n\n{skill.body}"

        registry.register(
            "load_skill",
            "Load the full instructions for a named skill before performing that task.",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            load_skill_tool,
            source="skill",
        )
