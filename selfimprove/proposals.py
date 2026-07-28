"""Constrained patch proposals for the Gemma self-improvement loop.

Gemma is good enough to identify useful mechanisms, but the real run archive
shows that it is not reliable at carrying a long text-emulated tool protocol
through to a valid edit. This module reduces the mutation operation to one
typed action:

    quote exact source -> propose replacement -> validate -> apply

The model proposes. Deterministic code owns path policy, anchor validity,
fixture separation, syntax, and mutation. Public research is distilled into
small provenance-carrying strategy cards; raw webpages never enter the
privileged editing context.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from morph.tools import ToolRegistry

from .guard import is_protected
from .memory import enrich_entry, render_experience_memory

STRATEGY_PATH = Path(__file__).with_name("strategies.json")
MAX_CONTEXT_CHARS = 20_000
MAX_FILE_CONTEXT_CHARS = 7_000
MAX_REPLACEMENT_CHARS = 10_000
MAX_CHANGED_LINES = 180

PROPOSAL_SYSTEM_PROMPT = """\
You are Morph's patch-proposal stage. You do not edit files and you do not call
general tools. You have exactly one action, `propose_patch`.

Propose one small causal change to Morph's real implementation. `old_string`
must be copied byte-for-byte from one supplied editable source excerpt and must
identify exactly one location. Never invent a file, source line, benchmark
fixture, or completed action. If the evidence is insufficient, do not guess:
finish with a short explanation and make no proposal.

The benchmark fixture is evidence about agent behavior, not Morph source. Never
implement its requested function inside Morph. The controller independently
checks every field and applies the patch only if it is structurally valid.
"""


@dataclass(frozen=True)
class PatchProposal:
    hypothesis: str
    strategy_id: str
    path: str
    old_string: str
    new_string: str
    expected_effect: str
    test: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "PatchProposal":
        return cls(
            hypothesis=str(value.get("hypothesis") or "").strip(),
            strategy_id=str(value.get("strategy_id") or "").strip(),
            path=str(value.get("path") or "").strip(),
            old_string=str(value.get("old_string") or ""),
            new_string=str(value.get("new_string") or ""),
            expected_effect=str(value.get("expected_effect") or "").strip(),
            test=str(value.get("test") or "").strip(),
        )

    def public_record(self) -> dict[str, Any]:
        """Compact history record; the audit log does not need full source text."""
        return {
            "hypothesis": self.hypothesis[:600],
            "strategy_id": self.strategy_id,
            "path": self.path,
            "expected_effect": self.expected_effect[:600],
            "test": self.test[:300],
            "old_chars": len(self.old_string),
            "new_chars": len(self.new_string),
        }


@dataclass
class ProposalReview:
    proposal: PatchProposal | None
    valid: bool
    errors: list[str] = field(default_factory=list)
    score: float = 0.0
    candidate: int = 0
    assigned_strategy: str = ""

    def record(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "assigned_strategy": self.assigned_strategy,
            "valid": self.valid,
            "score": round(self.score, 2),
            "errors": self.errors[:6],
            "proposal": self.proposal.public_record() if self.proposal else None,
        }


@dataclass(frozen=True)
class SourceContext:
    text: str
    paths: tuple[str, ...]


def load_strategy_cards(path: Path = STRATEGY_PATH) -> list[dict[str, Any]]:
    """Load curated, provenance-carrying ideas; malformed cards are ignored."""
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    required = {"id", "title", "failure_kinds", "mechanism", "experiment", "sources"}
    return [
        card
        for card in payload
        if isinstance(card, dict) and required.issubset(card)
    ]


def select_strategy_cards(
    history: list[dict[str, Any]],
    target: dict[str, Any] | str | None,
    *,
    limit: int = 3,
    cards: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve strategies matching all target history, not just its last row."""
    available = list(cards if cards is not None else load_strategy_cards())
    target_name = (
        str(target.get("name") or "") if isinstance(target, dict) else str(target or "")
    )
    failure_counts: dict[str, int] = {}
    for raw in history:
        experience = enrich_entry(raw)["experience"]
        if target_name and experience.get("target") != target_name:
            continue
        if experience.get("outcome") == "accepted":
            continue
        kind = str(experience.get("failure_kind") or "other_rejection")
        failure_counts[kind] = failure_counts.get(kind, 0) + 1

    declared_order = {
        str(card.get("id") or ""): index for index, card in enumerate(available)
    }

    def relevance(card: dict[str, Any]) -> tuple[int, int, int]:
        kinds = {str(kind) for kind in card.get("failure_kinds") or []}
        weighted = sum(failure_counts.get(kind, 0) for kind in kinds)
        return (
            -weighted,
            -len(kinds & failure_counts.keys()),
            declared_order[str(card.get("id") or "")],
        )

    ranked = sorted(available, key=relevance)
    if not ranked:
        return []
    # With no target-specific failure yet, retain the most generally useful
    # mutation primitives in the curated file's declared order.
    if not failure_counts:
        return available[:limit]
    return ranked[:limit]


def make_proposal_registry(captured: list[PatchProposal]) -> ToolRegistry:
    """Expose one narrow action to Gemma instead of a general editing surface."""
    registry = ToolRegistry()

    def propose_patch(
        hypothesis: str,
        strategy_id: str,
        path: str,
        old_string: str,
        new_string: str,
        expected_effect: str,
        test: str,
    ) -> str:
        captured.append(
            PatchProposal(
                hypothesis=hypothesis.strip(),
                strategy_id=strategy_id.strip(),
                path=path.strip(),
                old_string=old_string,
                new_string=new_string,
                expected_effect=expected_effect.strip(),
                test=test.strip(),
            )
        )
        return "Patch proposal captured for deterministic validation. Stop now."

    registry.register(
        "propose_patch",
        (
            "Submit one exact source replacement for deterministic validation. "
            "This records a proposal; it does not edit the file."
        ),
        {
            "type": "object",
            "properties": {
                "hypothesis": {"type": "string"},
                "strategy_id": {"type": "string"},
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "expected_effect": {"type": "string"},
                "test": {"type": "string"},
            },
            "required": [
                "hypothesis",
                "strategy_id",
                "path",
                "old_string",
                "new_string",
                "expected_effect",
                "test",
            ],
        },
        propose_patch,
    )
    return registry


def parse_patch_proposal(text: str) -> PatchProposal | None:
    """Salvage a plain or fenced JSON proposal when Gemma skips the action."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            value, _end = decoder.raw_decode((text or "")[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("proposal"), dict):
            value = value["proposal"]
        if not isinstance(value, dict):
            continue
        required = {
            "hypothesis",
            "strategy_id",
            "path",
            "old_string",
            "new_string",
            "expected_effect",
            "test",
        }
        if required.issubset(value):
            return PatchProposal.from_mapping(value)
    return None


def build_source_context(
    root: Path,
    diagnosis: str,
    target: dict[str, Any] | None,
) -> SourceContext:
    """Supply exact bounded source, chosen from diagnosis files and likely layers."""
    root = Path(root)
    category = str((target or {}).get("category") or "")
    defaults = {
        "coding": ["morph/agent.py", "morph/tools/files.py", "morph/llm/base.py"],
        "tool_use": ["morph/agent.py", "morph/tools/__init__.py", "morph/tools/files.py"],
        "mcp": ["morph/mcp.py", "morph/agent.py", "morph/config.py"],
        "skills": ["morph/skills.py", "morph/agent.py", "morph/tools/__init__.py"],
    }.get(category, ["morph/agent.py", "morph/tools/files.py", "morph/llm/base.py"])

    mentioned = re.findall(r"(?<![\w/])(morph/[A-Za-z0-9_./-]+\.py)", diagnosis or "")
    candidates: list[str] = []
    for path in [*mentioned, *defaults]:
        normalised = str(PurePosixPath(path))
        if normalised not in candidates:
            candidates.append(normalised)

    paths = [
        path
        for path in candidates
        if path.startswith("morph/")
        and not is_protected(path)
        and (root / path).is_file()
    ][:3]
    keywords = _context_keywords(diagnosis, target)
    sections: list[str] = []
    kept_paths: list[str] = []
    remaining = MAX_CONTEXT_CHARS
    for path in paths:
        content = (root / path).read_text("utf-8")
        excerpt = _relevant_excerpt(content, keywords, MAX_FILE_CONTEXT_CHARS)
        section = (
            f"## Editable source: `{path}`\n\n"
            "Copy `old_string` exactly from this raw excerpt. Line-number labels "
            "are deliberately absent.\n\n"
            f"```python\n{excerpt.rstrip()}\n```"
        )
        if len(section) > remaining:
            continue
        sections.append(section)
        kept_paths.append(path)
        remaining -= len(section)
    return SourceContext(text="\n\n".join(sections), paths=tuple(kept_paths))


def build_proposal_prompt(
    target: dict[str, Any] | None,
    diagnosis: str,
    history: list[dict[str, Any]],
    strategies: list[dict[str, Any]],
    source: SourceContext,
    *,
    assigned_strategy: str,
    candidate: int,
    focus: str | None = None,
) -> str:
    """One clean-context mutation request with an assigned, sourced prior."""
    target_name = str((target or {}).get("name") or "no measured target")
    score = float((target or {}).get("score") or 0)
    rendered_cards: list[str] = []
    for card in strategies:
        sources = "; ".join(
            f"{source_item.get('name')}: {source_item.get('url')}"
            for source_item in card.get("sources") or []
        )
        rendered_cards.append(
            f"### `{card['id']}` — {card['title']}\n"
            f"- mechanism: {card['mechanism']}\n"
            f"- experiment: {card['experiment']}\n"
            f"- risks: {card.get('risks', '')}\n"
            f"- provenance: {sources}"
        )

    sections = [
        "# Produce one executable patch proposal",
        (
            f"Candidate {candidate}; measured target `{target_name}` currently scores "
            f"{score:.0%}. Prefer strategy `{assigned_strategy}`. You may use another "
            "listed strategy only when the supplied source makes the assigned one "
            "inapplicable."
        ),
        "# Diagnosis hypothesis\n\n" + (diagnosis.strip()[:3500] or "(none)"),
        render_experience_memory(history, target),
        "# Curated strategy cards\n\n" + "\n\n".join(rendered_cards),
        source.text or "# Editable source\n\nNo valid editable source was resolved.",
        (
            "# Meta-planning gate\n\n"
            "A proposal is rejected before benchmarking unless: its path is one of "
            f"{list(source.paths)}; `old_string` occurs exactly once; the replacement "
            "parses as Python; it changes at most 180 lines; it does not implement "
            "the temporary benchmark fixture; and its strategy is one of "
            f"{[card['id'] for card in strategies]}.\n\n"
            "Call `propose_patch` exactly once. Do not claim that the patch has "
            "already been applied or tested."
        ),
    ]
    if focus:
        sections.insert(2, f"# Human focus\n\n{focus}")
    return "\n\n---\n\n".join(sections)


def review_proposal(
    root: Path,
    proposal: PatchProposal | None,
    *,
    allowed_paths: tuple[str, ...],
    strategy_ids: set[str],
    target: dict[str, Any] | None,
    candidate: int = 0,
    assigned_strategy: str = "",
) -> ProposalReview:
    """Cheap meta-evaluation: reject impossible plans before model benchmarking."""
    errors: list[str] = []
    if proposal is None:
        return ProposalReview(
            None,
            False,
            ["no parseable propose_patch action or JSON object"],
            candidate=candidate,
            assigned_strategy=assigned_strategy,
        )

    path = str(PurePosixPath(proposal.path))
    if proposal.path.startswith("/") or ".." in PurePosixPath(proposal.path).parts:
        errors.append("path escapes the repository")
    if path not in allowed_paths:
        errors.append(f"path {path!r} was not supplied as editable source")
    if not path.startswith("morph/") or is_protected(path):
        errors.append("only unprotected implementation files under morph/ are eligible")
    if proposal.strategy_id not in strategy_ids:
        errors.append(f"unknown or unretrieved strategy {proposal.strategy_id!r}")
    if not proposal.hypothesis:
        errors.append("hypothesis is empty")
    if not proposal.expected_effect:
        errors.append("expected_effect is empty")
    if not proposal.old_string.strip():
        errors.append("old_string is empty")
    if proposal.old_string == proposal.new_string:
        errors.append("old_string and new_string are identical")
    if len(proposal.new_string) > MAX_REPLACEMENT_CHARS:
        errors.append("replacement is too large for one causal mutation")
    changed_lines = max(
        len(proposal.old_string.splitlines()), len(proposal.new_string.splitlines())
    )
    if changed_lines > MAX_CHANGED_LINES:
        errors.append(f"proposal changes {changed_lines} lines; limit is {MAX_CHANGED_LINES}")

    target_path = Path(root) / path
    original = ""
    if target_path.is_file():
        original = target_path.read_text("utf-8")
        occurrences = original.count(proposal.old_string)
        if occurrences != 1:
            errors.append(
                f"old_string occurs {occurrences} times in {path}; exactly one is required"
            )
    elif path in allowed_paths:
        errors.append(f"source file {path!r} no longer exists")

    fixture_hits = _fixture_definition_hits(proposal.new_string, target)
    if fixture_hits:
        errors.append(
            "replacement defines benchmark fixture symbol(s): " + ", ".join(fixture_hits)
        )

    if original and proposal.old_string in original:
        updated = original.replace(proposal.old_string, proposal.new_string, 1)
        if path.endswith(".py"):
            try:
                ast.parse(updated, filename=path)
            except SyntaxError as exc:
                errors.append(f"replacement does not parse as Python: {exc.msg}")

    score = 100.0
    score -= min(changed_lines, MAX_CHANGED_LINES) * 0.08
    score += 5.0 if proposal.strategy_id == assigned_strategy else 0.0
    score += 2.0 if "pytest" in proposal.test.lower() else 0.0
    score += min(len(proposal.hypothesis.split()), 30) * 0.05
    return ProposalReview(
        proposal=proposal,
        valid=not errors,
        errors=errors,
        score=score if not errors else 0.0,
        candidate=candidate,
        assigned_strategy=assigned_strategy,
    )


def apply_proposal(root: Path, proposal: PatchProposal) -> None:
    """Apply a proposal already accepted by :func:`review_proposal`."""
    path = Path(root) / str(PurePosixPath(proposal.path))
    original = path.read_text("utf-8")
    occurrences = original.count(proposal.old_string)
    if occurrences != 1:
        raise ValueError(
            f"proposal anchor changed before application: found {occurrences} occurrences"
        )
    path.write_text(
        original.replace(proposal.old_string, proposal.new_string, 1),
        "utf-8",
    )


def _context_keywords(
    diagnosis: str, target: dict[str, Any] | None
) -> set[str]:
    text = " ".join(
        [
            diagnosis or "",
            str((target or {}).get("name") or ""),
            str((target or {}).get("detail") or ""),
        ]
    )
    words = {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
        if token.lower()
        not in {
            "the",
            "and",
            "for",
            "that",
            "this",
            "with",
            "from",
            "agent",
            "file",
            "change",
            "none",
            "code",
            "morph",
            "test",
        }
    }
    return words


def _relevant_excerpt(content: str, keywords: set[str], limit: int) -> str:
    if len(content) <= limit:
        return content
    lines = content.splitlines(keepends=True)
    scored: list[tuple[int, int]] = []
    lowered_keywords = {word.lower() for word in keywords}
    for index, line in enumerate(lines):
        tokens = {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line)}
        score = len(tokens & lowered_keywords)
        if score:
            scored.append((score, index))
    centers = [index for _score, index in sorted(scored, reverse=True)]
    if not centers:
        centers = [
            next(
                (i for i, line in enumerate(lines) if "class Agent" in line),
                len(lines) // 2,
            )
        ]

    ranges: list[tuple[int, int]] = []
    for center in centers:
        start, end = max(0, center - 28), min(len(lines), center + 45)
        if any(start < old_end + 12 and end > old_start - 12 for old_start, old_end in ranges):
            continue
        ranges.append((start, end))
        if len(ranges) >= 3:
            break
    ranges.sort()

    chunks: list[str] = []
    used = 0
    for start, end in ranges:
        chunk = "".join(lines[start:end])
        if used + len(chunk) > limit:
            chunk = chunk[: max(0, limit - used)]
        if not chunk:
            break
        chunks.append(chunk)
        used += len(chunk)
    return "\n# ... unrelated source omitted ...\n".join(chunks)


def _fixture_definition_hits(
    replacement: str, target: dict[str, Any] | None
) -> list[str]:
    target_name = str((target or {}).get("name") or "")
    if not target_name:
        return []
    try:
        from bench.tasks import ALL_TASKS

        task = next(item for item in ALL_TASKS if item.label == target_name)
    except (ImportError, StopIteration):
        return []

    symbols: set[str] = set()
    for content in task.files.values():
        symbols.update(
            match.group(2)
            for match in re.finditer(
                r"(?m)^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", content
            )
        )
    return sorted(
        symbol
        for symbol in symbols
        if re.search(rf"(?m)^\s*(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b", replacement)
    )
