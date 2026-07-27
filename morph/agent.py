"""The agent loop (R-101 … R-109).

    while steps < max_steps:
        response = provider.complete(conversation, tools)
        if not response.tool_calls:
            return response.text
        for call in response.tool_calls:
            conversation.append(registry.call(call))

Everything else in Morph — skills, MCP, images, the server, the self-improvement
loop — is a supplier of tools or context to this loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .llm import ModelResponse, Provider, ProviderError, ToolCall, get_provider
from .mcp import MCPManager
from .session import Session, SessionStore
from .skills import SkillRegistry
from .tools import ToolRegistry, ToolResult, build_default_registry

log = logging.getLogger("morph.agent")

SYSTEM_PROMPT = """\
You are Morph, a coding agent running on the user's own hardware.

You have tools for reading and writing files, running shell commands, searching
the web, and generating images. Use them; do not guess at file contents or
command output.

Working rules:
- Read before you edit. `edit_file` fails if the text you target is not present.
- Prefer small, verifiable steps. Run the tests after you change code.
- When a tool returns an error, read it and adapt — errors are information, not
  a reason to stop.
- Answer in plain prose when you are done. Do not narrate every tool call.
"""


@dataclass
class AgentEvent:
    """One item in the run's event stream (R-107)."""

    type: str  # text | tool_use | tool_result | error | done
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "ts": self.ts, **self.data}


@dataclass
class RunResult:
    text: str = ""
    stop_reason: str = "end_turn"
    steps: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0
    session_id: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.stop_reason not in {"error"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "duration_ms": round(self.duration_ms, 2),
            "session_id": self.session_id,
            "error": self.error,
        }


class Agent:
    """Wires a provider, a tool registry, skills, MCP and sessions together."""

    def __init__(
        self,
        config: Config | None = None,
        provider: Provider | None = None,
        tools: ToolRegistry | None = None,
        skills: SkillRegistry | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        require_edit: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.provider = provider or get_provider(
            self.config.provider,
            model=self.config.model,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
        )
        self.tools = tools if tools is not None else build_default_registry(self.config)
        self.skills = skills if skills is not None else SkillRegistry()
        self.sessions = SessionStore(self.config.path(self.config.sessions_dir))
        self.mcp = MCPManager(self.tools)
        self.base_system_prompt = system_prompt
        # Ordinary Morph sessions include read-only questions, so prose is
        # usually a valid completion. A self-improvement iteration exists to
        # attempt a code change. It opts into this stricter completion policy.
        self.require_edit = require_edit
        self._started = False

    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Discover skills and connect MCP servers. Safe to call twice."""
        if self._started:
            return
        self._started = True
        self.skills.discover([Path(p) for p in self.config.skill_paths])
        self.skills.register_tool(self.tools)
        if self.config.mcp_servers:
            await self.mcp.connect_all(self.config.mcp_servers)

    async def close(self) -> None:
        await self.mcp.close_all()

    async def __aenter__(self) -> "Agent":
        await self.start()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        parts = [self.base_system_prompt, f"\nWorkspace root: {self.config.root}"]
        skill_section = self.skills.prompt_section()
        if skill_section:
            parts.append("\n" + skill_section)
        return "\n".join(parts)

    def _conversation(self, session: Session) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system_prompt()}, *session.for_model()]

    # ------------------------------------------------------------------
    async def run(
        self,
        prompt: str,
        session: Session | str | None = None,
        max_steps: int | None = None,
    ) -> RunResult:
        """Run to completion and return the final result."""
        result = RunResult()
        async for event in self.stream(prompt, session=session, max_steps=max_steps):
            if event.type == "done":
                result = RunResult(**event.data["result"])
        return result

    async def stream(
        self,
        prompt: str,
        session: Session | str | None = None,
        max_steps: int | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run the loop, yielding events as they happen (R-101, R-107)."""
        await self.start()

        if isinstance(session, Session):
            active = session
        else:
            active = self.sessions.get_or_create(session)
        active.user(prompt)

        budget = max_steps if max_steps is not None else self.config.max_steps
        started = time.perf_counter()
        tool_log: list[dict[str, Any]] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        final_text = ""
        stop_reason = "end_turn"
        error: str | None = None
        steps = 0
        edits_made = 0
        act_nudges = 0

        specs = self.tools.specs()

        while steps < budget:  # bounded — the loop always terminates (R-102)
            steps += 1
            try:
                response: ModelResponse = await self.provider.complete(
                    self._conversation(active), tools=specs
                )
            except ProviderError as exc:
                error, stop_reason = str(exc), "error"
                yield AgentEvent("error", {"message": error, "recoverable": False})
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never take the process down
                error, stop_reason = f"{type(exc).__name__}: {exc}", "error"
                log.exception("Provider call failed")
                yield AgentEvent("error", {"message": error, "recoverable": False})
                break

            for key, value in (response.usage or {}).items():
                usage[key] = usage.get(key, 0) + value

            if response.text:
                final_text = response.text
                yield AgentEvent("text", {"text": response.text, "step": steps})

            if not response.tool_calls:
                # A tool_call block the parser could not read is not an answer.
                # Silently treating it as prose ends the run at this step with
                # no changes made and no explanation to the model — the single
                # most common way a small model's iteration used to die.
                if response.malformed_calls and steps < budget:
                    active.assistant(response.text)
                    active.tool(
                        "tool_call_parser",
                        _malformed_call_guidance(response.malformed_calls),
                        ok=False,
                    )
                    yield AgentEvent(
                        "error",
                        {
                            "message": "tool call could not be parsed; asking the model to retry",
                            "details": response.malformed_calls,
                            "recoverable": True,
                            "step": steps,
                        },
                    )
                    continue

                # Describing a change is not making one (R-720). Small models
                # end a turn either announcing an edit ("I will edit X to do Y")
                # or reporting one they never performed — and a text-only turn
                # reads as completion, so the run ends having changed nothing.
                # Across eight recorded iterations the mean was 3 steps of a
                # 60-step budget: not running out of room, just stopping.
                # Gated on "no editing tool has succeeded", so once real work
                # exists this can never fire.
                narration = None
                if not edits_made and act_nudges < MAX_ACT_NUDGES and steps < budget:
                    narration = _describes_work_that_did_not_happen(response.text)
                    if (
                        narration is None
                        and self.require_edit
                        and not _explicitly_declines_a_change(response.text)
                    ):
                        narration = "attempt"
                if narration:
                    act_nudges += 1
                    active.assistant(response.text)
                    active.tool(
                        "supervisor", _act_on_your_plan_guidance(narration), ok=False
                    )
                    yield AgentEvent(
                        "error",
                        {
                            "message": (
                                "the model reported an edit it never made; asking it to act"
                                if narration == "claim"
                                else (
                                    "the model described an action without taking it; asking it to act"
                                    if narration == "plan"
                                    else "the change attempt ended without an edit; asking it to act"
                                )
                            ),
                            "kind": narration,
                            "recoverable": True,
                            "step": steps,
                        },
                    )
                    continue

                active.assistant(response.text)
                stop_reason = response.stop_reason or "end_turn"
                break

            active.assistant(
                response.text,
                tool_calls=[
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in response.tool_calls
                ],
            )

            for call in response.tool_calls:
                yield AgentEvent(
                    "tool_use",
                    {"id": call.id, "name": call.name, "arguments": call.arguments, "step": steps},
                )
                result = await self._invoke(call)
                tool_log.append(result.to_dict())  # every call is recorded (R-108)
                edits_made += int(result.ok and call.name in EDITING_TOOLS)
                active.tool(call.name, result.content, call_id=call.id, ok=result.ok)
                yield AgentEvent(
                    "tool_result",
                    {
                        "id": call.id,
                        "name": call.name,
                        "ok": result.ok,
                        "content": result.content,
                        "duration_ms": round(result.duration_ms, 2),
                        "meta": result.meta,
                        "step": steps,
                    },
                )
        else:
            # Budget exhausted: end cleanly, do not hang or raise (R-102).
            stop_reason = "max_steps"
            final_text = final_text or (
                f"Stopped after {budget} steps without reaching a final answer."
            )
            active.assistant(final_text)

        outcome = RunResult(
            text=final_text,
            stop_reason=stop_reason,
            steps=steps,
            tool_calls=tool_log,
            usage=usage,
            duration_ms=(time.perf_counter() - started) * 1000,
            session_id=active.id,
            error=error,
        )
        yield AgentEvent("done", {"result": outcome.to_dict()})

    async def _invoke(self, call: ToolCall) -> ToolResult:
        """Execute one tool call. Failures become results, not exceptions (R-109)."""
        return await self.tools.call(call.name, call.arguments)


#: Tools whose success means a file actually changed. Reading, grepping and
#: listing do not clear the guard below — an iteration that read the right file
#: and then reported an edit it never made is the exact case this catches.
EDITING_TOOLS = frozenset({"edit_file", "write_file"})

#: How many times one run may be told that its description is not a result.
#: More than one because these models repeat themselves; bounded because a wrong
#: guess should cost a turn, not the iteration.
MAX_ACT_NUDGES = 3

#: Phrases announcing work not yet done. Required to co-occur with an editing
#: word, so "I should note that this returns None" in an answer to a read-only
#: question is not mistaken for an abandoned plan.
INTENT_PHRASES = (
    "i will ",
    "i'll ",
    "i am going to",
    "i'm going to",
    "let me ",
    "let's ",
    "i plan to",
    "i intend to",
    "i need to",
    "i must ",
    "we need to",
    "next, i",
    "my plan",
    "here's my plan",
    "i should ",
    "we should ",
)

#: Words that make a sentence about changing code rather than about anything
#: else. Stems, so "modify"/"modified"/"modification" all count.
EDITING_WORDS = (
    "edit",
    "write",
    "replace",
    "modif",
    "chang",
    "fix",
    "updat",
    "patch",
    "rewrit",
    "refactor",
    # not "implement": "this module implements a loop" is description, not intent
    " add ",
    "remove",
    "delete",
)

#: Phrases reporting work as already done. These need no second condition —
#: each is a first-person or summary-style assertion that an edit happened, and
#: the call site only consults them when no editing tool has succeeded, so the
#: assertion is necessarily false.
CLAIM_PHRASES = (
    "i modified",
    "i changed",
    "i updated",
    "i added",
    "i fixed",
    "i edited",
    "i replaced",
    "i removed",
    "i've modified",
    "i've changed",
    "i've updated",
    "i've added",
    "i've fixed",
    "i have modified",
    "i have changed",
    "i have updated",
    "i have added",
    "i have fixed",
    "i made the change",
    "modified the",
    "changed the",
    "updated the",
    "replaced the",
    "the change to",
    "this change",
    "after making",
    "**change:**",
    "changes made",
)


#: Models write prose, and prose uses typographic punctuation. Run #8 slipped
#: past the whole guard on "I’ve added the check" — U+2019, so none of "i've",
#: "i'll", "let's" or "here's my plan" could ever match.
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'"})

# The strict policy still needs an honest escape hatch. A model that inspected
# the target and concluded that changing code would be wrong may say so
# explicitly; analysis, an apology, or another future-tense plan is not enough.
NO_CHANGE_PHRASES = (
    "no code change is needed",
    "no change is needed",
    "no change is warranted",
    "nothing worth changing",
    "nothing to change",
    "changed nothing",
    "should not change the code",
    "should not modify the code",
)


def _describes_work_that_did_not_happen(text: str) -> str | None:
    """``"claim"``, ``"plan"``, or ``None``.

    Only meaningful when paired with "no editing tool has succeeded" at the call
    site — that is what makes a claim provably false and a plan provably
    unexecuted. Claims are checked first: a reply asserting a finished edit is
    worse than one merely proposing it, because the loop records the summary.
    """
    body = " ".join((text or "").lower().translate(APOSTROPHES).split())
    if len(body) < 40:  # "Done." and friends are answers, not narration
        return None
    if any(phrase in body for phrase in CLAIM_PHRASES):
        return "claim"
    if any(phrase in body for phrase in INTENT_PHRASES) and any(
        word in body for word in EDITING_WORDS
    ):
        return "plan"
    return None


def _explicitly_declines_a_change(text: str) -> bool:
    """Whether a strict edit attempt deliberately concludes with no change."""
    body = " ".join((text or "").lower().translate(APOSTROPHES).split())
    return any(phrase in body for phrase in NO_CHANGE_PHRASES)


def _act_on_your_plan_guidance(kind: str) -> str:
    """What the model is told when it narrates instead of acting."""
    opening = (
        "You have described a change as though it were finished, but no editing "
        "tool has run in this session: `edit_file` and `write_file` have not "
        "been called successfully even once. Reading a file does not change it, "
        "and neither does describing the change. The edit you summarised does "
        "not exist."
        if kind == "claim"
        else (
            "You described what you were going to do, but no editing tool has "
            "run in this session — `edit_file` and `write_file` have not been "
            "called successfully even once — so none of it has happened yet."
            if kind == "plan"
            else "This is a code-change attempt, but your reply ended without a "
            "successful `edit_file` or `write_file` call. Analysis and apologies "
            "are not a result; either act on the analysis or explicitly conclude "
            "that no code change is warranted."
        )
    )
    return (
        f"{opening}\n\n"
        "Do not explain this message. Your next reply should be a tool call. "
        "For this provider the exact syntax is:\n\n"
        "```tool_call\n"
        '{"name": "read_file", "arguments": {"path": "the/real/path.py"}}\n'
        "```\n\n"
        "Substitute the real path. If you already know the exact text, call "
        "`edit_file` instead with `path`, `old_string`, and `new_string`.\n\n"
        "If you have genuinely concluded there is nothing worth changing, say so "
        "using the exact words `no code change is warranted` and explain why. "
        "That is a valid outcome and it will be recorded. What is not useful is "
        "a change that exists only in prose."
    )


def _malformed_call_guidance(errors: list[str]) -> str:
    """What the model is told when its tool_call block would not parse."""
    listed = "\n".join(f"- {error}" for error in errors)
    return (
        "Your tool_call block was not valid JSON, so no tool ran:\n"
        f"{listed}\n\n"
        "Send it again as exactly one JSON object per ```tool_call``` block. "
        "Remember that backslashes must be doubled inside JSON strings: a regex "
        'for `average(` is written "average\\\\(" and one for a digit is "\\\\d". '
        "If you meant to answer rather than call a tool, reply in plain prose "
        "with no tool_call block."
    )


async def run_once(prompt: str, config: Config | None = None, **kwargs: Any) -> RunResult:
    """Convenience: build an agent, run one prompt, tear down."""
    async with Agent(config=config, **kwargs) as agent:
        return await agent.run(prompt)
