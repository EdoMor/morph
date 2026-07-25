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
        successful_calls = 0
        nudged_to_act = False

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

                # A statement of intent is not a result. Small models routinely
                # reply "I will edit X to do Y" and stop; the loop then reads a
                # text-only turn as completion and ends having changed nothing.
                # Across eight recorded iterations the mean was 3 steps of a
                # 60-step budget — not running out of room, just stopping.
                # Nudge once, then respect the answer.
                if (
                    not successful_calls
                    and not nudged_to_act
                    and steps < budget
                    and _reads_like_an_unexecuted_plan(response.text)
                ):
                    nudged_to_act = True
                    active.assistant(response.text)
                    active.tool("supervisor", _act_on_your_plan_guidance(), ok=False)
                    yield AgentEvent(
                        "error",
                        {
                            "message": "the model described an action without taking it; asking it to act",
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
                successful_calls += int(result.ok)
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


#: Phrases a model uses to announce an action it has not taken. Paired with
#: "has not successfully called a single tool yet", these are a reliable signal
#: that a turn is a plan rather than an answer.
INTENT_PHRASES = (
    "i will ",
    "i'll ",
    "i am going to",
    "i'm going to",
    "let me ",
    "let's ",
    "i plan to",
    "i intend to",
    "next, i",
    "my plan",
    "here's my plan",
    "i should ",
    "we should ",
)


def _reads_like_an_unexecuted_plan(text: str) -> bool:
    """True when a reply announces work rather than reporting it.

    Deliberately paired with a zero successful-tool-call count at the call site:
    on its own this would misfire on a legitimate answer that happens to say
    "I'll leave that alone". After real work has happened, it never fires.
    """
    body = " ".join((text or "").lower().split())
    if len(body) < 40:  # "Done." and friends are answers, not plans
        return False
    return any(phrase in body for phrase in INTENT_PHRASES)


def _act_on_your_plan_guidance() -> str:
    """What the model is told when it narrates instead of acting."""
    return (
        "You described what you were going to do, but you have not run a single "
        "tool yet, so none of it has happened: no file has been read, and nothing "
        "has been changed. Describing an edit does not perform it.\n\n"
        "Do the next concrete step now as a tool call. If the step you described "
        "needs information you do not have, read or search for it first.\n\n"
        "If you have genuinely concluded there is nothing worth changing, say so "
        "plainly and explain why — that is a valid outcome, and it will be "
        "recorded. What is not useful is a plan nobody carries out."
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
