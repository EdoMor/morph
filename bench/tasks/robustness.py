"""Robustness checks: does the system survive things going wrong? (15 points)

These run offline and deterministically. They are the checks that catch the
failure mode a self-improving system is most prone to — code that scores well on
the happy path and shatters on the first error.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from morph.agent import Agent
from morph.config import Config, MCPServerConfig
from morph.llm.echo import EchoProvider
from morph.mcp import MCPManager
from morph.skills import SkillRegistry, load_skill
from morph.tools import ToolError, ToolRegistry, build_default_registry
from morph.tools.files import resolve_in_root
from morph.tools.image import ImageRequest, get_backend, run_image_flow

from .spec import TaskOutcome

Check = Callable[[Path], Awaitable[TaskOutcome]]


@dataclass
class RobustnessCheck:
    name: str
    run: Check
    requirement_ids: list[str]
    weight: float = 1.0


def _config(root: Path) -> Config:
    return Config(workspace=root, provider="echo", image_backend="stub", max_steps=6)


# ---------------------------------------------------------------------------


async def _path_escape(root: Path) -> TaskOutcome:
    """Traversal outside the workspace must be refused before any I/O (R-205)."""
    registry = build_default_registry(_config(root))
    attempts = ["../../../../etc/passwd", "/etc/passwd", "subdir/../../../etc/hosts"]
    for attempt in attempts:
        result = await registry.call("read_file", {"path": attempt})
        if result.ok:
            return TaskOutcome.fail(f"read_file({attempt!r}) was allowed to escape the workspace")
        if "outside the workspace" not in result.content and "not found" not in result.content.lower():
            return TaskOutcome.fail(f"unclear refusal for {attempt!r}: {result.content[:200]}")

    write = await registry.call("write_file", {"path": "../escaped.txt", "content": "x"})
    if write.ok:
        return TaskOutcome.fail("write_file escaped the workspace root")

    # A symlink pointing out of the workspace must also be refused.
    outside = root.parent / "outside-secret.txt"
    outside.write_text("secret", "utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        return TaskOutcome.ok("path traversal refused (symlink case not testable here)")
    linked = await registry.call("read_file", {"path": "link.txt"})
    if linked.ok:
        return TaskOutcome.fail("a symlink pointing outside the workspace was followed")
    return TaskOutcome.ok("traversal, absolute paths and escaping symlinks all refused")


async def _tool_error_recovery(root: Path) -> TaskOutcome:
    """A raising tool becomes a tool result the model can react to (R-109)."""
    registry = build_default_registry(_config(root))

    def explode() -> str:
        raise RuntimeError("boom")

    registry.register("explode", "Always fails", {"type": "object", "properties": {}}, explode)

    provider = EchoProvider(
        script=[
            EchoProvider.call("explode"),
            EchoProvider.text_response("The tool failed; carrying on."),
        ]
    )
    agent = Agent(config=_config(root), provider=provider, tools=registry, skills=SkillRegistry())
    try:
        result = await agent.run("Call explode, then recover.")
    finally:
        await agent.close()

    if result.error is not None:
        return TaskOutcome.fail(f"a failing tool aborted the run: {result.error}")
    if not result.tool_calls or result.tool_calls[0]["ok"]:
        return TaskOutcome.fail("the failing tool was not recorded as failed")
    if "boom" not in result.tool_calls[0]["content"]:
        return TaskOutcome.fail("the tool error text never reached the model")
    if "carrying on" not in result.text:
        return TaskOutcome.fail(f"the run did not continue after the error: {result.text!r}")
    return TaskOutcome.ok("tool failure surfaced to the model and the run continued")


async def _loop_terminates(root: Path) -> TaskOutcome:
    """A model that never stops calling tools still terminates (R-102)."""
    registry = ToolRegistry()
    registry.register("noop", "Does nothing", {"type": "object", "properties": {}}, lambda: "ok")

    class Forever:
        name = "forever"
        supports_native_tools = True

        async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
            return EchoProvider.call("noop")

    agent = Agent(config=_config(root), provider=Forever(), tools=registry, skills=SkillRegistry())
    try:
        result = await asyncio.wait_for(agent.run("Loop forever.", max_steps=5), timeout=30)
    except asyncio.TimeoutError:
        return TaskOutcome.fail("the agent loop hung instead of hitting its step budget")
    finally:
        await agent.close()

    if result.stop_reason != "max_steps":
        return TaskOutcome.fail(f"expected stop_reason='max_steps', got {result.stop_reason!r}")
    if result.steps != 5:
        return TaskOutcome.fail(f"budget was 5 but the loop ran {result.steps} steps")
    return TaskOutcome.ok("step budget enforced, loop exited cleanly")


async def _provider_failure(root: Path) -> TaskOutcome:
    """A provider that throws ends the run with an error, not a crash (R-109)."""

    class Broken:
        name = "broken"
        supports_native_tools = True

        async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
            raise ConnectionError("model host unreachable")

    agent = Agent(
        config=_config(root), provider=Broken(), tools=ToolRegistry(), skills=SkillRegistry()
    )
    try:
        result = await agent.run("Anything.")
    except Exception as exc:  # noqa: BLE001
        return TaskOutcome.fail(f"provider failure propagated out of the agent: {exc!r}")
    finally:
        await agent.close()

    if result.error is None:
        return TaskOutcome.fail("a dead provider produced no error on the result")
    if "unreachable" not in result.error:
        return TaskOutcome.fail(f"error text lost the cause: {result.error!r}")
    return TaskOutcome.ok("provider failure reported cleanly")


async def _bad_arguments(root: Path) -> TaskOutcome:
    """Invalid tool arguments become tool errors, never exceptions (R-207)."""
    registry = build_default_registry(_config(root))
    cases = [
        ("read_file", {}, "missing required"),
        ("read_file", {"path": 42}, "must be string"),
        ("edit_file", {"path": "a.txt", "old_string": "x"}, "missing required"),
        ("nonexistent_tool", {"x": 1}, "no tool named"),
    ]
    for tool, arguments, expected in cases:
        result = await registry.call(tool, arguments)
        if result.ok:
            return TaskOutcome.fail(f"{tool}({arguments}) should have failed but succeeded")
        if expected not in result.content.lower():
            return TaskOutcome.fail(
                f"{tool}({arguments}) error was unhelpful: {result.content[:200]!r}"
            )
    return TaskOutcome.ok("argument validation rejects bad calls with readable errors")


async def _edit_is_loud(root: Path) -> TaskOutcome:
    """edit_file never silently no-ops (R-203)."""
    registry = build_default_registry(_config(root))
    (root / "sample.py").write_text("alpha\nbeta\nalpha\n", "utf-8")

    missing = await registry.call(
        "edit_file", {"path": "sample.py", "old_string": "gamma", "new_string": "delta"}
    )
    if missing.ok:
        return TaskOutcome.fail("editing an absent string reported success")
    if (root / "sample.py").read_text("utf-8") != "alpha\nbeta\nalpha\n":
        return TaskOutcome.fail("a failed edit still modified the file")

    ambiguous = await registry.call(
        "edit_file", {"path": "sample.py", "old_string": "alpha", "new_string": "omega"}
    )
    if ambiguous.ok:
        return TaskOutcome.fail("an ambiguous edit (2 matches) was applied anyway")

    forced = await registry.call(
        "edit_file",
        {"path": "sample.py", "old_string": "alpha", "new_string": "omega", "replace_all": True},
    )
    if not forced.ok:
        return TaskOutcome.fail(f"replace_all edit failed: {forced.content}")
    if (root / "sample.py").read_text("utf-8") != "omega\nbeta\nomega\n":
        return TaskOutcome.fail("replace_all did not replace every occurrence")
    return TaskOutcome.ok("absent and ambiguous edits refused; replace_all works")


async def _malformed_skill(root: Path) -> TaskOutcome:
    """A broken skill is skipped, never fatal (R-304)."""
    skills_dir = root / "skills"
    (skills_dir / "good").mkdir(parents=True, exist_ok=True)
    (skills_dir / "good" / "SKILL.md").write_text(
        "---\nname: good\ndescription: A working skill.\n---\n\nBody.\n", "utf-8"
    )
    (skills_dir / "broken").mkdir(parents=True, exist_ok=True)
    (skills_dir / "broken" / "SKILL.md").write_text("no frontmatter at all\n", "utf-8")
    (skills_dir / "empty").mkdir(parents=True, exist_ok=True)

    registry = SkillRegistry()
    try:
        found = registry.discover([skills_dir])
    except Exception as exc:  # noqa: BLE001
        return TaskOutcome.fail(f"discovery raised on a malformed skill: {exc!r}")

    names = {s.name for s in found}
    if "good" not in names:
        return TaskOutcome.fail("the valid skill was not discovered")
    if "broken" in names:
        return TaskOutcome.fail("a skill without frontmatter was accepted")
    if load_skill(skills_dir / "empty") is not None:
        return TaskOutcome.fail("a directory with no SKILL.md was accepted as a skill")
    return TaskOutcome.ok("malformed skills skipped, valid skill loaded")


async def _mcp_isolation(root: Path) -> TaskOutcome:
    """A dead MCP server is isolated, not fatal (R-403)."""
    registry = build_default_registry(_config(root))
    baseline = len(registry)
    manager = MCPManager(registry)

    await manager.connect_all(
        [
            MCPServerConfig(name="ghost", transport="stdio", command="definitely-not-a-real-binary"),
            MCPServerConfig(name="nourl", transport="http", url=None, timeout=2.0),
            MCPServerConfig(name="weird", transport="carrier-pigeon", timeout=2.0),
        ]
    )

    if len(registry) != baseline:
        return TaskOutcome.fail("a failed MCP server still injected tools")
    if len(manager.failures) != 3:
        return TaskOutcome.fail(f"expected 3 recorded failures, got {manager.failures}")
    if manager.connections:
        return TaskOutcome.fail("a failed server was recorded as connected")

    status = manager.status()
    if "ghost" not in status["failed"]:
        return TaskOutcome.fail("status() does not report the failure")
    return TaskOutcome.ok("three broken servers isolated; native tools untouched")


async def _image_backend_errors(root: Path) -> TaskOutcome:
    """Missing keys and bad inputs produce actionable errors (R-502, R-505)."""
    config = _config(root)

    config.image_backend = "flux"
    import os

    saved = os.environ.pop("FLUX_API_KEY", None)
    try:
        try:
            await run_image_flow(ImageRequest(prompt="anything"), config)
            return TaskOutcome.fail("the flux backend ran without an API key")
        except ToolError as exc:
            if "FLUX_API_KEY" not in str(exc):
                return TaskOutcome.fail(f"error does not name the env var: {exc}")
    finally:
        if saved is not None:
            os.environ["FLUX_API_KEY"] = saved

    try:
        get_backend("does-not-exist")
        return TaskOutcome.fail("an unknown image backend was accepted")
    except ToolError:
        pass

    config.image_backend = "stub"
    for bad in (
        ImageRequest(prompt=""),
        ImageRequest(prompt="x", width=0),
        ImageRequest(prompt="x", width=99999),
        ImageRequest(prompt="x", count=0),
    ):
        try:
            await run_image_flow(bad, config)
            return TaskOutcome.fail(f"invalid image request accepted: {bad}")
        except ToolError:
            continue
    return TaskOutcome.ok("missing keys, unknown backends and bad inputs all rejected clearly")


async def _image_determinism(root: Path) -> TaskOutcome:
    """Same prompt + same seed => identical bytes on the stub backend (R-503)."""
    config = _config(root)
    request = ImageRequest(prompt="a red barn", width=48, height=48, seed=11)

    first = await run_image_flow(request, config)
    second = await run_image_flow(request, config)
    third = await run_image_flow(
        ImageRequest(prompt="a blue barn", width=48, height=48, seed=11), config
    )

    a = (root / first.paths[0]).read_bytes()
    b = (root / second.paths[0]).read_bytes()
    c = (root / third.paths[0]).read_bytes()
    if a != b:
        return TaskOutcome.fail("identical requests produced different bytes")
    if a == c:
        return TaskOutcome.fail("different prompts produced identical images")
    if not first.previews or not first.previews[0].startswith("data:image/png;base64,"):
        return TaskOutcome.fail("no usable data-URI preview was returned")
    return TaskOutcome.ok("stub backend is deterministic and returns previews")


async def _session_resume(root: Path) -> TaskOutcome:
    """A resumed session keeps its tool calls and results (R-106)."""
    config = _config(root)
    registry = build_default_registry(config)
    provider = EchoProvider(
        script=[
            EchoProvider.call("write_file", path="note.txt", content="hello"),
            EchoProvider.text_response("Wrote the note."),
        ]
    )
    agent = Agent(config=config, provider=provider, tools=registry, skills=SkillRegistry())
    try:
        first = await agent.run("Write note.txt containing hello.")
        reloaded = agent.sessions.load(first.session_id)
    finally:
        await agent.close()

    roles = [m.get("role") for m in reloaded.messages]
    if "tool" not in roles:
        return TaskOutcome.fail(f"tool results were lost on reload: {roles}")
    tool_messages = [m for m in reloaded.messages if m.get("role") == "tool"]
    if not any(m.get("tool_call_id") for m in tool_messages):
        return TaskOutcome.fail("tool_call_id was not persisted")
    if not any("write_file" == m.get("name") for m in tool_messages):
        return TaskOutcome.fail("the tool name was not persisted")
    return TaskOutcome.ok(f"session round-tripped {len(reloaded.messages)} messages intact")


async def _workspace_guard_unit(root: Path) -> TaskOutcome:
    """resolve_in_root is correct in isolation (R-205)."""
    inside = root / "a" / "b"
    inside.mkdir(parents=True, exist_ok=True)
    if resolve_in_root(root, "a/b") != inside.resolve():
        return TaskOutcome.fail("a legitimate relative path did not resolve correctly")
    if resolve_in_root(root, str(inside)) != inside.resolve():
        return TaskOutcome.fail("a legitimate absolute path inside the root was rejected")
    for bad in ["..", "../..", "/etc", "a/../../..", "a/b/../../../.."]:
        try:
            resolve_in_root(root, bad)
            return TaskOutcome.fail(f"resolve_in_root allowed {bad!r}")
        except ToolError:
            continue
    return TaskOutcome.ok("workspace confinement holds for relative, absolute and traversal paths")


ROBUSTNESS_CHECKS: list[RobustnessCheck] = [
    RobustnessCheck("robustness/path-escape", _path_escape, ["R-205", "R-605"], weight=2.0),
    RobustnessCheck("robustness/workspace-guard", _workspace_guard_unit, ["R-205"]),
    RobustnessCheck("robustness/tool-error-recovery", _tool_error_recovery, ["R-109"], weight=2.0),
    RobustnessCheck("robustness/loop-terminates", _loop_terminates, ["R-102"], weight=2.0),
    RobustnessCheck("robustness/provider-failure", _provider_failure, ["R-109"]),
    RobustnessCheck("robustness/bad-arguments", _bad_arguments, ["R-207"]),
    RobustnessCheck("robustness/edit-is-loud", _edit_is_loud, ["R-203"], weight=2.0),
    RobustnessCheck("robustness/malformed-skill", _malformed_skill, ["R-304"]),
    RobustnessCheck("robustness/mcp-isolation", _mcp_isolation, ["R-403"], weight=2.0),
    RobustnessCheck("robustness/image-errors", _image_backend_errors, ["R-502", "R-505"]),
    RobustnessCheck("robustness/image-determinism", _image_determinism, ["R-503", "R-504"]),
    RobustnessCheck("robustness/session-resume", _session_resume, ["R-106"]),
]
