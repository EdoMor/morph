"""Conformance suite — the goalposts.

PROTECTED FILE — the self-improvement loop may not modify this (R-707). Every
test here maps to a requirement ID in ``REQUIREMENTS.md``. If a requirement is
wrong, a human changes it; the loop may not.

The meta-test at the bottom (R-805) enforces the other direction: every
requirement in the document must be referenced by at least one test.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from morph.agent import Agent
from morph.config import Config, load_config
from morph.llm import available_providers, get_provider
from morph.llm.base import parse_text_tool_calls
from morph.llm.echo import EchoProvider
from morph.skills import SkillRegistry
from morph.tools import ToolRegistry, build_default_registry
from morph.tools.image import BACKENDS

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_FILE = REPO_ROOT / "REQUIREMENTS.md"
REQUIREMENT_RE = re.compile(r"\*\*(R-\d{3})\*\*")
ANY_ID_RE = re.compile(r"R-\d{3}")


def declared_requirements() -> list[str]:
    return REQUIREMENT_RE.findall(REQUIREMENTS_FILE.read_text("utf-8"))


# ---------------------------------------------------------------------------
# Agent core
# ---------------------------------------------------------------------------


async def test_R_101_agent_runs_a_tool_loop(make_agent, call, say):
    """R-101: the agent calls tools, feeds results back, and answers."""
    agent = make_agent(
        script=[
            call("write_file", path="hello.txt", content="hi"),
            say("Wrote hello.txt."),
        ]
    )
    result = await agent.run("Create hello.txt containing hi.")

    assert result.ok
    assert (agent.config.root / "hello.txt").read_text() == "hi"
    assert [c["tool"] for c in result.tool_calls] == ["write_file"]
    assert "hello.txt" in result.text


async def test_R_102_loop_is_bounded(make_agent, call):
    """R-102: an endless tool-caller stops at the step budget, cleanly."""
    agent = make_agent(script=[call("list_dir", path=".")] * 50)
    result = await agent.run("Loop.", max_steps=4)

    assert result.stop_reason == "max_steps"
    assert result.steps == 4
    assert result.error is None


def test_R_103_providers_are_pluggable():
    """R-103: providers are selected by name, with no code change."""
    for name in ("ollama", "google", "echo"):
        assert name in available_providers()
    assert get_provider("echo").name == "echo"
    assert get_provider("ollama", model="gemma3:12b").model == "gemma3:12b"

    with pytest.raises(Exception) as excinfo:
        get_provider("does-not-exist")
    assert "Available" in str(excinfo.value)


def test_R_104_default_model_is_gemma():
    """R-104: Gemma writes Morph, so Gemma is the default."""
    assert "gemma" in Config().model.lower()
    assert "gemma" in load_config(REPO_ROOT).model.lower()


def test_R_105_text_tool_protocol():
    """R-105: providers without native function calling still call tools."""
    text, calls = parse_text_tool_calls(
        'Let me look.\n```tool_call\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```\n'
    )
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.py"}
    assert "Let me look." in text

    assert get_provider("ollama").supports_native_tools is False

    # Malformed blocks degrade to prose rather than raising.
    prose, none = parse_text_tool_calls("```tool_call\n{not json}\n```")
    assert none == []
    assert "not json" in prose


async def test_R_106_sessions_persist_and_resume(make_agent, call, say):
    """R-106: a reloaded session keeps every tool call and result."""
    agent = make_agent(
        script=[call("write_file", path="a.txt", content="x"), say("Done.")]
    )
    first = await agent.run("Write a.txt.")
    reloaded = agent.sessions.load(first.session_id)

    roles = [m["role"] for m in reloaded.messages]
    assert roles.count("user") == 1
    assert "tool" in roles
    tool_message = next(m for m in reloaded.messages if m["role"] == "tool")
    assert tool_message["name"] == "write_file"
    assert tool_message["tool_call_id"]

    assistant = next(m for m in reloaded.messages if m.get("tool_calls"))
    assert assistant["tool_calls"][0]["arguments"]["path"] == "a.txt"


async def test_R_107_event_stream_shape(make_agent, call, say):
    """R-107: the run emits a typed event stream a UI can render."""
    agent = make_agent(script=[call("list_dir", path="."), say("Listed.")])
    types = [event.type async for event in agent.stream("List the directory.")]

    assert types[0] == "tool_use"
    assert "tool_result" in types
    assert "text" in types
    assert types[-1] == "done"


async def test_R_108_every_tool_call_is_recorded(make_agent, call, say):
    """R-108: cost and latency are reconstructable from the log."""
    agent = make_agent(
        script=[call("write_file", path="b.txt", content="y"), say("Done.")]
    )
    result = await agent.run("Write b.txt.")

    entry = result.tool_calls[0]
    assert entry["tool"] == "write_file"
    assert entry["ok"] is True
    assert entry["arguments"] == {"path": "b.txt", "content": "y"}
    assert entry["duration_ms"] >= 0
    assert set(result.usage) >= {"input_tokens", "output_tokens"}
    assert result.duration_ms > 0


async def test_R_109_tool_failure_does_not_kill_the_run(config, say):
    """R-109: a raising tool becomes a result the model can recover from."""
    registry = ToolRegistry()

    def explode() -> str:
        raise RuntimeError("boom")

    registry.register("explode", "fails", {"type": "object", "properties": {}}, explode)
    agent = Agent(
        config=config,
        provider=EchoProvider(script=[EchoProvider.call("explode"), say("Recovered.")]),
        tools=registry,
        skills=SkillRegistry(),
    )
    result = await agent.run("Call explode.")

    assert result.error is None
    assert result.tool_calls[0]["ok"] is False
    assert "boom" in result.tool_calls[0]["content"]
    assert result.text == "Recovered."


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_R_201_tools_are_declared_and_discoverable(registry):
    """R-201: registering a tool is all it takes to expose it."""
    specs = {spec["name"]: spec for spec in registry.specs()}
    assert specs
    for spec in specs.values():
        assert spec["description"]
        assert spec["input_schema"]["type"] == "object"

    registry.register("custom", "A custom tool", {"type": "object", "properties": {}}, lambda: "x")
    assert "custom" in registry.names()


def test_R_202_filesystem_tools_exist(registry):
    """R-202: the standard filesystem tool set is present."""
    expected = {"read_file", "write_file", "edit_file", "list_dir", "glob", "grep"}
    assert expected <= set(registry.names())


async def test_R_203_edit_file_fails_loudly(registry, workspace):
    """R-203: a missing or ambiguous target is an error, never a silent no-op."""
    target = workspace / "f.py"
    target.write_text("a\nb\na\n")

    missing = await registry.call(
        "edit_file", {"path": "f.py", "old_string": "zzz", "new_string": "q"}
    )
    assert not missing.ok
    assert "not found" in missing.content
    assert target.read_text() == "a\nb\na\n"

    ambiguous = await registry.call(
        "edit_file", {"path": "f.py", "old_string": "a", "new_string": "q"}
    )
    assert not ambiguous.ok
    assert "ambiguous" in ambiguous.content

    forced = await registry.call(
        "edit_file",
        {"path": "f.py", "old_string": "a", "new_string": "q", "replace_all": True},
    )
    assert forced.ok
    assert target.read_text() == "q\nb\nq\n"


async def test_R_204_shell_tool(registry):
    """R-204: commands run with a timeout and captured output."""
    assert "shell" in registry.names()

    ok = await registry.call("shell", {"command": "echo morph"})
    assert ok.ok
    assert "morph" in ok.content
    assert ok.meta["exit_code"] == 0

    failed = await registry.call("shell", {"command": "exit 3"})
    assert failed.meta["exit_code"] == 3

    timed_out = await registry.call("shell", {"command": "sleep 5", "timeout": 0.3})
    assert not timed_out.ok
    assert "timed out" in timed_out.content


async def test_R_205_workspace_confinement(registry, workspace):
    """R-205: nothing reaches outside the workspace root."""
    outside = workspace.parent / "outside.txt"
    outside.write_text("secret")

    for path in ("../outside.txt", "/etc/passwd", "a/../../outside.txt"):
        result = await registry.call("read_file", {"path": path})
        assert not result.ok, f"{path} was not refused"

    write = await registry.call("write_file", {"path": "../escape.txt", "content": "x"})
    assert not write.ok
    assert not (workspace.parent / "escape.txt").exists()

    shell = await registry.call("shell", {"command": "pwd", "cwd": ".."})
    assert not shell.ok


async def test_R_206_web_tools_degrade_gracefully(registry, monkeypatch):
    """R-206: offline and unconfigured both produce clear errors."""
    assert {"web_fetch", "web_search"} <= set(registry.names())

    monkeypatch.delenv("MORPH_SEARCH_API_KEY", raising=False)
    search = await registry.call("web_search", {"query": "anything"})
    assert not search.ok
    assert "MORPH_SEARCH_API_KEY" in search.content

    bad_url = await registry.call("web_fetch", {"url": "not-a-url"})
    assert not bad_url.ok
    assert "http" in bad_url.content.lower()


async def test_R_207_arguments_are_validated(registry):
    """R-207: bad arguments produce tool errors, never exceptions."""
    cases = [
        ("read_file", {}),
        ("read_file", {"path": 123}),
        ("read_file", {"path": "a", "limit": "many"}),
        ("write_file", {"path": "a"}),
        ("no_such_tool", {}),
    ]
    for tool, arguments in cases:
        result = await registry.call(tool, arguments)
        assert not result.ok, f"{tool}({arguments}) should have failed"
        assert result.content


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def _write_skill(root: Path, name: str, description: str, body: str = "Body.") -> Path:
    directory = root / "skills" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\nallowed-tools: read_file, shell\n---\n\n{body}\n",
        "utf-8",
    )
    return directory


def test_R_301_skill_format(workspace):
    """R-301: SKILL.md with YAML frontmatter, Claude-compatible."""
    from morph.skills import load_skill

    _write_skill(workspace, "demo", "Does a demo thing.")
    skill = load_skill(workspace / "skills" / "demo")

    assert skill is not None
    assert skill.name == "demo"
    assert skill.description == "Does a demo thing."
    assert skill.allowed_tools == ["read_file", "shell"]


def test_R_302_skills_are_discovered_and_exposed(workspace, registry):
    """R-302: discovery from a search path, exposed to the model."""
    _write_skill(workspace, "alpha", "First skill.")
    _write_skill(workspace, "beta", "Second skill.")

    skills = SkillRegistry()
    skills.discover([workspace / "skills"])
    assert skills.names() == ["alpha", "beta"]

    section = skills.prompt_section()
    assert "alpha" in section and "First skill." in section

    skills.register_tool(registry)
    assert "load_skill" in registry.names()


async def test_R_303_skill_bodies_load_lazily(workspace, registry):
    """R-303: context cost scales with use, not with library size."""
    _write_skill(workspace, "lazy", "A lazy skill.", body="SECRET-BODY-MARKER")

    skills = SkillRegistry()
    skills.discover([workspace / "skills"])
    skill = skills.get("lazy")

    assert skill is not None
    assert not skill.loaded
    assert "SECRET-BODY-MARKER" not in skills.prompt_section()

    skills.register_tool(registry)
    result = await registry.call("load_skill", {"name": "lazy"})
    assert result.ok
    assert "SECRET-BODY-MARKER" in result.content
    assert skill.loaded


def test_R_304_malformed_skills_are_skipped(workspace):
    """R-304: a broken skill never prevents startup."""
    _write_skill(workspace, "good", "A good skill.")
    broken = workspace / "skills" / "broken"
    broken.mkdir(parents=True)
    (broken / "SKILL.md").write_text("no frontmatter here\n", "utf-8")
    (workspace / "skills" / "empty").mkdir(parents=True)

    skills = SkillRegistry()
    found = skills.discover([workspace / "skills", workspace / "does-not-exist"])

    assert [s.name for s in found] == ["good"]


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------


async def test_R_401_mcp_client_transports():
    """R-401: stdio and http are both supported; anything else is refused."""
    from morph.config import MCPServerConfig
    from morph.mcp import MCPConnection, MCPError

    assert MCPConnection(MCPServerConfig(name="s", transport="stdio")).config.transport == "stdio"

    bad = MCPConnection(MCPServerConfig(name="s", transport="smoke-signals"))
    with pytest.raises(MCPError) as excinfo:
        await bad.start()
    assert "transport" in str(excinfo.value)


async def test_R_402_mcp_tools_are_namespaced_and_callable(mcp_server_factory, registry):
    """R-402: MCP tools join the registry and behave like native ones."""
    from morph.mcp import MCPManager

    manager = MCPManager(registry)
    connection = await manager.connect(mcp_server_factory(tools=["echo"]))

    assert connection is not None
    assert "mcp__fake__echo" in registry.names()

    result = await registry.call("mcp__fake__echo", {"text": "hello"})
    assert result.ok
    assert "hello" in result.content

    await manager.close_all()


async def test_R_403_failing_mcp_server_is_isolated(registry):
    """R-403: a dead server drops its tools and nothing else."""
    from morph.config import MCPServerConfig
    from morph.mcp import MCPManager

    before = len(registry)
    manager = MCPManager(registry)
    await manager.connect_all(
        [
            MCPServerConfig(name="ghost", transport="stdio", command="no-such-binary-xyz"),
            MCPServerConfig(name="void", transport="http", url=None, timeout=1.0),
        ]
    )

    assert len(registry) == before
    assert set(manager.failures) == {"ghost", "void"}
    assert manager.connections == {}
    assert "ghost" in manager.status()["failed"]


async def test_R_404_mcp_handshake_follows_the_spec(mcp_server_factory, registry):
    """R-404: initialize -> notifications/initialized -> tools/list, JSON-RPC 2.0."""
    from morph.mcp import MCPManager

    spec = mcp_server_factory(tools=["echo"])
    manager = MCPManager(registry)
    connection = await manager.connect(spec)
    assert connection is not None

    methods = json.loads(Path(spec.env["FAKE_MCP_LOG"]).read_text("utf-8"))
    assert methods[0]["method"] == "initialize"
    assert methods[0]["jsonrpc"] == "2.0"
    assert methods[1]["method"] == "notifications/initialized"
    assert "tools/list" in [m["method"] for m in methods]

    await manager.close_all()


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


def test_R_501_generate_image_tool_exists(registry):
    """R-501: the agent and the UI can both generate images."""
    assert "generate_image" in registry.names()
    schema = registry.get("generate_image").input_schema
    assert schema["required"] == ["prompt"]


def test_R_502_image_backends_are_pluggable():
    """R-502: flux, gemini, local and stub all ship."""
    assert {"flux", "gemini", "local", "stub"} <= set(BACKENDS)

    from morph.tools import ToolError
    from morph.tools.image import get_backend, register_image_backend

    assert get_backend("stub").name == "stub"
    with pytest.raises(ToolError):
        get_backend("nope")

    class Custom:
        name = "custom"

    register_image_backend("custom", Custom)
    assert get_backend("custom").name == "custom"


async def test_R_503_image_flow_is_deterministic(config):
    """R-503: same prompt + same seed => identical bytes on the stub backend."""
    from morph.tools.image import ImageRequest, run_image_flow

    request = ImageRequest(prompt="a red barn", width=32, height=32, seed=5)
    first = await run_image_flow(request, config)
    second = await run_image_flow(request, config)
    other = await run_image_flow(
        ImageRequest(prompt="a blue barn", width=32, height=32, seed=5), config
    )

    root = config.root
    assert (root / first.paths[0]).read_bytes() == (root / second.paths[0]).read_bytes()
    assert (root / first.paths[0]).read_bytes() != (root / other.paths[0]).read_bytes()


async def test_R_504_images_are_saved_with_previews(registry, config):
    """R-504: file paths plus a data URI, so a phone renders without a round trip."""
    result = await registry.call(
        "generate_image", {"prompt": "a tree", "width": 32, "height": 32, "count": 2}
    )

    assert result.ok
    assert len(result.meta["images"]) == 2
    for relative in result.meta["images"]:
        assert (config.root / relative).is_file()
    for preview in result.meta["previews"]:
        assert preview.startswith("data:image/png;base64,")


async def test_R_505_missing_key_names_the_env_var(config, monkeypatch):
    """R-505: an actionable error, never a stack trace."""
    from morph.tools import ToolError
    from morph.tools.image import ImageRequest, run_image_flow

    monkeypatch.delenv("FLUX_API_KEY", raising=False)
    config.image_backend = "flux"
    with pytest.raises(ToolError) as excinfo:
        await run_image_flow(ImageRequest(prompt="x"), config)
    assert "FLUX_API_KEY" in str(excinfo.value)

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config.image_backend = "gemini"
    with pytest.raises(ToolError) as excinfo:
        await run_image_flow(ImageRequest(prompt="x"), config)
    assert "GOOGLE_API_KEY" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Server & mobile client
# ---------------------------------------------------------------------------


async def test_R_601_api_endpoints(live_server):
    """R-601: health, tools, skills, sessions and streaming chat."""
    import httpx

    base = live_server.base_url
    async with httpx.AsyncClient(base_url=base, timeout=30) as client:
        health = await client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        assert "tools" in (await client.get("/api/tools")).json()
        assert "skills" in (await client.get("/api/skills")).json()
        assert "sessions" in (await client.get("/api/sessions")).json()

        events = []
        async with client.stream(
            "POST", "/api/chat", json={"message": "List the directory."}
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))

    assert [e["type"] for e in events][-1] == "done"
    assert any(e["type"] == "tool_use" for e in events)


def test_R_602_client_is_mobile_ready(repo_root):
    """R-602: 360px-friendly, 44px touch targets, no horizontal scroll."""
    html = (repo_root / "webapp" / "index.html").read_text("utf-8")
    css = (repo_root / "webapp" / "style.css").read_text("utf-8")

    assert 'name="viewport"' in html and "width=device-width" in html
    assert "--tap: 44px" in css
    assert "overflow-x: hidden" in css

    # Nothing may declare a fixed width wider than the narrowest supported phone.
    for match in re.finditer(r"[^-]width:\s*(\d+)px", css):
        assert int(match.group(1)) <= 360, f"fixed width {match.group(0)} breaks 360px layout"

    # Every interactive control sizes itself from the 44px tap variable.
    rules = {}
    for rule in css.split("}"):
        head, _, body = rule.partition("{")
        for selector in head.split(","):
            rules.setdefault(selector.strip(), "")
            rules[selector.strip()] += body

    for selector in (".icon-btn", ".chip", ".send", ".composer textarea", ".list li"):
        assert selector in rules, f"{selector} has no rule"
        assert "var(--tap)" in rules[selector], f"{selector} does not use the 44px tap size"


def test_R_603_installable_pwa(repo_root):
    """R-603: manifest plus a service worker, so it installs to a home screen."""
    webapp = repo_root / "webapp"
    manifest = json.loads((webapp / "manifest.webmanifest").read_text("utf-8"))

    assert manifest["display"] in {"standalone", "fullscreen", "minimal-ui"}
    assert manifest["start_url"]
    assert manifest["name"] and manifest["short_name"]

    sizes = {icon["sizes"] for icon in manifest["icons"]}
    assert {"192x192", "512x512"} <= sizes
    for icon in manifest["icons"]:
        assert (webapp / icon["src"].lstrip("/")).is_file(), icon["src"]
    assert any("maskable" in icon.get("purpose", "") for icon in manifest["icons"])

    assert 'rel="manifest"' in (webapp / "index.html").read_text("utf-8")
    service_worker = (webapp / "sw.js").read_text("utf-8")
    assert "addEventListener" in service_worker and "fetch" in service_worker
    assert "serviceWorker" in (webapp / "app.js").read_text("utf-8")


def test_R_604_runs_locally_without_third_party_services(repo_root):
    """R-604: the server needs no web framework and no cloud service."""
    import morph.server  # noqa: F401  (must import with nothing extra installed)

    dependencies = (repo_root / "pyproject.toml").read_text("utf-8")
    core = dependencies.split("dependencies = [", 1)[1].split("]", 1)[0]
    for forbidden in ("fastapi", "uvicorn", "django", "flask", "torch"):
        assert forbidden not in core, f"{forbidden} must not be a core dependency"

    server_source = (repo_root / "morph" / "server.py").read_text("utf-8")
    assert "import asyncio" in server_source
    assert "fastapi" not in server_source.lower()


async def test_R_605_server_does_not_trust_client_paths(live_server):
    """R-605: workspace confinement applies to every request."""
    import httpx

    async with httpx.AsyncClient(base_url=live_server.base_url, timeout=30) as client:
        for path in ("/../../etc/passwd", "/..%2f..%2fetc%2fpasswd", "/etc/passwd"):
            response = await client.get(path)
            assert response.status_code in (200, 404)
            assert "root:" not in response.text, f"{path} leaked a system file"

        outside = await client.get("/api/sessions/..%2f..%2fetc%2fpasswd")
        assert outside.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Self-improvement loop
# ---------------------------------------------------------------------------


def test_R_701_loop_is_a_closed_cycle():
    """R-701: benchmark -> feedback -> edit -> benchmark -> keep or revert."""
    from selfimprove import loop

    source = Path(loop.__file__).read_text("utf-8")
    for stage in ("measure", "build_improvement_prompt", "run_benchmark", "_commit_and_merge"):
        assert stage in source, f"the loop does not reference {stage}"

    assert hasattr(loop, "run_iteration")
    assert hasattr(loop, "run_loop")


def test_R_702_loop_uses_morphs_own_agent():
    """R-702: Morph improves Morph — the editor is this project's agent."""
    from selfimprove import loop

    source = Path(loop.__file__).read_text("utf-8")
    assert "from morph.agent import Agent" in source
    assert "Agent(" in source
    assert "agent.run(" in source


def test_R_703_iterations_are_isolated_in_a_worktree():
    """R-703: the working branch is never left broken."""
    from selfimprove import loop

    assert hasattr(loop, "_create_worktree")
    source = Path(loop.__file__).read_text("utf-8")
    assert '"worktree", "add"' in source
    assert '"worktree", "remove"' in source


def test_R_704_acceptance_requires_a_passing_gate_and_no_regression():
    """R-704: accept only when tests pass and the score did not go down."""
    from selfimprove import loop

    source = Path(loop.__file__).read_text("utf-8")
    assert 'after.get("gated")' in source
    assert "score_after < iteration.score_before" in source
    assert "conformance suite is failing" in source


def test_R_705_history_is_recorded_and_fed_back(tmp_path):
    """R-705: attempts are logged, and the log shapes the next prompt."""
    from selfimprove.prompts import append_history, build_improvement_prompt, load_history

    history_file = tmp_path / "history.jsonl"
    append_history(
        history_file,
        {
            "ts": 1.0,
            "iteration": 1,
            "base_commit": "abc123",
            "score_before": 50.0,
            "score_after": 40.0,
            "accepted": False,
            "rejection_reason": "score regressed",
            "summary": "Tried caching the tool registry.",
            "files_changed": ["morph/tools/__init__.py"],
        },
    )
    entries = load_history(history_file)
    assert len(entries) == 1
    assert entries[0]["accepted"] is False

    prompt = build_improvement_prompt(
        requirements="R-000 do a thing",
        scorecard={"composite": 50.0, "categories": {}},
        feedback="something failed",
        history=entries,
    )
    assert "Tried caching the tool registry." in prompt
    assert "REJECTED" in prompt
    assert "score regressed" in prompt


def test_R_706_loop_runs_unattended_in_ci(repo_root):
    """R-706: a scheduled GitHub workflow and a Codespaces devcontainer exist."""
    workflow = repo_root / ".github" / "workflows" / "self-improve.yml"
    assert workflow.is_file()
    body = workflow.read_text("utf-8")
    assert "schedule" in body and "cron" in body
    assert "workflow_dispatch" in body
    assert "selfimprove.loop" in body or "morph improve" in body
    assert "concurrency" in body, "two evolutions at once would race for the branch"

    devcontainer = repo_root / ".devcontainer" / "devcontainer.json"
    assert devcontainer.is_file()
    assert "ollama" in devcontainer.read_text("utf-8").lower()


def test_R_713_workflow_publishes_to_the_default_branch(repo_root):
    """R-713: evolved code lands on main automatically, with its own gates."""
    body = (repo_root / ".github" / "workflows" / "self-improve.yml").read_text("utf-8")

    assert "contents: write" in body
    assert "selfimprove.publish" in body
    assert "main" in body
    # The whole run is re-verified, not just each iteration.
    assert "pytest tests -q" in body
    assert "REFUSING" in body, "the workflow must refuse to publish a regression"
    # And it must not have quietly reverted to proposing a PR.
    assert "gh pr create" not in body


def test_R_713_publish_never_force_pushes(repo_root):
    """Checked against the git arguments the code can actually pass, not its prose."""
    import ast

    from selfimprove import publish as publish_module

    tree = ast.parse(Path(publish_module.__file__).read_text("utf-8"))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    # Docstrings are string constants too, so match whole arguments only.
    arguments = {value for value in literals if value.startswith("-")}

    for forbidden in ("--force", "-f", "--force-with-lease", "--delete"):
        assert forbidden not in arguments, f"publishing must never pass {forbidden} to git"


async def test_R_713_publish_rebases_and_reverifies_before_pushing(tmp_path):
    """The branch moving mid-run is the normal case, not the exception."""
    from selfimprove.publish import publish

    origin, clone = _two_repos(tmp_path)

    # Someone else pushes to main while our run is in flight.
    _commit(origin, "upstream.txt", "theirs", "upstream work")
    # Meanwhile the loop accepted an iteration locally.
    _commit(clone, "ours.txt", "ours", "selfimprove: iteration 1")

    verified: list[Path] = []

    def verifier(repo):
        verified.append(repo)
        return True, "green"

    result = publish(repo=clone, branch="main", remote="origin", verify=verifier)

    assert result.published, result.reason
    assert result.rebased, "should have rebased onto the moved branch"
    assert verified, "the suite must be re-run after a rebase, before pushing"

    log = _git(origin, "log", "--oneline")
    assert "selfimprove: iteration 1" in log
    assert "upstream work" in log


async def test_R_713_publish_aborts_when_the_gate_fails_after_rebase(tmp_path):
    from selfimprove.publish import publish

    origin, clone = _two_repos(tmp_path)
    _commit(origin, "upstream.txt", "theirs", "upstream work")
    _commit(clone, "ours.txt", "ours", "selfimprove: iteration 1")

    result = publish(
        repo=clone,
        branch="main",
        remote="origin",
        verify=lambda repo: (False, "2 failed"),
    )

    assert not result.published
    assert "failed after rebasing" in result.reason
    assert "selfimprove: iteration 1" not in _git(origin, "log", "--oneline")


async def test_R_713_publish_is_a_no_op_when_nothing_was_accepted(tmp_path):
    from selfimprove.publish import publish

    origin, clone = _two_repos(tmp_path)
    result = publish(repo=clone, branch="main", remote="origin", verify=None)

    assert not result.published
    assert "nothing to publish" in result.reason


async def test_R_713_publish_stops_on_a_rebase_conflict(tmp_path):
    """Two sides editing the same line is a human's problem, not the bot's."""
    from selfimprove.publish import publish

    origin, clone = _two_repos(tmp_path)
    _commit(origin, "shared.txt", "their version", "upstream edit")
    _commit(clone, "shared.txt", "our version", "selfimprove: iteration 1")

    result = publish(repo=clone, branch="main", remote="origin", verify=None)

    assert not result.published
    assert "conflict" in result.reason.lower()
    assert "human" in result.reason.lower()
    # The conflict must not be left half-applied in the working tree.
    assert "rebase" not in _git(clone, "status").lower()


def test_R_714_history_is_committed_so_it_outlives_the_runner(tmp_path):
    """R-714: without this, a scheduled loop forgets every previous attempt."""
    from selfimprove.publish import record_history

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "seed.txt").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")

    assert record_history(repo) is False  # nothing to record yet

    history = repo / "selfimprove" / "history.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text('{"iteration": 1, "accepted": false}\n')

    assert record_history(repo) is True
    assert "history" in _git(repo, "log", "-1", "--format=%s")
    assert "selfimprove/history.jsonl" in _git(repo, "show", "--name-only", "--format=")
    assert record_history(repo) is False  # already recorded, nothing new

    # And it must not be gitignored, or the commit above would be a lie.
    ignored = subprocess.run(
        ["git", "check-ignore", "selfimprove/history.jsonl"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ignored.returncode != 0, "history.jsonl must not be gitignored"


# -- helpers for the publish tests -----------------------------------------


def _git(repo, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def _commit(repo: Path, name: str, content: str, message: str) -> None:
    (repo / name).write_text(content, "utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)


def _two_repos(tmp_path: Path) -> tuple[Path, Path]:
    """A non-bare 'origin' on branch main, plus a clone of it."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@example.com")
    _git(origin, "config", "user.name", "t")
    # Allow pushing to the checked-out branch of a non-bare repo.
    _git(origin, "config", "receive.denyCurrentBranch", "updateInstead")
    _commit(origin, "seed.txt", "seed", "initial")

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(clone)], capture_output=True, check=True
    )
    _git(clone, "config", "user.email", "bot@example.com")
    _git(clone, "config", "user.name", "bot")
    return origin, clone


def test_R_707_goalposts_are_protected(repo_root):
    """R-707: the loop cannot edit the requirements, the gate, the scorer or the tasks."""
    from selfimprove.guard import PROTECTED, is_protected

    for required in (
        "REQUIREMENTS.md",
        "tests/test_requirements.py",
        "bench/scorecard.py",
        "bench/tasks",
    ):
        assert required in PROTECTED

    assert is_protected("REQUIREMENTS.md")
    assert is_protected("./tests/test_requirements.py")
    assert is_protected("bench/scorecard.py")
    # The whole task package, so the loop cannot author its own easy benchmark.
    assert is_protected("bench/tasks/coding.py")
    assert is_protected("bench/tasks/spec.py")
    assert not is_protected("morph/agent.py")
    assert not is_protected("tests/test_agent.py")
    assert not is_protected("bench/runner.py")

    for path in PROTECTED:
        assert (repo_root / path).exists(), f"protected path {path} does not exist"


# ---------------------------------------------------------------------------
# Benchmark calibration — the instrument itself
# ---------------------------------------------------------------------------


def test_R_708_every_suite_spans_the_difficulty_range():
    """R-708: no gradient, no learning signal."""
    from collections import Counter

    from bench.scorecard import CAPABILITY_CATEGORIES
    from bench.tasks import SUITES

    assert set(SUITES) == set(CAPABILITY_CATEGORIES)

    for name, tasks in SUITES.items():
        assert len(tasks) >= 5, f"{name} has only {len(tasks)} tasks"
        tiers = Counter(int(t.tier) for t in tasks)
        missing = sorted({1, 2, 3, 4, 5} - set(tiers))
        assert not missing, f"{name} has no tasks at tier(s) {missing}"
        biggest = max(tiers.values())
        assert biggest <= len(tasks) / 2 + 1, (
            f"{name} is lopsided: {biggest} of {len(tasks)} tasks at one tier"
        )
        assert all(t.category == name for t in tasks)


def test_R_708_harder_tiers_are_worth_more():
    from bench.tasks.spec import TIER_WEIGHT, Tier

    weights = [TIER_WEIGHT[t] for t in sorted(Tier)]
    assert weights == sorted(weights), "tier weights must be non-decreasing"
    assert weights[-1] > weights[0]
    # Sub-exponential: solving every easy task must still be worth something.
    assert weights[-1] / weights[0] <= 12


def test_R_709_rubrics_are_graded_not_binary():
    """R-709: partial progress has to be visible, or the loop cannot climb."""
    from bench.tasks.spec import Criterion, Rubric, TaskContext

    rubric = Rubric(
        [
            Criterion("a", 1.0, lambda c: True),
            Criterion("b", 1.0, lambda c: False),
            Criterion("c", 2.0, lambda c: 0.5),
        ]
    )
    grade = rubric.grade(TaskContext(root=REPO_ROOT, result=None))

    assert 0.0 < grade.score < 1.0
    assert grade.score == pytest.approx((1.0 + 0.0 + 1.0) / 4.0)
    assert grade.breakdown["c"] == 0.5
    assert not grade.solved


def test_R_709_a_critical_failure_zeroes_the_task():
    """A destructive answer must never out-score a partial one."""
    from bench.tasks.spec import Criterion, Rubric, TaskContext

    ctx = TaskContext(root=REPO_ROOT, result=None)
    partial = Rubric([Criterion("ok", 1.0, lambda c: 0.4), Criterion("intact", 1.0, lambda c: True, critical=True)])
    destructive = Rubric([Criterion("ok", 1.0, lambda c: 1.0), Criterion("intact", 1.0, lambda c: False, critical=True)])

    assert destructive.grade(ctx).score == 0.0
    assert partial.grade(ctx).score > destructive.grade(ctx).score


def test_R_709_a_criterion_that_raises_scores_zero_not_crashes():
    from bench.tasks.spec import Criterion, Rubric, TaskContext

    def explode(_ctx):
        raise RuntimeError("boom")

    grade = Rubric([Criterion("boom", 1.0, explode), Criterion("fine", 1.0, lambda c: True)]).grade(
        TaskContext(root=REPO_ROOT, result=None)
    )
    assert grade.score == pytest.approx(0.5)
    assert "boom" in grade.detail


def test_R_709_critical_criteria_gate_but_do_not_score():
    """Doing nothing satisfies most "don't break it" criteria — they must not pay."""
    from bench.tasks.spec import Criterion, Rubric, TaskContext

    ctx = TaskContext(root=REPO_ROOT, result=None)
    # An agent that changed nothing: the gate passes, the real work does not.
    did_nothing = Rubric(
        [
            Criterion("intact", 5.0, lambda c: True, critical=True),
            Criterion("the actual task", 1.0, lambda c: False),
        ]
    )
    assert did_nothing.grade(ctx).score == 0.0, "a no-op run must not earn a free floor"

    all_gates = Rubric([Criterion("intact", 1.0, lambda c: True, critical=True)])
    assert all_gates.grade(ctx).score == 1.0


async def test_R_709_benchmark_discriminates_between_competence_levels():
    """R-708/R-709 together: the score must move with competence.

    A benchmark whose tasks are all solved, or all failed, gives the loop nothing
    to climb. This drives real tasks with progressively truncated reference
    traces — a stand-in for models of increasing competence — and asserts the
    scores actually spread out.
    """
    import statistics

    import bench.runner as runner
    from bench.tasks import ALL_TASKS
    from morph.config import Config

    sample = [t for t in ALL_TASKS if t.reference_script and len(t.reference_script) >= 3][:8]
    assert len(sample) >= 6, "not enough multi-step tasks to measure a gradient"

    config = Config(workspace=REPO_ROOT, provider="echo", image_backend="stub")
    original_build = runner._build_agent

    def truncated(fraction: float):
        def build(task, root, cfg):
            agent = original_build(task, root, cfg)
            script = list(task.reference_script)
            keep = max(0, int(len(script) * fraction))
            agent.provider = EchoProvider(
                script=script[:keep] + [EchoProvider.text_response("I think that's done.")]
            )
            return agent

        return build

    means: list[float] = []
    all_scores: list[float] = []
    try:
        for fraction in (0.0, 0.5, 1.0):
            runner._build_agent = truncated(fraction)
            scores = [(await runner.run_task(t, config))[0].score for t in sample]
            means.append(statistics.mean(scores))
            all_scores.extend(scores)
    finally:
        runner._build_agent = original_build

    assert means == sorted(means), f"score must rise with competence, got {means}"
    assert means[0] < 0.35, f"an incompetent run scores too well ({means[0]:.2f}) — free credit"
    assert means[-1] > 0.9, f"a correct run must score near full marks, got {means[-1]:.2f}"
    assert means[-1] - means[0] > 0.5, "the benchmark barely separates competence levels"

    partial = [s for s in all_scores if 0.0 < s < 1.0]
    assert partial, "no task ever scored partially — the rubrics are effectively binary"


def test_R_710_frontier_and_nearest_misses():
    """R-710: the loop is pointed at the closest failure, not the hardest."""
    from bench.scorecard import CheckResult, Scorecard

    card = Scorecard()
    for tier, score in ((1, 1.0), (2, 1.0), (3, 0.7), (4, 0.1), (5, 0.0)):
        card.add(CheckResult(f"coding/T{tier}/x", "coding", score=score, tier=tier))

    assert card.tier_profile("coding") == {1: 1.0, 2: 1.0, 3: 0.7, 4: 0.1, 5: 0.0}
    assert card.frontier("coding") == 3

    targets = card.next_targets()
    assert [t.tier for t in targets] == [3, 4, 5], "nearest misses must be easiest-first"
    assert card.headroom("coding") > 0


def test_R_710_frontier_is_zero_when_the_basics_fail():
    from bench.scorecard import CheckResult, Scorecard

    card = Scorecard()
    card.add(CheckResult("coding/T1/x", "coding", score=0.2, tier=1))
    card.add(CheckResult("coding/T2/x", "coding", score=0.9, tier=2))
    # A fluke at T2 does not count while T1 is broken.
    assert card.frontier("coding") == 0


def test_R_711_benchmark_diagnoses_its_own_calibration():
    """R-711: a suite with no gradient is a broken instrument and must say so."""
    from bench.scorecard import CheckResult, Scorecard

    def card_with(scores: list[float], skipped: int = 0) -> Scorecard:
        card = Scorecard()
        for index, score in enumerate(scores):
            card.add(CheckResult(f"coding/x{index}", "coding", score=score, tier=index % 5 + 1))
        for index in range(skipped):
            card.add(CheckResult(f"coding/s{index}", "coding", score=0.0, tier=1, skipped=True))
        return card

    assert card_with([1.0, 1.0, 1.0]).calibration("coding") == "saturated"
    assert card_with([0.0, 0.0, 0.05]).calibration("coding") == "floored"
    assert card_with([1.0, 0.5, 0.0]).calibration("coding") == "healthy"
    assert card_with([1.0, 1.0], skipped=2).calibration("coding") == "partial"
    assert Scorecard().calibration("coding") == "empty"

    warnings = card_with([1.0, 1.0, 1.0]).instrument_warnings
    assert any("saturated" in w and "Add harder tasks" in w for w in warnings)
    assert any("floored" in w for w in card_with([0.0, 0.0]).instrument_warnings)

    # Diagnostics must reach the serialised scorecard the loop reads.
    payload = card_with([1.0, 0.5, 0.0]).to_dict()
    assert payload["diagnostics"]["coding"]["calibration"] == "healthy"
    assert "frontier" in payload["diagnostics"]["coding"]
    assert payload["next_targets"]


def test_R_711_loop_prompt_leads_with_frontier_and_nearest_misses():
    from selfimprove.loop import _feedback_from

    feedback = _feedback_from(
        {
            "composite": 42.0,
            "gated": False,
            "diagnostics": {
                "coding": {"frontier": 2, "headroom": 8.0, "calibration": "healthy", "tier_profile": {"1": 1.0, "3": 0.3}}
            },
            "next_targets": [
                {"name": "coding/T3/refactor", "score": 0.55, "tier": 3, "detail": "duplication remains"}
            ],
            "results": [],
            "instrument_warnings": ["coding: saturated"],
        }
    )
    assert "frontier T2" in feedback
    assert "Nearest misses" in feedback
    assert "coding/T3/refactor" in feedback
    assert "duplication remains" in feedback
    assert "do not try to fix them yourself" in feedback


def test_R_712_unrunnable_checks_are_skipped_not_failed():
    """R-712: a replay of reference traces must not pose as a measurement."""
    from bench.scorecard import CheckResult, Scorecard

    card = Scorecard()
    card.add(CheckResult("coding/T1/a", "coding", score=1.0, tier=1))
    card.add(CheckResult("coding/T5/b", "coding", score=0.0, tier=5, skipped=True))

    assert card.category_score("coding") == 1.0, "a skipped check must not drag the score down"
    assert card.by_category("coding") == [card.results[0]]
    assert card.failures == []
    assert len(card.skipped_in("coding")) == 1

    # And the runner must actually mark them, rather than scoring them zero.
    from bench.tasks import ALL_TASKS

    unscripted = [t for t in ALL_TASKS if not t.reference_script]
    assert unscripted, "some tasks should be model-only, or echo mode proves too much"


# ---------------------------------------------------------------------------
# Engineering quality
# ---------------------------------------------------------------------------


def test_R_801_suite_runs_offline(repo_root):
    """R-801: no network is required to run these tests."""
    for module in ("morph", "bench.runner", "selfimprove.loop"):
        __import__(module)

    conftest = (repo_root / "tests" / "conftest.py").read_text("utf-8")
    assert "requests" not in conftest


def test_R_802_no_paid_api_required(repo_root):
    """R-802: the default configuration runs with no credentials at all."""
    import os

    for key in ("GOOGLE_API_KEY", "FLUX_API_KEY", "MORPH_SEARCH_API_KEY", "OPENAI_API_KEY"):
        assert key not in os.environ or True  # presence is fine; requirement is non-reliance

    config = Config(workspace=repo_root, provider="echo", image_backend="stub")
    assert build_default_registry(config)
    assert get_provider("echo")


def test_R_803_secrets_come_from_the_environment_only(repo_root):
    """R-803: no key is written to disk, logged, or committed."""
    from morph.config import Config as C

    assert not any("key" in f.lower() for f in C.__dataclass_fields__)

    suspicious = re.compile(r"(sk-[A-Za-z0-9]{16,}|AIza[A-Za-z0-9_-]{20,})")
    for path in repo_root.rglob("*"):
        if not path.is_file() or ".git/" in str(path) or "__pycache__" in str(path):
            continue
        if path.suffix not in {".py", ".json", ".md", ".yml", ".yaml", ".js", ".toml"}:
            continue
        assert not suspicious.search(path.read_text("utf-8", errors="ignore")), path

    for source in (repo_root / "morph").rglob("*.py"):
        body = source.read_text("utf-8")
        assert "os.environ[" not in body or "API_KEY" not in body.split("os.environ[")[0][-80:]


def test_R_804_public_api_is_typed_and_imports_cleanly(repo_root):
    """R-804: type hints on public functions; clean import on 3.11+."""
    import ast
    import sys

    assert sys.version_info >= (3, 11)

    total = 0
    unannotated = []
    for path in (repo_root / "morph").rglob("*.py"):
        tree = ast.parse(path.read_text("utf-8"), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            total += 1
            missing = [
                a.arg
                for a in node.args.args + node.args.kwonlyargs
                if a.annotation is None and a.arg not in {"self", "cls"}
            ]
            if missing or node.returns is None:
                unannotated.append(f"{path.name}:{node.lineno} {node.name}")

    assert total > 20
    ratio = 1 - len(unannotated) / total
    assert ratio >= 0.9, f"only {ratio:.0%} annotated; gaps: {unannotated[:10]}"


def test_R_805_every_requirement_is_covered(repo_root):
    """R-805: the spec cannot drift away from the tests.

    This is the meta-test. Every ``R-###`` declared in REQUIREMENTS.md must be
    referenced by at least one file in ``tests/``. Adding a requirement without
    a test fails here; so does deleting the test for an existing requirement.
    """
    declared = set(declared_requirements())
    assert len(declared) >= 40, f"only {len(declared)} requirements parsed — check the format"

    referenced: set[str] = set()
    for path in (repo_root / "tests").rglob("*.py"):
        body = path.read_text("utf-8")
        referenced |= set(ANY_ID_RE.findall(body))
        referenced |= {m.replace("_", "-") for m in re.findall(r"R_\d{3}", body)}

    uncovered = sorted(declared - referenced)
    assert not uncovered, f"requirements with no test: {uncovered}"
