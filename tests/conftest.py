"""Shared fixtures. Everything here works offline (R-801, R-802)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from morph.agent import Agent  # noqa: E402
from morph.config import Config  # noqa: E402
from morph.llm.echo import EchoProvider  # noqa: E402
from morph.skills import SkillRegistry  # noqa: E402
from morph.tools import build_default_registry  # noqa: E402


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An empty workspace root."""
    return tmp_path


@pytest.fixture
def config(workspace: Path) -> Config:
    return Config(
        workspace=workspace,
        provider="echo",
        model="test-model",
        image_backend="stub",
        max_steps=8,
        skill_paths=[workspace / "skills"],
    )


@pytest.fixture
def registry(config: Config):
    return build_default_registry(config)


@pytest.fixture
def make_agent(config: Config, registry):
    """Build an agent driven by a scripted, deterministic provider.

    No teardown needed: these agents have no MCP servers, so nothing to close.
    """

    def _make(script=None, reflexes=None, **overrides) -> Agent:
        for key, value in overrides.items():
            setattr(config, key, value)
        return Agent(
            config=config,
            provider=EchoProvider(script=script or [], reflexes=reflexes or []),
            tools=registry,
            skills=SkillRegistry(),
        )

    return _make


@pytest.fixture
def call():
    return EchoProvider.call


@pytest.fixture
def say():
    return EchoProvider.text_response


@pytest.fixture
def mcp_server_factory(tmp_path: Path):
    """Build an MCPServerConfig pointing at the real stdio server in tests/."""
    from morph.config import MCPServerConfig

    script = Path(__file__).resolve().parent / "fake_mcp_server.py"
    counter = {"n": 0}

    def _make(name: str = "fake", tools: list[str] | None = None, crash_after: int = 0):
        counter["n"] += 1
        log = tmp_path / f"mcp-log-{counter['n']}.json"
        env = {
            "FAKE_MCP_TOOLS": ",".join(tools or ["echo"]),
            "FAKE_MCP_LOG": str(log),
            "PYTHONUNBUFFERED": "1",
        }
        if crash_after:
            env["FAKE_MCP_CRASH"] = str(crash_after)
        return MCPServerConfig(
            name=name,
            transport="stdio",
            command=sys.executable,
            args=[str(script)],
            env=env,
            timeout=20.0,
        )

    return _make


class _LiveServer:
    def __init__(self, server, port: int) -> None:
        self.server = server
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"


@pytest.fixture
async def live_server(config: Config, registry):
    """A real MorphServer on an ephemeral port, backed by a scripted provider."""
    from morph.api import MorphAPI
    from morph.server import MorphServer

    provider = EchoProvider(
        script=[
            EchoProvider.call("list_dir", path="."),
            EchoProvider.text_response("Listed the directory."),
        ]
    )
    agent = Agent(
        config=config, provider=provider, tools=registry, skills=SkillRegistry()
    )
    server = MorphServer(MorphAPI(agent), webapp_dir=REPO_ROOT / "webapp")
    port = await server.start("127.0.0.1", 0)
    try:
        yield _LiveServer(server, port)
    finally:
        await server.stop()
