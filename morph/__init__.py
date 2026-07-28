"""Morph — a self-hosted coding agent platform that improves itself with Gemma.

See ``REQUIREMENTS.md`` for the contract this package implements.
"""

from __future__ import annotations

__version__ = "0.1.1"

from .agent import Agent, AgentEvent, RunResult, run_once
from .config import Config, MCPServerConfig, load_config
from .session import Session, SessionStore
from .skills import Skill, SkillRegistry
from .tools import Tool, ToolError, ToolRegistry, ToolResult

__all__ = [
    "Agent",
    "AgentEvent",
    "Config",
    "MCPServerConfig",
    "RunResult",
    "Session",
    "SessionStore",
    "Skill",
    "SkillRegistry",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "__version__",
    "load_config",
    "run_once",
]
