"""MCP suite — can the agent use tools it did not ship with?

Every task here connects to a real MCP server over stdio (``mcp_fixture.py``)
and grades what the agent does with tools it has never seen before: discovering
them, respecting a schema it was handed at runtime, chaining stateful calls,
retrying a transient failure, and carrying on when a server dies mid-run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from morph.config import MCPServerConfig
from morph.llm.echo import EchoProvider

from .spec import Rubric, Task, Tier, answer_mentions, behaviour, within_steps

call = EchoProvider.call
say = EchoProvider.text_response

FIXTURE = Path(__file__).resolve().parent / "mcp_fixture.py"


def _server(
    name: str = "bench",
    tools: list[str] | None = None,
    flaky_n: int = 1,
    die_after: int = 0,
):
    """Build the per-workspace MCP server spec for a task."""

    def factory(root: Path) -> list[MCPServerConfig]:
        env = {
            "MCP_STATE": str(root / ".mcp-state.json"),
            "MCP_FLAKY_N": str(flaky_n),
            "PYTHONUNBUFFERED": "1",
        }
        if tools:
            env["MCP_TOOLS"] = ",".join(tools)
        if die_after:
            env["MCP_DIE_AFTER"] = str(die_after)
        return [
            MCPServerConfig(
                name=name,
                transport="stdio",
                command=sys.executable,
                args=[str(FIXTURE)],
                env=env,
                timeout=30.0,
            )
        ]

    return factory


def _state(ctx) -> dict:
    path = ctx.root / ".mcp-state.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

DISCOVER = Task(
    name="discover-mcp-tools",
    category="mcp",
    tier=Tier.TRIVIAL,
    prompt=(
        "You are connected to an MCP server called 'bench'. Store the value "
        "'morph' under the key 'project' using its key-value tool."
    ),
    mcp_servers=_server(tools=["kv_set", "kv_get", "kv_keys"]),
    rubric=Rubric(
        [
            behaviour(
                "called the namespaced MCP tool",
                lambda c: c.used("mcp__bench__kv_set"),
                weight=3.0,
                critical=True,
            ),
            behaviour(
                "the value actually landed in the store",
                lambda c: _state(c).get("project") == "morph",
                weight=3.0,
            ),
            within_steps(3, weight=1.0),
        ]
    ),
    budget_steps=6,
    requirement_ids=["R-401", "R-402"],
    reference_script=[
        call("mcp__bench__kv_set", key="project", value="morph"),
        say("Stored 'morph' under the key 'project'."),
    ],
)


# ---------------------------------------------------------------------------
# T2
# ---------------------------------------------------------------------------

SCHEMA_CONFORMANCE = Task(
    name="respect-a-runtime-schema",
    category="mcp",
    tier=Tier.BASIC,
    prompt=(
        "Using the MCP server's unit conversion tool, convert 5 kilometres to "
        "miles. Report the number it gives you."
    ),
    mcp_servers=_server(tools=["convert_units"]),
    rubric=Rubric(
        [
            behaviour(
                "called the conversion tool",
                lambda c: c.used("mcp__bench__convert_units"),
                weight=2.0,
                critical=True,
            ),
            behaviour(
                "used the enum values the schema declares",
                lambda c: any(
                    x.get("arguments", {}).get("unit_from") == "km"
                    and x.get("arguments", {}).get("unit_to") == "mi"
                    for x in c.calls_to("mcp__bench__convert_units")
                ),
                weight=3.0,
            ),
            behaviour(
                "the call succeeded",
                lambda c: any(x.get("ok") for x in c.calls_to("mcp__bench__convert_units")),
                weight=2.0,
            ),
            answer_mentions("3.1", weight=2.0),
        ]
    ),
    budget_steps=8,
    requirement_ids=["R-402", "R-207"],
    reference_script=[
        call("mcp__bench__convert_units", value=5, unit_from="km", unit_to="mi"),
        say("5 km is 3.1069 miles."),
    ],
)


# ---------------------------------------------------------------------------
# T3
# ---------------------------------------------------------------------------

CHAIN_STATEFUL = Task(
    name="chain-stateful-mcp-calls",
    category="mcp",
    tier=Tier.INTERMEDIATE,
    prompt=(
        "Using the MCP key-value tools: store 'alpha' under key 'one' and 'beta' "
        "under key 'two'. Then list the keys and read back whichever key sorts "
        "last alphabetically. Tell me its value."
    ),
    mcp_servers=_server(tools=["kv_set", "kv_get", "kv_keys"]),
    rubric=Rubric(
        [
            behaviour(
                "both values stored",
                lambda c: _state(c).get("one") == "alpha" and _state(c).get("two") == "beta",
                weight=3.0,
                critical=True,
            ),
            behaviour("listed the keys", lambda c: c.used("mcp__bench__kv_keys"), weight=2.0),
            behaviour("read a key back", lambda c: c.used("mcp__bench__kv_get"), weight=2.0),
            behaviour(
                "read back the right key ('two')",
                lambda c: any(
                    x.get("arguments", {}).get("key") == "two"
                    for x in c.calls_to("mcp__bench__kv_get")
                ),
                weight=2.0,
            ),
            answer_mentions("beta", weight=2.0),
        ]
    ),
    budget_steps=12,
    requirement_ids=["R-402", "R-101"],
    reference_script=[
        call("mcp__bench__kv_set", key="one", value="alpha"),
        call("mcp__bench__kv_set", key="two", value="beta"),
        call("mcp__bench__kv_keys"),
        call("mcp__bench__kv_get", key="two"),
        say("The last key alphabetically is 'two', and its value is beta."),
    ],
)


MIX_NATIVE_AND_MCP = Task(
    name="mix-native-and-mcp-tools",
    category="mcp",
    tier=Tier.INTERMEDIATE,
    prompt=(
        "distances.txt contains one number: a distance in kilometres. Read it, "
        "convert it to miles using the MCP conversion tool, and write the result "
        "into miles.txt."
    ),
    files={"distances.txt": "42\n"},
    mcp_servers=_server(tools=["convert_units"]),
    rubric=Rubric(
        [
            behaviour("read the input with a native tool", lambda c: c.used("read_file"), weight=2.0),
            behaviour(
                "converted via MCP rather than doing the arithmetic itself",
                lambda c: c.used("mcp__bench__convert_units"),
                weight=3.0,
                critical=True,
            ),
            behaviour(
                "passed the value it actually read",
                lambda c: any(
                    float(x.get("arguments", {}).get("value", 0)) == 42.0
                    for x in c.calls_to("mcp__bench__convert_units")
                ),
                weight=2.0,
            ),
            behaviour(
                "miles.txt holds roughly 26.1",
                lambda c: _approx_in(c.read("miles.txt"), 26.0976, tolerance=0.05),
                weight=3.0,
            ),
        ]
    ),
    budget_steps=12,
    requirement_ids=["R-402", "R-202"],
    reference_script=[
        call("read_file", path="distances.txt"),
        call("mcp__bench__convert_units", value=42, unit_from="km", unit_to="mi"),
        call("write_file", path="miles.txt", content="26.0976\n"),
        say("42 km is 26.0976 miles; written to miles.txt."),
    ],
)


def _approx_in(text: str, expected: float, tolerance: float) -> bool:
    import re

    for match in re.finditer(r"-?\d+\.?\d*", text):
        try:
            if abs(float(match.group()) - expected) <= tolerance:
                return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# T4
# ---------------------------------------------------------------------------

RETRY_TRANSIENT = Task(
    name="retry-a-transient-mcp-failure",
    category="mcp",
    tier=Tier.HARD,
    prompt=(
        "Fetch record 'r-1' using the MCP fetch tool and tell me its owner. "
        "The upstream is unreliable — if the call fails with a transient error, "
        "retry it."
    ),
    mcp_servers=_server(tools=["flaky_fetch"], flaky_n=1),
    rubric=Rubric(
        [
            behaviour(
                "called the fetch tool",
                lambda c: c.used("mcp__bench__flaky_fetch"),
                weight=2.0,
                critical=True,
            ),
            behaviour(
                "retried after the failure",
                lambda c: c.call_count("mcp__bench__flaky_fetch") >= 2,
                weight=4.0,
            ),
            behaviour(
                "eventually succeeded",
                lambda c: any(x.get("ok") for x in c.calls_to("mcp__bench__flaky_fetch")),
                weight=3.0,
            ),
            answer_mentions("morph", weight=2.0),
            within_steps(6, weight=1.0),
        ]
    ),
    budget_steps=12,
    requirement_ids=["R-109", "R-403"],
    reference_script=[
        call("mcp__bench__flaky_fetch", record_id="r-1"),  # fails once
        call("mcp__bench__flaky_fetch", record_id="r-1"),
        say("Retried after the 503; the owner is morph."),
    ],
)


SURVIVE_DEATH = Task(
    name="survive-an-mcp-server-death",
    category="mcp",
    tier=Tier.HARD,
    prompt=(
        "Store 'first' under key 'a' using the MCP tool, then store 'second' "
        "under key 'b'. If the MCP server stops responding, do not keep retrying "
        "it — write whatever you managed to store into progress.txt and tell me "
        "what happened."
    ),
    mcp_servers=_server(tools=["kv_set", "kv_get"], die_after=1),
    rubric=Rubric(
        [
            behaviour(
                "the run did not error out",
                lambda c: getattr(c.result, "error", None) is None,
                weight=3.0,
                critical=True,
            ),
            behaviour(
                "did not spin on the dead server",
                lambda c: max(0.0, 1.0 - max(0, c.call_count("mcp__bench__kv_set") - 3) / 4.0),
                weight=2.0,
            ),
            behaviour(
                "wrote progress.txt with a native tool",
                lambda c: c.exists("progress.txt"),
                weight=3.0,
            ),
            behaviour(
                "reported the failure honestly",
                lambda c: c.mentions("fail", "unavailable", "stopped", "error", "disconnect", "crash"),
                weight=2.0,
            ),
        ]
    ),
    budget_steps=14,
    requirement_ids=["R-403", "R-109"],
)


# ---------------------------------------------------------------------------
# T5
# ---------------------------------------------------------------------------

TWO_SERVERS = Task(
    name="two-servers-one-answer",
    category="mcp",
    tier=Tier.EXPERT,
    prompt=(
        "Two MCP servers are connected: 'store' and 'units'. Read the number in "
        "route.txt (kilometres), convert it to feet with the units server, and "
        "save the result in the store server under the key 'route_ft'. Then read "
        "it back from the store to confirm, and tell me the value."
    ),
    files={"route.txt": "3\n"},
    mcp_servers=lambda root: (
        _server(name="store", tools=["kv_set", "kv_get", "kv_keys"])(root)
        + _server(name="units", tools=["convert_units"])(root)
    ),
    rubric=Rubric(
        [
            behaviour(
                "used the units server",
                lambda c: c.used("mcp__units__convert_units"),
                weight=2.0,
                critical=True,
            ),
            behaviour(
                "used the store server",
                lambda c: c.used("mcp__store__kv_set"),
                weight=2.0,
                critical=True,
            ),
            behaviour(
                "did not confuse the two namespaces",
                lambda c: not any(
                    t.startswith("mcp__units__kv") or t.startswith("mcp__store__convert")
                    for t in c.tools_used
                ),
                weight=2.0,
            ),
            behaviour(
                "stored roughly 9842.5 ft",
                lambda c: _approx_in(str(_state(c).get("route_ft", "")), 9842.5197, tolerance=1.0),
                weight=4.0,
            ),
            behaviour("read it back", lambda c: c.used("mcp__store__kv_get"), weight=2.0),
            answer_mentions("9842", "9,842", weight=2.0),
        ]
    ),
    budget_steps=20,
    budget_seconds=300.0,
    requirement_ids=["R-401", "R-402"],
)


TIMEOUT_DISCIPLINE = Task(
    name="mcp-timeout-discipline",
    category="mcp",
    tier=Tier.EXPERT,
    prompt=(
        "Try the MCP scan tool with pattern 'x'. It is known to hang. If it does "
        "not come back promptly, abandon it, fall back to searching the files "
        "yourself with the native tools, and tell me which file contains 'needle'."
    ),
    files={
        "hay1.txt": "nothing here\n",
        "hay2.txt": "still nothing\n",
        "hay3.txt": "the needle is here\n",
    },
    mcp_servers=_server(tools=["slow_scan"]),
    rubric=Rubric(
        [
            behaviour(
                "the run completed rather than hanging",
                lambda c: getattr(c.result, "stop_reason", "") != "error",
                weight=3.0,
                critical=True,
            ),
            behaviour("fell back to native search", lambda c: c.used("grep", "read_file"), weight=3.0),
            answer_mentions("hay3", weight=3.0),
            behaviour(
                "did not retry the hanging tool repeatedly",
                lambda c: c.call_count("mcp__bench__slow_scan") <= 1,
                weight=2.0,
            ),
        ]
    ),
    budget_steps=14,
    budget_seconds=180.0,
    requirement_ids=["R-403", "R-102"],
)


MCP_TASKS: list[Task] = [
    DISCOVER,
    SCHEMA_CONFORMANCE,
    CHAIN_STATEFUL,
    MIX_NATIVE_AND_MCP,
    RETRY_TRANSIENT,
    SURVIVE_DEATH,
    TWO_SERVERS,
    TIMEOUT_DISCIPLINE,
]
