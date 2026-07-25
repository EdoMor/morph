"""Capability tasks: can the agent actually get work done? (30 points)

Each task ships a ``reference_script`` — the tool sequence a competent model
would emit. With ``echo`` that script replays, so CI measures whether the
*harness* can execute the work. With Gemma configured the script is ignored and
the score measures the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from morph.llm.echo import EchoProvider

from .types import AgentTask, TaskOutcome

call = EchoProvider.call
say = EchoProvider.text_response


# ---------------------------------------------------------------------------
# 1. Read, understand, fix
# ---------------------------------------------------------------------------

BUGGY = '''\
def average(values):
    """Return the mean of a list of numbers."""
    return sum(values) / len(values)
'''

FIXED = '''\
def average(values):
    """Return the mean of a list of numbers, or 0.0 when empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)
'''


def _verify_bugfix(root: Path, result: Any) -> TaskOutcome:
    source = (root / "calc.py").read_text("utf-8")
    namespace: dict[str, Any] = {}
    try:
        exec(compile(source, "calc.py", "exec"), namespace)  # noqa: S102 - benchmark fixture
    except SyntaxError as exc:
        return TaskOutcome.fail(f"calc.py no longer parses: {exc}")

    average = namespace.get("average")
    if not callable(average):
        return TaskOutcome.fail("average() was removed from calc.py")
    try:
        empty = average([])
    except ZeroDivisionError:
        return TaskOutcome.fail("average([]) still raises ZeroDivisionError")
    except Exception as exc:  # noqa: BLE001
        return TaskOutcome.fail(f"average([]) raised {type(exc).__name__}: {exc}")

    if empty != 0:
        return TaskOutcome.fail(f"average([]) returned {empty!r}, expected 0")
    if average([2, 4]) != 3:
        return TaskOutcome.fail("average([2, 4]) regressed; it must still return 3")
    return TaskOutcome.ok("empty-list guard added, existing behaviour preserved")


FIX_A_BUG = AgentTask(
    name="capability/fix-a-bug",
    prompt=(
        "calc.py has a bug: average([]) raises ZeroDivisionError. "
        "Read the file, then edit it so an empty list returns 0.0. "
        "Do not change the behaviour for non-empty lists."
    ),
    files={"calc.py": BUGGY},
    verify=_verify_bugfix,
    requirement_ids=["R-101", "R-202", "R-203"],
    reference_script=[
        call("read_file", path="calc.py"),
        call(
            "edit_file",
            path="calc.py",
            old_string="    return sum(values) / len(values)",
            new_string="    if not values:\n        return 0.0\n    return sum(values) / len(values)",
        ),
        say("Guarded the empty case in calc.py; non-empty behaviour is unchanged."),
    ],
    weight=2.0,
)


# ---------------------------------------------------------------------------
# 2. Create a file from a specification
# ---------------------------------------------------------------------------


def _verify_scaffold(root: Path, result: Any) -> TaskOutcome:
    target = root / "config" / "settings.json"
    if not target.is_file():
        return TaskOutcome.fail("config/settings.json was not created")
    try:
        data = json.loads(target.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        return TaskOutcome.fail(f"settings.json is not valid JSON: {exc}")
    missing = {"name", "version", "debug"} - set(data)
    if missing:
        return TaskOutcome.fail(f"settings.json is missing keys: {sorted(missing)}")
    if data.get("debug") is not False:
        return TaskOutcome.fail(f"'debug' must be false, got {data.get('debug')!r}")
    return TaskOutcome.ok("settings.json created with the required shape")


SCAFFOLD = AgentTask(
    name="capability/scaffold-file",
    prompt=(
        'Create config/settings.json containing exactly this JSON object: '
        '{"name": "morph", "version": "0.1.0", "debug": false}. '
        "Create the parent directory if needed."
    ),
    verify=_verify_scaffold,
    requirement_ids=["R-202", "R-205"],
    reference_script=[
        call(
            "write_file",
            path="config/settings.json",
            content='{\n  "name": "morph",\n  "version": "0.1.0",\n  "debug": false\n}\n',
        ),
        say("Wrote config/settings.json."),
    ],
)


# ---------------------------------------------------------------------------
# 3. Search a codebase and report
# ---------------------------------------------------------------------------

SEARCH_FILES = {
    "app/handlers.py": "def handle_login(request):\n    return legacy_auth(request)\n",
    "app/auth.py": "def legacy_auth(request):\n    return None  # TODO: replace\n",
    "app/util.py": "def slugify(text):\n    return text.lower().replace(' ', '-')\n",
}


def _verify_search(root: Path, result: Any) -> TaskOutcome:
    calls = [c["tool"] for c in getattr(result, "tool_calls", [])]
    if not any(c in {"grep", "glob"} for c in calls):
        return TaskOutcome.fail(f"agent never searched; tools used: {calls}")
    text = (getattr(result, "text", "") or "").lower()
    if "auth.py" not in text and "legacy_auth" not in text:
        return TaskOutcome.fail(f"answer does not identify the definition site: {text[:300]!r}")
    return TaskOutcome.ok("located legacy_auth via search and reported it")


SEARCH = AgentTask(
    name="capability/search-codebase",
    prompt=(
        "Where is legacy_auth defined in this repository? "
        "Search for it and answer with the file path."
    ),
    files=SEARCH_FILES,
    verify=_verify_search,
    requirement_ids=["R-202", "R-108"],
    reference_script=[
        call("grep", pattern="def legacy_auth", path="."),
        say("legacy_auth is defined in app/auth.py, and called from app/handlers.py."),
    ],
)


# ---------------------------------------------------------------------------
# 4. Run a command and interpret the result
# ---------------------------------------------------------------------------

TEST_FILE = '''\
from calc import average


def test_average():
    assert average([1, 2, 3]) == 2


def test_average_empty():
    assert average([]) == 0.0
'''


def _verify_shell(root: Path, result: Any) -> TaskOutcome:
    calls = [c["tool"] for c in getattr(result, "tool_calls", [])]
    if "shell" not in calls:
        return TaskOutcome.fail(f"agent never ran a command; tools used: {calls}")
    shell_results = [c for c in result.tool_calls if c["tool"] == "shell"]
    if not any(c["ok"] for c in shell_results):
        return TaskOutcome.fail("every shell invocation failed")
    text = (getattr(result, "text", "") or "").lower()
    if "2" not in text and "pass" not in text:
        return TaskOutcome.fail(f"agent did not report the outcome: {text[:300]!r}")
    return TaskOutcome.ok("ran the suite and reported the result")


RUN_TESTS = AgentTask(
    name="capability/run-tests",
    prompt=(
        "Run the test file test_calc.py with `python -m pytest test_calc.py -q` "
        "and tell me how many tests passed."
    ),
    files={"calc.py": FIXED, "test_calc.py": TEST_FILE},
    verify=_verify_shell,
    requirement_ids=["R-204", "R-109"],
    reference_script=[
        call("shell", command="python -m pytest test_calc.py -q"),
        say("Ran the suite: 2 tests passed."),
    ],
)


# ---------------------------------------------------------------------------
# 5. Generate an image
# ---------------------------------------------------------------------------


def _verify_image(root: Path, result: Any) -> TaskOutcome:
    images = list((root / ".morph" / "images").glob("*.png"))
    if not images:
        return TaskOutcome.fail("no image was written to .morph/images")
    blob = images[0].read_bytes()
    if blob[:8] != b"\x89PNG\r\n\x1a\n":
        return TaskOutcome.fail("output is not a valid PNG")
    if len(blob) < 100:
        return TaskOutcome.fail(f"PNG is implausibly small ({len(blob)} bytes)")
    previews = [
        c for c in getattr(result, "tool_calls", []) if c.get("meta", {}).get("previews")
    ]
    if not previews:
        return TaskOutcome.fail("no inline preview returned; a phone client needs one (R-504)")
    return TaskOutcome.ok(f"generated {len(images)} image(s) with previews")


GENERATE_IMAGE = AgentTask(
    name="capability/generate-image",
    prompt="Generate a 128x128 image of a lighthouse at dusk. Use seed 7.",
    verify=_verify_image,
    requirement_ids=["R-501", "R-503", "R-504"],
    reference_script=[
        call("generate_image", prompt="a lighthouse at dusk", width=128, height=128, seed=7),
        say("Generated the image and saved it under .morph/images."),
    ],
)


# ---------------------------------------------------------------------------
# 6. Use a skill
# ---------------------------------------------------------------------------

SKILL_MD = """\
---
name: changelog
description: House style for writing CHANGELOG entries.
---

# Changelog style

Entries are written as `- <verb in past tense> <what changed>`.
Always append to the `## Unreleased` section.
"""


def _verify_skill(root: Path, result: Any) -> TaskOutcome:
    calls = [c["tool"] for c in getattr(result, "tool_calls", [])]
    if "load_skill" not in calls:
        return TaskOutcome.fail(f"the changelog skill was never loaded; tools used: {calls}")
    changelog = root / "CHANGELOG.md"
    if not changelog.is_file():
        return TaskOutcome.fail("CHANGELOG.md was not written")
    body = changelog.read_text("utf-8")
    if "## Unreleased" not in body:
        return TaskOutcome.fail("CHANGELOG.md has no '## Unreleased' section")
    if not any(line.strip().startswith("- ") for line in body.splitlines()):
        return TaskOutcome.fail("CHANGELOG.md has no bullet entry")
    return TaskOutcome.ok("loaded the skill and followed its house style")


USE_SKILL = AgentTask(
    name="capability/use-skill",
    prompt=(
        "Using the changelog skill, create CHANGELOG.md with an Unreleased section "
        "recording that we added an empty-list guard to average()."
    ),
    files={"skills/changelog/SKILL.md": SKILL_MD},
    verify=_verify_skill,
    requirement_ids=["R-301", "R-302", "R-303"],
    reference_script=[
        call("load_skill", name="changelog"),
        call(
            "write_file",
            path="CHANGELOG.md",
            content="# Changelog\n\n## Unreleased\n\n- Added an empty-list guard to average().\n",
        ),
        say("Wrote CHANGELOG.md in the house style."),
    ],
)


# ---------------------------------------------------------------------------
# 7. Multi-step: change code, then prove it works
# ---------------------------------------------------------------------------


def _verify_multistep(root: Path, result: Any) -> TaskOutcome:
    source = (root / "strutil.py").read_text("utf-8")
    if "def titlecase" not in source:
        return TaskOutcome.fail("titlecase() was not added to strutil.py")
    namespace: dict[str, Any] = {}
    try:
        exec(compile(source, "strutil.py", "exec"), namespace)  # noqa: S102
    except SyntaxError as exc:
        return TaskOutcome.fail(f"strutil.py does not parse: {exc}")
    titlecase = namespace["titlecase"]
    if titlecase("hello world") != "Hello World":
        return TaskOutcome.fail(
            f"titlecase('hello world') returned {titlecase('hello world')!r}"
        )
    if titlecase("") != "":
        return TaskOutcome.fail("titlecase('') must return ''")
    steps = getattr(result, "steps", 0)
    if steps > 10:
        return TaskOutcome.fail(f"solved, but took {steps} steps (budget 10)")
    return TaskOutcome.ok(f"added titlecase() in {steps} steps")


MULTI_STEP = AgentTask(
    name="capability/multi-step",
    prompt=(
        "Add a function titlecase(text) to strutil.py that capitalises the first "
        "letter of every word. Empty input returns an empty string. "
        "Then read the file back to confirm it is correct."
    ),
    files={"strutil.py": "def slugify(text):\n    return text.lower().replace(' ', '-')\n"},
    verify=_verify_multistep,
    budget_steps=10,
    requirement_ids=["R-101", "R-102", "R-202"],
    reference_script=[
        call("read_file", path="strutil.py"),
        call(
            "edit_file",
            path="strutil.py",
            old_string="def slugify(text):",
            new_string=(
                "def titlecase(text):\n"
                "    if not text:\n"
                "        return ''\n"
                "    return ' '.join(word[:1].upper() + word[1:] for word in text.split(' '))\n"
                "\n"
                "\n"
                "def slugify(text):"
            ),
        ),
        call("read_file", path="strutil.py"),
        say("Added titlecase() to strutil.py and verified the file."),
    ],
    weight=2.0,
)


CAPABILITY_TASKS: list[AgentTask] = [
    FIX_A_BUG,
    SCAFFOLD,
    SEARCH,
    RUN_TESTS,
    GENERATE_IMAGE,
    USE_SKILL,
    MULTI_STEP,
]
