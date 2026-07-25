"""Skills suite — does the agent find, load and actually follow instructions?

The interesting failures here are specific. An agent can load a skill and ignore
it. It can follow a skill it happened to load while the right one sat unused. It
can load six skills "to be safe" and burn the context that lazy loading exists to
save. Each of those gets its own criterion, because each is a different fix.
"""

from __future__ import annotations

from morph.llm.echo import EchoProvider

from .spec import (
    Rubric,
    Task,
    Tier,
    answer_mentions,
    behaviour,
    file_exists,
    within_steps,
)

call = EchoProvider.call
say = EchoProvider.text_response


# ---------------------------------------------------------------------------
# Verifiers
#
# Defined before the tasks: `behaviour(...)` captures the function object at
# module-import time, not at grading time.
# ---------------------------------------------------------------------------


def _entries(ctx) -> list[str]:
    return [
        line.strip()[2:].strip()
        for line in ctx.read("CHANGELOG.md").splitlines()
        if line.strip().startswith("- ")
    ]


def _past_tense_entry(ctx) -> bool:
    verbs = ("added", "fixed", "changed", "removed", "guarded", "updated", "introduced")
    return any(entry.lower().startswith(verbs) for entry in _entries(ctx))


def _ends_with_period(ctx) -> bool:
    entries = _entries(ctx)
    return bool(entries) and all(entry.endswith(".") for entry in entries)


def _api_error(ctx):
    return ctx.module("endpoint.py")["get_user"]("missing", {})


def _api_error_shape(ctx) -> float:
    result = _api_error(ctx)
    if not isinstance(result, dict) or "error" not in result:
        return 0.0
    error = result["error"]
    if not isinstance(error, dict):
        return 0.3
    has_code = isinstance(error.get("code"), str)
    has_message = isinstance(error.get("message"), str)
    return 0.5 + 0.25 * has_code + 0.25 * has_message


def _api_error_code_style(ctx) -> bool:
    result = _api_error(ctx)
    code = (result or {}).get("error", {}).get("code", "")
    return isinstance(code, str) and bool(code) and code == code.upper() and " " not in code



def skill(name: str, description: str, body: str, allowed: str = "") -> str:
    tools = f"allowed-tools: {allowed}\n" if allowed else ""
    return f"---\nname: {name}\ndescription: {description}\n{tools}---\n\n{body}\n"


CHANGELOG_SKILL = skill(
    "changelog",
    "House style for CHANGELOG entries. Use when recording a user-visible change.",
    "# Changelog style\n\n"
    "- New work goes under `## Unreleased`.\n"
    "- One line per change, formatted exactly as `- <PastTenseVerb> <what changed>`.\n"
    "- Every entry must end with a full stop.\n"
    "- Never write a bare version number as a heading without a date.\n",
)

COMMIT_SKILL = skill(
    "commit-message",
    "House style for git commit messages. Use when writing a commit.",
    "# Commit style\n\n"
    "- Subject line is imperative mood, at most 50 characters, no trailing period.\n"
    "- Then a blank line, then a body wrapped at 72 columns.\n"
    "- The body explains why, not what.\n",
)

SQL_SKILL = skill(
    "sql-review",
    "Checklist for reviewing SQL migrations. Use when a .sql file changes.",
    "# SQL review checklist\n\n"
    "1. Every migration must be reversible — a `-- DOWN` section is mandatory.\n"
    "2. No `DROP COLUMN` without a two-release deprecation.\n"
    "3. Indexes are created `CONCURRENTLY`.\n",
)

API_SKILL = skill(
    "api-errors",
    "How this project reports API errors. Use when adding or changing an endpoint.",
    "# API error style\n\n"
    "Errors are returned as `{\"error\": {\"code\": <string>, \"message\": <string>}}`.\n"
    "The `code` is SCREAMING_SNAKE_CASE. Never return a bare string.\n"
    "HTTP status is always set to match the code.\n",
)

SECRET_MARKER_SKILL = skill(
    "report-format",
    "The required format for status reports. Use before writing any report file.",
    "# Report format\n\n"
    "Every report must begin with the exact line `STATUS-REPORT-V2` and then a\n"
    "blank line, before any other content. Reports without this header are\n"
    "rejected by the downstream parser.\n",
)


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

LOAD_A_SKILL = Task(
    name="load-a-skill",
    category="skills",
    tier=Tier.TRIVIAL,
    prompt="What does the 'commit-message' skill say the maximum subject length is?",
    skills={"commit-message": COMMIT_SKILL},
    rubric=Rubric(
        [
            behaviour("loaded the skill", lambda c: c.used("load_skill"), weight=3.0, critical=True),
            answer_mentions("50", weight=3.0),
            within_steps(3, weight=1.0),
        ]
    ),
    budget_steps=6,
    requirement_ids=["R-301", "R-302", "R-303"],
    reference_script=[
        call("load_skill", name="commit-message"),
        say("The commit-message skill caps the subject line at 50 characters."),
    ],
)


# ---------------------------------------------------------------------------
# T2
# ---------------------------------------------------------------------------

FOLLOW_A_SKILL = Task(
    name="follow-a-skill",
    category="skills",
    tier=Tier.BASIC,
    prompt=(
        "Using the changelog skill, create CHANGELOG.md recording that we added "
        "an empty-list guard to average()."
    ),
    skills={"changelog": CHANGELOG_SKILL},
    rubric=Rubric(
        [
            behaviour("loaded the skill", lambda c: c.used("load_skill"), weight=2.0),
            file_exists("CHANGELOG.md", weight=1.0, critical=True),
            behaviour(
                "has an Unreleased section",
                lambda c: "## Unreleased" in c.read("CHANGELOG.md"),
                weight=2.0,
            ),
            behaviour(
                "entry is a bullet",
                lambda c: any(
                    line.strip().startswith("- ") for line in c.read("CHANGELOG.md").splitlines()
                ),
                weight=2.0,
            ),
            behaviour("entry uses past tense", _past_tense_entry, weight=2.0),
            behaviour("entry ends with a full stop", _ends_with_period, weight=1.5),
            answer_mentions("changelog", weight=0.5),
        ]
    ),
    budget_steps=10,
    requirement_ids=["R-301", "R-303"],
    reference_script=[
        call("load_skill", name="changelog"),
        call(
            "write_file",
            path="CHANGELOG.md",
            content="# Changelog\n\n## Unreleased\n\n- Added an empty-list guard to average().\n",
        ),
        say("Wrote CHANGELOG.md following the changelog skill."),
    ],
)


# ---------------------------------------------------------------------------
# T3
# ---------------------------------------------------------------------------

CHOOSE_THE_RIGHT_SKILL = Task(
    name="choose-among-skills",
    category="skills",
    tier=Tier.INTERMEDIATE,
    prompt=(
        "Add an error response to endpoint.py for the case where the requested "
        "user does not exist. Follow this project's conventions."
    ),
    files={
        "endpoint.py": (
            "def get_user(user_id, users):\n"
            "    if user_id in users:\n"
            "        return {'user': users[user_id]}\n"
            "    return None\n"
        )
    },
    skills={
        "api-errors": API_SKILL,
        "changelog": CHANGELOG_SKILL,
        "commit-message": COMMIT_SKILL,
        "sql-review": SQL_SKILL,
    },
    rubric=Rubric(
        [
            behaviour("loaded a skill", lambda c: c.used("load_skill"), weight=1.5),
            behaviour(
                "loaded the *right* skill",
                lambda c: any(
                    x.get("arguments", {}).get("name") == "api-errors"
                    for x in c.calls_to("load_skill")
                ),
                weight=3.0,
            ),
            behaviour(
                "did not bulk-load every skill",
                lambda c: max(0.0, 1.0 - max(0, c.call_count("load_skill") - 1) / 3.0),
                weight=2.0,
            ),
            behaviour(
                "returns the documented error shape",
                _api_error_shape,
                weight=4.0,
            ),
            behaviour(
                "code is SCREAMING_SNAKE_CASE",
                _api_error_code_style,
                weight=2.0,
            ),
            behaviour(
                "the happy path still works",
                lambda c: c.module("endpoint.py")["get_user"]("u1", {"u1": "Ada"}) == {"user": "Ada"},
                weight=2.0,
                critical=True,
            ),
        ]
    ),
    budget_steps=14,
    requirement_ids=["R-302", "R-303"],
    reference_script=[
        call("load_skill", name="api-errors"),
        call("read_file", path="endpoint.py"),
        call(
            "edit_file",
            path="endpoint.py",
            old_string="    return None",
            new_string="    return {'error': {'code': 'USER_NOT_FOUND', 'message': f'No user {user_id}'}}",
        ),
        say("Added a USER_NOT_FOUND error following the api-errors skill."),
    ],
)


LAZY_LOADING = Task(
    name="respects-lazy-loading",
    category="skills",
    tier=Tier.INTERMEDIATE,
    prompt=(
        "Write a one-line summary of what this project does into summary.txt. "
        "Base it on README.md. You should not need any skill for this."
    ),
    files={"README.md": "# Widgets\n\nA library for assembling widgets from parts.\n"},
    skills={
        "changelog": CHANGELOG_SKILL,
        "commit-message": COMMIT_SKILL,
        "sql-review": SQL_SKILL,
        "api-errors": API_SKILL,
    },
    rubric=Rubric(
        [
            file_exists("summary.txt", weight=1.0, critical=True),
            behaviour(
                "summary mentions widgets",
                lambda c: "widget" in c.read("summary.txt").lower(),
                weight=3.0,
            ),
            behaviour("read the README", lambda c: c.used("read_file"), weight=1.5),
            behaviour(
                "loaded no skills",
                lambda c: max(0.0, 1.0 - c.call_count("load_skill") / 2.0),
                weight=3.0,
            ),
            within_steps(4, weight=1.5),
        ]
    ),
    budget_steps=8,
    requirement_ids=["R-303"],
    reference_script=[
        call("read_file", path="README.md"),
        call("write_file", path="summary.txt", content="A library for assembling widgets from parts.\n"),
        say("Summarised the README into summary.txt without loading any skills."),
    ],
)


# ---------------------------------------------------------------------------
# T4
# ---------------------------------------------------------------------------

OBEY_EXACT_INSTRUCTION = Task(
    name="obey-an-exact-instruction",
    category="skills",
    tier=Tier.HARD,
    prompt=(
        "Write a short status report to status.txt saying the build is green and "
        "two tests were added. Follow the project's report format."
    ),
    skills={
        "report-format": SECRET_MARKER_SKILL,
        "changelog": CHANGELOG_SKILL,
        "commit-message": COMMIT_SKILL,
    },
    rubric=Rubric(
        [
            behaviour(
                "loaded the report-format skill",
                lambda c: any(
                    x.get("arguments", {}).get("name") == "report-format"
                    for x in c.calls_to("load_skill")
                ),
                weight=2.0,
                critical=True,
            ),
            file_exists("status.txt", weight=1.0, critical=True),
            behaviour(
                "first line is the exact required header",
                lambda c: c.read("status.txt").splitlines()[:1] == ["STATUS-REPORT-V2"],
                weight=4.0,
            ),
            behaviour(
                "blank line after the header",
                lambda c: c.read("status.txt").splitlines()[1:2] == [""],
                weight=1.5,
            ),
            behaviour(
                "actually contains the report",
                lambda c: "green" in c.read("status.txt").lower(),
                weight=2.0,
            ),
        ]
    ),
    budget_steps=12,
    requirement_ids=["R-301", "R-303"],
    reference_script=[
        call("load_skill", name="report-format"),
        call(
            "write_file",
            path="status.txt",
            content="STATUS-REPORT-V2\n\nBuild is green. Two tests were added.\n",
        ),
        say("Wrote status.txt with the required STATUS-REPORT-V2 header."),
    ],
)


SKILL_PLUS_TOOLS = Task(
    name="skill-plus-tools",
    category="skills",
    tier=Tier.HARD,
    prompt=(
        "Fix the bug in math_utils.py (clamp() ignores the lower bound), then "
        "record the fix in CHANGELOG.md following the project's changelog style."
    ),
    files={
        "math_utils.py": (
            "def clamp(value, low, high):\n"
            "    if value > high:\n"
            "        return high\n"
            "    return value\n"
        ),
        "CHANGELOG.md": "# Changelog\n\n## Unreleased\n\n- Added clamp().\n",
    },
    skills={"changelog": CHANGELOG_SKILL, "commit-message": COMMIT_SKILL},
    rubric=Rubric(
        [
            behaviour(
                "clamp respects the lower bound",
                lambda c: c.module("math_utils.py")["clamp"](-5, 0, 10) == 0,
                weight=4.0,
            ),
            behaviour(
                "clamp still respects the upper bound",
                lambda c: c.module("math_utils.py")["clamp"](50, 0, 10) == 10,
                weight=2.0,
                critical=True,
            ),
            behaviour("loaded the changelog skill", lambda c: c.used("load_skill"), weight=1.5),
            behaviour(
                "appended rather than replaced the changelog",
                lambda c: "- Added clamp()." in c.read("CHANGELOG.md"),
                weight=2.0,
                critical=True,
            ),
            behaviour(
                "new entry describes the fix",
                lambda c: any(
                    "clamp" in e.lower() and ("fix" in e.lower() or "lower" in e.lower() or "bound" in e.lower())
                    for e in _entries(c)
                ),
                weight=3.0,
            ),
            behaviour("all entries end with a full stop", _ends_with_period, weight=1.5),
        ]
    ),
    budget_steps=16,
    requirement_ids=["R-301", "R-303", "R-203"],
    reference_script=[
        call("read_file", path="math_utils.py"),
        call(
            "edit_file",
            path="math_utils.py",
            old_string="    if value > high:\n        return high\n    return value",
            new_string=(
                "    if value > high:\n"
                "        return high\n"
                "    if value < low:\n"
                "        return low\n"
                "    return value"
            ),
        ),
        call("load_skill", name="changelog"),
        call(
            "edit_file",
            path="CHANGELOG.md",
            old_string="- Added clamp().",
            new_string="- Added clamp().\n- Fixed clamp() ignoring its lower bound.",
        ),
        say("Fixed the lower bound and recorded it in the changelog."),
    ],
)


# ---------------------------------------------------------------------------
# T5
# ---------------------------------------------------------------------------

CONFLICTING_SKILLS = Task(
    name="resolve-conflicting-guidance",
    category="skills",
    tier=Tier.EXPERT,
    prompt=(
        "Add a `-- DOWN` section to migration.sql that reverses the migration, "
        "and record the change in CHANGELOG.md. Follow the project's conventions "
        "for both. If two skills disagree, say so rather than silently picking one."
    ),
    files={
        "migration.sql": "-- UP\nALTER TABLE users ADD COLUMN nickname text;\n",
        "CHANGELOG.md": "# Changelog\n\n## Unreleased\n\n",
    },
    skills={
        "sql-review": SQL_SKILL,
        "changelog": CHANGELOG_SKILL,
        "commit-message": COMMIT_SKILL,
        "api-errors": API_SKILL,
    },
    rubric=Rubric(
        [
            behaviour(
                "loaded both relevant skills",
                lambda c: len(
                    {
                        x.get("arguments", {}).get("name")
                        for x in c.calls_to("load_skill")
                    }
                    & {"sql-review", "changelog"}
                )
                / 2.0,
                weight=3.0,
            ),
            behaviour(
                "migration has a DOWN section",
                lambda c: "-- DOWN" in c.read("migration.sql").upper()
                or "-- down" in c.read("migration.sql"),
                weight=3.0,
            ),
            behaviour(
                "the DOWN actually reverses the UP",
                lambda c: "DROP COLUMN" in c.read("migration.sql").upper()
                and "nickname" in c.read("migration.sql"),
                weight=3.0,
            ),
            behaviour(
                "kept the original UP",
                lambda c: "ADD COLUMN nickname" in c.read("migration.sql"),
                weight=2.0,
                critical=True,
            ),
            behaviour(
                "changelog entry added in house style",
                lambda c: bool(_entries(c)) and _ends_with_period(c) and _past_tense_entry(c),
                weight=3.0,
            ),
            behaviour(
                "noticed the DROP COLUMN tension with the deprecation rule",
                lambda c: c.mentions("deprecat", "conflict", "tension", "two-release", "disagree"),
                weight=2.0,
            ),
        ]
    ),
    budget_steps=20,
    budget_seconds=300.0,
    requirement_ids=["R-302", "R-303"],
)


DISCOVER_UNPROMPTED = Task(
    name="discover-a-skill-unprompted",
    category="skills",
    tier=Tier.EXPERT,
    prompt=(
        "Record in the changelog that we removed the deprecated legacy_auth "
        "function. Get the formatting right."
    ),
    # The prompt never says "use a skill" — the agent has to notice one applies.
    files={"CHANGELOG.md": "# Changelog\n\n## Unreleased\n\n"},
    skills={
        "changelog": CHANGELOG_SKILL,
        "commit-message": COMMIT_SKILL,
        "sql-review": SQL_SKILL,
        "api-errors": API_SKILL,
        "report-format": SECRET_MARKER_SKILL,
    },
    rubric=Rubric(
        [
            behaviour(
                "noticed the changelog skill applied",
                lambda c: any(
                    x.get("arguments", {}).get("name") == "changelog"
                    for x in c.calls_to("load_skill")
                ),
                weight=4.0,
            ),
            behaviour(
                "did not load unrelated skills",
                lambda c: max(0.0, 1.0 - max(0, c.call_count("load_skill") - 1) / 3.0),
                weight=2.0,
            ),
            behaviour("entry exists", lambda c: bool(_entries(c)), weight=2.0, critical=True),
            behaviour("past tense", _past_tense_entry, weight=2.0),
            behaviour("ends with a full stop", _ends_with_period, weight=2.0),
            behaviour(
                "mentions what was removed",
                lambda c: any("legacy_auth" in e for e in _entries(c)),
                weight=2.0,
            ),
        ]
    ),
    budget_steps=14,
    requirement_ids=["R-302", "R-303"],
)


SKILLS_TASKS: list[Task] = [
    LOAD_A_SKILL,
    FOLLOW_A_SKILL,
    CHOOSE_THE_RIGHT_SKILL,
    LAZY_LOADING,
    OBEY_EXACT_INSTRUCTION,
    SKILL_PLUS_TOOLS,
    CONFLICTING_SKILLS,
    DISCOVER_UNPROMPTED,
]
