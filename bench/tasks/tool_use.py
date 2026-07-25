"""Tool-use suite — does the agent drive its tools well?

Distinct from the coding suite: here the code change is incidental and what is
measured is the *calling* — picking the right tool, getting arguments right the
first time, recovering from an error, chaining results, and knowing when to stop
calling tools at all.

Several tasks grade restraint, which nothing else does. An agent that solves
everything in twenty calls is not solving it well, and without a criterion for
that the loop has no reason to get more efficient.
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
    no_failed_calls,
    still_parses,
    used_tool,
    within_steps,
)

call = EchoProvider.call
say = EchoProvider.text_response


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

PICK_THE_RIGHT_TOOL = Task(
    name="pick-the-right-tool",
    category="tool_use",
    tier=Tier.TRIVIAL,
    prompt="List the files in the current directory.",
    files={"a.txt": "a", "b.txt": "b", "sub/c.txt": "c"},
    rubric=Rubric(
        [
            used_tool("list_dir", "glob", weight=3.0),
            behaviour("did not shell out for it", lambda c: not c.used("shell"), weight=1.0),
            answer_mentions("a.txt", weight=1.5),
            within_steps(2, weight=1.0),
            no_failed_calls(weight=1.0),
        ]
    ),
    budget_steps=5,
    requirement_ids=["R-201", "R-202"],
    reference_script=[call("list_dir", path="."), say("The directory contains a.txt, b.txt and sub/.")],
)


NO_TOOL_NEEDED = Task(
    name="knows-when-not-to-call",
    category="tool_use",
    tier=Tier.TRIVIAL,
    prompt="What is 17 multiplied by 3? Answer directly — do not use any tools for this.",
    rubric=Rubric(
        [
            behaviour("called no tools", lambda c: not c.tool_calls, weight=4.0),
            answer_mentions("51", weight=3.0),
            within_steps(1, weight=1.0),
        ]
    ),
    budget_steps=4,
    requirement_ids=["R-101"],
    reference_script=[say("51.")],
)


# ---------------------------------------------------------------------------
# T2
# ---------------------------------------------------------------------------

ARGUMENT_PRECISION = Task(
    name="argument-precision",
    category="tool_use",
    tier=Tier.BASIC,
    prompt=(
        "In config.ini, change the line `timeout = 30` to `timeout = 60`. "
        "Change nothing else — the file has several similar-looking lines."
    ),
    files={
        "config.ini": (
            "[server]\n"
            "timeout = 30\n"
            "retries = 30\n"
            "\n"
            "[client]\n"
            "connect_timeout = 30\n"
        )
    },
    rubric=Rubric(
        [
            behaviour(
                "timeout is now 60",
                lambda c: "timeout = 60" in c.read("config.ini"),
                weight=3.0,
            ),
            behaviour(
                "retries untouched",
                lambda c: "retries = 30" in c.read("config.ini"),
                weight=2.0,
                critical=True,
            ),
            behaviour(
                "connect_timeout untouched",
                lambda c: "connect_timeout = 30" in c.read("config.ini"),
                weight=2.0,
                critical=True,
            ),
            behaviour("read before editing", lambda c: c.used("read_file", "grep"), weight=1.0),
            no_failed_calls(weight=1.0),
        ]
    ),
    budget_steps=8,
    requirement_ids=["R-203", "R-207"],
    reference_script=[
        call("read_file", path="config.ini"),
        call("edit_file", path="config.ini", old_string="timeout = 30\nretries", new_string="timeout = 60\nretries"),
        say("Changed the server timeout to 60; retries and connect_timeout untouched."),
    ],
)


CHAIN_TOOLS = Task(
    name="chain-tool-results",
    category="tool_use",
    tier=Tier.BASIC,
    prompt=(
        "Find the file that contains the string 'DEPRECATED', read it, and write "
        "its first line into found.txt."
    ),
    files={
        "one.py": "# fine\nvalue = 1\n",
        "two.py": "# DEPRECATED do not use\nvalue = 2\n",
        "three.py": "# also fine\nvalue = 3\n",
    },
    rubric=Rubric(
        [
            used_tool("grep", weight=2.0),
            file_exists("found.txt", weight=1.0, critical=True),
            behaviour(
                "found.txt holds the right line",
                lambda c: "DEPRECATED" in c.read("found.txt"),
                weight=3.0,
            ),
            behaviour(
                "used the search result rather than guessing",
                lambda c: c.used("read_file") or "two.py" in str(c.tool_calls),
                weight=1.5,
            ),
            no_failed_calls(weight=1.0),
        ]
    ),
    budget_steps=10,
    requirement_ids=["R-101", "R-202"],
    reference_script=[
        call("grep", pattern="DEPRECATED", path="."),
        call("read_file", path="two.py"),
        call("write_file", path="found.txt", content="# DEPRECATED do not use\n"),
        say("two.py was the deprecated file; wrote its first line to found.txt."),
    ],
)


# ---------------------------------------------------------------------------
# T3
# ---------------------------------------------------------------------------

RECOVER_FROM_ERROR = Task(
    name="recover-from-tool-error",
    category="tool_use",
    tier=Tier.INTERMEDIATE,
    prompt=(
        "Update the version string in version.py from 1.0.0 to 2.0.0. "
        "Note: the file may not be exactly where or how you expect — if a tool "
        "call fails, read the error and adapt."
    ),
    # The obvious first guess (version.py at the root) does not exist; it is in src/.
    files={
        "src/version.py": '__version__ = "1.0.0"\n',
        "README.md": "See src/version.py for the current version.\n",
    },
    rubric=Rubric(
        [
            behaviour(
                "version updated",
                lambda c: '"2.0.0"' in c.read("src/version.py"),
                weight=4.0,
            ),
            still_parses("src/version.py"),
            behaviour(
                "recovered rather than giving up",
                lambda c: 1.0 if not c.failed_calls else (1.0 if '"2.0.0"' in c.read("src/version.py") else 0.0),
                weight=2.0,
            ),
            behaviour(
                "did not create a stray version.py at the root",
                lambda c: not c.exists("version.py"),
                weight=2.0,
            ),
        ]
    ),
    budget_steps=12,
    requirement_ids=["R-109", "R-207"],
    reference_script=[
        call("read_file", path="version.py"),  # fails: no such file
        call("glob", pattern="**/version.py"),
        call("edit_file", path="src/version.py", old_string='"1.0.0"', new_string='"2.0.0"'),
        say("version.py was under src/; bumped it to 2.0.0."),
    ],
)


RESTRAINT = Task(
    name="restraint-under-budget",
    category="tool_use",
    tier=Tier.INTERMEDIATE,
    prompt=(
        "How many Python files are in this repository, and which one is largest? "
        "Answer efficiently — do not read every file."
    ),
    files={
        f"mod/file{i}.py": ("x = 1\n" * (i * 4 + 1)) for i in range(8)
    }
    | {"README.md": "docs\n"},
    rubric=Rubric(
        [
            answer_mentions("8", "eight", weight=3.0),
            behaviour("used a listing tool", lambda c: c.used("glob", "list_dir", "shell"), weight=1.5),
            behaviour(
                "did not read every file",
                lambda c: max(0.0, 1.0 - max(0, c.call_count("read_file") - 2) / 6.0),
                weight=3.0,
            ),
            within_steps(6, weight=2.0),
        ]
    ),
    budget_steps=12,
    requirement_ids=["R-108"],
    reference_script=[
        call("shell", command="ls mod/*.py | wc -l; wc -l mod/*.py | sort -n | tail -2"),
        say("There are 8 Python files; mod/file7.py is the largest."),
    ],
)


IMAGE_TOOL = Task(
    name="image-generation",
    category="tool_use",
    tier=Tier.INTERMEDIATE,
    prompt=(
        "Generate a 128x128 image of a lighthouse at dusk using seed 7, then tell "
        "me where you saved it."
    ),
    rubric=Rubric(
        [
            used_tool("generate_image", weight=2.0),
            behaviour(
                "a PNG was written",
                lambda c: any(
                    p.suffix == ".png" and p.read_bytes()[:4] == b"\x89PNG"
                    for p in (c.root / ".morph" / "images").glob("*.png")
                ),
                weight=3.0,
                critical=True,
            ),
            behaviour(
                "honoured the requested size",
                lambda c: any(
                    call_.get("arguments", {}).get("width") == 128
                    for call_ in c.calls_to("generate_image")
                ),
                weight=1.5,
            ),
            behaviour(
                "honoured the seed",
                lambda c: any(
                    call_.get("arguments", {}).get("seed") == 7
                    for call_ in c.calls_to("generate_image")
                ),
                weight=1.5,
            ),
            behaviour(
                "returned an inline preview",
                lambda c: any(call_.get("meta", {}).get("previews") for call_ in c.tool_calls),
                weight=1.5,
            ),
            answer_mentions(".morph/images", "images/", ".png", weight=1.0),
        ]
    ),
    budget_steps=8,
    requirement_ids=["R-501", "R-503", "R-504"],
    reference_script=[
        call("generate_image", prompt="a lighthouse at dusk", width=128, height=128, seed=7),
        say("Saved it under .morph/images/."),
    ],
)


# ---------------------------------------------------------------------------
# T4
# ---------------------------------------------------------------------------

AMBIGUOUS_EDIT = Task(
    name="disambiguate-an-edit",
    category="tool_use",
    tier=Tier.HARD,
    prompt=(
        "In handlers.py, the second occurrence of `return None` (the one inside "
        "handle_delete) should become `return 204`. Leave the first one alone."
    ),
    files={
        "handlers.py": (
            "def handle_get(request):\n"
            "    if not request:\n"
            "        return None\n"
            "    return 200\n"
            "\n\n"
            "def handle_delete(request):\n"
            "    if not request:\n"
            "        return None\n"
            "    return 200\n"
        )
    },
    rubric=Rubric(
        [
            still_parses("handlers.py"),
            behaviour(
                "handle_delete returns 204",
                lambda c: c.module("handlers.py")["handle_delete"](None) == 204,
                weight=4.0,
            ),
            behaviour(
                "handle_get still returns None",
                lambda c: c.module("handlers.py")["handle_get"](None) is None,
                weight=3.0,
                critical=True,
            ),
            behaviour(
                "did not blanket-replace",
                lambda c: c.read("handlers.py").count("return 204") == 1,
                weight=2.0,
            ),
            behaviour(
                "recovered if the first edit was ambiguous",
                lambda c: 1.0 if c.module("handlers.py")["handle_delete"](None) == 204 else 0.0,
                weight=1.0,
            ),
        ]
    ),
    budget_steps=14,
    requirement_ids=["R-203"],
    reference_script=[
        call("read_file", path="handlers.py"),
        call(
            "edit_file",
            path="handlers.py",
            old_string="def handle_delete(request):\n    if not request:\n        return None",
            new_string="def handle_delete(request):\n    if not request:\n        return 204",
        ),
        say("Disambiguated by including the enclosing function in the match."),
    ],
)


VERIFY_YOUR_WORK = Task(
    name="verify-your-work",
    category="tool_use",
    tier=Tier.HARD,
    prompt=(
        "Add a function is_palindrome(text) to words.py that ignores case, spaces "
        "and punctuation. Then verify it actually works by running it — do not "
        "just assert that it does."
    ),
    files={"words.py": "def word_count(text):\n    return len(text.split())\n"},
    rubric=Rubric(
        [
            still_parses("words.py"),
            behaviour(
                "handles a simple palindrome",
                lambda c: c.module("words.py")["is_palindrome"]("racecar") is True,
                weight=2.0,
            ),
            behaviour(
                "ignores case, spaces and punctuation",
                lambda c: c.module("words.py")["is_palindrome"]("A man, a plan, a canal: Panama") is True,
                weight=3.0,
            ),
            behaviour(
                "rejects a non-palindrome",
                lambda c: c.module("words.py")["is_palindrome"]("hello world") is False,
                weight=2.0,
            ),
            behaviour("actually executed it", lambda c: c.used("shell"), weight=3.0),
            behaviour(
                "word_count still works",
                lambda c: c.module("words.py")["word_count"]("a b c") == 3,
                weight=1.0,
                critical=True,
            ),
        ]
    ),
    budget_steps=16,
    requirement_ids=["R-204", "R-101"],
    reference_script=[
        call("read_file", path="words.py"),
        call(
            "edit_file",
            path="words.py",
            old_string="def word_count(text):",
            new_string=(
                "def is_palindrome(text):\n"
                "    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
                "    return cleaned == cleaned[::-1]\n"
                "\n"
                "\n"
                "def word_count(text):"
            ),
        ),
        call(
            "shell",
            command=(
                "python -c \"from words import is_palindrome as p; "
                "print(p('racecar'), p('A man, a plan, a canal: Panama'), p('hello world'))\""
            ),
        ),
        say("Added is_palindrome() and verified it prints True True False."),
    ],
)


# ---------------------------------------------------------------------------
# T5
# ---------------------------------------------------------------------------

MULTI_TOOL_INVESTIGATION = Task(
    name="multi-tool-investigation",
    category="tool_use",
    tier=Tier.EXPERT,
    prompt=(
        "Something in this project writes to a file called audit.log, but nobody "
        "knows what. Find every place that writes to it, work out which one runs "
        "in production (config.py decides), and disable only that one by setting "
        "its enabled flag to False. Report what you found."
    ),
    files={
        "config.py": "MODE = 'production'\nAUDIT_BACKEND = 'file'\n",
        "audit/file_backend.py": (
            "ENABLED = True\n\n\n"
            "def write(entry):\n"
            "    with open('audit.log', 'a') as handle:\n"
            "        handle.write(entry + chr(10))\n"
        ),
        "audit/null_backend.py": "ENABLED = False\n\n\ndef write(entry):\n    return None\n",
        "tools/debug_dump.py": (
            "# development helper, never runs in production\n"
            "ENABLED = True\n\n\n"
            "def dump(entries):\n"
            "    with open('audit.log', 'w') as handle:\n"
            "        handle.writelines(entries)\n"
        ),
        "audit/__init__.py": "",
    },
    rubric=Rubric(
        [
            behaviour("searched rather than guessed", lambda c: c.used("grep", "glob"), weight=2.0),
            behaviour("read config.py", lambda c: "config.py" in str(c.tool_calls), weight=1.5),
            behaviour(
                "disabled the file backend",
                lambda c: "ENABLED = False" in c.read("audit/file_backend.py"),
                weight=4.0,
            ),
            behaviour(
                "left the debug helper alone",
                lambda c: "ENABLED = True" in c.read("tools/debug_dump.py"),
                weight=3.0,
                critical=True,
            ),
            behaviour(
                "did not break the module",
                lambda c: c.parses("audit/file_backend.py") and "def write" in c.read("audit/file_backend.py"),
                weight=1.0,
                critical=True,
            ),
            answer_mentions("file_backend", "audit", weight=1.5),
        ]
    ),
    budget_steps=22,
    budget_seconds=300.0,
    requirement_ids=["R-101", "R-202"],
)


BUDGET_DISCIPLINE = Task(
    name="budget-discipline",
    category="tool_use",
    tier=Tier.EXPERT,
    prompt=(
        "There is exactly one TODO comment in this project. Find it and tell me "
        "the file and what it says. You have a tight step budget — be surgical."
    ),
    files={
        **{f"pkg/mod{i}.py": f"# module {i}\nvalue = {i}\n" for i in range(20)},
        "pkg/mod13.py": "# module 13\n# TODO: replace the placeholder rate table\nvalue = 13\n",
        "pkg/__init__.py": "",
    },
    rubric=Rubric(
        [
            answer_mentions("mod13", weight=3.0),
            answer_mentions("rate table", "placeholder", weight=2.0),
            behaviour("used search, not enumeration", lambda c: c.used("grep"), weight=3.0),
            behaviour(
                "kept read_file calls low",
                lambda c: max(0.0, 1.0 - max(0, c.call_count("read_file") - 1) / 4.0),
                weight=2.0,
            ),
            within_steps(4, weight=3.0),
        ]
    ),
    budget_steps=10,
    requirement_ids=["R-108", "R-102"],
)


TOOL_USE_TASKS: list[Task] = [
    PICK_THE_RIGHT_TOOL,
    NO_TOOL_NEEDED,
    ARGUMENT_PRECISION,
    CHAIN_TOOLS,
    RECOVER_FROM_ERROR,
    RESTRAINT,
    IMAGE_TOOL,
    AMBIGUOUS_EDIT,
    VERIFY_YOUR_WORK,
    MULTI_TOOL_INVESTIGATION,
    BUDGET_DISCIPLINE,
]
