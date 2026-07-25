"""Coding suite — can the agent actually change software correctly?

Twelve tasks from "write one file" to "find a bug that only shows up under a
specific input, across three modules". Rubrics are graded, and every task that
touches existing code carries a critical criterion for "it still parses" and
"existing behaviour is intact", so a destructive answer can never out-score a
partial one.
"""

from __future__ import annotations

from morph.llm.echo import EchoProvider

from .spec import (
    Criterion,
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
# Verifiers
#
# Defined before the tasks because `behaviour(...)` captures the function object
# at module-import time, not at grading time.
# ---------------------------------------------------------------------------


def _lru(ctx, capacity: int = 2):
    return ctx.module("cache.py")["LRUCache"](capacity)


def _lru_basic(ctx) -> bool:
    cache = _lru(ctx)
    cache.put("a", 1)
    return cache.get("a") == 1 and cache.get("zzz") is None


def _lru_recency(ctx) -> bool:
    cache = _lru(ctx)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' is now most recently used
    cache.put("c", 3)  # should evict 'b'
    return cache.get("a") == 1 and cache.get("b") is None


def _lru_capacity(ctx) -> bool:
    cache = _lru(ctx, 2)
    for key in ("a", "b", "c"):
        cache.put(key, key)
    return len(cache.data) == 2


def _lru_reput(ctx) -> bool:
    cache = _lru(ctx, 2)
    cache.put("a", 1)
    cache.put("a", 2)
    cache.put("b", 3)
    return cache.get("a") == 2 and cache.get("b") == 3



def _stage(ctx):
    namespace = ctx.module("stages.py")
    return namespace["Stage"](lambda v: v * 2)


def _pipeline_skips_malformed(ctx) -> bool:
    stage = _stage(ctx)
    try:
        stage.apply({"id": 1})  # no 'value'
    except Exception:  # noqa: BLE001 - any raise means it was not handled
        return False
    return True


def _pipeline_handles_valid(ctx) -> bool:
    stage = _stage(ctx)
    return stage.apply({"id": 1, "value": 21}) == 42


def _import_store(ctx):
    """Import the task's `store` package fresh from the workspace."""
    import sys

    sys.path.insert(0, str(ctx.root))
    try:
        for stale in [m for m in sys.modules if m == "store" or m.startswith("store.")]:
            del sys.modules[stale]
        import store.models as models
        import store.pricing as pricing
        import store.report as report

        return pricing, report, models
    finally:
        sys.path.remove(str(ctx.root))


def _gbp_rate(ctx) -> bool:
    pricing, _, _ = _import_store(ctx)
    return abs(pricing.convert(10, "GBP") - 8.0) < 1e-9


def _usd_rate(ctx) -> bool:
    pricing, _, _ = _import_store(ctx)
    return abs(pricing.convert(10, "USD") - 10.0) < 1e-9


def _unknown_currency(ctx) -> bool:
    pricing, _, _ = _import_store(ctx)
    try:
        pricing.convert(10, "XYZ")
    except ValueError:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _error_names_currency(ctx) -> bool:
    pricing, _, _ = _import_store(ctx)
    try:
        pricing.convert(10, "XYZ")
    except ValueError as exc:
        return "XYZ" in str(exc)
    except Exception:  # noqa: BLE001
        return False
    return False


def _summary(ctx):
    _, report, models = _import_store(ctx)
    orders = [models.Order(1, [10, 5]), models.Order(2, [20], currency="XYZ")]
    return report.summarise(orders)


def _summarise_degrades(ctx) -> bool:
    try:
        return _summary(ctx).get(2) is None
    except Exception:  # noqa: BLE001 - crashing the report is the failure mode
        return False


def _summarise_good(ctx) -> bool:
    try:
        return abs((_summary(ctx).get(1) or 0) - 15.0) < 1e-9
    except Exception:  # noqa: BLE001
        return False


def _is_linear(ctx) -> bool:
    """Verify by measurement, not by reading the source for `set`."""
    import time

    dedupe = ctx.module("dedupe.py")["dedupe"]
    data = list(range(6000)) * 2

    started = time.perf_counter()
    dedupe(data)
    elapsed = time.perf_counter() - started
    # The quadratic version takes seconds on 12k items; a linear one is ~1ms.
    return elapsed < 0.25


# ---------------------------------------------------------------------------
# T1 — trivial
# ---------------------------------------------------------------------------

WRITE_FILE = Task(
    name="write-a-file",
    category="coding",
    tier=Tier.TRIVIAL,
    prompt=(
        "Create a file greet.py containing a single function greet(name) that "
        "returns the string 'Hello, <name>!'."
    ),
    rubric=Rubric(
        [
            file_exists("greet.py", weight=1.0, critical=True),
            still_parses("greet.py"),
            behaviour(
                "greet('Ada') == 'Hello, Ada!'",
                lambda c: c.module("greet.py")["greet"]("Ada") == "Hello, Ada!",
                weight=3.0,
            ),
            within_steps(3, weight=0.5),
        ]
    ),
    budget_steps=6,
    requirement_ids=["R-202", "R-101"],
    reference_script=[
        call(
            "write_file",
            path="greet.py",
            content='def greet(name):\n    return f"Hello, {name}!"\n',
        ),
        say("Created greet.py."),
    ],
)


READ_AND_REPORT = Task(
    name="read-and-report",
    category="coding",
    tier=Tier.TRIVIAL,
    prompt="What is the default timeout in settings.py? Read the file and tell me the number.",
    files={"settings.py": "DEBUG = False\nDEFAULT_TIMEOUT = 42\nRETRIES = 3\n"},
    rubric=Rubric(
        [
            used_tool("read_file", "grep", weight=1.0),
            answer_mentions("42", weight=3.0),
            behaviour(
                "did not modify the file",
                lambda c: c.read("settings.py") == "DEBUG = False\nDEFAULT_TIMEOUT = 42\nRETRIES = 3\n",
                weight=1.0,
                critical=True,
            ),
        ]
    ),
    budget_steps=5,
    requirement_ids=["R-202"],
    reference_script=[
        call("read_file", path="settings.py"),
        say("The default timeout is 42."),
    ],
)


# ---------------------------------------------------------------------------
# T2 — basic
# ---------------------------------------------------------------------------

BUGGY_AVERAGE = '''\
def average(values):
    """Return the mean of a list of numbers."""
    return sum(values) / len(values)


def total(values):
    return sum(values)
'''

FIX_EDGE_CASE = Task(
    name="fix-edge-case",
    category="coding",
    tier=Tier.BASIC,
    prompt=(
        "calc.py has a bug: average([]) raises ZeroDivisionError. Fix it so an "
        "empty list returns 0.0. Do not change behaviour for non-empty lists."
    ),
    files={"calc.py": BUGGY_AVERAGE},
    rubric=Rubric(
        [
            still_parses("calc.py"),
            behaviour(
                "average([]) returns 0.0",
                lambda c: c.module("calc.py")["average"]([]) == 0.0,
                weight=3.0,
            ),
            behaviour(
                "average([2, 4]) still returns 3",
                lambda c: c.module("calc.py")["average"]([2, 4]) == 3,
                weight=2.0,
                critical=True,
            ),
            behaviour(
                "total() left intact",
                lambda c: c.module("calc.py")["total"]([1, 2]) == 3,
                weight=1.0,
                critical=True,
            ),
            used_tool("read_file", weight=0.5),
        ]
    ),
    budget_steps=8,
    requirement_ids=["R-202", "R-203"],
    reference_script=[
        call("read_file", path="calc.py"),
        call(
            "edit_file",
            path="calc.py",
            old_string="    return sum(values) / len(values)",
            new_string="    if not values:\n        return 0.0\n    return sum(values) / len(values)",
        ),
        say("Guarded the empty case; non-empty behaviour unchanged."),
    ],
)


ADD_FUNCTION = Task(
    name="add-function",
    category="coding",
    tier=Tier.BASIC,
    prompt=(
        "Add a function titlecase(text) to strutil.py that capitalises the first "
        "letter of every word, leaving the rest of each word alone. Empty input "
        "returns an empty string. Keep the existing slugify() working."
    ),
    files={"strutil.py": "def slugify(text):\n    return text.lower().replace(' ', '-')\n"},
    rubric=Rubric(
        [
            still_parses("strutil.py"),
            behaviour(
                "titlecase('hello world') == 'Hello World'",
                lambda c: c.module("strutil.py")["titlecase"]("hello world") == "Hello World",
                weight=3.0,
            ),
            behaviour(
                "titlecase('') == ''",
                lambda c: c.module("strutil.py")["titlecase"]("") == "",
                weight=1.5,
            ),
            behaviour(
                "does not lowercase the rest ('iPhone app' -> 'IPhone App')",
                lambda c: c.module("strutil.py")["titlecase"]("iPhone app") == "IPhone App",
                weight=1.0,
            ),
            behaviour(
                "slugify() still works",
                lambda c: c.module("strutil.py")["slugify"]("a b") == "a-b",
                weight=1.0,
                critical=True,
            ),
        ]
    ),
    budget_steps=8,
    requirement_ids=["R-202"],
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
                "    return ' '.join(w[:1].upper() + w[1:] for w in text.split(' '))\n"
                "\n"
                "\n"
                "def slugify(text):"
            ),
        ),
        say("Added titlecase() to strutil.py."),
    ],
)


SEARCH_CODEBASE = Task(
    name="search-codebase",
    category="coding",
    tier=Tier.BASIC,
    prompt="Where is legacy_auth defined in this repository? Search for it and answer with the file path.",
    files={
        "app/handlers.py": "from app.auth import legacy_auth\n\n\ndef handle_login(request):\n    return legacy_auth(request)\n",
        "app/auth.py": "def legacy_auth(request):\n    return None  # TODO: replace\n",
        "app/util.py": "def slugify(text):\n    return text.lower()\n",
        "docs/notes.md": "We should remove legacy_auth eventually.\n",
    },
    rubric=Rubric(
        [
            used_tool("grep", "glob", weight=1.5),
            answer_mentions("auth.py", weight=3.0),
            behaviour(
                "does not misreport handlers.py as the definition",
                lambda c: "defined in app/handlers.py" not in c.answer,
                weight=1.0,
            ),
            no_failed_calls(weight=0.5),
        ]
    ),
    budget_steps=8,
    requirement_ids=["R-202"],
    reference_script=[
        call("grep", pattern="def legacy_auth", path="."),
        say("legacy_auth is defined in app/auth.py and called from app/handlers.py."),
    ],
)


# ---------------------------------------------------------------------------
# T3 — intermediate
# ---------------------------------------------------------------------------

FAILING_SUITE = '''\
from inventory import restock, Item


def test_restock_adds():
    item = Item("bolt", 3)
    restock(item, 5)
    assert item.count == 8


def test_restock_rejects_negative():
    item = Item("bolt", 3)
    try:
        restock(item, -1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
'''

INVENTORY = '''\
class Item:
    def __init__(self, name, count):
        self.name = name
        self.count = count


def restock(item, amount):
    item.count += amount
'''

MAKE_TESTS_PASS = Task(
    name="make-tests-pass",
    category="coding",
    tier=Tier.INTERMEDIATE,
    prompt=(
        "Run `python -m pytest test_inventory.py -q`. One test fails. "
        "Fix inventory.py so both tests pass, then re-run to confirm."
    ),
    files={"inventory.py": INVENTORY, "test_inventory.py": FAILING_SUITE},
    rubric=Rubric(
        [
            still_parses("inventory.py"),
            behaviour("ran the suite", lambda c: c.used("shell"), weight=1.5),
            behaviour(
                "restock rejects negatives",
                lambda c: _raises_value_error(c),
                weight=3.0,
            ),
            behaviour(
                "restock still adds",
                lambda c: _restock_adds(c),
                weight=2.0,
                critical=True,
            ),
            behaviour(
                "did not edit the test file",
                lambda c: "expected ValueError" in c.read("test_inventory.py"),
                weight=2.0,
                critical=True,
            ),
            behaviour(
                "re-ran the suite after fixing",
                lambda c: c.call_count("shell") >= 2,
                weight=1.0,
            ),
        ]
    ),
    budget_steps=14,
    requirement_ids=["R-204", "R-109"],
    reference_script=[
        call("shell", command="python -m pytest test_inventory.py -q"),
        call("read_file", path="inventory.py"),
        call(
            "edit_file",
            path="inventory.py",
            old_string="def restock(item, amount):\n    item.count += amount",
            new_string=(
                "def restock(item, amount):\n"
                "    if amount < 0:\n"
                "        raise ValueError('amount must not be negative')\n"
                "    item.count += amount"
            ),
        ),
        call("shell", command="python -m pytest test_inventory.py -q"),
        say("Both tests pass now: restock rejects negative amounts."),
    ],
)


def _raises_value_error(ctx) -> bool:
    namespace = ctx.module("inventory.py")
    item = namespace["Item"]("bolt", 3)
    try:
        namespace["restock"](item, -1)
    except ValueError:
        return True
    return False


def _restock_adds(ctx) -> bool:
    namespace = ctx.module("inventory.py")
    item = namespace["Item"]("bolt", 3)
    namespace["restock"](item, 5)
    return item.count == 8


WRITE_TESTS = Task(
    name="write-tests",
    category="coding",
    tier=Tier.INTERMEDIATE,
    prompt=(
        "parser.py has a function parse_range('1-5') that returns (1, 5). "
        "Write test_parser.py covering the normal case, a single number '7' "
        "(which returns (7, 7)), and malformed input (which raises ValueError). "
        "Run the tests and make sure they pass."
    ),
    files={
        "parser.py": (
            "def parse_range(text):\n"
            "    if '-' in text:\n"
            "        lo, hi = text.split('-', 1)\n"
            "        return int(lo), int(hi)\n"
            "    if not text.strip().isdigit():\n"
            "        raise ValueError(f'bad range: {text!r}')\n"
            "    value = int(text)\n"
            "    return value, value\n"
        )
    },
    rubric=Rubric(
        [
            file_exists("test_parser.py", weight=1.0, critical=True),
            still_parses("test_parser.py"),
            behaviour(
                "covers the range case",
                lambda c: "1-5" in c.read("test_parser.py") or "parse_range" in c.read("test_parser.py"),
                weight=1.0,
            ),
            behaviour(
                "covers the single-number case",
                lambda c: "7" in c.read("test_parser.py"),
                weight=1.5,
            ),
            behaviour(
                "covers the error case",
                lambda c: "ValueError" in c.read("test_parser.py")
                or "raises" in c.read("test_parser.py"),
                weight=2.0,
            ),
            behaviour(
                "has at least three test functions",
                lambda c: min(c.read("test_parser.py").count("def test_") / 3.0, 1.0),
                weight=2.0,
            ),
            behaviour("ran the tests", lambda c: c.used("shell"), weight=1.5),
            behaviour(
                "did not weaken parser.py",
                lambda c: "raise ValueError" in c.read("parser.py"),
                weight=1.0,
                critical=True,
            ),
        ]
    ),
    budget_steps=14,
    requirement_ids=["R-202", "R-204"],
    reference_script=[
        call("read_file", path="parser.py"),
        call(
            "write_file",
            path="test_parser.py",
            content=(
                "import pytest\n\n"
                "from parser import parse_range\n\n\n"
                "def test_range():\n    assert parse_range('1-5') == (1, 5)\n\n\n"
                "def test_single():\n    assert parse_range('7') == (7, 7)\n\n\n"
                "def test_malformed():\n"
                "    with pytest.raises(ValueError):\n        parse_range('abc')\n"
            ),
        ),
        call("shell", command="python -m pytest test_parser.py -q"),
        say("Wrote three tests covering ranges, single values and malformed input. All pass."),
    ],
)


REFACTOR = Task(
    name="refactor-duplication",
    category="coding",
    tier=Tier.INTERMEDIATE,
    prompt=(
        "report.py has the same currency-formatting logic copy-pasted in three "
        "functions. Extract it into one helper and use it everywhere. The output "
        "of every function must stay byte-for-byte identical."
    ),
    files={
        "report.py": (
            "def line_total(amount):\n"
            '    return "$" + format(round(amount, 2), ",.2f")\n'
            "\n\n"
            "def subtotal(amounts):\n"
            '    return "$" + format(round(sum(amounts), 2), ",.2f")\n'
            "\n\n"
            "def grand_total(amounts, tax):\n"
            '    return "$" + format(round(sum(amounts) * (1 + tax), 2), ",.2f")\n'
        )
    },
    rubric=Rubric(
        [
            still_parses("report.py"),
            behaviour(
                "line_total unchanged",
                lambda c: c.module("report.py")["line_total"](1234.5) == "$1,234.50",
                weight=2.0,
                critical=True,
            ),
            behaviour(
                "subtotal unchanged",
                lambda c: c.module("report.py")["subtotal"]([1000, 234.5]) == "$1,234.50",
                weight=2.0,
                critical=True,
            ),
            behaviour(
                "grand_total unchanged",
                lambda c: c.module("report.py")["grand_total"]([100], 0.1) == "$110.00",
                weight=2.0,
                critical=True,
            ),
            behaviour(
                "duplication actually removed",
                lambda c: c.read("report.py").count('format(round') <= 1,
                weight=3.0,
            ),
            behaviour(
                "introduced a named helper",
                lambda c: c.read("report.py").count("def ") >= 4,
                weight=1.0,
            ),
        ]
    ),
    budget_steps=12,
    requirement_ids=["R-202", "R-203"],
    reference_script=[
        call("read_file", path="report.py"),
        call(
            "write_file",
            path="report.py",
            content=(
                "def money(amount):\n"
                '    return "$" + format(round(amount, 2), ",.2f")\n'
                "\n\n"
                "def line_total(amount):\n    return money(amount)\n"
                "\n\n"
                "def subtotal(amounts):\n    return money(sum(amounts))\n"
                "\n\n"
                "def grand_total(amounts, tax):\n    return money(sum(amounts) * (1 + tax))\n"
            ),
        ),
        say("Extracted money() and routed all three functions through it."),
    ],
)


# ---------------------------------------------------------------------------
# T4 — hard
# ---------------------------------------------------------------------------

CACHE_BUG = '''\
class LRUCache:
    """Least-recently-used cache."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.data = {}
        self.order = []

    def get(self, key):
        if key not in self.data:
            return None
        return self.data[key]

    def put(self, key, value):
        if key not in self.data and len(self.data) >= self.capacity:
            oldest = self.order.pop(0)
            del self.data[oldest]
        self.data[key] = value
        self.order.append(key)
'''

DEBUG_LRU = Task(
    name="debug-lru-eviction",
    category="coding",
    tier=Tier.HARD,
    prompt=(
        "cache.py implements an LRU cache, but it evicts the wrong entry: after "
        "filling it and then reading an old key, that key gets evicted anyway. "
        "Find the bug and fix it. Keep the public interface (get/put) the same."
    ),
    files={"cache.py": CACHE_BUG},
    rubric=Rubric(
        [
            still_parses("cache.py"),
            behaviour(
                "basic get/put still works",
                lambda c: _lru_basic(c),
                weight=2.0,
                critical=True,
            ),
            behaviour("reading a key protects it from eviction", _lru_recency, weight=4.0),
            behaviour("capacity is still enforced", _lru_capacity, weight=2.0),
            behaviour(
                "re-putting an existing key does not duplicate order entries",
                _lru_reput,
                weight=2.0,
            ),
            behaviour(
                "kept the public interface",
                lambda c: c.defines("cache.py", "LRUCache")
                and "def get" in c.read("cache.py")
                and "def put" in c.read("cache.py"),
                weight=1.0,
                critical=True,
            ),
        ]
    ),
    budget_steps=18,
    requirement_ids=["R-101", "R-202"],
    reference_script=[
        call("read_file", path="cache.py"),
        call(
            "edit_file",
            path="cache.py",
            old_string="        if key not in self.data:\n            return None\n        return self.data[key]",
            new_string=(
                "        if key not in self.data:\n"
                "            return None\n"
                "        self.order.remove(key)\n"
                "        self.order.append(key)\n"
                "        return self.data[key]"
            ),
        ),
        call(
            "edit_file",
            path="cache.py",
            old_string="        self.data[key] = value\n        self.order.append(key)",
            new_string=(
                "        if key in self.order:\n"
                "            self.order.remove(key)\n"
                "        self.data[key] = value\n"
                "        self.order.append(key)"
            ),
        ),
        say("get() now marks the key as most-recently-used, so it survives eviction."),
    ],
)


TRACEBACK_TEXT = """\
Traceback (most recent call last):
  File "pipeline.py", line 22, in run
    return stage.apply(payload)
  File "stages.py", line 14, in apply
    return self.transform(record["value"])
KeyError: 'value'
"""

DIAGNOSE_FROM_TRACEBACK = Task(
    name="diagnose-from-traceback",
    category="coding",
    tier=Tier.HARD,
    prompt=(
        "This traceback came from production:\n\n"
        + TRACEBACK_TEXT
        + "\nThe records come from load_records() in loader.py. Find why the key is "
        "missing and fix it so the pipeline skips malformed records instead of "
        "crashing. Do not silence the error for records that are actually valid."
    ),
    files={
        "pipeline.py": (
            "from stages import Stage\n\n\n"
            "def run(payload, stages):\n"
            "    for stage in stages:\n"
            "        payload = stage.apply(payload)\n"
            "    return payload\n"
        ),
        "stages.py": (
            "class Stage:\n"
            "    def __init__(self, transform):\n"
            "        self.transform = transform\n\n"
            "    def apply(self, record):\n"
            '        return self.transform(record["value"])\n'
        ),
        "loader.py": (
            "def load_records(rows):\n"
            "    records = []\n"
            "    for row in rows:\n"
            "        records.append({'id': row[0], 'value': row[1]} if len(row) > 1 else {'id': row[0]})\n"
            "    return records\n"
        ),
    },
    rubric=Rubric(
        [
            still_parses("stages.py"),
            still_parses("loader.py"),
            behaviour("investigated more than one file", lambda c: c.call_count("read_file") + c.call_count("grep") >= 2, weight=1.5),
            behaviour(
                "malformed record no longer raises KeyError",
                _pipeline_skips_malformed,
                weight=4.0,
            ),
            behaviour(
                "valid records still transform",
                _pipeline_handles_valid,
                weight=3.0,
                critical=True,
            ),
            answer_mentions("value", "missing", "malformed", weight=1.0),
        ]
    ),
    budget_steps=20,
    requirement_ids=["R-101", "R-109", "R-202"],
)


# ---------------------------------------------------------------------------
# T5 — expert
# ---------------------------------------------------------------------------

CROSS_MODULE = {
    "store/models.py": (
        "class Order:\n"
        "    def __init__(self, id, items, currency='USD'):\n"
        "        self.id = id\n"
        "        self.items = items\n"
        "        self.currency = currency\n"
    ),
    "store/pricing.py": (
        "RATES = {'USD': 1.0, 'EUR': 0.9}\n\n\n"
        "def convert(amount, currency):\n"
        "    return amount * RATES[currency]\n\n\n"
        "def order_total(order):\n"
        "    return convert(sum(order.items), order.currency)\n"
    ),
    "store/report.py": (
        "from store.pricing import order_total\n\n\n"
        "def summarise(orders):\n"
        "    return {o.id: order_total(o) for o in orders}\n"
    ),
    "store/__init__.py": "",
}

ADD_CURRENCY = Task(
    name="add-currency-support",
    category="coding",
    tier=Tier.EXPERT,
    prompt=(
        "Add support for GBP at a rate of 0.8, and make convert() raise a clear "
        "ValueError naming the currency when it is unknown, instead of raising "
        "KeyError. Every existing call site must keep working, and summarise() "
        "must not crash the whole report when one order has a bad currency — "
        "that order should be reported as None instead."
    ),
    files=CROSS_MODULE,
    rubric=Rubric(
        [
            still_parses("store/pricing.py"),
            still_parses("store/report.py"),
            behaviour("read more than one module", lambda c: c.call_count("read_file") >= 2, weight=1.0),
            behaviour("GBP converts at 0.8", _gbp_rate, weight=2.0),
            behaviour("USD still converts at 1.0", _usd_rate, weight=2.0, critical=True),
            behaviour("unknown currency raises ValueError", _unknown_currency, weight=3.0),
            behaviour("the ValueError names the currency", _error_names_currency, weight=1.5),
            behaviour("summarise degrades to None for a bad order", _summarise_degrades, weight=4.0),
            behaviour("summarise still totals good orders", _summarise_good, weight=2.0, critical=True),
        ]
    ),
    budget_steps=24,
    budget_seconds=300.0,
    requirement_ids=["R-101", "R-102", "R-202"],
)


PERFORMANCE = Task(
    name="fix-quadratic-hotspot",
    category="coding",
    tier=Tier.EXPERT,
    prompt=(
        "dedupe.py is too slow on large inputs — it is quadratic. Make it linear "
        "while preserving the exact output, including the order of first "
        "appearance. Then prove it: run it on 20000 items and report the time."
    ),
    files={
        "dedupe.py": (
            "def dedupe(items):\n"
            '    """Return items with duplicates removed, preserving first-seen order."""\n'
            "    out = []\n"
            "    for item in items:\n"
            "        if item not in out:\n"
            "            out.append(item)\n"
            "    return out\n"
        )
    },
    rubric=Rubric(
        [
            still_parses("dedupe.py"),
            behaviour(
                "output identical on a small case",
                lambda c: c.module("dedupe.py")["dedupe"]([3, 1, 3, 2, 1]) == [3, 1, 2],
                weight=3.0,
                critical=True,
            ),
            behaviour("handles an empty list", lambda c: c.module("dedupe.py")["dedupe"]([]) == [], weight=1.0),
            behaviour("no longer scans a list per item", _is_linear, weight=4.0),
            behaviour("actually ran it", lambda c: c.used("shell"), weight=1.5),
            behaviour("reported a timing", lambda c: c.mentions("second", "ms", "s "), weight=1.0),
        ]
    ),
    budget_steps=20,
    budget_seconds=300.0,
    requirement_ids=["R-204"],
)


CODING_TASKS: list[Task] = [
    WRITE_FILE,
    READ_AND_REPORT,
    FIX_EDGE_CASE,
    ADD_FUNCTION,
    SEARCH_CODEBASE,
    MAKE_TESTS_PASS,
    WRITE_TESTS,
    REFACTOR,
    DEBUG_LRU,
    DIAGNOSE_FROM_TRACEBACK,
    ADD_CURRENCY,
    PERFORMANCE,
]
