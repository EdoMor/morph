"""Tests for the scoring and self-improvement machinery itself.

The loop is only as trustworthy as its scorer and its guard rails, so those get
tested directly rather than only through the conformance suite.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from bench.scorecard import WEIGHTS, CheckResult, Scorecard, compare
from selfimprove.guard import is_protected, violations
from selfimprove.prompts import build_improvement_prompt, select_target


# ---------------------------------------------------------------------------
# Scorecard maths
# ---------------------------------------------------------------------------


def _card(**counts: tuple[int, int]) -> Scorecard:
    """Build a scorecard with `passed` of `total` checks solved per category."""
    card = Scorecard()
    for category, (passed, total) in counts.items():
        for index in range(total):
            card.add(
                CheckResult.binary(
                    name=f"{category}/{index}",
                    category=category,
                    passed=index < passed,
                    tier=index % 5 + 1,
                )
            )
    return card


def test_weights_sum_to_one_hundred():
    assert sum(WEIGHTS.values()) == 100.0


def test_perfect_scorecard_is_one_hundred():
    card = _card(**{cat: (3, 3) for cat in WEIGHTS})
    assert card.composite == 100.0
    assert not card.gated


def test_failing_conformance_clamps_the_score():
    """The gate is the whole point: a red suite cannot be traded for other points."""
    card = _card(requirements=(9, 10), coding=(5, 5), robustness=(5, 5))
    assert card.gated
    assert card.composite == 0.0


def test_partial_credit_is_weighted():
    card = _card(requirements=(4, 4), coding=(1, 2))
    # requirements 25 + coding 10 (half of 20); other categories are empty => 0
    assert card.composite == pytest.approx(WEIGHTS["requirements"] + WEIGHTS["coding"] / 2)


def test_graded_scores_land_between_pass_and_fail():
    """The whole point of R-709: half-done must sit between done and nothing."""
    nothing = Scorecard()
    nothing.add(CheckResult("coding/x", "coding", score=0.0, tier=3))
    halfway = Scorecard()
    halfway.add(CheckResult("coding/x", "coding", score=0.5, tier=3))
    done = Scorecard()
    done.add(CheckResult("coding/x", "coding", score=1.0, tier=3))

    assert nothing.composite < halfway.composite < done.composite


def test_tier_weighting_favours_harder_tasks():
    easy = Scorecard()
    easy.add(CheckResult("coding/a", "coding", score=1.0, tier=1, weight=1.0))
    easy.add(CheckResult("coding/b", "coding", score=0.0, tier=5, weight=8.0))

    hard = Scorecard()
    hard.add(CheckResult("coding/a", "coding", score=0.0, tier=1, weight=1.0))
    hard.add(CheckResult("coding/b", "coding", score=1.0, tier=5, weight=8.0))

    assert hard.composite > easy.composite


def test_pytest_skips_neither_help_nor_hurt():
    """The runner records a pytest skip as passed with weight 0."""
    card = Scorecard()
    card.add(CheckResult.binary("passing", "requirements", passed=True, weight=1.0))
    card.add(CheckResult.binary("skipped", "requirements", passed=True, weight=0.0))

    assert card.category_score("requirements") == 1.0
    assert not card.gated


def test_the_gate_counts_any_non_pass_regardless_of_weight():
    """Weight discounts a check's contribution; it never exempts it from the gate."""
    card = Scorecard()
    card.add(CheckResult.binary("passing", "requirements", passed=True, weight=1.0))
    card.add(CheckResult.binary("failing", "requirements", passed=False, weight=0.0))

    assert card.category_score("requirements") == 1.0
    assert card.gated
    assert card.composite == 0.0


def test_empty_category_scores_zero():
    assert Scorecard().category_score("capability") == 0.0


def test_compare_reports_per_category_deltas():
    before = _card(requirements=(4, 4), coding=(1, 2)).to_dict()
    after = _card(requirements=(4, 4), coding=(2, 2)).to_dict()
    delta = compare(before, after)

    assert delta["delta"] == pytest.approx(WEIGHTS["coding"] / 2)
    assert delta["categories"]["coding"] == pytest.approx(WEIGHTS["coding"] / 2)
    assert delta["categories"]["requirements"] == 0.0


def test_compare_reports_frontier_movement():
    """A frontier that moves is the headline result of an iteration."""
    before = Scorecard()
    before.add(CheckResult("coding/T1/a", "coding", score=1.0, tier=1))
    before.add(CheckResult("coding/T2/b", "coding", score=0.2, tier=2))

    after = Scorecard()
    after.add(CheckResult("coding/T1/a", "coding", score=1.0, tier=1))
    after.add(CheckResult("coding/T2/b", "coding", score=0.9, tier=2))

    moves = compare(before.to_dict(), after.to_dict())["frontier_moves"]
    assert moves["coding"] == "T1 -> T2"


def test_scorecard_round_trips_through_json(tmp_path: Path):
    card = _card(requirements=(2, 3), coding=(1, 1))
    path = card.write(tmp_path / "sub" / "scorecard.json")
    data = Scorecard.read(path)

    assert data["composite"] == card.composite
    assert len(data["results"]) == 4
    assert json.dumps(data)  # serialisable end to end


def test_render_and_feedback_name_the_failures():
    card = Scorecard()
    card.add(
        CheckResult(
            "coding/T2/fix-a-bug",
            "coding",
            score=0.4,
            tier=2,
            detail="average([]) still raises ZeroDivisionError",
            requirement_ids=["R-203"],
        )
    )
    rendered = card.render()
    assert "coding/T2/fix-a-bug" in rendered
    assert "nearest misses" in rendered

    feedback = card.feedback()
    assert "ZeroDivisionError" in feedback
    assert "R-203" in feedback
    assert "40%" in feedback, "the feedback must show how close it got"


def test_feedback_when_everything_passes_asks_for_harder_tasks():
    """A saturated benchmark is a broken instrument, and the loop must say so."""
    card = _card(**{cat: (2, 2) for cat in WEIGHTS})
    feedback = card.feedback()

    assert "needs harder tasks" in feedback
    assert "human" in feedback


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "REQUIREMENTS.md",
        "./REQUIREMENTS.md",
        "tests/test_requirements.py",
        "bench/scorecard.py",
        "selfimprove/loop.py",
        "selfimprove/guard.py",
        ".github/workflows/self-improve.yml",
    ],
)
def test_protected_paths_are_recognised(path: str):
    assert is_protected(path)


@pytest.mark.parametrize(
    "path",
    [
        "morph/agent.py",
        "tests/test_agent.py",
        "bench/runner.py",
        "bench/tasks_notes.md",
        "REQUIREMENTS-notes.md",
        "webapp/app.js",
    ],
)
def test_unprotected_paths_are_editable(path: str):
    assert not is_protected(path)


def test_violations_detects_an_edit_to_the_goalposts(tmp_path: Path):
    """A real git repo, a real edit, and the guard catching it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "REQUIREMENTS.md").write_text("original\n")
    (repo / "morph").mkdir()
    (repo / "morph" / "agent.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)

    assert violations(repo) == []

    (repo / "morph" / "agent.py").write_text("x = 2\n")
    assert violations(repo) == []  # editing real code is fine

    (repo / "REQUIREMENTS.md").write_text("moved the goalposts\n")
    assert violations(repo) == ["REQUIREMENTS.md"]


def test_untracked_protected_file_is_caught(tmp_path: Path):
    repo = tmp_path / "repo2"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "seed.txt").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)

    (repo / "bench").mkdir()
    (repo / "bench" / "scorecard.py").write_text("WEIGHTS = {'requirements': 100}\n")
    assert violations(repo) == ["bench/scorecard.py"]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def test_prompt_leads_with_the_gate_when_the_suite_is_red():
    prompt = build_improvement_prompt(
        requirements="R-001 something",
        scorecard={"composite": 0.0, "gated": True, "categories": {}},
        feedback="tests/test_agent.py::test_x failed",
        history=[],
    )
    assert "conformance suite is failing" in prompt
    assert "clamped to 0" in prompt


def test_prompt_includes_requirements_and_protected_list():
    prompt = build_improvement_prompt(
        requirements="R-999 the contract text",
        scorecard={"composite": 80.0, "categories": {}},
        feedback="nothing",
        history=[],
    )
    assert "R-999 the contract text" in prompt
    assert "REQUIREMENTS.md" in prompt
    assert "must not modify" in prompt


def test_prompt_carries_a_focus_when_given():
    prompt = build_improvement_prompt(
        requirements="x",
        scorecard={"composite": 1.0, "categories": {}},
        feedback="y",
        history=[],
        focus="MCP reconnection",
    )
    assert "MCP reconnection" in prompt


def test_prompt_history_is_ordered_newest_first():
    history = [
        {"accepted": False, "score_before": 10, "score_after": 9, "summary": "oldest attempt"},
        {"accepted": True, "score_before": 10, "score_after": 20, "summary": "newest attempt"},
    ]
    prompt = build_improvement_prompt(
        requirements="x", scorecard={"composite": 20.0, "categories": {}}, feedback="y", history=history
    )
    assert prompt.index("newest attempt") < prompt.index("oldest attempt")


def test_target_rotates_away_from_a_recent_rejection():
    targets = [
        {"name": "coding/T2/fix-edge-case", "category": "coding", "score": 0.14},
        {"name": "tool_use/T2/chain-tool-results", "category": "tool_use", "score": 0.4},
    ]
    history = [
        {
            "accepted": False,
            "target": "coding/T2/fix-edge-case",
            "summary": "Tried fix-edge-case and made no edit.",
        }
    ]

    assert select_target({"next_targets": targets}, history) == targets[1]


def test_prompt_gives_gemma_one_synthetic_target_and_a_first_action():
    target = {
        "name": "coding/T2/fix-edge-case",
        "category": "coding",
        "score": 0.14,
        "detail": "average([]) raised ZeroDivisionError",
    }
    prompt = build_improvement_prompt(
        requirements="R-001 something",
        scorecard={"composite": 60.0, "categories": {}, "next_targets": [target]},
        feedback="nearest miss",
        history=[],
    )

    assert "work on this one only" in prompt
    assert 'name="fix-edge-case"' in prompt
    assert "bench/tasks/coding.py" in prompt
    assert "evidence, not source code" in prompt


def test_benchmark_temperature_defaults_to_zero(monkeypatch, tmp_path):
    from morph.config import Config
    from selfimprove.loop import _bench_config

    monkeypatch.delenv("MORPH_BENCH_TEMPERATURE", raising=False)
    measured = _bench_config(tmp_path, Config(workspace=tmp_path, temperature=0.2))
    assert measured.temperature == 0.0

    monkeypatch.setenv("MORPH_BENCH_TEMPERATURE", "0.1")
    assert _bench_config(tmp_path, Config(workspace=tmp_path)).temperature == 0.1


@pytest.mark.asyncio
async def test_strict_edit_attempt_cannot_end_on_analysis(config, registry):
    from morph.agent import Agent
    from morph.llm.base import ModelResponse
    from morph.llm.echo import EchoProvider

    class AnalystThenEditor:
        name = "analyst"
        supports_native_tools = True

        def __init__(self):
            self.turns = []

        async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
            self.turns.append(list(messages))
            if len(self.turns) == 1:
                return ModelResponse(text="The likely cause is in the retry policy.")
            if len(self.turns) == 2:
                return EchoProvider.call("write_file", path="fix.txt", content="done\n")
            return ModelResponse(text="Implemented and verified the change.")

    provider = AnalystThenEditor()
    agent = Agent(
        config=config,
        provider=provider,
        tools=registry,
        require_edit=True,
    )
    result = await agent.run("Make an improvement.", max_steps=8)

    assert (config.root / "fix.txt").read_text("utf-8") == "done\n"
    assert result.steps == 3
    assert "code-change attempt" in json.dumps(provider.turns[1])


# ---------------------------------------------------------------------------
# Iteration decisions
#
# These drive the real loop against the real repository, with the benchmark
# stubbed so a case costs milliseconds instead of a full scorecard run. What is
# under test is the decision: when does an iteration get kept, and when reverted.
# ---------------------------------------------------------------------------


@pytest.fixture
def iteration_harness(monkeypatch, repo_root, tmp_path):
    """Run `run_iteration` with a scripted editor and a controllable score."""
    import selfimprove.loop as loop_module
    from morph.config import Config
    from morph.llm.echo import EchoProvider
    from morph.trace import ProgressFile

    def _run(script, score_after: float = 100.0, gated: bool = False, **kwargs):
        # These iterations run against the real repository, because they need
        # real worktrees — so their heartbeat must go somewhere else. Without
        # this the suite overwrites the progress file of any run in flight, and
        # the benchmark runs this suite on every pass.
        kwargs.setdefault("progress_file", ProgressFile(tmp_path / "progress.json"))
        monkeypatch.setattr(
            loop_module, "get_provider", lambda *a, **k: EchoProvider(script=script)
        )

        async def fake_measure(root, config, progress=None):
            return {"composite": score_after, "gated": gated, "categories": {}, "results": []}

        monkeypatch.setattr(loop_module, "measure", fake_measure)

        config = Config(workspace=repo_root, provider="echo", image_backend="stub")
        baseline = {"composite": 50.0, "gated": False, "categories": {}, "results": []}
        return asyncio.run(
            loop_module.run_iteration(
                index=99,
                repo=repo_root,
                config=config,
                baseline=baseline,
                history=[],
                dry_run=True,
                **kwargs,
            )
        )

    return _run


def test_iteration_rejected_when_nothing_changed(iteration_harness, say):
    result = iteration_harness([say("I looked around and changed nothing.")])

    assert not result.accepted
    assert result.rejection_reason == "the agent made no changes"
    assert result.files_changed == []


def test_iteration_rejected_for_touching_the_goalposts(iteration_harness, call, say):
    """R-707 in practice: editing REQUIREMENTS.md voids the whole iteration."""
    result = iteration_harness(
        [
            call("write_file", path="REQUIREMENTS.md", content="R-001 be great\n"),
            say("Simplified the requirements."),
        ]
    )

    assert not result.accepted
    assert "protected files" in result.rejection_reason
    assert "REQUIREMENTS.md" in result.rejection_reason
    assert result.score_after == 0.0, "a goalpost breach scores zero, not partial credit"


def test_iteration_rejected_when_the_gate_is_red(iteration_harness, call, say):
    result = iteration_harness(
        [call("write_file", path="NOTES.md", content="x"), say("Edited.")],
        score_after=90.0,
        gated=True,
    )

    assert not result.accepted
    assert "conformance suite is failing" in result.rejection_reason


def test_iteration_rejected_on_score_regression(iteration_harness, call, say):
    result = iteration_harness(
        [call("write_file", path="NOTES.md", content="x"), say("Edited.")],
        score_after=49.9,
    )

    assert not result.accepted
    assert "regressed" in result.rejection_reason


def test_iteration_accepted_when_the_score_holds_or_improves(iteration_harness, call, say):
    result = iteration_harness(
        [call("write_file", path="NOTES.md", content="# Notes\n"), say("Added NOTES.md.")],
        score_after=61.0,
    )

    assert result.accepted
    assert result.files_changed == ["NOTES.md"]
    assert result.score_after == 61.0
    assert "Added NOTES.md." in result.summary


def test_iteration_leaves_no_worktree_or_branch_behind(iteration_harness, repo_root, say):
    """Every iteration cleans up after itself, accepted or not.

    Asserted as a delta, not as absolute git state. When the benchmark runs
    *inside* a self-improvement iteration, the parent loop's own worktree is
    legitimately present — a test that demanded a globally clean worktree list
    would fail there, and since the conformance suite gates acceptance, that
    would make the loop reject every iteration it ever produced.
    """
    import subprocess as sp

    def git_state() -> tuple[set[str], set[str]]:
        worktrees = sp.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        branches = sp.run(
            ["git", "branch", "--list", "selfimprove/*"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        paths = {
            line.split(" ", 1)[1]
            for line in worktrees.splitlines()
            if line.startswith("worktree ")
        }
        return paths, {b.strip(" *+") for b in branches.splitlines() if b.strip()}

    before_worktrees, before_branches = git_state()
    iteration_harness([say("Nothing to do.")])
    after_worktrees, after_branches = git_state()

    assert after_worktrees - before_worktrees == set(), "an iteration leaked a worktree"
    assert after_branches - before_branches == set(), "an iteration leaked a branch"
