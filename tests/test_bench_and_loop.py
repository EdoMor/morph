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
from selfimprove.prompts import build_improvement_prompt


# ---------------------------------------------------------------------------
# Scorecard maths
# ---------------------------------------------------------------------------


def _card(**counts: tuple[int, int]) -> Scorecard:
    card = Scorecard()
    for category, (passed, total) in counts.items():
        for index in range(total):
            card.add(
                CheckResult(
                    name=f"{category}/{index}",
                    category=category,
                    passed=index < passed,
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
    card = _card(requirements=(9, 10), capability=(5, 5), robustness=(5, 5))
    assert card.gated
    assert card.composite == 0.0


def test_partial_credit_is_weighted():
    card = _card(requirements=(4, 4), capability=(1, 2))
    # requirements 40 + capability 15 (half of 30); other categories are empty => 0
    assert card.composite == pytest.approx(55.0)


def test_skipped_tests_neither_help_nor_hurt():
    """The runner records a skip as passed with weight 0 — it cannot move the score."""
    card = Scorecard()
    card.add(CheckResult("passing", "requirements", passed=True, weight=1.0))
    card.add(CheckResult("skipped", "requirements", passed=True, weight=0.0))

    assert card.category_score("requirements") == 1.0
    assert not card.gated


def test_the_gate_counts_any_non_pass_regardless_of_weight():
    """Weight discounts a check's contribution; it never exempts it from the gate."""
    card = Scorecard()
    card.add(CheckResult("passing", "requirements", passed=True, weight=1.0))
    card.add(CheckResult("failing", "requirements", passed=False, weight=0.0))

    assert card.category_score("requirements") == 1.0
    assert card.gated
    assert card.composite == 0.0


def test_empty_category_scores_zero():
    assert Scorecard().category_score("capability") == 0.0


def test_compare_reports_per_category_deltas():
    before = _card(requirements=(4, 4), capability=(1, 2)).to_dict()
    after = _card(requirements=(4, 4), capability=(2, 2)).to_dict()
    delta = compare(before, after)

    assert delta["delta"] == pytest.approx(15.0)
    assert delta["categories"]["capability"] == pytest.approx(15.0)
    assert delta["categories"]["requirements"] == 0.0


def test_scorecard_round_trips_through_json(tmp_path: Path):
    card = _card(requirements=(2, 3), capability=(1, 1))
    path = card.write(tmp_path / "sub" / "scorecard.json")
    data = Scorecard.read(path)

    assert data["composite"] == card.composite
    assert len(data["results"]) == 4
    assert json.dumps(data)  # serialisable end to end


def test_render_and_feedback_name_the_failures():
    card = Scorecard()
    card.add(
        CheckResult(
            "capability/fix-a-bug",
            "capability",
            passed=False,
            detail="average([]) still raises ZeroDivisionError",
            requirement_ids=["R-203"],
        )
    )
    rendered = card.render()
    assert "capability/fix-a-bug" in rendered
    assert "R-203" in rendered

    feedback = card.feedback()
    assert "ZeroDivisionError" in feedback
    assert "R-203" in feedback


def test_feedback_when_everything_passes():
    card = _card(**{cat: (2, 2) for cat in WEIGHTS})
    assert "efficiency" in card.feedback()


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
        "bench/tasks/capability.py",
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


# ---------------------------------------------------------------------------
# Iteration decisions
#
# These drive the real loop against the real repository, with the benchmark
# stubbed so a case costs milliseconds instead of a full scorecard run. What is
# under test is the decision: when does an iteration get kept, and when reverted.
# ---------------------------------------------------------------------------


@pytest.fixture
def iteration_harness(monkeypatch, repo_root):
    """Run `run_iteration` with a scripted editor and a controllable score."""
    import selfimprove.loop as loop_module
    from morph.config import Config
    from morph.llm.echo import EchoProvider

    def _run(script, score_after: float = 100.0, gated: bool = False, **kwargs):
        monkeypatch.setattr(
            loop_module, "get_provider", lambda *a, **k: EchoProvider(script=script)
        )

        async def fake_measure(root, config):
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
    """Every iteration cleans up after itself, accepted or not."""
    import subprocess as sp

    iteration_harness([say("Nothing to do.")])

    worktrees = sp.run(
        ["git", "worktree", "list"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    branches = sp.run(
        ["git", "branch", "--list", "selfimprove/*"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert "selfimprove" not in worktrees
    assert branches.strip() == ""
