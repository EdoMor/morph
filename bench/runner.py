"""Benchmark runner: produces the score the self-improvement loop optimises.

    python -m bench.runner [--output scorecard.json]

Five categories, defined in :mod:`bench.scorecard`:

``requirements``  the conformance suite in ``tests/`` — also the gate
``capability``    end-to-end agent tasks
``robustness``    error injection
``efficiency``    steps and wall time against per-task budgets
``health``        import cleanliness, annotations, obvious code smells
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from morph.agent import Agent
from morph.config import Config
from morph.llm import get_provider
from morph.llm.echo import EchoProvider
from morph.skills import SkillRegistry
from morph.tools import build_default_registry

from .scorecard import CheckResult, Scorecard
from .tasks import CAPABILITY_TASKS, ROBUSTNESS_CHECKS
from .tasks.types import AgentTask

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTEST_TIMEOUT = 900


# ---------------------------------------------------------------------------
# 1. requirements — the conformance suite (gate)
# ---------------------------------------------------------------------------


def run_requirements(scorecard: Scorecard, repo: Path = REPO_ROOT) -> None:
    """Run pytest and turn every test into a check (R-801, R-805)."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests",
                    "-q",
                    "--tb=short",
                    f"--junitxml={report}",
                    "-p",
                    "no:cacheprovider",
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=PYTEST_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            scorecard.add(
                CheckResult(
                    name="requirements/pytest",
                    category="requirements",
                    passed=False,
                    detail=f"pytest did not finish within {PYTEST_TIMEOUT}s",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            )
            return
        except FileNotFoundError as exc:
            scorecard.add(
                CheckResult(
                    name="requirements/pytest",
                    category="requirements",
                    passed=False,
                    detail=f"pytest is not installed: {exc}",
                )
            )
            return

        if not report.is_file():
            scorecard.add(
                CheckResult(
                    name="requirements/pytest",
                    category="requirements",
                    passed=False,
                    detail=(process.stdout + process.stderr)[-4000:],
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            )
            return

        for case in ET.parse(report).getroot().iter("testcase"):
            classname = (case.get("classname") or "").replace(".", "/")
            name = f"{classname}::{case.get('name')}" if classname else str(case.get("name"))
            failure = case.find("failure")
            error = case.find("error")
            problem = failure if failure is not None else error
            skipped = case.find("skipped") is not None
            scorecard.add(
                CheckResult(
                    name=f"requirements/{name}",
                    category="requirements",
                    passed=problem is None,
                    detail="" if problem is None else (problem.get("message", "") + "\n" + (problem.text or ""))[:2000],
                    duration_ms=float(case.get("time") or 0) * 1000,
                    weight=0.0 if skipped else 1.0,
                    requirement_ids=_requirement_ids_from_name(str(case.get("name"))),
                )
            )


def _requirement_ids_from_name(test_name: str) -> list[str]:
    """Extract ``R-###`` ids embedded in a parametrised test id."""
    import re

    return re.findall(r"R-\d{3}", test_name)


# ---------------------------------------------------------------------------
# 2. capability — end-to-end agent tasks
# ---------------------------------------------------------------------------


def _build_agent(task: AgentTask, root: Path, config: Config) -> Agent:
    if config.provider == "echo":
        # Deterministic replay: measures the harness, not a model.
        provider: Any = EchoProvider(script=list(task.reference_script))
    else:
        provider = get_provider(
            config.provider,
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
        )
    task_config = Config(
        workspace=root,
        provider=config.provider,
        model=config.model,
        base_url=config.base_url,
        image_backend=config.image_backend,
        max_steps=task.budget_steps,
        skill_paths=[root / "skills"],
        allow_shell=True,
    )
    return Agent(
        config=task_config,
        provider=provider,
        tools=build_default_registry(task_config),
        skills=SkillRegistry(),
    )


async def run_capability(scorecard: Scorecard, config: Config) -> list[dict[str, Any]]:
    """Run every capability task in a throwaway workspace. Returns timing data."""
    timings: list[dict[str, Any]] = []

    for task in CAPABILITY_TASKS:
        workspace = Path(tempfile.mkdtemp(prefix="morph-bench-"))
        started = time.perf_counter()
        try:
            task.materialise(workspace)
            agent = _build_agent(task, workspace, config)
            try:
                result = await asyncio.wait_for(
                    agent.run(task.prompt, max_steps=task.budget_steps),
                    timeout=task.budget_seconds,
                )
            finally:
                await agent.close()

            if result.error:
                outcome_passed, detail = False, f"run errored: {result.error}"
            else:
                outcome = task.verify(workspace, result)
                outcome_passed, detail = outcome.passed, outcome.detail

            elapsed = time.perf_counter() - started
            timings.append(
                {
                    "task": task.name,
                    "steps": result.steps,
                    "budget_steps": task.budget_steps,
                    "seconds": elapsed,
                    "budget_seconds": task.budget_seconds,
                    "passed": outcome_passed,
                }
            )
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - started
            outcome_passed = False
            detail = f"timed out after {task.budget_seconds:g}s"
            timings.append(
                {
                    "task": task.name,
                    "steps": task.budget_steps,
                    "budget_steps": task.budget_steps,
                    "seconds": elapsed,
                    "budget_seconds": task.budget_seconds,
                    "passed": False,
                }
            )
        except Exception as exc:  # noqa: BLE001 - a broken task must not stop the benchmark
            elapsed = time.perf_counter() - started
            outcome_passed = False
            detail = f"{type(exc).__name__}: {exc}"
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

        scorecard.add(
            CheckResult(
                name=task.name,
                category="capability",
                passed=outcome_passed,
                detail=detail,
                duration_ms=elapsed * 1000,
                weight=task.weight,
                requirement_ids=task.requirement_ids,
            )
        )

    return timings


# ---------------------------------------------------------------------------
# 3. robustness
# ---------------------------------------------------------------------------


async def run_robustness(scorecard: Scorecard) -> None:
    for check in ROBUSTNESS_CHECKS:
        workspace = Path(tempfile.mkdtemp(prefix="morph-rob-"))
        started = time.perf_counter()
        try:
            outcome = await asyncio.wait_for(check.run(workspace), timeout=120)
            passed, detail = outcome.passed, outcome.detail
        except asyncio.TimeoutError:
            passed, detail = False, "check timed out after 120s"
        except Exception as exc:  # noqa: BLE001
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

        scorecard.add(
            CheckResult(
                name=check.name,
                category="robustness",
                passed=passed,
                detail=detail,
                duration_ms=(time.perf_counter() - started) * 1000,
                weight=check.weight,
                requirement_ids=check.requirement_ids,
            )
        )


# ---------------------------------------------------------------------------
# 4. efficiency
# ---------------------------------------------------------------------------


def run_efficiency(scorecard: Scorecard, timings: list[dict[str, Any]]) -> None:
    """Reward solving tasks in fewer steps and less wall time."""
    if not timings:
        scorecard.add(
            CheckResult(
                name="efficiency/no-data",
                category="efficiency",
                passed=False,
                detail="no capability timings were collected",
            )
        )
        return

    solved = [t for t in timings if t["passed"]]
    if solved:
        step_ratio = sum(t["steps"] / max(t["budget_steps"], 1) for t in solved) / len(solved)
        scorecard.add(
            CheckResult(
                name="efficiency/step-budget",
                category="efficiency",
                passed=step_ratio <= 0.75,
                detail=f"solved tasks used {step_ratio:.0%} of their step budget on average",
                weight=2.0,
                requirement_ids=["R-108"],
            )
        )
        time_ratio = sum(t["seconds"] / max(t["budget_seconds"], 1) for t in solved) / len(solved)
        scorecard.add(
            CheckResult(
                name="efficiency/time-budget",
                category="efficiency",
                passed=time_ratio <= 0.5,
                detail=f"solved tasks used {time_ratio:.0%} of their time budget on average",
                requirement_ids=["R-108"],
            )
        )
    else:
        scorecard.add(
            CheckResult(
                name="efficiency/step-budget",
                category="efficiency",
                passed=False,
                detail="no capability task was solved, so efficiency cannot be earned",
                weight=2.0,
            )
        )

    timeouts = [t for t in timings if t["seconds"] >= t["budget_seconds"]]
    scorecard.add(
        CheckResult(
            name="efficiency/no-timeouts",
            category="efficiency",
            passed=not timeouts,
            detail=", ".join(t["task"] for t in timeouts) or "no task hit its time budget",
            requirement_ids=["R-102"],
        )
    )


# ---------------------------------------------------------------------------
# 5. health
# ---------------------------------------------------------------------------

PUBLIC_PACKAGES = ("morph", "bench", "selfimprove")


def run_health(scorecard: Scorecard, repo: Path = REPO_ROOT) -> None:
    modules = sorted(
        path
        for package in PUBLIC_PACKAGES
        for path in (repo / package).rglob("*.py")
        if "__pycache__" not in path.parts
    )

    # -- every module parses ------------------------------------------
    broken: list[str] = []
    trees: dict[Path, ast.Module] = {}
    for path in modules:
        try:
            trees[path] = ast.parse(path.read_text("utf-8"), str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            broken.append(f"{path.relative_to(repo)}: {exc}")
    scorecard.add(
        CheckResult(
            name="health/parses",
            category="health",
            passed=not broken,
            detail="\n".join(broken) or f"{len(trees)} modules parse",
            weight=2.0,
            requirement_ids=["R-804"],
        )
    )

    # -- the package imports cleanly ----------------------------------
    probe = subprocess.run(
        [sys.executable, "-c", "import morph, bench.runner, selfimprove.loop"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
    )
    scorecard.add(
        CheckResult(
            name="health/imports",
            category="health",
            passed=probe.returncode == 0,
            detail=(probe.stderr or probe.stdout)[-1500:] or "imports clean",
            weight=2.0,
            requirement_ids=["R-804"],
        )
    )

    # -- public functions are annotated -------------------------------
    unannotated: list[str] = []
    total = 0
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            total += 1
            missing = [
                arg.arg
                for arg in node.args.args + node.args.kwonlyargs
                if arg.annotation is None and arg.arg not in {"self", "cls"}
            ]
            if missing or node.returns is None:
                unannotated.append(f"{path.relative_to(repo)}:{node.lineno} {node.name}")
    ratio = 1 - (len(unannotated) / total) if total else 1.0
    scorecard.add(
        CheckResult(
            name="health/type-hints",
            category="health",
            passed=ratio >= 0.9,
            detail=f"{ratio:.0%} of {total} public functions fully annotated"
            + (f"; first gaps: {unannotated[:5]}" if unannotated else ""),
            requirement_ids=["R-804"],
        )
    )

    # -- no silently swallowed exceptions -----------------------------
    swallowed: list[str] = []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                body = node.body
                only_pass = len(body) == 1 and isinstance(body[0], ast.Pass)
                if only_pass and node.type is None:
                    swallowed.append(f"{path.relative_to(repo)}:{node.lineno}")
    scorecard.add(
        CheckResult(
            name="health/no-bare-swallow",
            category="health",
            passed=not swallowed,
            detail=", ".join(swallowed) or "no bare `except: pass`",
        )
    )


# ---------------------------------------------------------------------------


async def run_benchmark(
    config: Config | None = None,
    repo: Path = REPO_ROOT,
    skip_requirements: bool = False,
) -> Scorecard:
    """Run every category and return the scorecard."""
    cfg = config or Config(workspace=repo, provider="echo", image_backend="stub")
    scorecard = Scorecard(
        metadata={
            "provider": cfg.provider,
            "model": cfg.model,
            "image_backend": cfg.image_backend,
            "python": sys.version.split()[0],
            "repo": str(repo),
            "note": (
                "provider=echo: capability replays reference traces, so the score "
                "measures harness integrity rather than model skill"
                if cfg.provider == "echo"
                else f"capability measured against {cfg.provider}/{cfg.model}"
            ),
        }
    )

    if not skip_requirements:
        run_requirements(scorecard, repo)
    timings = await run_capability(scorecard, cfg)
    await run_robustness(scorecard)
    run_efficiency(scorecard, timings)
    run_health(scorecard, repo)
    return scorecard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench", description="Run the Morph benchmark")
    parser.add_argument("--output", help="Write the scorecard JSON here")
    parser.add_argument("--provider", default="echo", help="echo | ollama | google")
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--image-backend", default="stub")
    parser.add_argument(
        "--skip-requirements",
        action="store_true",
        help="Skip the pytest gate (useful when the benchmark is invoked from pytest)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = Config(
        workspace=REPO_ROOT,
        provider=args.provider,
        model=args.model,
        image_backend=args.image_backend,
    )
    scorecard = asyncio.run(
        run_benchmark(config, skip_requirements=args.skip_requirements)
    )

    if not args.quiet:
        print(scorecard.render())
    if args.output:
        path = scorecard.write(args.output)
        print(f"scorecard written to {path}")
    return 0 if not scorecard.gated else 1


if __name__ == "__main__":
    raise SystemExit(main())
