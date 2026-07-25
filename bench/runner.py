"""Benchmark runner: produces the score the self-improvement loop optimises.

    python -m bench.runner [--provider ollama --model gemma3:4b] [--output card.json]

Categories, defined in :mod:`bench.scorecard`:

``requirements``  the conformance suite in ``tests/`` — also the gate
``coding``        agent tasks: change software correctly (T1-T5)
``tool_use``      agent tasks: drive the tools well (T1-T5)
``mcp``           agent tasks: use runtime-discovered MCP tools (T1-T5)
``skills``        agent tasks: find, load and follow packaged instructions (T1-T5)
``robustness``    error injection
``efficiency``    steps and wall time against per-task budgets
``health``        import cleanliness, annotations, obvious code smells

Expect stderr noise from the robustness suite — it deliberately injects
failures, and the logging that produces is the system behaving correctly.
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
from morph.trace import ProgressFile, TraceRenderer

from .scorecard import CAPABILITY_CATEGORIES, CheckResult, Scorecard
from .tasks import ALL_TASKS, ROBUSTNESS_CHECKS, SUITES
from .tasks.spec import Task, TaskContext

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
                    score=0.0,
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
                    score=0.0,
                    detail=f"pytest is not installed: {exc}",
                )
            )
            return

        if not report.is_file():
            scorecard.add(
                CheckResult(
                    name="requirements/pytest",
                    category="requirements",
                    score=0.0,
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
                    score=1.0 if problem is None else 0.0,
                    detail=""
                    if problem is None
                    else (problem.get("message", "") + "\n" + (problem.text or ""))[:2000],
                    duration_ms=float(case.get("time") or 0) * 1000,
                    weight=0.0 if skipped else 1.0,
                    requirement_ids=_requirement_ids_from_name(str(case.get("name"))),
                )
            )


def _requirement_ids_from_name(test_name: str) -> list[str]:
    """Extract ``R-###`` ids embedded in a test id (``test_R_205_...``)."""
    import re

    return [m.replace("_", "-") for m in re.findall(r"R[_-]\d{3}", test_name)]


# ---------------------------------------------------------------------------
# 2-5. capability suites
# ---------------------------------------------------------------------------


def _build_agent(task: Task, root: Path, config: Config) -> Agent:
    task_config = Config(
        workspace=root,
        provider=config.provider,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        image_backend=config.image_backend,
        max_steps=task.budget_steps,
        skill_paths=[root / "skills"],
        mcp_servers=task.mcp_servers(root) if task.mcp_servers else [],
        allow_shell=True,
    )

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

    return Agent(
        config=task_config,
        provider=provider,
        tools=build_default_registry(task_config),
        skills=SkillRegistry(),
    )


async def run_task(task: Task, config: Config) -> tuple[CheckResult, dict[str, Any]]:
    """Run one capability task in a throwaway workspace and grade it."""
    # In echo mode a task with no reference trace has nothing to replay. Skip it
    # rather than score it zero — a plumbing check must not masquerade as a
    # capability measurement.
    if config.provider == "echo" and not task.reference_script:
        return (
            CheckResult(
                name=task.label,
                category=task.category,
                score=0.0,
                detail="skipped: no reference trace for the deterministic provider",
                weight=task.weight,
                tier=int(task.tier),
                requirement_ids=task.requirement_ids,
                skipped=True,
            ),
            {},
        )

    workspace = Path(tempfile.mkdtemp(prefix="morph-bench-"))
    started = time.perf_counter()
    agent: Agent | None = None
    try:
        task.materialise(workspace)
        agent = _build_agent(task, workspace, config)
        result = await asyncio.wait_for(
            agent.run(task.prompt, max_steps=task.budget_steps), timeout=task.budget_seconds
        )

        if result.error:
            score, detail = 0.0, f"run errored: {result.error}"
            steps = result.steps
        else:
            grade = task.rubric.grade(TaskContext(root=workspace, result=result))
            score, detail = grade.score, grade.detail
            steps = result.steps

    except asyncio.TimeoutError:
        score, detail, steps = 0.0, f"timed out after {task.budget_seconds:g}s", task.budget_steps
    except Exception as exc:  # noqa: BLE001 - a broken task must not stop the benchmark
        score, detail, steps = 0.0, f"harness error: {type(exc).__name__}: {exc}", 0
    finally:
        if agent is not None:
            await agent.close()
        shutil.rmtree(workspace, ignore_errors=True)

    elapsed = time.perf_counter() - started
    check = CheckResult(
        name=task.label,
        category=task.category,
        score=score,
        detail=detail,
        duration_ms=elapsed * 1000,
        weight=task.weight,
        tier=int(task.tier),
        requirement_ids=task.requirement_ids,
    )
    timing = {
        "task": task.label,
        "steps": steps,
        "budget_steps": task.budget_steps,
        "seconds": elapsed,
        "budget_seconds": task.budget_seconds,
        "score": score,
    }
    return check, timing


async def run_capability_suites(
    scorecard: Scorecard,
    config: Config,
    only: str | None = None,
    tracer: TraceRenderer | None = None,
    progress: ProgressFile | None = None,
) -> list[dict[str, Any]]:
    """Run the capability tasks, reporting each one as it finishes.

    Against a real model this is most of the run's wall time. Printing a line
    per task is the difference between a benchmark you can watch and forty
    silent minutes (R-718).
    """
    timings: list[dict[str, Any]] = []
    tasks = ALL_TASKS if only is None else SUITES.get(only, [])
    suite = None

    for index, task in enumerate(tasks, 1):
        if tracer and task.category != suite:
            suite = task.category
            tracer.header(f"{suite} suite")
        if progress:
            progress.update(activity=f"benchmark {index}/{len(tasks)}: {task.label}")

        check, timing = await run_task(task, config)
        scorecard.add(check)
        if timing:
            timings.append(timing)

        if tracer:
            if check.skipped:
                tracer.note(f"[{index:>2}/{len(tasks)}] {task.label}  skipped")
            else:
                mark = "solved" if check.passed else "      "
                tracer.note(
                    f"[{index:>2}/{len(tasks)}] {task.label}  {check.score:.2f} {mark}"
                    f"  {check.duration_ms / 1000:.0f}s"
                )
    return timings


# ---------------------------------------------------------------------------
# 6. robustness
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
            CheckResult.binary(
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
# 7. efficiency
# ---------------------------------------------------------------------------


def run_efficiency(scorecard: Scorecard, timings: list[dict[str, Any]]) -> None:
    """Reward solving tasks in fewer steps and less wall time.

    Graded, so shaving two steps off shows up. Only solved tasks count —
    otherwise failing fast would score as efficiency.
    """
    solved = [t for t in timings if t["score"] >= 0.8]
    if not solved:
        scorecard.add(
            CheckResult(
                name="efficiency/step-budget",
                category="efficiency",
                score=0.0,
                detail="no task was solved, so efficiency cannot be earned",
                weight=2.0,
            )
        )
        return

    step_ratio = sum(t["steps"] / max(t["budget_steps"], 1) for t in solved) / len(solved)
    scorecard.add(
        CheckResult(
            name="efficiency/step-budget",
            category="efficiency",
            score=max(0.0, min(1.0, (1.0 - step_ratio) / 0.5)),
            detail=f"solved tasks used {step_ratio:.0%} of their step budget on average "
            f"(full marks at 50% or below)",
            weight=2.0,
            requirement_ids=["R-108"],
        )
    )

    time_ratio = sum(t["seconds"] / max(t["budget_seconds"], 1) for t in solved) / len(solved)
    scorecard.add(
        CheckResult(
            name="efficiency/time-budget",
            category="efficiency",
            score=max(0.0, min(1.0, (1.0 - time_ratio) / 0.7)),
            detail=f"solved tasks used {time_ratio:.0%} of their time budget on average",
            requirement_ids=["R-108"],
        )
    )

    timeouts = [t for t in timings if t["seconds"] >= t["budget_seconds"]]
    scorecard.add(
        CheckResult.binary(
            name="efficiency/no-timeouts",
            category="efficiency",
            passed=not timeouts,
            detail=", ".join(t["task"] for t in timeouts) or "no task hit its time budget",
            requirement_ids=["R-102"],
        )
    )


# ---------------------------------------------------------------------------
# 8. health
# ---------------------------------------------------------------------------

PUBLIC_PACKAGES = ("morph", "bench", "selfimprove")


def run_health(scorecard: Scorecard, repo: Path = REPO_ROOT) -> None:
    modules = sorted(
        path
        for package in PUBLIC_PACKAGES
        for path in (repo / package).rglob("*.py")
        if "__pycache__" not in path.parts
    )

    broken: list[str] = []
    trees: dict[Path, ast.Module] = {}
    for path in modules:
        try:
            trees[path] = ast.parse(path.read_text("utf-8"), str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            broken.append(f"{path.relative_to(repo)}: {exc}")
    scorecard.add(
        CheckResult.binary(
            name="health/parses",
            category="health",
            passed=not broken,
            detail="\n".join(broken) or f"{len(trees)} modules parse",
            weight=2.0,
            requirement_ids=["R-804"],
        )
    )

    probe = subprocess.run(
        [sys.executable, "-c", "import morph, bench.runner, selfimprove.loop"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
    )
    scorecard.add(
        CheckResult.binary(
            name="health/imports",
            category="health",
            passed=probe.returncode == 0,
            detail=(probe.stderr or probe.stdout)[-1500:] or "imports clean",
            weight=2.0,
            requirement_ids=["R-804"],
        )
    )

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
            # Graded: going from 70% to 85% annotated should register.
            score=max(0.0, min(1.0, (ratio - 0.5) / 0.4)),
            detail=f"{ratio:.0%} of {total} public functions fully annotated"
            + (f"; first gaps: {unannotated[:5]}" if unannotated else ""),
            requirement_ids=["R-804"],
        )
    )

    swallowed: list[str] = []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                body = node.body
                only_pass = len(body) == 1 and isinstance(body[0], ast.Pass)
                if only_pass and node.type is None:
                    swallowed.append(f"{path.relative_to(repo)}:{node.lineno}")
    scorecard.add(
        CheckResult.binary(
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
    only: str | None = None,
    trace: bool = True,
    progress: ProgressFile | None = None,
) -> Scorecard:
    """Run every category and return the scorecard.

    ``trace`` prints progress to stderr as each suite and task completes;
    stdout stays clean for the scorecard JSON that CI pipes into a file.
    """
    cfg = config or Config(workspace=repo, provider="echo", image_backend="stub")
    tracer = TraceRenderer() if trace else None
    scorecard = Scorecard(
        metadata={
            "provider": cfg.provider,
            "model": cfg.model,
            "image_backend": cfg.image_backend,
            "python": sys.version.split()[0],
            "repo": str(repo),
            "suites": list(CAPABILITY_CATEGORIES) if only is None else [only],
            "note": (
                "provider=echo: capability tasks replay reference traces where one "
                "exists and are skipped where none does. This measures harness "
                "integrity, not model skill — run with a real model for a real score."
                if cfg.provider == "echo"
                else f"capability measured against {cfg.provider}/{cfg.model}"
            ),
        }
    )

    if not skip_requirements:
        if tracer:
            tracer.header("conformance suite")
        run_requirements(scorecard, repo)
        if tracer:
            gate = scorecard.by_category("requirements")
            failed = [r for r in gate if not r.passed]
            tracer.note(
                f"{len(gate) - len(failed)}/{len(gate)} tests pass"
                + (f" — FAILING: {failed[0].name}" if failed else "")
            )

    timings = await run_capability_suites(
        scorecard, cfg, only=only, tracer=tracer, progress=progress
    )

    if only is None:
        if tracer:
            tracer.header("robustness")
        await run_robustness(scorecard)
        if tracer:
            checks = scorecard.by_category("robustness")
            tracer.note(f"{sum(1 for r in checks if r.passed)}/{len(checks)} survived")

    run_efficiency(scorecard, timings)
    run_health(scorecard, repo)
    if tracer:
        tracer.note(f"composite {scorecard.composite:.1f}/100")
    return scorecard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench", description="Run the Morph benchmark")
    parser.add_argument("--output", help="Write the scorecard JSON here")
    parser.add_argument("--provider", default="echo", help="echo | ollama | google")
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--base-url")
    parser.add_argument("--image-backend", default="stub")
    parser.add_argument(
        "--only",
        choices=sorted(SUITES),
        help="Run a single capability suite (skips requirements and robustness)",
    )
    parser.add_argument(
        "--skip-requirements",
        action="store_true",
        help="Skip the pytest gate (useful when the benchmark is invoked from pytest)",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print the scorecard")
    parser.add_argument(
        "--no-trace", action="store_true", help="Do not print per-task progress to stderr"
    )
    args = parser.parse_args(argv)

    config = Config(
        workspace=REPO_ROOT,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        image_backend=args.image_backend,
    )
    scorecard = asyncio.run(
        run_benchmark(
            config,
            skip_requirements=args.skip_requirements or args.only is not None,
            only=args.only,
            trace=not args.no_trace,
        )
    )

    if not args.quiet:
        print(scorecard.render())
    if args.output:
        path = scorecard.write(args.output)
        print(f"scorecard written to {path}")
    return 0 if not scorecard.gated else 1


if __name__ == "__main__":
    raise SystemExit(main())
