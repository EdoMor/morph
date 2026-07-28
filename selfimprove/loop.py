"""The self-improvement loop (R-701 … R-707).

    benchmark  →  feedback  →  Gemma edits the code  →  benchmark  →  keep or revert

Each iteration runs in its own git worktree, so the working branch is never left
broken (R-703). The edits are made by **Morph's own agent** (R-702): if a change
degrades the agent, the next iteration feels it immediately.

PROTECTED FILE — the loop may not modify this (R-707).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.scorecard import compare
from morph.agent import Agent
from morph.config import Config
from morph.llm import get_provider
from morph.tools import ToolRegistry, build_default_registry
from morph.trace import EventLog, ProgressFile, TraceRenderer

from .guard import violations
from .prompts import (
    DIAGNOSIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    append_history,
    build_diagnosis_prompt,
    build_implementation_prompt,
    load_history,
    select_target,
)
from .release import cut_release

log = logging.getLogger("selfimprove")

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = REPO_ROOT / "selfimprove" / "history.jsonl"
PROGRESS_PATH = REPO_ROOT / "selfimprove" / "progress.json"
WORKTREE_ROOT = Path(".morph") / "worktrees"
AGENT_MAX_STEPS = 60
AGENT_TIMEOUT = 3600.0
DIAGNOSIS_MAX_STEPS = 12
DIAGNOSIS_TIMEOUT = 1200.0
MIN_TARGET_GAIN = 0.02


class GitError(RuntimeError):
    pass


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


@dataclass
class Iteration:
    index: int
    base_commit: str
    branch: str
    worktree: Path
    score_before: float = 0.0
    score_after: float = 0.0
    accepted: bool = False
    rejection_reason: str = ""
    summary: str = ""
    files_changed: list[str] = field(default_factory=list)
    agent_steps: int = 0
    duration_s: float = 0.0
    deltas: dict[str, Any] = field(default_factory=dict)
    version: str = ""
    tag: str = ""
    target: str = ""
    diagnosis: str = ""
    target_score_before: float | None = None
    target_score_after: float | None = None
    tool_failures: list[str] = field(default_factory=list)
    measurement_after: dict[str, Any] | None = field(default=None, repr=False)

    def to_entry(self) -> dict[str, Any]:
        return {
            "ts": time.time(),
            "iteration": self.index,
            "base_commit": self.base_commit,
            "branch": self.branch,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "summary": self.summary,
            "files_changed": self.files_changed,
            "agent_steps": self.agent_steps,
            "duration_s": round(self.duration_s, 1),
            "deltas": self.deltas,
            "version": self.version,
            "tag": self.tag,
            "target": self.target,
            "diagnosis": self.diagnosis,
            "target_score_before": self.target_score_before,
            "target_score_after": self.target_score_after,
            "tool_failures": self.tool_failures,
        }


# ---------------------------------------------------------------------------


def _bench_config(root: Path, config: Config) -> Config:
    """Config for measuring.

    Capability is measured against the *same* model that does the editing, so the
    score is a real signal the loop can move. Measuring with ``echo`` instead
    replays reference traces, which pins capability at full marks and leaves the
    loop able to detect regressions but not improvements — useful as a fast
    smoke test, not as an objective. Set ``MORPH_BENCH_PROVIDER=echo`` to opt
    into that cheaper, fully deterministic mode.

    The image backend stays on ``stub`` either way: determinism there costs no
    signal and no GPU.
    """
    provider = os.environ.get("MORPH_BENCH_PROVIDER") or config.provider
    # A stochastic objective cannot distinguish improvement from sampling
    # noise. The same commit scored 58.9 and 64.1 on consecutive scheduled
    # runs at 0.2. Editing stays exploratory; measurement defaults to zero.
    temperature = float(os.environ.get("MORPH_BENCH_TEMPERATURE", "0"))
    return Config(
        workspace=root,
        provider=provider,
        model=config.model,
        base_url=config.base_url,
        temperature=temperature,
        image_backend="stub",
    )


def _agent_config(root: Path, config: Config) -> Config:
    """Config for the editing agent. This is where Gemma actually runs."""
    return Config(
        workspace=root,
        provider=config.provider,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        max_steps=AGENT_MAX_STEPS,
        image_backend=config.image_backend,
        # Meta-improvement is a local code task. Extra skill/MCP schemas consume
        # scarce attention in Gemma 4B and are not needed by either stage.
        skill_paths=[],
        allow_shell=True,
        shell_timeout=900.0,
    )


async def measure(
    root: Path, config: Config, progress: ProgressFile | None = None
) -> dict[str, Any]:
    """Score ``root`` in a fresh interpreter rooted at that candidate.

    This isolation is essential. Importing :mod:`bench.runner` once in the
    parent process also imports ``morph.agent`` once. The old implementation
    then evaluated every worktree with the *parent checkout's* Agent class, so
    candidate capability edits literally could not affect their own score.
    SICA evaluates each archived agent version from that version's code
    directory; the subprocess is Morph's lightweight equivalent.
    """
    return await _benchmark_subprocess(root, config, progress=progress)


async def measure_target(
    root: Path,
    config: Config,
    target_name: str,
    progress: ProgressFile | None = None,
) -> dict[str, Any] | None:
    """Re-run one exact capability task from the candidate checkout."""
    card = await _benchmark_subprocess(
        root, config, progress=progress, task_labels=[target_name]
    )
    return next(
        (result for result in card.get("results", []) if result.get("name") == target_name),
        None,
    )


async def _benchmark_subprocess(
    root: Path,
    config: Config,
    progress: ProgressFile | None = None,
    task_labels: list[str] | None = None,
) -> dict[str, Any]:
    cfg = _bench_config(root, config)
    with tempfile.TemporaryDirectory(prefix="morph-score-") as temporary:
        output = Path(temporary) / "scorecard.json"
        command = [
            sys.executable,
            "-m",
            "bench.runner",
            "--provider",
            cfg.provider,
            "--model",
            cfg.model,
            "--temperature",
            str(cfg.temperature),
            "--image-backend",
            cfg.image_backend,
            "--output",
            str(output),
            "--quiet",
        ]
        if cfg.base_url:
            command += ["--base-url", cfg.base_url]
        for label in task_labels or []:
            command += ["--task", label]

        env = dict(os.environ)
        env["PYTHONPATH"] = str(root) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def drain(
            reader: asyncio.StreamReader | None,
            stream: Any | None,
            heartbeat: bool,
        ) -> list[str]:
            lines: list[str] = []
            if reader is None:
                return lines
            while True:
                raw = await reader.readline()
                if not raw:
                    return lines
                line = raw.decode("utf-8", errors="replace")
                lines.append(line)
                if stream is not None:
                    stream.write(line)
                    stream.flush()
                activity = line.strip()
                if heartbeat and progress is not None and activity:
                    progress.update(activity=activity[-240:])

        # bench.runner reports its output path on stdout. The outer loop's
        # stdout is reserved for the final history JSON that CI tees into
        # loop-output.json, so capture benchmark stdout without forwarding it.
        stdout_task = asyncio.create_task(drain(process.stdout, None, False))
        stderr_task = asyncio.create_task(drain(process.stderr, sys.stderr, True))
        returncode = await process.wait()
        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)

        if output.is_file():
            return json.loads(output.read_text("utf-8"))
        detail = "".join((stdout + stderr)[-30:]).strip()
        raise RuntimeError(
            f"benchmark subprocess exited {returncode} without a scorecard"
            + (f":\n{detail}" if detail else "")
        )


def _create_worktree(repo: Path, branch: str, base: str) -> Path:
    path = repo / WORKTREE_ROOT / branch.replace("/", "-")
    if path.exists():
        _remove_worktree(repo, path, branch)
    path.parent.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "add", "-f", "-b", branch, str(path), base)
    return path


def _remove_worktree(repo: Path, path: Path, branch: str) -> None:
    git(repo, "worktree", "remove", "--force", str(path), check=False)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    git(repo, "worktree", "prune", check=False)
    git(repo, "branch", "-D", branch, check=False)


def _restricted_tools(config: Config, allowed: set[str]) -> ToolRegistry:
    """Give each stage only the tools needed for its role.

    Gemma 4B is materially more reliable with a small schema surface. The
    diagnosis stage cannot accidentally edit; the implementation stage does
    not see web, image, MCP, or skill tools unrelated to this local task.
    """
    registry = build_default_registry(config)
    for name in registry.names():
        if name not in allowed:
            registry.remove(name)
    return registry


def _score_for(scorecard: dict[str, Any], target_name: str) -> float | None:
    for result in scorecard.get("results", []):
        if result.get("name") == target_name and not result.get("skipped"):
            return float(result.get("score", 0.0))
    return None


def _is_capability_target(target_name: str) -> bool:
    try:
        from bench.tasks import ALL_TASKS

        return any(task.label == target_name for task in ALL_TASKS)
    except ImportError:
        return False


def _fixture_leaks(
    worktree: Path,
    base_commit: str,
    target_name: str,
    changed: list[str],
) -> list[str]:
    """Reject newly-created Morph files copied from a synthetic fixture.

    This is deliberately narrow: an existing ``morph/config.py`` may
    legitimately be the implementation to improve even if a benchmark also has
    a ``config.py`` fixture. A *new* ``morph/calc.py`` while working on the
    benchmark's temporary ``calc.py`` is unambiguous goal confusion.
    """
    try:
        from bench.tasks import ALL_TASKS

        task = next(item for item in ALL_TASKS if item.label == target_name)
    except (ImportError, StopIteration):
        return []

    fixture_names = {Path(relative).name for relative in task.files}
    leaks: list[str] = []
    for name in changed:
        normalised = name.replace("\\", "/")
        if not normalised.startswith("morph/") or Path(normalised).name not in fixture_names:
            continue
        existed = git(
            worktree,
            "ls-tree",
            "--name-only",
            base_commit,
            "--",
            normalised,
            check=False,
        )
        if not existed:
            leaks.append(normalised)
    return leaks


# ---------------------------------------------------------------------------


async def run_iteration(
    index: int,
    repo: Path,
    config: Config,
    baseline: dict[str, Any],
    history: list[dict[str, Any]],
    dry_run: bool = False,
    focus: str | None = None,
    keep_worktree: bool = False,
    progress_file: ProgressFile | None = None,
) -> Iteration:
    """One full cycle: edit in isolation, measure, keep or revert.

    ``keep_worktree`` leaves the iteration's worktree and branch in place so a
    human can inspect the diff. Off by default — otherwise every dry run
    accumulates a branch.
    """
    started = time.perf_counter()
    base_commit = git(repo, "rev-parse", "HEAD")
    branch = f"selfimprove/iter-{index}-{int(time.time())}"
    worktree = _create_worktree(repo, branch, base_commit)

    tracer = TraceRenderer()
    # Relative to the repository being worked on, not to this module's location.
    # PROGRESS_PATH points at the real checkout, so defaulting to it meant the
    # conformance suite — which drives run_iteration with a scratch repo, and
    # which the benchmark runs on every pass — overwrote the live heartbeat of
    # whatever run was in flight, and dirtied a file the loop commits.
    progress = progress_file or ProgressFile(Path(repo) / "selfimprove" / "progress.json")
    tracer.header(f"iteration {index} — {config.provider}/{config.model}")
    tracer.note(f"base {base_commit[:8]}, score to beat {baseline.get('composite', 0):.1f}")
    progress.update(
        phase="editing",
        iteration=index,
        base_commit=base_commit[:8],
        score_before=baseline.get("composite", 0.0),
        activity="building the prompt",
    )

    iteration = Iteration(
        index=index,
        base_commit=base_commit,
        branch=branch,
        worktree=worktree,
        score_before=baseline.get("composite", 0.0),
    )

    try:
        target = select_target(baseline, history)
        iteration.target = str((target or {}).get("name") or "")
        agent_config = _agent_config(worktree, config)

        # As in SICA's archive-explorer -> software-engineer sequence, planning
        # and implementation use separate contexts. Gemma no longer has to hold
        # the whole scorecard, history, task definition, repository inspection,
        # and edit syntax in one fragile conversation.
        diagnosis = ""
        if target is not None:
            tracer.header(f"diagnosis — {iteration.target}")
            progress.update(phase="diagnosing", activity="inspecting the target and real code")
            planner = Agent(
                config=agent_config,
                provider=get_provider(
                    config.provider,
                    model=config.model,
                    base_url=config.base_url,
                    temperature=0.0,
                ),
                tools=_restricted_tools(
                    agent_config, {"read_file", "list_dir", "glob", "grep"}
                ),
                system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
            )
            try:
                diagnosis_result = await _run_traced(
                    planner,
                    build_diagnosis_prompt(target, history, focus=focus),
                    DIAGNOSIS_MAX_STEPS,
                    tracer,
                    progress,
                    DIAGNOSIS_TIMEOUT,
                )
            finally:
                await planner.close()
            if diagnosis_result.error:
                iteration.rejection_reason = (
                    f"diagnosis failed: {diagnosis_result.error}"
                )
                iteration.score_after = iteration.score_before
                return iteration
            diagnosis = (diagnosis_result.text or "").strip()
            iteration.diagnosis = diagnosis[:3000]

        prompt = build_implementation_prompt(target, diagnosis, focus=focus)
        tracer.header(f"implementation — {iteration.target or 'fallback'}")
        progress.update(phase="editing", activity="implementing the diagnosis")

        # Morph improves Morph: the editing agent is this project's own agent
        # (R-702), in the isolated candidate worktree.
        agent = Agent(
            config=agent_config,
            provider=get_provider(
                config.provider,
                model=config.model,
                base_url=config.base_url,
                temperature=config.temperature,
            ),
            tools=_restricted_tools(
                agent_config,
                {
                    "read_file",
                    "write_file",
                    "edit_file",
                    "list_dir",
                    "glob",
                    "grep",
                    "shell",
                },
            ),
            system_prompt=SYSTEM_PROMPT,
            require_edit=True,
        )
        try:
            result = await _run_traced(
                agent, prompt, AGENT_MAX_STEPS, tracer, progress, AGENT_TIMEOUT
            )
        finally:
            await agent.close()

        iteration.summary = (result.text or "").strip()[:2000]
        iteration.agent_steps = result.steps
        iteration.tool_failures = [
            f"{call.get('tool', '?')}: {str(call.get('content') or '')[:400]}"
            for call in result.tool_calls
            if not call.get("ok")
        ][:8]

        if result.error:
            iteration.rejection_reason = f"agent failed: {result.error}"
            iteration.score_after = iteration.score_before
            return iteration

        iteration.files_changed = [
            name
            for name in _changed(worktree, base_commit)
            if not name.startswith(".morph/")
        ]
        if not iteration.files_changed:
            iteration.rejection_reason = "the agent made no changes"
            iteration.score_after = iteration.score_before
            return iteration

        # R-707: touching the goalposts rejects the whole iteration.
        breached = violations(worktree, base_commit)
        if breached:
            iteration.rejection_reason = f"modified protected files: {', '.join(breached)}"
            iteration.score_after = 0.0
            return iteration

        leaks = _fixture_leaks(
            worktree, base_commit, iteration.target, iteration.files_changed
        )
        if leaks:
            iteration.rejection_reason = (
                "copied synthetic benchmark fixture into Morph: "
                + ", ".join(leaks)
            )
            iteration.score_after = iteration.score_before
            return iteration

        # Cheap selection stage: a candidate must improve the exact task it was
        # designed for before spending another hour on a full sweep. This also
        # stops unrelated edits from winning on benchmark sampling noise.
        if iteration.target and _is_capability_target(iteration.target):
            iteration.target_score_before = _score_for(baseline, iteration.target)
            tracer.note(f"preflight: re-running {iteration.target}")
            progress.update(
                phase="measuring",
                activity=f"candidate preflight: {iteration.target}",
            )
            target_after = await measure_target(
                worktree, config, iteration.target, progress=progress
            )
            iteration.target_score_after = (
                float(target_after.get("score", 0.0)) if target_after else 0.0
            )
            before = iteration.target_score_before
            if before is None:
                iteration.rejection_reason = (
                    f"baseline has no comparable result for {iteration.target}"
                )
                iteration.score_after = iteration.score_before
                return iteration
            if iteration.target_score_after < before + MIN_TARGET_GAIN:
                iteration.rejection_reason = (
                    f"target did not improve ({before:.0%} → "
                    f"{iteration.target_score_after:.0%}; need +{MIN_TARGET_GAIN:.0%})"
                )
                iteration.score_after = iteration.score_before
                return iteration

        tracer.note(f"changed {len(iteration.files_changed)} file(s); re-measuring")
        progress.update(phase="measuring", activity="running the benchmark on the result")
        after = await measure(worktree, config, progress=progress)
        iteration.measurement_after = after
        iteration.score_after = after.get("composite", 0.0)
        iteration.deltas = compare(baseline, after)

        if after.get("gated"):
            iteration.rejection_reason = "conformance suite is failing after the change"
            return iteration
        if iteration.score_after < iteration.score_before:
            iteration.rejection_reason = (
                f"score regressed ({iteration.score_before:.1f} → {iteration.score_after:.1f})"
            )
            return iteration

        # R-704: accepted.
        if dry_run:
            iteration.accepted = True
            iteration.rejection_reason = "(dry run: not merged)"
            return iteration

        tracer.note(f"ACCEPTED {iteration.score_before:.1f} -> {iteration.score_after:.1f}")
        _commit_and_merge(repo, worktree, branch, iteration)

        # Every accepted iteration is a new version of the agent, cut before the
        # next iteration starts — so the next one improves the version that was
        # just released, not the one before it (R-715). The bump is made here
        # rather than by the model: nothing is gained by letting the change under
        # test also choose its own version number.
        release = cut_release(
            summary=iteration.summary,
            score_before=iteration.score_before,
            score_after=iteration.score_after,
            repo=repo,
        )
        iteration.version = str(release.version)
        iteration.tag = release.tag
        log.info("Cut %s", release.tag)

        iteration.accepted = True
        return iteration

    except asyncio.TimeoutError:
        iteration.rejection_reason = f"the agent exceeded {AGENT_TIMEOUT:g}s"
        iteration.score_after = iteration.score_before
        return iteration
    except Exception as exc:  # noqa: BLE001 - one bad iteration must not end the loop
        log.exception("Iteration %d failed", index)
        iteration.rejection_reason = f"{type(exc).__name__}: {exc}"
        iteration.score_after = iteration.score_before
        return iteration
    finally:
        iteration.duration_s = time.perf_counter() - started
        if not iteration.accepted and iteration.rejection_reason:
            tracer.note(f"rejected: {iteration.rejection_reason}")
        progress.update(
            phase="idle",
            activity=("accepted " + iteration.tag) if iteration.accepted else "rejected",
            accepted=iteration.accepted,
            score_after=iteration.score_after,
        )
        if keep_worktree:
            log.info("Worktree kept for inspection: %s (branch %s)", worktree, branch)
        else:
            _remove_worktree(repo, worktree, branch)


async def _run_traced(
    agent: Agent,
    prompt: str,
    max_steps: int,
    tracer: TraceRenderer,
    progress: ProgressFile,
    timeout: float,
):
    """Drive the agent through its event stream so the run can be watched live.

    ``agent.run()`` returns only the final result; consuming ``agent.stream()``
    gives the same result plus every step on the way to it (R-718).
    """
    from morph.agent import RunResult

    async def consume() -> RunResult:
        outcome = RunResult()
        async for event in agent.stream(prompt, max_steps=max_steps):
            tracer.event(event)
            progress.observe(event)
            if event.type == "done":
                outcome = RunResult(**event.data["result"])
        return outcome

    return await asyncio.wait_for(consume(), timeout=timeout)


def _changed(worktree: Path, base: str) -> list[str]:
    tracked = git(worktree, "diff", "--name-only", base, check=False)
    untracked = git(worktree, "ls-files", "--others", "--exclude-standard", check=False)
    return sorted(
        {line.strip() for line in f"{tracked}\n{untracked}".splitlines() if line.strip()}
    )


def _commit_and_merge(repo: Path, worktree: Path, branch: str, iteration: Iteration) -> None:
    """Commit the accepted work and fast-forward the main working branch."""
    git(worktree, "add", "-A")
    message = (
        f"selfimprove: iteration {iteration.index} "
        f"({iteration.score_before:.1f} -> {iteration.score_after:.1f})\n\n"
        f"{iteration.summary[:1200]}\n\n"
        f"Files: {', '.join(iteration.files_changed[:20])}"
    )
    git(worktree, "commit", "-m", message)
    git(repo, "merge", "--ff-only", branch)


def _feedback_from(scorecard: dict[str, Any]) -> str:
    """Rebuild a failure digest from a serialised scorecard.

    Leads with the difficulty frontier and the nearest misses rather than a flat
    list of everything red. Pointing a model at the hardest failure is how a loop
    stalls; pointing it at the closest one is how it climbs (R-710).
    """
    lines = [f"Composite score: {scorecard.get('composite', 0):.1f}/100.", ""]

    if scorecard.get("gated"):
        lines.append(
            "**The conformance suite is failing.** The composite is clamped to 0 until "
            "it is green. Fix that before anything else.\n"
        )

    diagnostics = scorecard.get("diagnostics") or {}
    if diagnostics:
        lines.append("Difficulty frontier per suite — the tier where you stop being reliable:")
        for category, data in diagnostics.items():
            profile = data.get("tier_profile") or {}
            shape = " ".join(f"T{t}:{v:.2f}" for t, v in sorted(profile.items()))
            lines.append(
                f"- **{category}** — frontier T{data.get('frontier', 0)}, "
                f"{data.get('headroom', 0):.1f} points unearned "
                f"[{data.get('calibration', '?')}] — {shape}"
            )
        lines.append(
            "\nThe cheapest points are one tier above each frontier. A suite already at "
            "T5 has nothing left to give; work where the profile drops off."
        )

    targets = scorecard.get("next_targets") or []
    if targets:
        lines += ["", "Nearest misses — closest to solved, attack these first:"]
        for target in targets:
            ids = target.get("requirement_ids") or []
            suffix = f" (requirements: {', '.join(ids)})" if ids else ""
            tier = f" [tier {target['tier']}]" if target.get("tier") else ""
            lines.append(
                f"\n### {target['name']} — scored {target.get('score', 0):.0%}{tier}{suffix}"
            )
            lines.append((target.get("detail") or "(no detail)")[:1200])

    target_names = {t.get("name") for t in targets}
    others = [
        r
        for r in scorecard.get("results", [])
        if not r.get("passed") and not r.get("skipped") and r.get("name") not in target_names
    ]
    if others:
        lines += ["", f"Also failing ({len(others)}):"]
        for failure in others[:12]:
            lines.append(
                f"- {failure['name']} — {failure.get('score', 0):.0%} — "
                f"{(failure.get('detail') or '')[:180]}"
            )

    warnings = scorecard.get("instrument_warnings") or []
    if warnings:
        lines += ["", "Benchmark calibration (report these; do not try to fix them yourself):"]
        lines += [f"- {w}" for w in warnings]

    if not others and not targets:
        lines.append(
            "\nEverything currently measured passes. Say so in your summary rather than "
            "inventing a change — the benchmark needs harder tasks, which is a human's call."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------


async def run_loop(
    config: Config | None = None,
    iterations: int = 1,
    repo: Path | None = None,
    dry_run: bool = False,
    focus: str | None = None,
    history_path: Path | None = None,
    keep_worktree: bool = False,
    scorecard_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run ``iterations`` improvement cycles. Returns one history entry each."""
    repo = Path(repo or REPO_ROOT)
    cfg = config or Config(workspace=repo)
    history_file = Path(history_path or HISTORY_PATH)

    if git(repo, "rev-parse", "--is-inside-work-tree", check=False) != "true":
        raise GitError(f"{repo} is not a git repository; the loop needs git for isolation")

    progress_path = Path(history_path).parent / "progress.json" if history_path else PROGRESS_PATH
    progress = ProgressFile(
        progress_path,
        # The same stream the console trace renders, kept as JSONL so the
        # dashboard can show the run as it happens rather than after it (R-721).
        events=EventLog(progress_path.parent / "live-trace.jsonl"),
    )
    progress.update(phase="baseline", activity="scoring the current code", iterations=iterations)

    scorecard_file = Path(
        scorecard_path or (repo / "selfimprove" / "scorecard.json")
    )
    previous_scorecard_file = (
        repo / "selfimprove" / "previous-scorecard.json"
        if scorecard_path is None
        else scorecard_file.with_name("previous-" + scorecard_file.name)
    )
    if scorecard_file.is_file():
        previous_scorecard_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scorecard_file, previous_scorecard_file)

    tracer = TraceRenderer()
    tracer.header(f"baseline — {cfg.provider}/{cfg.model}")
    baseline = await measure(repo, cfg, progress=progress)
    scorecard_file.parent.mkdir(parents=True, exist_ok=True)
    scorecard_file.write_text(json.dumps(baseline, indent=2), "utf-8")
    log.info("Baseline score: %.1f", baseline.get("composite", 0.0))
    tracer.note(f"baseline composite {baseline.get('composite', 0.0):.1f}")

    entries: list[dict[str, Any]] = []
    for index in range(1, iterations + 1):
        history = load_history(history_file)
        iteration = await run_iteration(
            index=index,
            repo=repo,
            config=cfg,
            baseline=baseline,
            history=history,
            dry_run=dry_run,
            focus=focus,
            keep_worktree=keep_worktree,
            progress_file=progress,
        )
        entry = iteration.to_entry()
        append_history(history_file, entry)  # R-705
        entries.append(entry)

        log.info(
            "Iteration %d: %s (%.1f -> %.1f) %s",
            index,
            "accepted" if iteration.accepted else "rejected",
            iteration.score_before,
            iteration.score_after,
            iteration.rejection_reason,
        )

        if (
            iteration.accepted
            and not dry_run
            and iteration.measurement_after is not None
        ):
            # The accepted candidate was already measured in its own fresh
            # interpreter. Re-running the same full sweep after merging doubled
            # wall time and introduced a second noisy number for identical code.
            baseline = iteration.measurement_after
            scorecard_file.write_text(json.dumps(baseline, indent=2), "utf-8")

    progress.update(phase="done", activity=f"{sum(1 for e in entries if e['accepted'])}"
                    f"/{len(entries)} accepted")
    return entries


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="selfimprove", description="Gemma improves Morph")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--base-url")
    parser.add_argument("--focus", help="Steer this run at a specific area")
    parser.add_argument("--dry-run", action="store_true", help="Measure but never merge")
    parser.add_argument(
        "--keep-worktree",
        action="store_true",
        help="Leave each iteration's worktree and branch in place for inspection",
    )
    parser.add_argument("--repo", default=str(REPO_ROOT))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = Config(
        workspace=Path(args.repo),
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )
    entries = asyncio.run(
        run_loop(
            config,
            iterations=args.iterations,
            repo=Path(args.repo),
            dry_run=args.dry_run,
            focus=args.focus,
            keep_worktree=args.keep_worktree,
        )
    )

    print(json.dumps(entries, indent=2, default=str))
    # No accepted attempt is a normal completed run. Real exceptions still
    # propagate and fail CI, so the workflow no longer needs a blanket
    # ``|| true`` that also hid crashes.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
