# Morph — Requirements

This document is the **contract**. It is the single source of truth for what the
system must do. Every requirement has a stable ID (`R-###`). Every requirement is
either:

- **testable** — there is at least one test in `tests/` referencing its ID, or
- **scored** — the benchmark in `bench/` produces a numeric signal for it.

The self-improvement loop (`selfimprove/`) reads this file verbatim into the
model's context. Changing this file changes what the system optimises for.

> **Rule for the improving model:** you may not edit this file, and you may not
> edit `tests/test_requirements.py`. Those are the goalposts. Move the code, not
> the goalposts.

---

## 0. Product definition

**Morph** is a self-hosted coding-agent platform. The target is parity with what
the Claude app can do, plus first-class image generation, running on hardware the
user controls — including a phone.

| Pillar | Requirement range |
| --- | --- |
| Agent core | R-100 |
| Tools | R-200 |
| Skills | R-300 |
| MCP | R-400 |
| Image generation | R-500 |
| Server & mobile client | R-600 |
| Self-improvement loop | R-700 |
| Engineering quality | R-800 |

---

## 1. Agent core (R-1xx)

- **R-101** — The agent runs a tool-use loop: send conversation to the model,
  execute any requested tools, append results, repeat until the model returns a
  final text answer or the step budget is exhausted.
- **R-102** — The loop must terminate. A configurable `max_steps` (default 24)
  bounds every run; exceeding it ends the run with `stop_reason="max_steps"`
  rather than hanging or raising.
- **R-103** — Model providers are pluggable behind one interface. At minimum:
  `ollama` (local Gemma), `google` (Gemma/Gemini API), and `echo` (deterministic,
  offline, used by tests). Selecting a provider must not require code changes.
- **R-104** — The default model is a **Gemma** model. Gemma is the model that
  writes Morph's code; Morph must be able to run on it.
- **R-105** — Providers that lack native function-calling must still support
  tools, via a documented text protocol the agent parses. Gemma models exposed
  through Ollama fall in this category. Because the protocol is JSON emitted by
  a language model, it must be **forgiving of the mistakes models actually
  make** and must never fail silently:
  - a stray backslash — what a model produces every time it writes a regex — is
    repaired rather than rejected, without corrupting escapes that were correct;
  - a block that still cannot be parsed is reported back to the model as a
    failed tool result so it can retry, never dropped and treated as prose.
    Dropping it ends the run at that step having done nothing, with the model
    never learning why.
- **R-106** — Conversations are persisted as append-only JSONL sessions and can
  be resumed by id, with no loss of tool calls or results.
- **R-107** — A run emits a structured event stream (`text`, `tool_use`,
  `tool_result`, `error`, `done`) suitable for streaming to a UI.
- **R-108** — Every tool call is recorded with its arguments, result, duration
  and success flag. A run's cost/latency must be reconstructable from the log.
- **R-109** — A failing tool must not kill the run. The error is returned to the
  model as a tool result so it can recover.

## 2. Tools (R-2xx)

- **R-201** — Tools are declared with a name, description and JSON Schema, and
  are discoverable through a registry. Registering a tool is the only step
  required to expose it to the model.
- **R-202** — Filesystem tools: `read_file`, `write_file`, `edit_file`,
  `list_dir`, `glob`, `grep`.
- **R-203** — `edit_file` performs exact string replacement and **fails loudly**
  when the target string is absent or ambiguous. Silent no-ops are a defect.
- **R-204** — A shell tool executes commands with a timeout and captured
  stdout/stderr/exit code.
- **R-205** — All filesystem and shell access is confined to a workspace root.
  Path traversal outside the root is rejected before any I/O happens.
- **R-206** — Web tools: `web_fetch` (URL → text) and `web_search`. Both degrade
  gracefully to a clear error when offline or unconfigured.
- **R-207** — Tool arguments are validated against their schema before
  execution; invalid arguments produce a tool error, never an exception.

## 3. Skills (R-3xx)

- **R-301** — Skills are directories containing a `SKILL.md` with YAML
  frontmatter (`name`, `description`, optional `allowed-tools`), compatible with
  the Claude skill format.
- **R-302** — Skills are discovered from a search path at startup and exposed to
  the model as callable capabilities.
- **R-303** — Skill bodies are loaded **lazily** — only the name and description
  enter the system prompt; the body loads when the skill is invoked. This keeps
  context cost proportional to use, not to the number of installed skills.
- **R-304** — A malformed skill is skipped with a warning and never prevents
  startup.

## 4. MCP (R-4xx)

- **R-401** — Morph is an MCP **client**: it connects to servers declared in
  config, over `stdio` and `http`.
- **R-402** — Tools exposed by MCP servers are merged into the tool registry
  under a namespaced name (`mcp__<server>__<tool>`) and are indistinguishable
  from native tools at the call site.
- **R-403** — A server that fails to start, times out, or crashes mid-session is
  isolated: its tools are dropped and the rest of the system continues.
- **R-404** — The MCP handshake follows the spec: `initialize` →
  `notifications/initialized` → `tools/list`, over JSON-RPC 2.0.

## 5. Image generation (R-5xx)

- **R-501** — A `generate_image` tool is available to the agent and to the UI.
- **R-502** — Image backends are pluggable. Shipped: `flux` (hosted FLUX
  endpoint), `gemini` (Google image model), `local` (Diffusers/ComfyUI), and
  `stub` (deterministic offline generator used by tests).
- **R-503** — The image flow supports prompt, negative prompt, size, seed and
  count; identical inputs with a fixed seed produce identical outputs on the
  `stub` backend.
- **R-504** — Generated images are written to the workspace and returned as file
  paths plus a data URI preview, so a phone client can display them without a
  second round trip.
- **R-505** — A missing API key produces an actionable error naming the exact
  env var required — never a stack trace.

## 6. Server & mobile client (R-6xx)

- **R-601** — An HTTP server exposes the agent: `POST /api/chat` (streaming
  SSE), `GET /api/sessions`, `GET /api/tools`, `GET /api/skills`,
  `GET /api/health`.
- **R-602** — A web client is served from the same origin and is usable on a
  phone: responsive down to 360 px, touch targets ≥ 44 px, no horizontal scroll.
- **R-603** — The client is an installable PWA (web app manifest + service
  worker) so it can be added to a phone home screen.
- **R-604** — The full stack runs locally with no third-party service beyond the
  chosen model backend. `morph serve` on a laptop or phone-accessible host is
  enough.
- **R-605** — The server never trusts client input for filesystem paths;
  workspace confinement (R-205) applies to every request.

## 7. Self-improvement loop (R-7xx)

- **R-701** — `selfimprove` runs a closed loop: **benchmark → feedback →
  Gemma edits the code → benchmark again → keep or revert.**
- **R-702** — The loop uses **Morph's own agent** to make the edits. Morph
  improves Morph; a regression in the agent is felt immediately by the loop.
- **R-703** — Each iteration works on an isolated git branch/worktree. The main
  branch is never left in a failing state.
- **R-704** — An iteration is **accepted** only if: the requirement suite passes,
  and the composite score is ≥ the previous best. Otherwise it is reverted and
  recorded as a failed attempt.
- **R-705** — Every iteration appends to `selfimprove/history.jsonl`: timestamp,
  base commit, score before/after, per-category deltas, accepted/rejected, and
  the model's stated rationale. The history is fed back into the next prompt so
  the model does not repeat a failed approach.
- **R-706** — The loop runs unattended on GitHub Codespaces / GitHub Actions and
  is safe to run on a schedule.
- **R-713** — Accepted iterations are **published to the default branch
  automatically**, as ordinary commits, with no human in the loop. Because
  nothing downstream reviews them, publishing carries its own guarantees:
  - the whole run is re-verified after the loop, not just each iteration —
    iterations compose, and two individually safe changes can break together;
  - a composite regression across the run refuses to publish, even if every
    individual iteration was accepted;
  - if the branch moved during the run, the work is rebased onto it and the
    conformance suite is **re-run before pushing** — a rebase is a merge;
  - the push is never forced, and a lost race is retried from a fresh fetch. A
    loop that can overwrite history is one bad iteration away from deleting the
    project;
  - a rebase conflict stops and leaves it for a human rather than guessing.
- **R-714** — The attempt history survives the machine that produced it. A
  scheduled loop on ephemeral runners must commit `selfimprove/history.jsonl`,
  or R-705's "do not repeat a rejected approach" holds only within a single run
  — which is never the interesting case.
- **R-715** — Every accepted iteration produces a **new version** of the agent,
  cut before the next iteration begins, so each iteration improves the version
  that was just released rather than the one before it. The version is a single
  source of truth (`morph.__version__`), bumped by the loop and not by the
  model — a change under test has nothing to gain from choosing its own version
  number. Version tags are created **after** publishing succeeds, never before:
  a rebase rewrites every SHA, and a tag made earlier would point at an orphan.
  A tag that already exists is never moved.
- **R-717** — Progress is **public and legible without reading the repository**.
  A GitHub Pages dashboard shows the current score, its category and difficulty
  breakdown, the score over time, the available releases, and **every attempt —
  accepted and rejected — with the reason for each decision**. A self-improving
  system that only publishes its successes is not reporting, it is advertising;
  the rejected iterations are the part that shows the guard rails working. The
  site is static, built from data the loop already commits, and renders on a
  phone.
- **R-716** — Every version is downloadable and installable on a phone. The loop
  builds an Android APK per version and attaches it to a GitHub Release, with
  the APK's `versionName`/`versionCode` derived from `morph.__version__` so the
  app and the agent that produced it always carry the same number. The APK is a
  **client**: it asks for the address of the machine running `morph serve`, and
  ships no agent, model or conversation data of its own.
- **R-707** — The loop must never modify `REQUIREMENTS.md`,
  `tests/test_requirements.py`, `bench/scorecard.py`, `bench/tasks/`, or its own
  acceptance criteria. Any iteration that touches them is rejected outright.
  Authoring benchmark tasks is included deliberately: a loop that can write its
  own tasks will write easy ones, which is the same failure as editing the
  scorer, one level down.

## 7b. Benchmark calibration (R-70x)

A self-improving loop is only as good as the gradient it climbs. These
requirements are about the *instrument*, not the system under test.

- **R-708** — Every capability suite (`coding`, `tool_use`, `mcp`, `skills`)
  must span five difficulty tiers, T1 (one tool call) to T5 (multi-file,
  adversarial, or requiring a plan), with at least one task at each of T1-T5 and
  no tier holding more than half the suite. If every task is the same difficulty
  there is no gradient: the loop either solves them all and stops learning, or
  solves none and gets no signal.
- **R-709** — Capability tasks are graded on a rubric of weighted criteria and
  score continuously in `[0, 1]`. Binary pass/fail discards the information the
  loop needs — "parses, preserves old behaviour, misses the edge case" must
  score above "deleted the file". Rubrics must include critical criteria such
  that a destructive answer can never out-score a partial one.
- **R-710** — The benchmark reports a **frontier** per suite: the hardest tier
  the system reliably handles. It also reports the **nearest misses** — unsolved
  checks ordered easiest-and-closest first — and the loop's prompt leads with
  them. Pointing the model at the hardest failure is how a loop stalls.
- **R-711** — The benchmark diagnoses its own calibration per suite and says so
  in the scorecard: `saturated` (everything passes — cannot show progress),
  `floored` (almost nothing passes — improvements will not register), `partial`
  (some checks did not run, so calibration cannot be judged), or `healthy`.
  A saturated or floored suite is a broken instrument, and the fix is new tasks
  by a human — which is why R-707 forbids the loop from writing them.
- **R-712** — Tasks that cannot run in a given mode are **skipped**, not scored
  zero. A deterministic replay of reference traces must not be presentable as a
  capability measurement.

## 8. Engineering quality (R-8xx)

- **R-801** — `pytest` passes from a clean checkout with no network access.
- **R-802** — No hard dependency on a paid API to run the test suite or the
  benchmark.
- **R-803** — Secrets come from the environment only. No key is ever written to
  disk, logged, or committed.
- **R-804** — Public functions carry type hints; the package imports cleanly on
  Python 3.11+.
- **R-805** — Every requirement ID in this document is referenced by at least one
  test. `tests/test_requirements.py::test_every_requirement_is_covered` enforces
  this, and is itself the guard against the spec drifting away from the code.

---

## Scoring

The benchmark (`python -m bench.runner`) produces a composite score in `[0, 100]`:

| Category | Weight | Signal |
| --- | --- | --- |
| `requirements` | 25 | fraction of `tests/` passing — **the gate** |
| `coding` | 20 | agent tasks: change software correctly (T1-T5) |
| `tool_use` | 15 | agent tasks: tool selection, precision, restraint, recovery (T1-T5) |
| `mcp` | 12 | agent tasks: use tools discovered at runtime from MCP servers (T1-T5) |
| `skills` | 12 | agent tasks: find, load and follow packaged instructions (T1-T5) |
| `robustness` | 10 | error-injection checks survived |
| `efficiency` | 4 | steps & wall-time vs. per-task budget |
| `health` | 2 | import cleanliness, type hints, dead code |

`requirements` is a **gate**: if any test in `tests/test_requirements.py` fails,
the composite score is clamped to 0 and the iteration cannot be accepted.

Capability checks are weighted by difficulty tier — T1:1, T2:2, T3:3, T4:5,
T5:8 — so the score reflects *what* was solved, not just how many. The weights
are deliberately sub-exponential: a system that solves every T1-T2 and nothing
else still earns a visible score, which is what makes early progress legible.

A task counts as **solved** at 80% of its rubric, leaving room for stylistic
variation without rewarding a near-miss as a win.
