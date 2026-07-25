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
  through Ollama fall in this category.
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
- **R-707** — The loop must never modify `REQUIREMENTS.md`,
  `tests/test_requirements.py`, `bench/scorecard.py`, or its own acceptance
  criteria. Any iteration that touches them is rejected outright.

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
| `requirements` | 40 | fraction of `tests/` passing |
| `capability` | 30 | fraction of `bench/tasks/` agent tasks solved |
| `robustness` | 15 | error-injection tasks survived |
| `efficiency` | 10 | steps & wall-time vs. per-task budget |
| `health` | 5 | import cleanliness, type hints, dead code |

`requirements` is a **gate**: if any test in `tests/test_requirements.py` fails,
the composite score is clamped to 0 and the iteration cannot be accepted.
