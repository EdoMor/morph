# Morph

A self-hosted coding agent platform that **improves itself**. Gemma reads the
requirements, edits Morph's source code through Morph's own agent, gets scored by
a benchmark, and the change is kept only if it actually made things better.

```
   ┌──────────────────────────────────────────────────────────┐
   │                                                          │
   ▼                                                          │
benchmark  ──►  feedback  ──►  Gemma edits the code  ──►  benchmark
   ▲              (failing checks,      (via Morph's           │
   │               score deltas,         own agent, in a       │
   │               past attempts)        git worktree)         │
   │                                                           ▼
   └────────────────────  keep if better, revert if not  ◄─────┘
```

The contract it optimises against is [`REQUIREMENTS.md`](REQUIREMENTS.md). The
loop cannot edit it.

---

## What Morph is

| | |
| --- | --- |
| **Agent** | Tool-use loop with file/search/shell/web tools, bounded steps, persisted sessions |
| **Models** | Gemma via Ollama (local), Gemma/Gemini via Google AI, or a deterministic offline stub |
| **Skills** | Claude-compatible `SKILL.md` directories, lazily loaded |
| **MCP** | Full client — stdio and HTTP servers, namespaced tools, failure isolation |
| **Images** | Pluggable image flow: FLUX, Gemini, local Diffusers/ComfyUI, or offline stub |
| **Client** | Installable mobile PWA served from the same process |
| **Deps** | `httpx` and `pyyaml`. No web framework, no ML stack — it runs on a phone |

## Quick start

```bash
pip install -e ".[dev]"

# 1. Everything works offline with the deterministic provider
morph --provider echo chat "hello"

# 2. Point it at a real Gemma
ollama serve & ollama pull gemma3:4b
morph chat "summarise this repository"

# 3. Serve the mobile app
morph serve            # http://127.0.0.1:8787
```

Open that URL on your phone (same network, or a forwarded Codespaces port) and
**Add to Home Screen**. It installs as a standalone app.

## Running it on your phone

Two options, both fully local:

**A — phone as client.** Run `morph serve --host 0.0.0.0` on a laptop, Pi, or
Codespace. Open the URL on the phone and install the PWA. The model runs on the
bigger machine; the phone is a real app.

**B — phone as host.** In Termux:

```bash
pkg install python git
pip install -e .
MORPH_PROVIDER=ollama MORPH_MODEL=gemma3:1b morph serve
```

The server is stdlib-only asyncio precisely so this works — there is no compiled
ASGI stack to build on-device.

## The self-improvement loop

```bash
python -m bench.runner                      # score the current code
python -m bench.runner --only mcp           # one suite, for a fast iteration
python -m selfimprove.loop --iterations 3   # let Gemma try to raise it
python -m selfimprove.loop --dry-run        # measure without merging
```

Each iteration:

1. **Measures** the current code → composite score + failure digest.
2. **Builds a prompt** from the requirements, the failures, and every previous
   attempt with its outcome — so the model stops re-trying dead ends.
3. **Runs Morph's own agent** with Gemma, in a fresh `git worktree`, with real
   file and shell tools.
4. **Re-measures**, then **accepts or reverts**:
   - conformance suite red → rejected;
   - score went down → rejected;
   - a protected file was touched → rejected outright;
   - otherwise → committed and fast-forwarded onto the working branch.
5. **Appends** the outcome to `selfimprove/history.jsonl`.

### Scoring

| Category | Weight | What it measures |
| --- | --- | --- |
| `requirements` | 25 | `tests/` — also a **gate**: red suite clamps the score to 0 |
| `coding` | 20 | change software correctly |
| `tool_use` | 15 | tool selection, argument precision, restraint, recovery |
| `mcp` | 12 | use tools discovered at runtime from MCP servers |
| `skills` | 12 | find, load and actually follow packaged instructions |
| `robustness` | 10 | twelve error-injection checks (path escape, dead MCP server, runaway loop…) |
| `efficiency` | 4 | steps and wall time against per-task budgets |
| `health` | 2 | parses, imports, annotations, no swallowed exceptions |

### Keeping the benchmark climbable

A self-improving loop is only as good as the gradient it climbs. A suite where
everything passes teaches nothing; so does one where nothing does. Three things
keep the instrument usable:

**Difficulty tiers.** Each of the four capability suites spans T1-T5 — from one
tool call to multi-file adversarial work — so at any level of competence some
tasks are solved, some are borderline, and some are out of reach. The borderline
band *is* the signal. Checks are weighted T1:1, T2:2, T3:3, T4:5, T5:8, so the
score reflects what was solved, not just how many.

**Graded rubrics, not pass/fail.** Every task scores continuously in `[0, 1]`
against weighted criteria. "Parses, preserves old behaviour, misses the edge
case" scores above "deleted the file", so a half-finished iteration is
distinguishable from a wasted one. Criteria marked *critical* — "the tests were
not edited", "existing behaviour intact" — **gate** the task rather than scoring
it: they zero a destructive answer, but earn nothing, because an agent that does
nothing at all would otherwise collect them for free.

Driving the real suites with progressively truncated reference traces (a
stand-in for models of increasing competence) gives:

| competence | mean score | solved | distinct scores |
| --- | --- | --- | --- |
| 0% | 0.11 | 0/29 | 9 |
| 50% | 0.41 | 6/29 | 18 |
| 75% | 0.83 | 21/29 | 11 |
| 100% | 1.00 | 29/29 | 1 |

Monotonic, with a near-zero floor and real resolution in the middle. That
property is itself a test — `test_R_709_benchmark_discriminates_between_competence_levels`
fails if the rubrics ever collapse back toward binary.

**Self-diagnosis.** The scorecard reports, per suite, a **frontier** (the hardest
tier handled reliably), the **nearest misses** (unsolved checks ordered
easiest-and-closest first — what the loop's prompt leads with, because pointing a
model at the hardest failure is how a loop stalls), and a **calibration** verdict:

- `healthy` — some solved, some not; the loop has somewhere to go
- `saturated` — everything passes; the suite can no longer show progress
- `floored` — almost nothing passes; improvements will not register
- `partial` — some checks did not run, so calibration cannot be judged

`saturated` and `floored` mean the benchmark is broken as an instrument, and the
fix is new tasks written by a human — which is why the loop is forbidden from
touching `bench/tasks/`.

> **Reading the score.** With `--provider echo` the capability tasks replay
> reference traces where one exists and are *skipped* where none does — never
> scored zero, since a replay must not pose as a measurement. A green echo run
> reports 100/100 and flags every suite `partial`. That measures the harness. The
> number that matters comes from running against real Gemma.

### Why the goalposts are protected

A system that can edit its own scorer will optimise by editing its scorer. So
`REQUIREMENTS.md`, `tests/test_requirements.py`, `bench/scorecard.py`,
**`bench/tasks/`** and the loop itself are off limits (`selfimprove/guard.py`),
and any iteration that touches them is rejected in full — not partially
credited.

The task suites are on that list deliberately. A loop allowed to author its own
benchmark tasks will author easy ones, which is the same failure as editing the
scorer, one level down. Growing the benchmark stays a human's job; the
calibration verdict tells the human when it needs growing.

If the model thinks a requirement is wrong, it says so in its summary and a
human decides.

## Running the loop on GitHub

**Codespaces** — open the repo in a Codespace. `.devcontainer/setup.sh` installs
Ollama, pulls Gemma, and verifies the suite. Then:

```bash
python -m selfimprove.loop --iterations 5
```

**Actions** — `.github/workflows/self-improve.yml` runs nightly (and on demand).
It scores the baseline, runs the loop, re-scores, re-runs the gate, pushes
accepted iterations to a branch, and opens a PR with the before/after scorecard.
A human still merges.

## Configuration

`morph.json` in the workspace root, overridden by environment variables:

```json
{
  "provider": "ollama",
  "model": "gemma3:12b",
  "max_steps": 24,
  "image_backend": "flux",
  "skill_paths": ["skills"],
  "mcp_servers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    },
    "remote": { "transport": "http", "url": "https://example.com/mcp" }
  }
}
```

| Variable | Purpose |
| --- | --- |
| `MORPH_PROVIDER` / `MORPH_MODEL` | model selection |
| `MORPH_IMAGE_BACKEND` | `stub` \| `flux` \| `gemini` \| `local` |
| `MORPH_BENCH_PROVIDER` | force the benchmark's provider (e.g. `echo` for a fast run) |
| `GOOGLE_API_KEY`, `FLUX_API_KEY`, `MORPH_SEARCH_API_KEY` | credentials — environment only, never written to disk |

## Skills

Drop a directory under `skills/`:

```
skills/changelog/SKILL.md
```

```markdown
---
name: changelog
description: House style for writing CHANGELOG entries.
allowed-tools: read_file, edit_file
---

Entries are `- <past-tense verb> <what changed>`, appended to `## Unreleased`.
```

Only the name and description reach the system prompt. The body loads when the
model calls `load_skill`, so a hundred installed skills cost about a hundred
lines of context, not a hundred documents.

## Layout

```
morph/          agent core, tools, skills, MCP, API, server, CLI
webapp/         mobile PWA (no build step)
bench/          scorecard, plus coding / tool_use / mcp / skills / robustness suites
selfimprove/    the loop, its prompts, and the guard rails
tests/          the conformance suite — one test per requirement ID
```

## Development

```bash
python -m pytest tests -q          # must be green, offline, no credentials
python -m bench.runner             # full scorecard
python scripts/make_icons.py       # regenerate PWA icons from icon.svg
```

Adding a requirement means adding an `R-###` to `REQUIREMENTS.md` **and** a test
that references it — `test_R_805_every_requirement_is_covered` fails otherwise.
