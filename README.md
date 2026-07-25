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
| **Client** | Installable mobile PWA, plus an Android APK released per version |
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
5. **Cuts a version** — every accepted iteration bumps `morph.__version__`,
   writes a changelog entry, and becomes a `release: vX.Y.Z` commit. The next
   iteration therefore improves the version that was just released, not the one
   before it. The bump is made by the loop, not the model: a change under test
   has nothing to gain from choosing its own version number.
6. **Appends** the outcome to `selfimprove/history.jsonl`.

Then, once per run: publish to `main`, tag each version, and **build an APK per
version and attach it to a GitHub Release**.

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

**Actions** — `.github/workflows/self-improve.yml` runs nightly (and on demand),
and **commits evolved code straight to `main`**. No PR, no human in the loop.

```
checkout main
  → baseline score
  → run the loop (each accepted iteration fast-forwards onto local main)
  → re-verify the whole run: suite green, composite not regressed
  → rebase onto origin/main if it moved, re-run the suite
  → push (never forced; a lost race is retried, not overridden)
```

Four things stand between a bad idea and `main`:

1. **Per-iteration** — tests green, score held, no protected file touched.
2. **Per-run** — the suite and benchmark run again over all accepted iterations
   together. Individually safe changes can break in combination.
3. **Post-rebase** — if `main` moved during the run, the work is rebased onto it
   and the suite runs *again*. A rebase is a merge, and both sides being green
   separately proves nothing about the combination.
4. **Never a force-push.** A race is retried from a fresh fetch; a rebase
   conflict stops and leaves it for a human. A loop that can overwrite history is
   one bad iteration away from deleting the project.

Run it with `dry_run: true` to see what it *would* push without pushing.

Three operational notes:

- **Branch protection on `main` will block the bot.** Either exempt the Actions
  bot, or point `publish_branch` at something like `selfimprove/nightly` and
  merge by hand.
- **Pushes made with `GITHUB_TOKEN` do not trigger other workflows**, so `ci.yml`
  will not run on the bot's commits. The loop already runs that exact suite
  three times before pushing; add a PAT or deploy key if you want an independent
  CI run as well.
- **The attempt history is committed** (`selfimprove/history.jsonl`). Runners are
  ephemeral, so without this the loop would forget every previous attempt at the
  end of each run and re-try the same dead ends nightly.

## Watching a run

A run takes hours, so it reports as it goes rather than only at the end.

**In the GitHub Actions log** — open the running `self-improve` job and you get a
live trace of every step:

```
──── iteration 1 — ollama/gemma3:4b ────────────────────────────────────
  10:56:00    0.0s  base c37fd7a7, score to beat 62.0
  10:56:00    0.0s           · Let me find the bug.
  10:56:00    0.0s           ! tool call could not be parsed (retrying)
  10:56:00    0.0s  step  2  → grep(pattern='def average', path='.')
  10:56:00    0.0s           ← ok     0.0s  calc.py:1: def average(values):
  10:56:00    0.0s  step  4  → edit_file(path='calc.py', old_string='return sum…')
  10:56:00    0.0s           ← ok     0.0s  Edited calc.py (1 replacement)
  10:56:01    0.9s           ⤷ end_turn after 6 step(s), 4 tool call(s)
```

The benchmark reports per task too — against a real model that is most of the
wall clock, and it used to be forty silent minutes:

```
──── skills suite ──────────────────────────────────────────────────────
  [ 1/8] skills/T1/load-a-skill              1.00 solved   3s
  [ 2/8] skills/T2/follow-a-skill            0.57          48s
```

**As a heartbeat** — `selfimprove/progress.json` is rewritten on every event and
uploaded as a run artefact, so a stalled run is distinguishable from a slow one
by whether its `updated_at` is still moving.

**On the dashboard** — a banner appears while a run is in flight, with a link
straight to the live log.

All trace output goes to stderr; stdout stays clean for the JSON that CI pipes
into a file.

Run it locally and watch the same thing:

```bash
python -m selfimprove.loop --iterations 1 --dry-run    # trace on stderr
python -m bench.runner --only coding                   # per-task progress
```

## Progress dashboard

Live scoreboard: **https://edomor.github.io/morph/** — current score, category
and difficulty breakdown, score over time, releases, and **every attempt the
loop made, accepted and rejected, with the reason for each decision**.

The rejected ones are the point. A self-improving system that publishes only its
successes is advertising, not reporting; the rejections are what shows the guard
rails working. It is a static page built by `scripts/build_site.py` from data the
loop already commits, and it refreshes at the end of every run.

Needs Pages switched on once — see [docs/SETUP.md](docs/SETUP.md).

## Getting it on your phone

Every version the loop produces is published as a GitHub Release with an APK
attached. Open **Releases**, download `morph-vX.Y.Z.apk`, and install it — you
will need to let your browser install unknown apps.

The APK is a **client**. Morph is self-hosted, so on first launch it asks for the
address of the machine running the agent:

```bash
morph serve --host 0.0.0.0     # then enter e.g. http://192.168.1.10:8787
```

The agent, the model and your conversations stay on hardware you control;
nothing about them ships inside the APK. A forwarded Codespaces URL works too,
and so does Termux on the phone itself (option B above) — in which case point it
at `http://127.0.0.1:8787`.

Without an `ANDROID_KEYSTORE_BASE64` repository secret the APK is **debug
signed**: it installs from a browser fine, but Android will warn about an
unknown developer and it cannot go through Play. Add the secret (plus
`ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD`) to get
properly signed builds.

Versioning is single-source: `morph.__version__` drives the Python package, the
git tag, the release, and the APK's `versionName`/`versionCode` — so the app and
the agent that produced it always carry the same number. Tags are created
**after** the push succeeds, never before: a rebase rewrites every SHA, and a tag
made earlier would point at an orphaned commit.

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
selfimprove/    the loop, its prompts, the guard rails, versioning, publishing
android/        the phone client (WebView shell, built into an APK per version)
site/           the public progress dashboard (static, GitHub Pages)
tests/          the conformance suite — one test per requirement ID
```

## Repository setup

Four things a workflow cannot do for itself — enabling Pages, checking branch
protection on `main`, optional APK signing, optional CI on bot commits. All in
**[docs/SETUP.md](docs/SETUP.md)**.

## Development

```bash
python -m pytest tests -q          # must be green, offline, no credentials
python -m bench.runner             # full scorecard
python scripts/make_icons.py       # regenerate PWA icons from icon.svg
python scripts/build_site.py       # rebuild the dashboard data, then serve site/
```

Adding a requirement means adding an `R-###` to `REQUIREMENTS.md` **and** a test
that references it — `test_R_805_every_requirement_is_covered` fails otherwise.
