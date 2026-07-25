# Setup — the parts only you can do

Everything else is automated. These four need repository settings or secrets,
which a workflow cannot grant itself.

Only **step 1** is required for the dashboard, and **step 2** is required for
the loop to actually land commits. Steps 3 and 4 are optional.

---

## 1. Turn on GitHub Pages — *required for the dashboard*

The dashboard workflow deploys via the Pages API, which is inert until Pages is
enabled and set to build from Actions.

1. Go to **Settings → Pages** (`https://github.com/EdoMor/morph/settings/pages`).
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
   Do *not* choose "Deploy from a branch" — there is no `gh-pages` branch and
   the site is built fresh from the committed history on each deploy.
3. That is the whole change. Nothing to save beyond the dropdown.

Then publish it once by hand:

- **Actions → pages → Run workflow → Run workflow** (on `main`).

The job summary prints the URL. It will be:

```
https://edomor.github.io/morph/
```

After that it refreshes itself: the `self-improve` workflow calls the `pages`
workflow at the end of every run.

> If the run fails with `Resource not accessible by integration` or
> `Pages site not found`, the source dropdown is still on the default — go back
> to step 2.

**A fresh site will say "No runs recorded yet".** That is correct until a run
publishes a scorecard; it is not an error.

---

## 2. Check branch protection on `main` — *required for the loop*

The loop pushes accepted iterations straight to `main` as
`morph-selfimprove[bot]`. If `main` is protected, those pushes are rejected and
every run ends having done nothing.

Go to **Settings → Branches** and look for a rule on `main`.

- **No rule?** Nothing to do. This is the default for a new repository, so most
  likely you are already fine.
- **Rule exists?** Pick one:
  - **Allow the bot through** — in the rule, under *Restrict who can push*, add
    the GitHub Actions app; or tick *Allow specified actors to bypass required
    pull requests*.
  - **Or keep `main` protected** and send the loop somewhere else: run the
    workflow with **publish_branch** set to `selfimprove/nightly`, then review
    and merge that branch yourself. The loop's own gates still apply — you are
    adding a human review on top, not replacing anything.

To change the *scheduled* run's target as well, edit the `publish_branch`
default in `.github/workflows/self-improve.yml`.

---

## 3. Sign the APK properly — *optional*

Without a keystore the release APK is signed with the Android **debug key**. It
installs fine from a phone browser; Android just warns about an unknown
developer, and it cannot go through the Play Store.

To sign it for real, generate a keystore once:

```bash
keytool -genkey -v -keystore morph-release.jks \
  -keyalg RSA -keysize 2048 -validity 10000 -alias morph
base64 -w0 morph-release.jks    # copy this output
```

Then add four **Settings → Secrets and variables → Actions → New repository
secret**:

| Secret | Value |
| --- | --- |
| `ANDROID_KEYSTORE_BASE64` | the base64 output above |
| `ANDROID_KEYSTORE_PASSWORD` | the store password you chose |
| `ANDROID_KEY_ALIAS` | `morph` |
| `ANDROID_KEY_PASSWORD` | the key password you chose |

Keep `morph-release.jks` somewhere safe and **out of the repository**. Losing it
means you can never update an installed app in place — Android will refuse an
APK signed with a different key.

---

## 4. Run CI on the bot's own commits — *optional*

GitHub deliberately does not start new workflow runs from pushes made with the
built-in `GITHUB_TOKEN`. So `ci.yml` does **not** run on commits the loop pushes.

This is mostly cosmetic: the loop runs that exact suite three times before
pushing (per iteration, per run, and again after any rebase). But if you want a
green tick on those commits from an independent run:

1. Create a fine-grained PAT with **Contents: read and write** on this repo.
2. Add it as a secret named `MORPH_PUSH_TOKEN`.
3. In `.github/workflows/self-improve.yml`, add to the `actions/checkout` step:
   ```yaml
   with:
     token: ${{ secrets.MORPH_PUSH_TOKEN }}
   ```

---

## Running the loop by hand

**Actions → self-improve → Run workflow**, with:

| Input | Meaning |
| --- | --- |
| `iterations` | how many improvement cycles to attempt (each takes ~10–40 min) |
| `provider` | `ollama` (Gemma on the runner) or `google` (needs `GOOGLE_API_KEY`) |
| `model` | e.g. `gemma3:4b`. Bigger models are better but slower on a CPU runner |
| `focus` | optional: steer the run at one area, e.g. "the mcp suite" |
| `publish_branch` | where accepted work lands. Defaults to `main` |
| `dry_run` | run and score everything, publish nothing. **Good for a first run** |

It also runs on its own at 03:17 UTC daily.

## What a healthy run looks like

- **Accepted: 0** is a normal outcome, not a failure. A small model rejected on
  every attempt still produced a record of *why*, and that record is fed into
  the next run's prompt.
- The score to trust is the one measured against a real model. The
  `echo` provider reports 100/100 and flags every suite `partial` — that is a
  harness check, not a capability measurement.
- If the dashboard shows a suite as `saturated` or `floored`, the benchmark has
  stopped discriminating and needs new tasks. The loop cannot write them —
  `bench/tasks/` is protected precisely so it cannot grade its own homework.
