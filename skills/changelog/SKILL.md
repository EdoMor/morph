---
name: changelog
description: House style for CHANGELOG entries. Use when recording a user-visible change.
allowed-tools: read_file, edit_file, write_file
---

# Changelog style

`CHANGELOG.md` follows Keep a Changelog, loosely.

## Rules

- New work goes under `## Unreleased`. Create that section if it is missing.
- One line per change: `- <past-tense verb> <what changed>`.
  - Good: `- Added an empty-list guard to average().`
  - Bad: `- average fix`
- Write for someone who does not know the codebase. Name the user-visible
  effect, not the internal refactor that caused it.
- Group under `### Added`, `### Changed`, `### Fixed`, `### Removed` once a
  section has more than about five entries. Below that, a flat list is fine.
- Never rewrite a released section. Releases are history.

## Example

```markdown
# Changelog

## Unreleased

- Added `generate_image` so the agent can produce images inline.
- Fixed `edit_file` silently succeeding when the target string was absent.

## 0.1.0 — 2026-01-15

- First release.
```
