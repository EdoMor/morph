# Changelog

## v0.1.2 — 2026-10-31

- Structured patch via bounded-context-management: Modify `morph/agent.py` to explicitly use the `kv_set` tool instead of the incorrect `mcp__bench__kv_set` tool.. Expected: The agent will use the correct tool name.

Composite score: 58.7 (was 58.1). Produced by Gemma editing Morph through Morph's own agent, accepted only after the conformance suite passed and the score held.

## v0.1.1 — 2026-07-28

- Okay, the `calc.py` file has been created successfully! Now let’s verify that the changes have taken effect. I will use `read_file` to read the contents of `morph/calc.py`. This should confirm that the code is now present in the file and that the `average([])` function returns 0.0 as expected.

Composite score: 61.8 (was 60.6). Produced by Gemma editing Morph through Morph's own agent, accepted only after the conformance suite passed and the score held.

Every version here was produced by the self-improvement loop.

