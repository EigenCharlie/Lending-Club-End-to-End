# Paper 4 Scripts F841 Unused-Variable Repair Batch v432

Generated: 2026-05-17T11:38:22.128932+00:00

v432 applies targeted unused-variable repairs across scripts F841 diagnostics.

## Result

- Repository diagnostics: `30` ->
  `23`.
- Repository F841 diagnostics: `7` ->
  `0`.
- Scripts diagnostics: `28` ->
  `21`.
- Changed script files: `6`.
- py_compile passed: `True`.
- Ruff fix command: `uv run ruff check scripts/papers --select F841 --fix --unsafe-fixes`.

## Required Caveat

v432 does not claim repository ruff clean, full pytest clean, Quarto render, or
Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v433_post_scripts_f841_repair_pytest_probe.md`.
