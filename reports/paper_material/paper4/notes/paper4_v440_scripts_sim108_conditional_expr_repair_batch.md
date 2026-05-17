# Paper 4 Scripts SIM108 Conditional-Expression Repair Batch v440

Generated: 2026-05-17T12:40:53.307176+00:00

v440 applies targeted conditional-expression repairs across scripts SIM108 diagnostics.

## Result

- Repository diagnostics: `11` ->
  `9`.
- Repository SIM108 diagnostics: `2` ->
  `0`.
- Scripts diagnostics: `11` ->
  `9`.
- Changed files: `2`.
- py_compile passed: `True`.
- Ruff fix command: `uv run ruff check scripts/papers/build_paper4_v10_resolution_wave.py scripts/papers/build_paper4_v41_v44_living_lab_wave.py --select SIM108 --fix --unsafe-fixes`.

## Required Caveat

v440 does not claim repository ruff clean, full pytest clean, Quarto render, or
Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v441_post_scripts_sim108_repair_pytest_probe.md`.
