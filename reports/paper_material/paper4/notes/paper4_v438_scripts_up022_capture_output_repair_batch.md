# Paper 4 Scripts UP022 Capture-Output Repair Batch v438

Generated: 2026-05-17T12:26:36.887735+00:00

v438 applies targeted `capture_output=True` repairs across scripts UP022 diagnostics.

## Result

- Repository diagnostics: `14` ->
  `11`.
- Repository UP022 diagnostics: `3` ->
  `0`.
- Scripts diagnostics: `14` ->
  `11`.
- Changed files: `3`.
- py_compile passed: `True`.
- Ruff fix command: `uv run ruff check scripts/papers/build_paper4_v39_v40_living_lab_execution.py scripts/papers/build_paper4_v41_v44_living_lab_wave.py scripts/papers/build_paper4_v45_v48_living_lab_wave.py --select UP022 --fix --unsafe-fixes`.

## Required Caveat

v438 does not claim repository ruff clean, full pytest clean, Quarto render, or
Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v439_post_scripts_up022_repair_pytest_probe.md`.
