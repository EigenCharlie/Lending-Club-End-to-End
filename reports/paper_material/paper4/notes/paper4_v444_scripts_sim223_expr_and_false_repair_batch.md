# Paper 4 Scripts SIM223 Expr-And-False Repair Batch v444

Generated: 2026-05-17T13:16:25.388825+00:00

v444 applies the targeted SIM223 expr-and-false repair.

## Result

- Repository diagnostics: `8` ->
  `7`.
- Repository SIM223 diagnostics: `1` ->
  `0`.
- Changed files: `1`.
- py_compile passed: `True`.
- Ruff fix command: `uv run ruff check scripts/papers/build_paper4_v41_v44_living_lab_wave.py --select SIM223 --fix --unsafe-fixes | uv run ruff check scripts/papers/build_paper4_v41_v44_living_lab_wave.py --select UP018 --fix`.

## Required Caveat

v444 does not claim repository ruff clean, full pytest clean, Quarto render, or
Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v445_post_scripts_sim223_repair_pytest_probe.md`.
