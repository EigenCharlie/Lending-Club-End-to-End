# Paper 4 Scripts/Book F401 Unused-Import Repair Batch v436

Generated: 2026-05-17T12:11:29.920785+00:00

v436 applies targeted unused-import repairs across scripts/book F401 diagnostics.

## Result

- Repository diagnostics: `18` ->
  `14`.
- Repository F401 diagnostics: `4` ->
  `0`.
- Scripts diagnostics: `17` ->
  `14`.
- Book diagnostics: `1` ->
  `0`.
- Changed files: `4`.
- py_compile passed: `True`.
- Ruff fix command: `uv run ruff check book/_helpers/plot_helpers.py scripts/papers/build_paper4_v398_notebook_historical_e402_policy.py scripts/papers/build_paper4_v399_notebook_e402_cell_refactor_plan.py scripts/papers/build_paper4_v406_post_sys_path_refactor_pytest_probe.py --select F401 --fix`.

## Required Caveat

v436 does not claim repository ruff clean, full pytest clean, Quarto render, or
Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v437_post_scripts_f401_repair_pytest_probe.md`.
