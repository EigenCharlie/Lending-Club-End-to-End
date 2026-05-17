# Paper 4 Notebook E402 Local Import Hoist Batch v400

Generated: 2026-05-17T08:15:15.317421+00:00

v400 applies only the first E402 batch selected by v399: local delayed imports
inside 6 cells.

## Result

- E402 diagnostics: `119` ->
  `112`.
- Global notebook diagnostics: `139` ->
  `132`.
- Changed notebook files: `5`.
- Roundtrip integrity passed: `True`.

## Required Caveat

v400 does not repair setup warning-filter E402 cells, does not clear notebook or
repository ruff, does not run full pytest, and does not create Paper 4 final
promotion.

## Next Executable Wave

Build `paper4_v401_notebook_e402_setup_warning_refactor_plan.md` for the setup warning-filter E402 cells.
