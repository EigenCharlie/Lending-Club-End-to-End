# Paper 4 Notebook Import Side-Effect and SIM115 Patch v397

Generated: 2026-05-17T07:54:47.758832+00:00

v397 cleans the small lint frontier introduced by v396 and the remaining SIM115
manual-refactor diagnostic.

## Result

- Global notebook diagnostics: `144` ->
  `139`.
- E402 diagnostics: `120` ->
  `119`.
- I001 diagnostics: `3` ->
  `0`.
- SIM115 diagnostics: `1` ->
  `0`.
- Changed notebook files: `3`.
- Roundtrip integrity passed: `True`.

## Required Caveat

v397 does not clear notebook lint, does not make repository-wide ruff clean, does
not run full pytest, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v398_notebook_historical_e402_policy.md` to govern the historical E402 notebook
frontier.
