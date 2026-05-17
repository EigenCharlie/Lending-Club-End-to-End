# Paper 4 Notebook Ruff-Unsafe Fix Roundtrip Application v396

Generated: 2026-05-17T07:44:29.336998+00:00

v396 applies the 5 B905/SIM105 fixes reviewed in v395 and validates notebook
roundtrip integrity.

## Result

- Residual selected diagnostics before: `6`.
- Residual selected diagnostics after: `1`.
- Approved unsafe fixes applied: `5`.
- Global notebook diagnostics: `145` ->
  `144`.
- Import-lint side effects detected: `4`.
- Changed notebook files: `2`.
- Roundtrip integrity passed: `True`.

## Required Caveat

v396 does not clear notebook lint, does not make repository-wide ruff clean, does
not run full pytest, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v397_notebook_import_side_effect_and_sim115_patch.md` for the remaining SIM115 manual refactor
and contextlib import-lint side effects.
