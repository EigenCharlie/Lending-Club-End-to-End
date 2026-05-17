# Paper 4 Notebook E402 Cell Refactor Plan v399

Generated: 2026-05-17T08:06:28.636069+00:00

v399 turns the v398 E402 policy into an executable cell-level plan without
mutating notebooks.

## Result

- Planned E402 cells: `15`.
- E402 diagnostics covered: `119`.
- First batch cells: `6`.
- First batch diagnostics: `7`.
- Setup warning-filter cells: `9`.
- Setup warning-filter diagnostics: `112`.
- Notebooks mutated: `False`.

## Required Caveat

v399 does not repair E402, does not make notebook or repository ruff clean, does
not run full pytest, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v400_notebook_e402_local_import_hoist_batch.md` for the local delayed-import hoist batch.
