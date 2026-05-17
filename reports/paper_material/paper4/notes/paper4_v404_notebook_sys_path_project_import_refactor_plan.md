# Paper 4 Notebook Sys.path/Project-Import Refactor Plan v404

Generated: 2026-05-17T08:42:57.647836+00:00

v404 plans the final E402 group: 3 setup cells where `sys.path.insert(...)`
precedes project imports.

## Result

- Sys.path/project-import cells planned: `3`.
- Current E402 diagnostics planned: `42`.
- Import viability probes passed: `True`.
- Current notebook diagnostics: `62`.
- Expected diagnostics after v405: `20`.
- Expected E402 after v405: `0`.
- Notebooks mutated: `False`.

## Required Caveat

v404 does not repair E402, does not make notebook or repository ruff clean, does
not run post-refactor pytest, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v405_notebook_sys_path_project_import_refactor_batch.md` for the sys.path/project-import refactor
batch.
