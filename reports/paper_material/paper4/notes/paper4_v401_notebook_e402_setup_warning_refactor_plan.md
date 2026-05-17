# Paper 4 Notebook E402 Setup Warning-Filter Refactor Plan v401

Generated: 2026-05-17T08:21:30.649174+00:00

v401 plans the remaining setup-cell E402 frontier after v400's local import
hoist batch. It does not mutate notebooks.

## Result

- Setup warning-filter cells planned: `9`.
- Setup warning-filter E402 diagnostics: `112`.
- Warning-filter-only first batch cells: `6`.
- Warning-filter-only first batch diagnostics: `70`.
- Sys.path/project-import cells deferred: `3`.
- Sys.path/project-import diagnostics deferred: `42`.
- Notebooks mutated: `False`.

## Required Caveat

v401 does not repair E402, does not make notebook or repository ruff clean, does
not run full pytest, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v402_notebook_warning_filter_only_reorder_batch.md` for the warning-filter-only setup cells.
