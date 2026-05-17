# Paper 4 Notebook Warning-Filter-Only Reorder Batch v402

Generated: 2026-05-17T08:28:24.883871+00:00

v402 applies the first setup-cell batch selected by v401: 6 warning-filter-only
cells. It moves warning filters below import blocks and applies import-sort
normalization only in those changed notebooks.

## Result

- E402 diagnostics: `112` ->
  `42`.
- Global notebook diagnostics: `132` ->
  `62`.
- I001 diagnostics after normalization: `0`.
- Changed notebook files: `6`.
- Roundtrip integrity passed: `True`.

## Required Caveat

v402 does not repair sys.path/project-import E402 cells, does not clear notebook
or repository ruff, does not run full pytest, and does not create Paper 4 final
promotion.

## Next Executable Wave

Build `paper4_v403_post_notebook_mutation_pytest_probe.md` as a post-notebook-mutation pytest probe.
