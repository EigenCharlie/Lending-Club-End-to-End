# Paper 4 Notebook Lint Policy v392

Generated: 2026-05-17T07:17:50.642771+00:00

v392 turns the v391 lint reduction into a notebook-specific mutation policy.

## Decision

Selected policy: `dry_run_first_no_bulk_notebook_mutation`.

- Notebook diagnostics: `158`.
- Notebook fixable diagnostics: `22`.
- Dominant notebook rule: `E402`
  (`119` findings).
- Notebook bulk mutation applied: `False`.

## Required Caveat

v392 does not repair notebooks, does not hide notebooks from global ruff, does
not claim global ruff cleanliness, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v393_notebook_lint_dry_run_manifest.csv` as a no-mutation dry-run manifest.
