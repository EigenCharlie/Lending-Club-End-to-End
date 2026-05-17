# Paper 4 Notebook Safe-Fix Roundtrip Batch v394

Generated: 2026-05-17T07:30:50.886446+00:00

v394 executes the conservative subset of the v393 dry-run manifest: only Ruff
fixes with `safe` applicability are applied to notebooks.

## Result

- Selected notebook diagnostics before: `19`.
- Selected notebook diagnostics after: `6`.
- Safe-applicability fixes applied: `13`.
- Ruff-unsafe fixes deferred: `5`.
- Global notebook diagnostics: `158` ->
  `145`.
- Changed notebook files: `5`.
- Roundtrip integrity passed: `True`.

## Required Caveat

v394 does not apply Ruff-unsafe fixes, does not clear notebook lint, does not
make repository-wide ruff clean, does not run full pytest, and does not create
Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v395_notebook_unsafe_fix_review.md` for the 5 Ruff-unsafe notebook fixes.
