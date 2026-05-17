# Paper 4 Notebook Ruff-Unsafe Fix Review v395

Generated: 2026-05-17T07:37:49.793763+00:00

v395 reviews the residual Ruff-unsafe notebook fixes after v394's safe-only
application batch.

## Result

- Residual selected diagnostics: `6`.
- Ruff-unsafe candidates reviewed: `5`.
- Approved for guarded application: `5`.
- Nonfixable SIM115 diagnostics deferred: `1`.
- Global notebook diagnostics remain: `145`.
- Notebooks mutated in v395: `False`.

## Decision

B905 and SIM105 are approved for v396 guarded application because the previewed
changes make existing behavior explicit rather than changing the intended
notebook workflow. SIM115 remains deferred because Ruff provides no automatic
fix.

## Required Caveat

v395 does not mutate notebooks, does not reduce lint, does not run full pytest,
and does not create Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v396_notebook_unsafe_fix_roundtrip_application.md` and rerun roundtrip integrity checks.
