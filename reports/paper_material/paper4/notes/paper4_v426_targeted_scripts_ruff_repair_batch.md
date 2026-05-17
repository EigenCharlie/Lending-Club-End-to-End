# Paper 4 Targeted Scripts Ruff Repair Batch v426

Generated: 2026-05-17T10:52:34.715645+00:00

v426 applies explicit `strict=False` to scripts/papers B905 `zip()` calls.

## Result

- Repository diagnostics: `57` ->
  `46`.
- Repository B905 diagnostics: `14` ->
  `3`.
- Scripts diagnostics: `47` ->
  `36`.
- Scripts B905 after: `0`.
- Changed script files: `6`.
- py_compile passed: `True`.
- Ruff fix command: `uv run ruff check scripts/papers --select B905 --fix --unsafe-fixes`.

## Required Caveat

v426 does not claim repository ruff clean, full pytest clean, Quarto render, or
Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v427_post_scripts_ruff_repair_pytest_probe.md`.
