# Paper 4 Repository Lint Frontier v390

Generated: 2026-05-17T07:04:05.037020+00:00

v390 probes `uv run ruff check .` after the full pytest suite became clean in
v389.

## Result

- Ruff command: `uv run ruff check .`.
- Ruff status: `fail`.
- Total diagnostics: `282`.
- Fixable diagnostics reported by ruff: `88`.
- Top rule: `E402` with `169` findings.
- Top file: `streamlit_app/pages/model_interpretability.py` with `22` findings.

## Required Caveat

v390 is a lint frontier only. It does not repair the 282 diagnostics, does not
claim global ruff cleanliness, does not run full Quarto render, and does not
create Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v391_targeted_lint_repair_batch.md` to start a targeted lint repair batch.
