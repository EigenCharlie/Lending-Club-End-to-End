# Paper 4 Repository Ruff Frontier After Notebook Clean v423

Generated: 2026-05-17T10:33:02.710848+00:00

v423 runs repository-wide ruff after v421/v422 cleared and validated notebook
lint.

## Result

- Ruff command: `uv run ruff check . --output-format json`.
- Ruff exit code: `1`.
- Total diagnostics: `107`.
- Fixable diagnostics: `49`.
- Notebook diagnostics: `0`.
- Top rule: `E402` with `50` diagnostics.
- Top file: `streamlit_app/pages/model_interpretability.py` with `22` diagnostics.
- Top surface: `streamlit_app`.

## Required Caveat

v423 is non-mutating. It does not repair repository ruff diagnostics, run Quarto
render, or create Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v424_targeted_repo_ruff_repair_batch.md`.
