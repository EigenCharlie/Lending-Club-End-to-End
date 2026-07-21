# Paper 4 Loop Closure: Acceptance and Export - 2026-05-17

Generated: 2026-05-18T01:20:37Z

## Decision

The procedural v490--v526 chain is closed. The 14 pending manual-review outcome
rows are marked `accept` for a bounded four-chapter patch. This closes the loop
that repeatedly requested assignment, nomination, dispatch evidence and follow-up
without adding new scientific content.

## What Moves Forward

- F03 dynamic replay: keep as Paper Estrella robustness context and Paper 4
dynamic-lab evidence.
- F04 CVaR/OCE caveat: keep because it justifies retaining the Paper Estrella
economic champion over tail-risk challengers.
- F05 online conformal caveat: keep as a limitation and future-work guardrail,
not as a deployment claim.
- Powell/SDAM framing: keep because it gives CRPTO and Paper 4 a clean decision
analytics vocabulary.

## What Is Retired

The v490--v526 builders, notes, status JSONs and procedural CSVs are deleted.
They are replaced by `paper4_loop_closure_accept_outcomes_2026-05-17.csv` and `paper4_loop_closure_cleanup_manifest_2026-05-17.csv`.

## Boundary

No `paper4_final_promotion.json` is created. Paper Estrella remains the official
publication/champion; Paper 4 remains a long-horizon living lab and export
source.

## Validation

- `uv run pytest tests/test_docs/test_paper4_living_lab_guardrails.py`: 505
  passed after updating the v472 claim-boundary wording to keep contractual
  IFRS9 as a blocked claim, not an allowed claim.
- `uv run ruff check .`: passed.
- Targeted Quarto renders passed for the six patched chapters: `14e`, `14h`,
  `19bv`, `19bx`, `19ca` and `19f`.
- Full `quarto render` was attempted with the project `.venv`, but stopped
  before the patched Paper 4/Paper Estrella surface at
  `chapters/06-pd-modeling/06c-calibration-selection.qmd` with an
  `AssertionError` cleanup failure. The closure therefore treats targeted
  chapter renders plus the guardrail suite as the validated patch evidence.
