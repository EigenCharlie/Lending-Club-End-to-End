# Quarto Book Blueprint

> **STATUS: LIVE AND UNDER CONTINUOUS MAINTENANCE** — The Quarto book is the official, citable surface of the project, but it must be maintained against the live canonical artifacts. The monotonic champion promotion plus the ADSFCR-inspired tranches made the earlier “book complete” status insufficient as a maintenance statement.

## Objective

- The Quarto book (`book/`) is the master's thesis and the official narrative layer.
- Streamlit is a reduced companion for interaction, not the primary source of project truth.
- The book must explain the current champion stack, the current monitoring/governance stack, and the research lanes without mixing them.

## Official narrative axes

- `Pipeline Operativo`
  Data -> PD -> calibration -> fairness semantics -> conformal -> survival/time series -> IFRS9 -> portfolio policy -> governance / MRM.
- `Insight Factory`
  Explainability, causal extensions, notebooks, RAPIDS benchmarks, side projects, extended figures, and paper drafts.
- `Modernización metodológica`
  The modern layers are integrated into the main narrative, not isolated in a single appendix: monotonic promotion, approval-based fairness semantics, C2ST, PD backtesting interpretation, bootstrap gap diagnostics, calibration mapping sidecars, model-shift semantics, IFRS9 diagnostics, and encoding stability.

## Editorial contract

- Quarto chapters must read canonical artifacts whenever possible.
- Hardcoded metric snapshots should be avoided unless clearly marked as historical context.
- Current-state claims must be traceable through:
  - canonical runtime artifacts;
  - `docs/CANONICAL_DOCUMENTATION_AND_QUARTO_TRACEABILITY_2026-03-30.md`;
  - primary external references.
- Research lanes may be described in the book, but must never be narrated as champion behavior when they are not promoted.

## Maintenance rules

- If a champion changes, the following chapters should be reviewed first:
  - executive map;
  - PD champion narrative;
  - conformal / fairness / MRM sections;
  - IFRS9 / sensitivity sections;
  - any paper or GPU chapter that names the active baseline.
- If a new diagnostic layer is added, the book should answer five questions:
  - what the technique is;
  - why it was added here;
  - what value it adds in this project;
  - what the current artifact says;
  - what limitation remains open.

## Primary maintenance companion

Use `docs/CANONICAL_DOCUMENTATION_AND_QUARTO_TRACEABILITY_2026-03-30.md` as the living editorial ledger that maps:

- technique -> artifact;
- artifact -> Quarto chapter;
- claim -> evidence;
- primary source -> bibliography target;
- stale claim -> rewrite/remove action.
