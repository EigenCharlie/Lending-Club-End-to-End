# Quarto Book Blueprint

> **STATUS: LIVE HISTORICAL/DIAGNOSTIC COMPANION (2026-07-20).** The Quarto
> book is this repository's principal editorial surface, but it is not the
> scientific authority for the autonomous CRPTO paper and does not convert
> runtime labels into current claims. Its navigation is governed by
> `book/_quarto.yml`; its interpretation is governed by
> `book/includes/_scientific-scope-contract.qmd` and the surface-specific claim
> contracts.

## Objective

- The Quarto book (`book/`) is the master's historical and diagnostic companion.
- Streamlit is a reduced companion for interaction, not the primary source of project truth.
- The book must distinguish the local runtime freeze, CRPTO, Paper 2 and Paper 4 without collapsing their estimands or statuses.

## Official narrative axes

- `Pipeline histórico ejecutable`
  Data -> score/calibration -> interval for binary $Y$ -> stress utilities, with survival, time-series, IFRS9, fairness and MRM retained as diagnostics.
- `Insight Factory`
  Explainability, causal extensions, notebooks, RAPIDS benchmarks, side projects, extended figures, and paper drafts.
- `Modernización metodológica`
  The modern layers are integrated into the main narrative, not isolated in a single appendix: monotonic promotion, approval-based fairness semantics, C2ST, PD backtesting interpretation, bootstrap gap diagnostics, calibration mapping sidecars, model-shift semantics, IFRS9 diagnostics, and encoding stability.

## Editorial contract

- Quarto chapters must read canonical artifacts whenever possible.
- Hardcoded metric snapshots should be avoided unless clearly marked as historical context.
- Current-state claims must be traceable through the relevant current contract,
  registered evidence and primary external references. The March canonical
  ledger is historical provenance only.
- Research lanes may be described, but must not be narrated as current winners,
  deployed policy, IFRS 9 evidence or independent validation.

## Maintenance rules

- If a runtime freeze or a claim contract changes, the following chapters should be reviewed first:
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

Use `docs/research/crpto_evolution_cross_project_audit_2026-07-20.md`, the
portable CRPTO contract, the Paper 2 claim contract and the current Paper 4
findings/boundaries as the live editorial hierarchy. The March ledger preserves
the historical mapping:

- technique -> artifact;
- artifact -> Quarto chapter;
- claim -> evidence;
- primary source -> bibliography target;
- stale claim -> rewrite/remove action.
