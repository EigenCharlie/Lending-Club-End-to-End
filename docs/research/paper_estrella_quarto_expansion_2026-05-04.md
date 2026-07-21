# Paper Estrella Quarto Expansion - 2026-05-04

> **HISTORICAL / SUPERSEDED — NO TRANSFER TO ACTIVE CRPTO (2026-07-20).** This memo preserves a past project decision or proposal. It does not govern the autonomous external CRPTO dossier, select a learner/comparator/policy, or reactivate Paper 2/Paper 4 claims. Current authority: `SESSION_STATE.md`, `docs/research/crpto_external_contract_2026-07-20.yml` and `docs/research/crpto_evolution_cross_project_audit_2026-07-20.md`.

This note records the book-side expansion added after the P1 hardening work. The
purpose is to keep the Quarto book richer than the eventual paper manuscript:
more explanation, more reviewer-facing context, and a local numeric reference
guide for the Paper Estrella section.

## Book Changes

- Added `book/chapters/14-paper-estrella/14f-editorial-claims-references.qmd`.
- Added `book/chapters/14-paper-estrella/14g-manuscript-blueprint.qmd`.
- Added `book/chapters/14-paper-estrella/14h-journal-appendix-robustness.qmd`.
- Added the new page to `book/_quarto.yml` under Part IV.
- Reworked the Paper Estrella landing page so it no longer depends on hard-coded
  chapter numbers and now explains how to read the book as an editorial dossier.
- Updated the introduction scope: the conditional Hoeffding/Bernstein tightening
  is now documented as appendix-level material, while Markov remains the main
  distribution-free theorem.
- Added a methodology table that maps every P1 evidence layer to its artifact and
  reviewer question.
- Linked the discussion back to the new editorial guide.
- Added a manuscript blueprint with target venue, abstract, claims C1--C7,
  paper outline, final table/figure plan, notation and claim-artifact-test
  location map.
- Added a journal appendix page that renders A12--A18 plus three new figures:
  CRPTO conceptual pipeline, alpha -> `Gamma_CP` -> funded set, and robust
  region heatmap.

## Journal Package Artifacts

The journal package is regenerated with:

```bash
uv run python scripts/build_paper1_journal_package.py
```

Generated tables:

- `paper1_tableA12_tail_risk_oce_cvar.csv`
- `paper1_tableA13_satisficing_margins.csv`
- `paper1_tableA14_dependency_cluster_diagnostics.csv`
- `paper1_tableA15_leave_one_period_stress.csv`
- `paper1_tableA16_bootstrap_funded_set_metrics.csv`
- `paper1_tableA17_budget_cap_lgd_sensitivity.csv`
- `paper1_tableA18_robust_region_policy_family.csv`

Generated figures:

- `reports/paper_material/figures_publication/estrella_fig12_crpto_conceptual_pipeline.png`
- `reports/paper_material/figures_publication/estrella_fig13_alpha_gamma_funded_set.png`
- `reports/paper_material/figures_publication/estrella_fig14_robust_region_heatmap.png`

Status artifact:

- `models/paper1_journal_package_status.json`

These outputs are diagnostics and manuscript-packaging evidence. They do not
replace `models/final_project_promotion.json` as the source of official champion
metrics.

## Why This Matters

The manuscript version should eventually be compressed, but the book should keep
the reasoning that justifies compression decisions. The new page separates:

- claims that belong in the paper body;
- robustness checks that belong in appendix;
- future work that should not be sold as current evidence;
- local numeric references `[1]`, `[2]`, ... for the Paper Estrella narrative.
- A12--A18 robustness evidence that can be pushed to appendix instead of
  crowding the paper body.

## Guardrails

The documentation tests should verify that:

- the new Quarto pages are registered in `book/_quarto.yml`;
- the page contains a claim ladder, reviewer Q&A, paper-placement table and local
  numbered references;
- the manuscript blueprint contains venue, claims C1--C7 and final table/figure
  plan;
- the appendix page references A12--A18 and the new figures;
- the Paper Estrella docs still point to the official economic champion and do
  not reopen the champion search.
