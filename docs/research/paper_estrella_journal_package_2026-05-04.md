# Paper Estrella Journal Package - 2026-05-04

This dossier records the journal-oriented tables and figures generated from
frozen Paper Estrella artifacts. It does not reopen the champion search.

## Generated artifacts

- `reports/paper_material/paper1/tables/paper1_tableA12_tail_risk_oce_cvar.csv`
- `reports/paper_material/paper1/tables/paper1_tableA12_tail_risk_oce_cvar.tex`
- `reports/paper_material/paper1/tables/paper1_tableA13_satisficing_margins.csv`
- `reports/paper_material/paper1/tables/paper1_tableA13_satisficing_margins.tex`
- `reports/paper_material/paper1/tables/paper1_tableA14_dependency_cluster_diagnostics.csv`
- `reports/paper_material/paper1/tables/paper1_tableA14_dependency_cluster_diagnostics.tex`
- `reports/paper_material/paper1/tables/paper1_tableA15_leave_one_period_stress.csv`
- `reports/paper_material/paper1/tables/paper1_tableA15_leave_one_period_stress.tex`
- `reports/paper_material/paper1/tables/paper1_tableA16_bootstrap_funded_set_metrics.csv`
- `reports/paper_material/paper1/tables/paper1_tableA16_bootstrap_funded_set_metrics.tex`
- `reports/paper_material/paper1/tables/paper1_tableA17_budget_cap_lgd_sensitivity.csv`
- `reports/paper_material/paper1/tables/paper1_tableA17_budget_cap_lgd_sensitivity.tex`
- `reports/paper_material/paper1/tables/paper1_tableA18_robust_region_policy_family.csv`
- `reports/paper_material/paper1/tables/paper1_tableA18_robust_region_policy_family.tex`
- `reports/paper_material/figures_publication/estrella_fig12_crpto_conceptual_pipeline.png`
- `reports/paper_material/figures_publication/estrella_fig12_crpto_conceptual_pipeline.pdf`
- `reports/paper_material/figures_publication/estrella_fig13_alpha_gamma_funded_set.png`
- `reports/paper_material/figures_publication/estrella_fig13_alpha_gamma_funded_set.pdf`
- `reports/paper_material/figures_publication/estrella_fig14_robust_region_heatmap.png`
- `reports/paper_material/figures_publication/estrella_fig14_robust_region_heatmap.pdf`

## Scope notes

- A12--A18 are diagnostic robustness and packaging tables.
- Budget and segment-cap sensitivity are funded-set diagnostics, not
  re-optimized portfolios.
- Tail-risk and bootstrap return columns are funded-set repricing diagnostics;
  the official champion return remains sourced from `final_project_promotion.json`.
- The official champion remains `bound_aware_276k_economic_champion`.

## Appendix map

| Artifact | Purpose | Caveat |
|---|---|---|
| A12 tail risk OCE/CVaR | Adds funded-set tail-risk diagnostics under LGD alternatives. | Diagnostic repricing, not a new champion metric. |
| A13 satisficing margins | Converts champion evidence into OR-style pass/margin checks. | Editorial thresholds must be justified if moved to body. |
| A14 dependency clusters | Documents period/grade concentration for the tightening appendix. | Does not prove conditional independence. |
| A15 leave-one-period stress | Reweights the funded set by period. | Not a re-optimized portfolio. |
| A16 bootstrap funded-set metrics | Adds empirical intervals for realized funded-set quantities. | Not a conformal guarantee. |
| A17 budget/LGD/cap sensitivity | Checks practical sensitivity to budget, LGD and segment caps. | Cap checks are diagnostics, not solver constraints. |
| A18 robust region by family | Summarizes the `45/45` alpha01-safe region by `risk_tolerance x gamma`. | Bound-aware family only. |

## Quarto integration

- `book/chapters/14-paper-estrella/14g-manuscript-blueprint.qmd` uses
  these artifacts to define the paper outline and final table/figure plan.
- `book/chapters/14-paper-estrella/14h-journal-appendix-robustness.qmd`
  renders A12--A18 and Figures 12--14.
