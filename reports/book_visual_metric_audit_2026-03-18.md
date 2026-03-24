# Auditoría de Cobertura Visual y Métrica del Libro Quarto

Fecha: 2026-03-18

## Objetivo

Verificar dos cosas:

1. Que el libro no dependa de imágenes compuestas que dañen el layout o reduzcan legibilidad.
2. Que las métricas y artefactos más importantes del proyecto sí estén representados en la narrativa del libro.

## Figuras compuestas reemplazadas

Se creó un generador dedicado: [`scripts/generate_book_editorial_figures.py`](../scripts/generate_book_editorial_figures.py).

Este script produce 17 figuras unitarias en [`book/assets/figures/editorial`](../book/assets/figures/editorial):

- `pipeline_pd_distribution_by_grade.png`
- `pipeline_interval_sample.png`
- `pipeline_ecl_by_scenario.png`
- `pd_roc_curve.png`
- `pd_pr_curve.png`
- `pd_reliability_curve.png`
- `conformal_coverage_by_grade.png`
- `conformal_width_vs_target.png`
- `ts_default_rate_series.png`
- `causal_policy_monthly_net.png`
- `shap_family_mass.png`
- `ale_int_rate.png`
- `ale_fico_score.png`
- `uncertainty_baselines_tradeoff.png`
- `alpha_pareto_frontier.png`
- `alpha_eligible_loans.png`
- `fairness_threshold_frontier.png`

## Capítulos corregidos

Las figuras unitarias quedaron integradas en:

- [`book/chapters/01-executive-map.qmd`](../book/chapters/01-executive-map.qmd)
- [`book/chapters/04-pipeline-overview/index.qmd`](../book/chapters/04-pipeline-overview/index.qmd)
- [`book/chapters/05-feature-engineering/index.qmd`](../book/chapters/05-feature-engineering/index.qmd)
- [`book/chapters/06-pd-modeling/index.qmd`](../book/chapters/06-pd-modeling/index.qmd)
- [`book/chapters/06-pd-modeling/06d-model-comparison-champion.qmd`](../book/chapters/06-pd-modeling/06d-model-comparison-champion.qmd)
- [`book/chapters/07-conformal/index.qmd`](../book/chapters/07-conformal/index.qmd)
- [`book/chapters/08-time-survival-causal/index.qmd`](../book/chapters/08-time-survival-causal/index.qmd)
- [`book/chapters/08-time-survival-causal/08c-causal-inference.qmd`](../book/chapters/08-time-survival-causal/08c-causal-inference.qmd)
- [`book/chapters/10-ifrs9-governance/10e-model-risk-management.qmd`](../book/chapters/10-ifrs9-governance/10e-model-risk-management.qmd)
- [`book/chapters/11-explainability/11a-global-explanations.qmd`](../book/chapters/11-explainability/11a-global-explanations.qmd)
- [`book/chapters/13-advanced-topics/13a-uncertainty-baselines.qmd`](../book/chapters/13-advanced-topics/13a-uncertainty-baselines.qmd)
- [`book/chapters/13-advanced-topics/13b-alpha-sweep-pareto.qmd`](../book/chapters/13-advanced-topics/13b-alpha-sweep-pareto.qmd)
- [`book/chapters/13-advanced-topics/13e-ts-ecl-intervals.qmd`](../book/chapters/13-advanced-topics/13e-ts-ecl-intervals.qmd)
- [`book/chapters/14-paper-estrella/14d-results.qmd`](../book/chapters/14-paper-estrella/14d-results.qmd)
- [`book/chapters/09-portfolio/09c-policy-selection.qmd`](../book/chapters/09-portfolio/09c-policy-selection.qmd)

## Métricas fuertes verificadas como cubiertas

### PD

Fuente:

- `data/processed/model_comparison.json`
- `data/processed/pipeline_summary.json`

Cubiertas en el libro:

- AUC
- Gini
- Brier
- ECE
- D² Brier
- KS
- PR-AUC
- Recall/F1 al threshold operativo
- Calibración con Venn-Abers

Capítulos principales:

- [`book/chapters/06-pd-modeling/06d-model-comparison-champion.qmd`](../book/chapters/06-pd-modeling/06d-model-comparison-champion.qmd)
- [`book/chapters/01-executive-map.qmd`](../book/chapters/01-executive-map.qmd)

### Conformal

Fuente:

- `models/conformal_policy_status.json`
- `data/processed/conformal_group_metrics_mondrian.parquet`
- `data/processed/alpha_sweep_pareto_both.parquet`

Cubiertas en el libro:

- coverage 90 / 95
- avg width
- min group coverage
- Winkler
- overall_pass
- strict_overall_pass
- warning alerts
- failing checks (Kupiec/Christoffersen)
- sensibilidad de alpha
- elegibilidad operativa por alpha

Capítulos principales:

- [`book/chapters/07-conformal/index.qmd`](../book/chapters/07-conformal/index.qmd)
- [`book/chapters/10-ifrs9-governance/10e-model-risk-management.qmd`](../book/chapters/10-ifrs9-governance/10e-model-risk-management.qmd)
- [`book/chapters/13-advanced-topics/13b-alpha-sweep-pareto.qmd`](../book/chapters/13-advanced-topics/13b-alpha-sweep-pareto.qmd)
- [`book/chapters/14-paper-estrella/14d-results.qmd`](../book/chapters/14-paper-estrella/14d-results.qmd)

### Portafolio y robustez

Fuente:

- `models/champion_portfolio_policy.json`
- `data/processed/portfolio_robustness_frontier.parquet`
- `data/processed/pipeline_summary.json`

Cubiertas en el libro:

- robust return
- funded loans
- price of robustness
- price of robustness pct
- worst-case PD reduction bps
- allocation similarity
- breadth score
- A/B no-regression

Capítulos principales:

- [`book/chapters/09-portfolio/09c-policy-selection.qmd`](../book/chapters/09-portfolio/09c-policy-selection.qmd)
- [`book/chapters/01-executive-map.qmd`](../book/chapters/01-executive-map.qmd)

### IFRS9

Fuente:

- `data/processed/ifrs9_scenario_summary.parquet`
- `data/processed/cif_ecl_impact.parquet`
- `data/processed/stage_misclassification_cost.parquet`
- `data/processed/ts_ecl_intervals.parquet`

Cubiertas en el libro:

- total ECL por escenario
- severidad prudencial
- Stage composition
- CIF vs KM y exceso de reserva
- costo de misclasificación Stage 2
- banda temporal de ECL

Capítulos principales:

- [`book/chapters/10-ifrs9-governance/index.qmd`](../book/chapters/10-ifrs9-governance/index.qmd)
- [`book/chapters/08-time-survival-causal/08a-survival-analysis.qmd`](../book/chapters/08-time-survival-causal/08a-survival-analysis.qmd)
- [`book/chapters/13-advanced-topics/13c-competing-risks-ecl.qmd`](../book/chapters/13-advanced-topics/13c-competing-risks-ecl.qmd)
- [`book/chapters/13-advanced-topics/13d-stage-misclassification-cost.qmd`](../book/chapters/13-advanced-topics/13d-stage-misclassification-cost.qmd)
- [`book/chapters/13-advanced-topics/13e-ts-ecl-intervals.qmd`](../book/chapters/13-advanced-topics/13e-ts-ecl-intervals.qmd)

### Supervivencia y temporal

Fuente:

- `data/processed/pipeline_summary.json`
- `data/processed/lifetime_pd_table.parquet`
- `data/processed/time_series.parquet`
- `data/processed/ts_cv_stats.parquet`

Cubiertas en el libro:

- Cox c-index
- RSF c-index
- lifetime PD / CIF
- serie agregada de default
- forecast fan chart
- champion temporal y multiplicador

Capítulos principales:

- [`book/chapters/08-time-survival-causal/index.qmd`](../book/chapters/08-time-survival-causal/index.qmd)
- [`book/chapters/08-time-survival-causal/08a-survival-analysis.qmd`](../book/chapters/08-time-survival-causal/08a-survival-analysis.qmd)
- [`book/chapters/08-time-survival-causal/08b-time-series-forecasting.qmd`](../book/chapters/08-time-survival-causal/08b-time-series-forecasting.qmd)

### Causal

Fuente:

- `data/processed/pipeline_summary.json`
- `data/processed/causal_policy_oot_backtest.parquet`
- `data/processed/causal_policy_oot_backtest_by_grade.parquet`

Cubiertas en el libro:

- ATE
- IC 95% del ATE
- CATE mean/std
- selected rule
- total net value
- action rate
- p05 monthly net
- backtest mensual OOT

Capítulos principales:

- [`book/chapters/08-time-survival-causal/08c-causal-inference.qmd`](../book/chapters/08-time-survival-causal/08c-causal-inference.qmd)
- [`book/chapters/01-executive-map.qmd`](../book/chapters/01-executive-map.qmd)

### Explainability y drift

Fuente:

- `data/processed/shap_summary.parquet`
- `data/processed/permutation_importance.parquet`
- `data/processed/ale_curves.parquet`
- `data/processed/explanation_drift.parquet`

Cubiertas en el libro:

- drivers SHAP top-5
- contraste SHAP vs permutation
- ALE en variables clave
- controlabilidad / familias
- monotonía
- drift explicativo

Capítulos principales:

- [`book/chapters/11-explainability/11a-global-explanations.qmd`](../book/chapters/11-explainability/11a-global-explanations.qmd)
- [`book/chapters/11-explainability/11c-explanation-drift.qmd`](../book/chapters/11-explainability/11c-explanation-drift.qmd)

### Gobernanza

Fuente:

- `models/fairness_audit_status.json`
- `data/processed/fairness_threshold_frontier.parquet`
- `reports/mrm/mrm_validation_report.json`

Cubiertas en el libro:

- fairness overall_pass
- atributos aprobados
- threshold operativo
- thresholds auditados
- strict_overall_pass conformal
- warning alerts conformal
- subsistemas MRM en PASS
- frontera de fairness por threshold

Capítulos principales:

- [`book/chapters/10-ifrs9-governance/10e-model-risk-management.qmd`](../book/chapters/10-ifrs9-governance/10e-model-risk-management.qmd)
- [`book/chapters/01-executive-map.qmd`](../book/chapters/01-executive-map.qmd)

## Hallazgos de la auditoría

- Los collages más problemáticos ya no se usan en el libro renderizado.
- Las métricas fuertes de discriminación, calibración, conformal, robustez, causalidad y gobernanza sí quedaron expuestas en capítulos relevantes.
- El libro ahora muestra explícitamente no solo resultados “buenos”, sino también estados de observación o warning (`strict_overall_pass`, `warning_alerts`, `MRM overall_pass = False`).
- Los PNG compuestos antiguos siguen existiendo en disco, pero dejaron de ser la base visual del libro.

## Gaps no críticos que permanecen

- La coherencia de `run_tag` del bloque causal sigue existiendo como artefacto de trazabilidad en `pipeline_summary.json`, pero no se promovió a narrativa principal porque es una alerta de consistencia operativa, no un resultado científico.
- Hay espacio para una segunda pasada opcional que agregue visuales específicos de fairness por atributo o comparación por grade de la policy causal, pero el libro ya no pierde información material por no mostrarlos.
