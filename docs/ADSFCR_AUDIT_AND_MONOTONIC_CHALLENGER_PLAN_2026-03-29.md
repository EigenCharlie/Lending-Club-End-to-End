# ADSFCR Audit And Monotonic Challenger Plan

Date: 2026-03-29

## Executive Summary

- Audit scope completed for the `adsfcr` repository surface referenced from its `README`.
- Blob links audited: 90.
- Live links audited: 8 primary public surfaces plus the main repository.
- Access conclusion: no material access blocker was found during the audit window. The primary public links responded, and the 90 blob links referenced in the `README` exist in the local clone.
- Immediate implementation decision:
  - implement now the fairness-semantics fix for the monotonic challenger search;
  - do not mix broad `adsfcr` method adoption into the 7-9 hour fairness-aware search;
  - complete this audit document now so that post-search decisions are traceable.
- High-value post-search candidates:
  - representativeness C2ST with drivers/materiality;
  - monotonicity disruption and heterogeneity audit;
  - PD backtesting suite with exact binomial two-sided, Jeffreys, z-score and HL;
  - IFRS9 diagnostics with recursive regressions, ADF power and sign-coherence checks;
  - encoding/binning stability diagnostics.

## Implementation Status In This Repo

- Fairness parity patch implemented in `scripts/search_monotonic_competitor.py`.
- The monotonic challenger search now mirrors the official fairness audit semantics:
  - `outcome_mode=approval`;
  - base plus intersectional groups;
  - quartile binning aligned with `configs/fairness_policy.yaml`;
  - threshold selection from the official global-threshold frontier.
- New search outputs now persist both:
  - `official_fairness_all_attributes`;
  - `official_fairness_base_only`.
- Guardrail tests were added for:
  - search/audit fairness parity;
  - audit-document coverage of the 90 blob links and relevant live links.
- Validation status on 2026-03-29:
  - targeted test suite passed: `8 passed`;
  - preview search rerun completed successfully with `outcome_mode=approval` across all evaluated candidates;
  - preview search reported `6` attributes in the official fairness view and `3` in the base-only view;
  - preview best variant was `bureau_capacity::reg_up::venn_abers`;
  - preview best variant passed fairness on all 6 official attributes at threshold `0.30`, but was still not promotable because `brier_increase_pct_mean=0.01888` exceeded the configured `0.015` limit.
- Operational next step:
  - launch the corrected full fairness-aware search;
  - reserve the expanded search for the case where the full run remains near-promotable or fairness still fails narrowly.

## Post-Promotion Tranche 1

- Implemented on top of the promoted monotonic champion:
  - richer C2ST output in governance with driver attribution and materiality;
  - monotonicity audit artifact;
  - PD backtesting suite artifact;
  - post-core orchestration hooks so future confirmatory rebuilds execute these diagnostics automatically.
- New artifacts:
  - `models/monotonicity_audit_status.json`
  - `models/pd_backtesting_status.json`
  - `data/processed/monotonicity_band_summary.parquet`
  - `data/processed/monotonicity_pair_report.parquet`
  - `data/processed/monotonicity_feature_report.parquet`
  - `data/processed/pd_backtesting_by_grade.parquet`
  - `data/processed/pd_backtesting_by_band.parquet`
- First real read on the promoted champion:
  - monotonicity audit: `PASS`, with zero adjacent band disruptions and zero feature-level monotonicity violations on the constrained features;
  - PD backtesting suite: `diagnostic fail`, with exact binomial p-value and HL p-value near zero, indicating statistically visible calibration deviation even though the champion is already promoted and operationally valid;
  - governance C2ST remains `severe` and now exposes drivers, with `int_rate` dominating the train-vs-test split.
- Interpretation:
  - the monotonic champion is structurally strong under the new monotonicity diagnostics;
  - the highest-value next ADSFCR tranche should focus on calibration / backtesting interpretation and IFRS9 diagnostics, not on re-opening monotonicity.

## Post-Promotion Tranche 2

- Implemented after the confirmatory monotonic rebuild:
  - IFRS9 diagnostics with recursive regressions, ADF power and stress-sign-coherence checks;
  - encoding/binning stability audit for WOE and bucketed features;
  - MRM run-tag hardening so confirmatory rebuilds stamp the correct official `run_tag`;
  - comparison refresh after the new ADSFCR diagnostics were added.
- New artifacts:
  - `models/ifrs9_diagnostics_status.json`
  - `models/encoding_stability_status.json`
  - `data/processed/ifrs9_recursive_regression_paths.parquet`
  - `data/processed/ifrs9_recursive_regression_summary.parquet`
  - `data/processed/ifrs9_sign_coherence.parquet`
  - `data/processed/woe_encoding_stability.parquet`
  - `data/processed/bucket_binning_stability.parquet`
- Real read on the confirmatory monotonic run `canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129`:
  - IFRS9 diagnostics: `diagnostic fail`, not because the stress ladder is incoherent, but because the temporal relationships are unstable and the ADF power at a near-unit-root alternative is weak;
  - recursive regression summary on `avg_int_rate`, `avg_dti` and `avg_loan_amnt` shows sign-match shares only between `0.616` and `0.704`, with up to `8` sign flips;
  - ADF level p-value is about `0.297`, first-difference p-value is about `0.00010`, and estimated ADF power at `phi=0.95` is only `0.275`;
  - sign-coherence across `baseline -> mild_stress -> adverse -> severe` passes on `pd_mult`, `stage2_share`, `stage3_share`, `total_ecl` and `total_ecl_high`;
  - encoding/binning stability: `PASS`, with `13` WOE features and `3` bucket features audited, zero failures, max WOE PSI about `0.106`, max bucket PSI about `0.100`, and no category rank-breaks.
- Operational interpretation:
  - the promoted monotonic champion remains structurally strong in the PD lane;
  - the next ADSFCR value is no longer in monotonicity or fairness search logic;
  - the highest-value follow-up is now IFRS9 model diagnostics / redesign and calibration interpretation, not challenger promotion mechanics;
  - encoding/binning does not currently justify a feature-contract rewrite.

## Confirmatory Closeout

- Confirmatory comparison refreshed after the tranche-2 diagnostics:
  - `reports/run_comparisons/canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129/comparison.json`
  - `overall_pass = true`
  - `operational_overall_pass = true`
  - `conformal_promotion_pass = true`
  - `ab_no_regression_pass = true`
- The MRM report and wrapper now carry the correct confirmatory `run_tag`:
  - `reports/mrm/mrm_validation_report.json`
  - `models/mrm_report_status.json`
- Decision:
  - keep `IFRS9 diagnostics` and `encoding/binning stability` as `diagnostic_only`;
  - do not reopen the monotonic promotion because of these additions.

## Post-Promotion Tranche 3

- Implemented after tranche 2:
  - rare-event calibration rerun with the correct confirmatory `run_tag`;
  - PD validation interpretation layer that converts raw statistical backtesting into `pass / warning / fail` style materiality language;
  - IFRS9 diagnostics extended with scenario-interval uncertainty and sensitivity-surface interpretation.
- New artifacts:
  - `models/pd_validation_interpretation_status.json`
  - `data/processed/pd_backtesting_quarter_materiality.parquet`
- Real read on the confirmatory monotonic run:
  - `pd_validation_interpretation_status.json` lands at `severity = warning`, not `fail`;
  - the global PD gap is only about `36.9 bp`, so the issue is not a large overall calibration break;
  - the real problem is slice persistence: `8` evaluated issue-quarters exceed the persistence threshold, with the worst recent gaps reaching roughly `1247 bp` in `2019Q4` and `1882 bp` in `2020Q1`;
  - the current statistical fail therefore reads as `material_slice_deviation`, not just `large_sample_significance`;
  - `pd_rare_event_calibration_status.json` is now stamped with the confirmatory `run_tag` and remains consistent with the rare-event report;
  - IFRS9 still fails diagnostically on recursive stability / ADF power, but the extension now shows that interval uncertainty is very wide (`mean_relative_width_90 ≈ 6.99x`) and the dominant sensitivity driver in the ECL grid is `lgd_mult`, not `pd_mult`.
- Operational interpretation:
  - the next PD work should prioritize cohort-sensitive calibration interpretation or mapping adjustments before any new champion search;
  - the next IFRS9 work should prioritize LGD / scenario uncertainty handling and temporal macro defensibility, not just alternative stress ladders;
  - these additions remain `diagnostic_only` and do not reopen the monotonic promotion.

## Decision Rules

- `implementar ahora`: worth implementing in the repo in the near-term roadmap.
- `usar como referencia metodologica`: valuable for design, governance or interpretation, but not coded immediately.
- `documentar pero no implementar`: useful context, but no direct roadmap action.
- `descartar por ahora`: low applicability to the current Lending Club pipeline.

## What Can Enter The Fairness-Aware Search Right Now

### Implement now inside the search relaunch

- Replicate official fairness audit semantics:
  - `outcome_mode=approval`;
  - base plus intersectional groups;
  - quartile binning aligned with `configs/fairness_policy.yaml`;
  - global-threshold frontier selection.
- Save richer fairness diagnostics per candidate:
  - `official_fairness_all_attributes`;
  - `official_fairness_base_only`;
  - selected threshold;
  - approval rate;
  - failed attributes list.

### Do not mix into the long search yet

- C2ST, model shift, blockwise designs, LDP, Vasicek theory, new binning schemes, discrete rating-scale remapping, bootstrap ranking logic, IFRS9 diagnostics and other broad `adsfcr` adoptions.

## Impact Matrix

| Technique family | Worth implementing | Earliest phase | Expected effect | Requires retraining | Requires rerun downstream |
|---|---|---|---|---|---|
| Official fairness parity in monotonic search | implementar ahora | before preview rerun | fixes comparability bug; reduces false fairness failure | No | No |
| Richer fairness candidate diagnostics | implementar ahora | before preview rerun | better ranking diagnosis and failure analysis | No | No |
| Representativeness C2ST with drivers | implementar ahora | post-search | stronger governance and promotion defense | No | Yes |
| Monotonicity disruption / heterogeneity audit | implementar ahora | post-search | stronger defense of monotonic champion | No | Yes |
| PD backtesting suite | implementar ahora | post-search | better calibration and MRM narrative | No | Yes |
| IFRS9 diagnostics | implementar ahora | post-search | more defensible macro/ECL layer | No | Yes |
| Encoding/binning stability diagnostics | implementar ahora | post-search | stronger structural robustness story | No by default | Yes |
| Blockwise regression designs | usar como referencia metodologica | future research | challenger design ideas | Yes | Yes |
| LDP / Vasicek / concentration / EIR / repayment | descartar por ahora | none | low relevance to current stack | No | No |

## Live Links Ledger

| Link | Archivo local | Acceso | Profundidad | Familia | Tema | Aplicabilidad | Prioridad | Accion | Impacto esperado | Requiere reentrenamiento | Requiere rerun downstream | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [Leanpub ADSFCR](https://leanpub.com/adsfcr) | n/a | 200 | skim | live | book surface | context only | low | documentar pero no implementar | context | No | No | confirms active public surface |
| [Leanpub PDRMWR](https://leanpub.com/pdrmwr) | n/a | 200 | skim | live | book surface | context only | low | documentar pero no implementar | context | No | No | useful reference only |
| [Leanpub Working Notes](https://leanpub.com/crmwn) | n/a | 200 | skim | live | working notes | medium context | low | documentar pero no implementar | context | No | No | can cite as evolving companion |
| [Vasicek Shiny](https://andrijadj.shinyapps.io/vasicek_distribution/) | n/a | 202 | skim | live | interactive Vasicek app | low current relevance | low | descartar por ahora | none | No | No | outside current tabular challenger work |
| [BCR dataset](https://andrija-djurovic.github.io/adsfcr/ldp/BCR_TABLES.xlsx) | `ldp/BCR_TABLES.xlsx` | 200 | skim | live | LDP dataset | low | low | descartar por ahora | none | No | No | LDP-specific |
| [Concentration dataset](https://raw.githubusercontent.com/andrija-djurovic/adsfcr/main/concentration_risk/db.csv) | `concentration_risk/db.csv` | 200 | skim | live | concentration dataset | low | low | descartar por ahora | none | No | No | not needed for Lending Club pipeline |
| [Bootstrap HT HTML](https://andrija-djurovic.github.io/adsfcr/model_dev_and_vld/bootstrap_ht.html) | local html equivalent not required | 200 | deep | live | bootstrap hypothesis tests | high | high | implementar ahora | can strengthen post-search diagnostics | No | Yes | do not inject into search ranking yet |
| [HL vs Z-score HTML](https://andrija-djurovic.github.io/adsfcr/model_dev_and_vld/hl_vs_zscore.html#/) | local html equivalent not required | 200 | deep | live | backtesting interpretation | high | medium | usar como referencia metodologica | helps PD validation narrative | No | Yes | supports test interpretation |
| [Effective interest rate HTML](https://andrija-djurovic.github.io/adsfcr/effective_interest_rate/eir.html) | local html equivalent not required | 200 | skim | live | EIR | low | low | descartar por ahora | none | No | No | outside current scope |
| [Loan repayment plan HTML](https://andrija-djurovic.github.io/adsfcr/loan_repayment_plan/lrp.html) | local html equivalent not required | 200 | skim | live | amortization | low | low | descartar por ahora | none | No | No | outside current scope |

## Blob Link Ledger

### Vasicek Distribution (Probability of Default Models)

| Link | Archivo local | Acceso | Profundidad | Tema | Aplicabilidad | Prioridad | Accion | Impacto esperado | Requiere reentrenamiento | Requiere rerun downstream | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [The Functional Form and Parameters Estimation Methods](https://github.com/andrija-djurovic/adsfcr/blob/main/vasicek_distribution/vasicek_distribution.pdf) | `vasicek_distribution/vasicek_distribution.pdf` | local clone | skim | Vasicek basics | low | low | descartar por ahora | none | No | No | theory-heavy, not needed for current challenger |
| [Asset Correlation Estimation - Analytical vs Numerical](https://github.com/andrija-djurovic/adsfcr/blob/main/vasicek_distribution/imm_vs_mle.pdf) | `vasicek_distribution/imm_vs_mle.pdf` | local clone | skim | asset correlation estimation | low | low | descartar por ahora | none | No | No | IRB-oriented |
| [The Logistic Vasicek Distribution](https://github.com/andrija-djurovic/adsfcr/blob/main/vasicek_distribution/logistic_vasicek_distribution.pdf) | `vasicek_distribution/logistic_vasicek_distribution.pdf` | local clone | skim | logistic Vasicek | low | low | descartar por ahora | none | No | No | not relevant to current PD stack |
| [Asset Correlation Estimation - Normal vs Logistic Vasicek](https://github.com/andrija-djurovic/adsfcr/blob/main/vasicek_distribution/rho_normal_vs_logistic_vasicek.pdf) | `vasicek_distribution/rho_normal_vs_logistic_vasicek.pdf` | local clone | skim | correlation estimation | low | low | descartar por ahora | none | No | No | no direct pipeline use |
| [Asset Correlation Estimators and Bias Quantification](https://github.com/andrija-djurovic/adsfcr/blob/main/vasicek_distribution/rho_bias_quantification.pdf) | `vasicek_distribution/rho_bias_quantification.pdf` | local clone | skim | bias quantification | low | low | descartar por ahora | none | No | No | not needed for monotonic challenger |
| [The Vasicek PD Model and Transition Matrices - Optimization of the Systemic Factor Z](https://github.com/andrija-djurovic/adsfcr/blob/main/vasicek_distribution/tr_and_z_factor.pdf) | `vasicek_distribution/tr_and_z_factor.pdf` | local clone | skim | transition matrices / systemic factor | medium | low | documentar pero no implementar | possible appendix context only | No | No | could inform academic discussion only |

### Loss Given Default

| Link | Archivo local | Acceso | Profundidad | Tema | Aplicabilidad | Prioridad | Accion | Impacto esperado | Requiere reentrenamiento | Requiere rerun downstream | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Loss Given Default as a Function of the Default Rate](https://github.com/andrija-djurovic/adsfcr/blob/main/lgd/lgd_as_a_function_of_dr.pdf) | `lgd/lgd_as_a_function_of_dr.pdf` | local clone | skim | LGD-default-rate linkage | medium | medium | usar como referencia metodologica | future LGD/EAD enrichment | No | No | not tied to immediate challenger relaunch |
| [The Vasicek LGD Model - Functional Form and Estimation](https://github.com/andrija-djurovic/adsfcr/blob/main/lgd/vasicek_lgd.pdf) | `lgd/vasicek_lgd.pdf` | local clone | skim | Vasicek LGD | low | low | documentar pero no implementar | future LGD theory only | No | No | not immediate |
| [The Vasicek LGD Model - Parameter Distribution](https://github.com/andrija-djurovic/adsfcr/blob/main/lgd/vasicek_lgd_params_dist.pdf) | `lgd/vasicek_lgd_params_dist.pdf` | local clone | skim | LGD parameter uncertainty | low | low | documentar pero no implementar | future context | No | No | not current priority |
| [The Vasicek LGD Model - Bias Quantification](https://github.com/andrija-djurovic/adsfcr/blob/main/lgd/vasicek_lgd_q_bias_quant.pdf) | `lgd/vasicek_lgd_q_bias_quant.pdf` | local clone | skim | LGD bias | low | low | documentar pero no implementar | future context | No | No | not current priority |
| [Enhancing IRB LGD Modeling with Survival Analysis](https://github.com/andrija-djurovic/adsfcr/blob/main/lgd/lgd_survival.pdf) | `lgd/lgd_survival.pdf` | local clone | deep | survival LGD | medium | medium | usar como referencia metodologica | useful for later LGD lane | Yes | Yes | not mix with monotonic PD search |
| [Component-Based IRB LGD Models - Probability of Cure Calibration](https://github.com/andrija-djurovic/adsfcr/blob/main/lgd/PoC_calibration.pdf) | `lgd/PoC_calibration.pdf` | local clone | deep | PoC calibration | medium | medium | usar como referencia metodologica | useful for LGD/EAD work | Yes | Yes | post-promotion research lane |

### Low Default Portfolios

| Link | Archivo local | Acceso | Profundidad | Tema | Aplicabilidad | Prioridad | Accion | Impacto esperado | Requiere reentrenamiento | Requiere rerun downstream | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Alan Forrest multi-year adjustment - PD domain search](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/ldp_adj_af_adjustment.pdf) | `ldp/ldp_adj_af_adjustment.pdf` | local clone | skim | LDP conservative PD | low | low | descartar por ahora | none | No | No | LDP-specific |
| [Alan Forrest multi-year adjustment R code](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/ldp_adj_af_multi_year.R) | `ldp/ldp_adj_af_multi_year.R` | local clone | catalog_only | LDP code | low | low | descartar por ahora | none | No | No | R-only and out of scope |
| [Alan Forrest multi-year adjustment Python code](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/ldp_adj_af_multi_year.py) | `ldp/ldp_adj_af_multi_year.py` | local clone | skim | LDP code | low | low | descartar por ahora | none | No | No | Python but still out of scope |
| [Alan Forrest multi-year optimization approach](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/ldp_adj_af_adjustment_opt.pdf) | `ldp/ldp_adj_af_adjustment_opt.pdf` | local clone | skim | LDP optimization | low | low | descartar por ahora | none | No | No | not relevant to current Lending Club stack |
| [Alan Forrest optimization R code](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/ldp_adj_af_adjustment_multi_year_opt.R) | `ldp/ldp_adj_af_adjustment_multi_year_opt.R` | local clone | catalog_only | LDP code | low | low | descartar por ahora | none | No | No | R-only |
| [Alan Forrest optimization Python code](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/ldp_adj_af_adjustment_multi_year_opt.py) | `ldp/ldp_adj_af_adjustment_multi_year_opt.py` | local clone | skim | LDP code | low | low | descartar por ahora | none | No | No | out of scope |
| [Pluto-Tasche approach](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/pt.pdf) | `ldp/pt.pdf` | local clone | skim | LDP PD estimation | low | low | descartar por ahora | none | No | No | not needed for retail challenger |
| [Pluto-Tasche R code](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/pt.R) | `ldp/pt.R` | local clone | catalog_only | LDP code | low | low | descartar por ahora | none | No | No | R-only |
| [Pluto-Tasche Python code](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/pt.py) | `ldp/pt.py` | local clone | skim | LDP code | low | low | descartar por ahora | none | No | No | out of scope |
| [Benjamin-Cathcart-Ryan approach](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/bcr.pdf) | `ldp/bcr.pdf` | local clone | skim | LDP conservative PD | low | low | descartar por ahora | none | No | No | not current stack |
| [BCR R code](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/bcr.R) | `ldp/bcr.R` | local clone | catalog_only | LDP code | low | low | descartar por ahora | none | No | No | R-only |
| [BCR Python code](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/bcr.py) | `ldp/bcr.py` | local clone | skim | LDP code | low | low | descartar por ahora | none | No | No | out of scope |
| [Benchmarking LDP tendency testing](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/ldp_benchmarking_tendency.pdf) | `ldp/ldp_benchmarking_tendency.pdf` | local clone | skim | LDP benchmarking | low | low | descartar por ahora | none | No | No | not current portfolio type |
| [Benchmarking LDP deviation testing](https://github.com/andrija-djurovic/adsfcr/blob/main/ldp/ldp_benchmarking_deviation.pdf) | `ldp/ldp_benchmarking_deviation.pdf` | local clone | skim | LDP benchmarking | low | low | descartar por ahora | none | No | No | not current portfolio type |

### Concentration Risk

| Link | Archivo local | Acceso | Profundidad | Tema | Aplicabilidad | Prioridad | Accion | Impacto esperado | Requiere reentrenamiento | Requiere rerun downstream | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Measuring Concentration Risk - A Partial Portfolio Approach](https://github.com/andrija-djurovic/adsfcr/blob/main/concentration_risk/cr.pdf) | `concentration_risk/cr.pdf` | local clone | skim | concentration risk | low | low | descartar por ahora | none | No | No | not part of current promotion blocker |

### Model Risk Management

| Link | Archivo local | Acceso | Profundidad | Tema | Aplicabilidad | Prioridad | Accion | Impacto esperado | Requiere reentrenamiento | Requiere rerun downstream | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Model Shift and Model Risk Management](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/ms_mrm.pdf) | `mrm/ms_mrm.pdf` | local clone | deep | model shift | high | medium | usar como referencia metodologica | strengthens governance framing | No | Yes | useful after search, not inside search |
| [The Instability of WoE Encoding in PD Modeling](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/pd_and_woe_encoding_instability.pdf) | `mrm/pd_and_woe_encoding_instability.pdf` | local clone | deep | encoding instability | high | high | implementar ahora | supports post-search binning/encoding stability diagnostics | No by default | Yes | do not alter feature space before rerun |
| [The Instability of Mean Target Encoding in LGD and EAD Modeling](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/ols_and_mte_instability.pdf) | `mrm/ols_and_mte_instability.pdf` | local clone | deep | encoding instability in LGD/EAD | medium | low | documentar pero no implementar | low current value | No | No | LGD/EAD lane only |
| [Discriminatory Power Shortfalls - RWA Impact](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/dp_shortfall_and_rwa_impact_analysis.pdf) | `mrm/dp_shortfall_and_rwa_impact_analysis.pdf` | local clone | deep | DP shortfalls | medium | low | documentar pero no implementar | committee context only | No | No | IRB framing, limited direct use |
| [Economic Value of Credit Rating Systems](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/evrs.pdf) | `mrm/evrs.pdf` | local clone | deep | economic value framing | medium | low | documentar pero no implementar | business narrative support | No | No | not a technical blocker |
| [Heterogeneity Testing in IRB Credit Risk Models](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/heterogeneity.pdf) | `mrm/heterogeneity.pdf` | local clone | deep | heterogeneity framework | high | high | implementar ahora | direct post-search structure audit candidate | No | Yes | strong fit with monotonic challenger |
| [Heterogeneity Testing - Statistical Power Analysis](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/heterogeneity_and_power_analysis.pdf) | `mrm/heterogeneity_and_power_analysis.pdf` | local clone | deep | power analysis | high | medium | usar como referencia metodologica | informs interpretation of heterogeneity tests | No | Yes | supports diagnostics, not initial gate |
| [Heterogeneity Testing - Disruption of Monotonicity](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/heterogeneity_and_monotonicity.pdf) | `mrm/heterogeneity_and_monotonicity.pdf` | local clone | deep | monotonicity disruption | very high | high | implementar ahora | direct defense and monitoring of monotonic champion | No | Yes | top-priority post-search technique |
| [Welch vs Mann-Whitney for LGD/EAD](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/welch_t_vs_mw_test.pdf) | `mrm/welch_t_vs_mw_test.pdf` | local clone | deep | LGD/EAD heterogeneity tests | medium | low | documentar pero no implementar | not tied to current blocker | No | No | LGD/EAD-specific |
| [When the P-Value > 50% is Informative](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/heterogeneity_and_p_value.pdf) | `mrm/heterogeneity_and_p_value.pdf` | local clone | deep | p-value interpretation | high | medium | usar como referencia metodologica | improves warning/gate semantics | No | Yes | useful in MRM narrative |
| [Heterogeneity Shortfalls - RWA Impact](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/heterogeneity_shortfall_and_rwa_impact_analysis.pdf) | `mrm/heterogeneity_shortfall_and_rwa_impact_analysis.pdf` | local clone | deep | heterogeneity economics | medium | low | documentar pero no implementar | committee context only | No | No | indirect fit |
| [Heterogeneity Shortfalls - Portfolio Returns Impact](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/heterogeneity_shortfall_and_portfolio_returns.pdf) | `mrm/heterogeneity_shortfall_and_portfolio_returns.pdf` | local clone | deep | heterogeneity economics | medium | low | documentar pero no implementar | committee context only | No | No | indirect fit |
| [Representativeness Analysis - Classifier Two-Sample Test](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/representativeness_c2st.pdf) | `mrm/representativeness_c2st.pdf` | local clone | deep | representativeness C2ST | very high | high | implementar ahora | top post-search governance addition | No | Yes | extends existing governance lane |
| [Somers' D for LGD/EAD Models - Dyx or Dxy?](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/lgd_ccf_dp_somersd_dxy_dyx.pdf) | `mrm/lgd_ccf_dp_somersd_dxy_dyx.pdf` | local clone | deep | LGD/EAD discriminatory power | medium | low | documentar pero no implementar | LGD/EAD context only | No | No | not immediate |
| [Somers' D in LGD/EAD Modeling - Input Binning Impact](https://github.com/andrija-djurovic/adsfcr/blob/main/mrm/lgd_ead_dp_somersd_and_binning.pdf) | `mrm/lgd_ead_dp_somersd_and_binning.pdf` | local clone | deep | LGD/EAD binning stability | medium | low | documentar pero no implementar | future LGD/EAD idea | No | No | not immediate |

### Model Development and Validation

| Link | Archivo local | Acceso | Profundidad | Tema | Aplicabilidad | Prioridad | Accion | Impacto esperado | Requiere reentrenamiento | Requiere rerun downstream | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Common Inconsistencies in Probability of Default Modeling](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/pd_modeling_inconsistencies.pdf) | `model_dev_and_vld/pd_modeling_inconsistencies.pdf` | local clone | deep | PD modeling inconsistencies | high | medium | usar como referencia metodologica | strengthens documentation and review criteria | No | No | narrative and audit value |
| [WoE Regression for PD Modeling - Intercept Estimation Uncertainty](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/woe_reg_intercept.pdf) | `model_dev_and_vld/woe_reg_intercept.pdf` | local clone | deep | intercept uncertainty | medium | low | documentar pero no implementar | low direct fit to CatBoost challenger | No | No | scorecard-oriented |
| [From Binary to Continuous: A WoE-Equivalent Encoding Method](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/cont_target_woe_equivalent.pdf) | `model_dev_and_vld/cont_target_woe_equivalent.pdf` | local clone | deep | continuous target encoding | low | low | documentar pero no implementar | not needed now | No | No | limited fit |
| [IRB PD Periodic Model Validation - Quantitative Testing Procedures](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/periodic_pd_vld.pdf) | `model_dev_and_vld/periodic_pd_vld.pdf` | local clone | deep | PD validation suite | very high | high | implementar ahora | direct post-search PD validation upgrade | No | Yes | key source for backtesting suite |
| [On Testing the Concentration in the Rating Grades - The Initial and Periodic PD Model Validation](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/hi_cv_testing.pdf) | `model_dev_and_vld/hi_cv_testing.pdf` | local clone | deep | rating grade concentration / HI-CV testing | high | medium | usar como referencia metodologica | useful post-search monitoring of grade concentration and validation drift | No | Yes | good complement to periodic PD validation, not needed inside fairness search ranking |
| [IRB Model Validation - Example of an Automated Validation Report](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/vld_report_example.pdf) | `model_dev_and_vld/vld_report_example.pdf` | local clone | deep | automated report design | high | medium | usar como referencia metodologica | useful for MRM/report layout | No | Yes | supports reporting, not modeling |
| [Does the P-value Provide Sufficient Insight?](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/vld_and_p_value.pdf) | `model_dev_and_vld/vld_and_p_value.pdf` | local clone | deep | p-value interpretation | high | medium | usar como referencia metodologica | improves gate vs warning separation | No | Yes | governance narrative |
| [On Favorable P-values in Statistical Tests](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/favorable_p_value.pdf) | `model_dev_and_vld/favorable_p_value.pdf` | local clone | deep | p-value interpretation | high | medium | usar como referencia metodologica | improves test interpretation | No | Yes | complements previous item |
| [MoC Type C with Autocorrelation](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/moc_type_c_ct_autocorrelation.pdf) | `model_dev_and_vld/moc_type_c_ct_autocorrelation.pdf` | local clone | deep | conservatism with autocorrelation | medium | medium | usar como referencia metodologica | useful for IFRS9 and calibration caution | No | Yes | not immediate search change |
| [OLS vs Yule-Walker for AR coefficients](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/aet.pdf) | `model_dev_and_vld/aet.pdf` | local clone | deep | AR estimation | medium | medium | usar como referencia metodologica | supports IFRS9 diagnostics | No | Yes | macro lane only |
| [Hypothesis Testing in Credit Risk - A Visual Approach](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/visual_support_for_ht.pdf) | `model_dev_and_vld/visual_support_for_ht.pdf` | local clone | deep | test interpretation | medium | low | documentar pero no implementar | narrative support only | No | No | useful but not immediate |
| [Independent and Correlated Binomial Distributions](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/independent_correlated_binomial_test.pdf) | `model_dev_and_vld/independent_correlated_binomial_test.pdf` | local clone | deep | correlated defaults backtesting | high | high | implementar ahora | strengthens post-search PD backtesting suite | No | Yes | direct relevance |
| [Model-Based Heterogeneity Testing](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/model_based_heterogeneity_testing.pdf) | `model_dev_and_vld/model_based_heterogeneity_testing.pdf` | local clone | deep | heterogeneity testing | high | high | implementar ahora | supports monotonicity/structure audit | No | Yes | complements MRM heterogeneity items |
| [Risk-Weighted Assets as a Function of Probability of Default](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/rwa_pd.pdf) | `model_dev_and_vld/rwa_pd.pdf` | local clone | deep | RWA-PD relation | low | low | documentar pero no implementar | academic context only | No | No | low direct value |
| [Third Party Ratings Adjustment](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/third_party_rating_treatment.pdf) | `model_dev_and_vld/third_party_rating_treatment.pdf` | local clone | deep | constrained threshold ideas | medium | low | documentar pero no implementar | future blockwise/off-policy idea | Yes | Yes | not now |
| [PCA for IFRS9 Forward-Looking Modeling](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/pca.pdf) | `model_dev_and_vld/pca.pdf` | local clone | deep | PCA sign coherence | high | medium | usar como referencia metodologica | supports post-search IFRS9 diagnostics | No | Yes | sign-coherence checks |
| [Supervised Macroeconomic Index](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/fli_smi.pdf) | `model_dev_and_vld/fli_smi.pdf` | local clone | deep | macro supervised index | high | high | implementar ahora | good candidate for IFRS9 diagnostics lane | No initially | Yes | diagnostic first, not replacement model |
| [SMI R package manual](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/smi_r/smi_r_manual.pdf) | `model_dev_and_vld/smi_r/smi_r_manual.pdf` | local clone | deep | SMI implementation details | medium | low | usar como referencia metodologica | useful if building Python analogue later | No | No | R-centric |
| [ADF Power in IFRS9](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/adf_power.pdf) | `model_dev_and_vld/adf_power.pdf` | local clone | deep | stationarity test power | very high | high | implementar ahora | direct IFRS9 diagnostics improvement | No | Yes | strong fit |
| [OLS Regression and Predictor Importance](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/ols_predictor_importance.pdf) | `model_dev_and_vld/ols_predictor_importance.pdf` | local clone | deep | importance taxonomy | high | medium | usar como referencia metodologica | improves explainability narrative | No | Yes | post-search explanation layer |
| [Do We Use OLS Regression Efficiently?](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/fli_ols.pdf) | `model_dev_and_vld/fli_ols.pdf` | local clone | deep | OLS IFRS9 design critique | medium | low | documentar pero no implementar | context only | No | No | secondary |
| [Dynamic Regression Models and Estimation Uncertainty](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/fli_dynamic_regression_models.pdf) | `model_dev_and_vld/fli_dynamic_regression_models.pdf` | local clone | deep | dynamic regression uncertainty | high | high | implementar ahora | post-search IFRS9 diagnostics improvement | No | Yes | diagnostic first |
| [Recursive Regressions in Practice](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/fli_recursive_reg.pdf) | `model_dev_and_vld/fli_recursive_reg.pdf` | local clone | deep | coefficient stability over time | very high | high | implementar ahora | direct IFRS9 diagnostics addition | No | Yes | strong fit |
| [Bootstrap Hypothesis Tests PDF](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/bootstrap_ht.pdf) | `model_dev_and_vld/bootstrap_ht.pdf` | local clone | deep | bootstrap tests | high | medium | implementar ahora | useful after search for warning/diagnostic layers | No | Yes | do not use as search ranking now |
| [Statistical Binning of Numeric Risk Factors - PD Modeling](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/nrf_binning.pdf) | `model_dev_and_vld/nrf_binning.pdf` | local clone | deep | numeric binning | high | medium | usar como referencia metodologica | supports future stability diagnostics | Yes if adopted | Yes | not before search rerun |
| [Statistical Binning of Categorical Risk Factors - PD Modeling](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/cat_rf_bin.pdf) | `model_dev_and_vld/cat_rf_bin.pdf` | local clone | deep | categorical binning | high | medium | usar como referencia metodologica | supports future stability diagnostics | Yes if adopted | Yes | not before search rerun |
| [Binning and Validation](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/binning_and_validation.pdf) | `model_dev_and_vld/binning_and_validation.pdf` | local clone | deep | binning stability and validation | very high | high | implementar ahora | direct post-search structural audit candidate | No by default | Yes | strong fit |
| [Hosmer-Lemeshow vs Z-score PDF](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/hl_vs_zscore.pdf) | `model_dev_and_vld/hl_vs_zscore.pdf` | local clone | deep | HL vs z-score interpretation | high | medium | usar como referencia metodologica | supports PD backtesting interpretation | No | Yes | complements backtesting suite |
| [Power Play: PD Predictive Ability Testing](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/statistical_power_of_pd_pp_tests.pdf) | `model_dev_and_vld/statistical_power_of_pd_pp_tests.pdf` | local clone | deep | power of predictive-ability tests | medium | medium | usar como referencia metodologica | can refine warning semantics | No | Yes | secondary |
| [Nested Dummy Encoding](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/nested_dummy_encoding.pdf) | `model_dev_and_vld/nested_dummy_encoding.pdf` | local clone | deep | encoding design | high | medium | usar como referencia metodologica | future encoding stability diagnostics | Yes if adopted | Yes | do not change features before search |
| [Marginal Information Value](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/marginal_information_value.pdf) | `model_dev_and_vld/marginal_information_value.pdf` | local clone | deep | feature value contribution | medium | low | documentar pero no implementar | minor narrative improvement only | No | No | secondary |
| [Scorecard Scaling](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/scorecard_scaling.pdf) | `model_dev_and_vld/scorecard_scaling.pdf` | local clone | deep | score scaling | low | low | descartar por ahora | none | No | No | scorecard-specific |
| [Level Importance of Risk Factors - WoE Regression and Scorecard Scaling](https://github.com/andrija-djurovic/adsfcr/blob/main/model_dev_and_vld/woe_level_importance.pdf) | `model_dev_and_vld/woe_level_importance.pdf` | local clone | deep | importance by level | medium | low | documentar pero no implementar | narrative only | No | No | secondary |

### Risk Quantification and Backtesting

| Link | Archivo local | Acceso | Profundidad | Tema | Aplicabilidad | Prioridad | Accion | Impacto esperado | Requiere reentrenamiento | Requiere rerun downstream | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Calibration of Discrete PD Rating Scale](https://github.com/andrija-djurovic/adsfcr/blob/main/risk_quantification/pd_drs_calibration.pdf) | `risk_quantification/pd_drs_calibration.pdf` | local clone | deep | discrete PD calibration | high | medium | usar como referencia metodologica | potential calibration mapping diagnostics | No initially | Yes | do not replace current calibrator before search |
| [Intercept Optimization for Discrete PD Calibration](https://github.com/andrija-djurovic/adsfcr/blob/main/risk_quantification/model_based_pd_calibration_intercept.pdf) | `risk_quantification/model_based_pd_calibration_intercept.pdf` | local clone | deep | intercept optimization | high | medium | usar como referencia metodologica | possible calibration challenger after search | No initially | Yes | diagnostic first |
| [MoC Type C in PD Models](https://github.com/andrija-djurovic/adsfcr/blob/main/risk_quantification/pd_moc_c_egim_para_327.pdf) | `risk_quantification/pd_moc_c_egim_para_327.pdf` | local clone | deep | conservatism quantification | medium | medium | usar como referencia metodologica | useful in governance/MRM narrative | No | Yes | not needed inside search |
| [Two-Sided Exact Binomial Test](https://github.com/andrija-djurovic/adsfcr/blob/main/risk_quantification/2_sided_exact_binomial_test.pdf) | `risk_quantification/2_sided_exact_binomial_test.pdf` | local clone | deep | exact binomial backtesting | very high | high | implementar ahora | direct PD backtesting upgrade | No | Yes | top post-search implementation |

### Business-Guided Regression Designs

| Link | Archivo local | Acceso | Profundidad | Tema | Aplicabilidad | Prioridad | Accion | Impacto esperado | Requiere reentrenamiento | Requiere rerun downstream | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Staged Blocks](https://github.com/andrija-djurovic/adsfcr/blob/main/bgrd/staged_blocks.pdf) | `bgrd/staged_blocks.pdf` | local clone | deep | staged blockwise design | medium | low | usar como referencia metodologica | future challenger design ideas | Yes | Yes | not for immediate fairness rerun |
| [Embedded Blocks](https://github.com/andrija-djurovic/adsfcr/blob/main/bgrd/embedded_blocks.pdf) | `bgrd/embedded_blocks.pdf` | local clone | deep | embedded blockwise design | medium | low | usar como referencia metodologica | future challenger design ideas | Yes | Yes | not for immediate fairness rerun |
| [Ensemble Blocks](https://github.com/andrija-djurovic/adsfcr/blob/main/bgrd/ensemble_blocks.pdf) | `bgrd/ensemble_blocks.pdf` | local clone | deep | ensemble blockwise design | medium | low | usar como referencia metodologica | future challenger design ideas | Yes | Yes | not for immediate fairness rerun |
| [Constrained Threshold Logistic Regression](https://github.com/andrija-djurovic/adsfcr/blob/main/bgrd/constrained_threshold_lr.pdf) | `bgrd/constrained_threshold_lr.pdf` | local clone | deep | constrained blending with thresholds | medium | low | usar como referencia metodologica | future policy-blend idea | Yes | Yes | too invasive for current search |

### OLS Regression

| Link | Archivo local | Acceso | Profundidad | Tema | Aplicabilidad | Prioridad | Accion | Impacto esperado | Requiere reentrenamiento | Requiere rerun downstream | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Ordinal, One-Hot and Nested Dummy Approaches in Linear Regression](https://github.com/andrija-djurovic/adsfcr/blob/main/ols/ols_and_rf_encoding.pdf) | `ols/ols_and_rf_encoding.pdf` | local clone | skim | encoding comparison | medium | low | documentar pero no implementar | only indirect relevance | No | No | LGD/EAD and OLS-oriented |
| [Violating the Normality Assumption for OLS Regression](https://github.com/andrija-djurovic/adsfcr/blob/main/ols/normality.pdf) | `ols/normality.pdf` | local clone | skim | OLS assumptions | low | low | descartar por ahora | none | No | No | too generic for current stack |
| [Heteroscedasticity for OLS Regression](https://github.com/andrija-djurovic/adsfcr/blob/main/ols/heteroscedasticity.pdf) | `ols/heteroscedasticity.pdf` | local clone | skim | OLS assumptions | low | low | descartar por ahora | none | No | No | too generic |
| [Multicollinearity for OLS Regression](https://github.com/andrija-djurovic/adsfcr/blob/main/ols/multicollinearity.pdf) | `ols/multicollinearity.pdf` | local clone | skim | OLS assumptions | low | low | descartar por ahora | none | No | No | too generic |
| [Autocorrelation for OLS Regression](https://github.com/andrija-djurovic/adsfcr/blob/main/ols/autocorrelation.pdf) | `ols/autocorrelation.pdf` | local clone | skim | OLS assumptions | low | low | descartar por ahora | none | No | No | too generic |

### Effective Interest Rate

| Link | Archivo local | Acceso | Profundidad | Tema | Aplicabilidad | Prioridad | Accion | Impacto esperado | Requiere reentrenamiento | Requiere rerun downstream | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Effective Interest Rate PDF](https://github.com/andrija-djurovic/adsfcr/blob/main/effective_interest_rate/eir.pdf) | `effective_interest_rate/eir.pdf` | local clone | skim | EIR calculation | low | low | descartar por ahora | none | No | No | accounting utility, not current pipeline need |

### Loan Repayment Plan

| Link | Archivo local | Acceso | Profundidad | Tema | Aplicabilidad | Prioridad | Accion | Impacto esperado | Requiere reentrenamiento | Requiere rerun downstream | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Loan Repayment Plan PDF](https://github.com/andrija-djurovic/adsfcr/blob/main/loan_repayment_plan/lrp.pdf) | `loan_repayment_plan/lrp.pdf` | local clone | skim | amortization / repayment schedule | low | low | descartar por ahora | none | No | No | not relevant to current fairness blocker |

## What To Implement After The New Search

### If the corrected fairness-aware search finds a promotable candidate

- Promote the monotonic challenger.
- Rerun downstream in this order:
  1. champion freeze and lineage update;
  2. predictions, calibration artifacts and threshold semantics;
  3. official fairness audit;
  4. conformal and backtests;
  5. governance, MRM and explainability;
  6. IFRS9/ECL;
  7. portfolio, tradeoff, A/B and policy outputs;
  8. causal, survival and surfaces to unify `run_tag`;
  9. Quarto, Streamlit and final reports.
- Then implement:
  - C2ST with drivers/materiality;
  - monotonicity disruption / heterogeneity audit;
  - PD backtesting suite;
  - IFRS9 diagnostics;
  - encoding/binning stability diagnostics.
- All of these enter first as `diagnostic_only` or `warning`, not as promotion-blocking hard gates.

### If the corrected fairness-aware search still fails on fairness

- Do not start the broad downstream adoptions yet.
- Run the corrected expanded search next.
- Use the richer search artifacts to answer:
  - which attributes fail;
  - at which threshold;
  - what approval rate is implied;
  - how close the candidate is to promotion.
- Keep the next workstream narrow:
  - fairness and monotonicity diagnostics only;
  - no blockwise redesign;
  - no LDP / Vasicek / concentration work;
  - no feature-engineering redesign before understanding the corrected search result.

## Acceptance Criteria

- The monotonic challenger preview runs with official fairness semantics:
  - `outcome_mode=approval`;
  - 6 fairness attributes, not 3;
  - selected threshold from the fairness-aware grid;
  - stable candidate ranking.
- The search output includes:
  - `official_fairness_all_attributes`;
  - `official_fairness_base_only`;
  - selected threshold;
  - approval rate;
  - failed attributes.
- This document remains the canonical human audit artifact and covers:
  - all 90 blob links;
  - the live links listed above;
  - decision per link.
