# Paper Estrella Bound-Improvement Intake 2026-05-21

External artifact root: `/mnt/d/crpto_experiments/regret_auditability/regret_auditability_20260513_v3_resource_tuned`.

## Decision

This intake does not replace the frozen champion. It records a credible PD/conformal
challenger package and defines gated parent-project runs for the bound-improvement lane.

## PD signal

Main challenger: `full_challenger_woe__bureau_behavior_15` with AUC `0.720679`, Brier `0.153161`, ECE `0.007689`. Delta AUC vs incumbent replay is `0.008001`.

## Conformal signal

The child-selected conformal configuration is usable, but not final: grade E fails
the strict 90% group gate and E/G are weak at 95%. The next parent action is a
focused conformal follow-up before any full cuOpt portfolio promotion.

## Portfolio quick signal

The quick CPU run produced an alpha01 pass candidate with return `75602.19`, `V=0.098750`, `Gamma_CP=0.206650` and zero violation, but it used only 25k candidates.

## Parent-project gates

- Do not compare the quick 25k return directly with the frozen 276k champion.
- Run final portfolio only after focused conformal follow-up.
- Use cuOpt/proxy-first broad search plus exact rerank; CPU exact-all is out of scope.
- Promote only if the challenger improves a declared metric without breaking coverage,
  min-group coverage, exact alpha01 pass, zero violation, and source/temporal caveats.

## Generated tables

- `reports/paper_material/paper1/tables/paper1_bound_improvement_pd_intake_2026-05-21.csv`
- `reports/paper_material/paper1/tables/paper1_bound_improvement_conformal_group_diagnostics_2026-05-21.csv`
- `reports/paper_material/paper1/tables/paper1_bound_improvement_conformal_config_candidates_2026-05-21.csv`
- `reports/paper_material/paper1/tables/paper1_bound_improvement_portfolio_quick_alpha01_2026-05-21.csv`
- `reports/paper_material/paper1/tables/paper1_bound_improvement_theory_fronts_2026-05-21.csv`

## Bound fronts

- `nested_prospective_confirmation`: Run only after PD/conformal/portfolio selection is frozen. Gate: Strict temporal or prospectively sealed split keeps alpha01 pass with zero violation.
- `direct_crc_ltt_decision_loss`: Calibrate monotone loss L=max(0,sum w_i Y_i - tau) or V directly. Gate: Decision-loss gate passes without weakening return or coverage.
- `dependency_aware_concentration`: Cluster by issue_month, grade, source/state and compare cluster-robust tail bounds. Gate: Cluster-aware bound is less vacuous than Markov and more credible than iid.
- `mondrian_funded_set_refinement`: Compute sum_g W_g alpha_g for selected and challenger policies. Gate: Weighted group bound improves over nominal alpha without hidden subgroup failure.
- `decision_aware_conformal_selector`: Select by coverage, width, robust_return, V, gamma_cp, violation and group gates. Gate: Selector changes or confirms conformal choice under decision metrics.
- `less_conservative_uncertainty_sets`: Compare grade x scoreband or polyhedral/contextual conformal candidates. Gate: Gamma_CP falls while coverage and min-group gates hold.
- `online_shift_aware_bound`: Use temporal replay or online/weighted conformal only as declared retrospective gate. Gate: Coverage and V remain stable under sealed temporal slices.
- `richer_financial_target`: Prototype LGD*default or ECL proxy loss if data quality is sufficient. Gate: Financial target improves interpretability without adding unsupported IFRS9 claims.
