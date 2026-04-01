# Model Risk Management Document — SR 11-7

## 1. Model Identification & Purpose

| Field | Value |
|-------|-------|
| **Model Name** | CorePDCanonical |
| **Model Type** | Probability of Default (PD) — Binary Classification |
| **Algorithm** | Monotonic CatBoost Gradient Boosting + probabilistic calibration (current champion artifact uses Venn-Abers) |
| **Uncertainty Quantification** | Mondrian Conformal Prediction (MAPIE 1.3) |
| **Owner** | Carlos Vergara |
| **Version** | See `models/pd_training_record.pkl` for current version |
| **Champion Artifact** | `models/pd_canonical.cbm` |
| **Calibrator Artifact** | `models/pd_canonical_calibrator.pkl` |
| **Feature Contract** | `models/pd_model_contract.json` (42 champion features; broader FE universe tracked in `data/processed/feature_manifest_v2.parquet`) |

### Intended Use
- **Primary**: PD estimation for credit portfolio optimization under uncertainty
- **Secondary**: IFRS9 ECL computation (Stage 1/2/3 classification), conformal interval generation for robust decision-making

### Out-of-Scope Uses
- Individual loan underwriting decisions without human review
- Real-time credit scoring for automated approval/denial
- Application to non-Lending-Club loan populations without recalibration
- Use as a sole regulatory capital model without independent validation

---

## 2. Model Development

### 2.1 Data

**Source**: Lending Club Loan Data (Kaggle), 2.26M loans, 2007-2020.

**Temporal Splits** (out-of-time, NOT random):

| Split | Rows | Default Rate | Date Range |
|-------|------|-------------|------------|
| Train | 1,346,311 | 18.52% | 2007-06 to 2017-03 |
| Calibration | 237,584 | 22.20% | 2017-03 to 2017-12 |
| Test (OOT) | 276,869 | 21.98% | 2018-01 to 2020-09 |

**Data Leakage Prevention**: Post-loan variables removed in `src/data/make_dataset.py`:
total_pymnt, total_rec_*, recoveries, collection_recovery_fee, out_prncp*, last_pymnt_*, settlement_*, hardship_*, funded_amnt*.

**Feature Engineering**: The rerun V2 introduced a canonical feature producer that materializes `train_fe`, `calibration_fe`, `test_fe`, `feature_config.pkl`, `woe_encoders.pkl`, and `feature_manifest_v2`. The official champion now freezes a **42-feature contract**, while the broader FE universe retains additional bureau, ratio, missingness-flag, and challenger-only variables. Schema enforced by Pandera (`src/features/schemas.py`).

### 2.2 Methodology

**Architecture**: Predict → Calibrate → Conformalize → Optimize

1. **Logistic Regression baseline** — regulatory interpretability benchmark
2. **CatBoost default** — gradient boosting with default hyperparameters
3. **CatBoost tuned** — Optuna HPO (when enabled by config)
4. **Calibration selection** — temporal multi-metric policy evaluates Platt Sigmoid, Isotonic Regression, Venn-Abers, and Beta Calibration; the current canonical artifact is Venn-Abers
5. **Fairness on approval decisions** — official fairness and threshold semantics are read on approval outcomes, not on internal PD search thresholds
6. **Mondrian Conformal Prediction** — group-conditional coverage by grade using MAPIE 1.3 SplitConformalRegressor
7. **Robust Portfolio Optimization** — Pyomo + HiGHS with box uncertainty sets from conformal intervals
8. **Post-promotion diagnostics** — C2ST with drivers, monotonicity audit, PD backtesting suite, PD validation interpretation, IFRS9 diagnostics, and encoding stability

### 2.3 Key Assumptions and Limitations

1. **Exchangeability** (Conformal): calibration and test observations are drawn from the same distribution. Temporal split mitigates but does not eliminate distribution shift risk.
2. **Grade A coverage**: Group A has fewer calibration samples, resulting in wider intervals. Coverage may be slightly below target for this subgroup.
3. **Time series exchangeability**: 118-month history is limited for long-horizon forecasting. IFRS9 scenarios should be treated as indicative, not precise.
4. **Cox PH proportional hazards**: The assumption is violated for some covariates; Random Survival Forest is used as a robustness check.
5. **No demographic data**: Lending Club does not provide race, gender, or age. Fairness analysis is limited to proxy attributes (home_ownership, income quartile, verification status).

---

## 3. Model Validation

### 3.1 Backtesting (Out-of-Time)
- **Metrics**: AUC-ROC, Gini, Brier score, KS statistic, ECE
- **Source**: `data/processed/pipeline_summary.json`
- The OOT test set (2018-2020) was never seen during training or calibration
- Additional diagnostic layers now exist for post-promotion validation:
  - `models/pd_backtesting_status.json` — exact binomial, Jeffreys, z-score, HL;
  - `models/bootstrap_validation_status.json` — bootstrap materiality layer for aggregate and slice-level calibration gaps;
  - `models/pd_validation_interpretation_status.json` — materiality-oriented interpretation of those tests plus quarter persistence;
  - `models/calibration_mapping_status.json` — shadow remap/intercept comparison without replacing the canonical calibrator;
  - `models/calibration_mapping_shadow_impact_status.json` — consolidated decision artifact for the executed shadow lane;
  - `models/pd_rare_event_calibration_status.json` — rare-event and slice calibration sidecar.

Current reading after execution:
- the shadow lane was executed on the confirmatory monotonic run and closed with `keep_current_calibrator`;
- no lightweight remap candidate improved cohort persistence without degrading calibration quality;
- therefore the remaining PD work is better framed as cohort-sensitive analytical interpretation, not as a simple calibrator replacement exercise.

### 3.2 Benchmarking
- **Comparison**: Logistic Regression vs CatBoost default vs CatBoost tuned
- **Source**: `data/processed/model_comparison.json`
- Logistic Regression serves as the mandatory interpretable baseline

### 3.3 Conformal Coverage
- **Target**: 90% coverage (alpha = 0.10)
- **Mondrian groups**: By grade (A-G) for group-conditional coverage
- **Policy gate**: the current canonical status is `overall_pass = true`, `strict_overall_pass = false`, and `methodological_justification_pass = true`; non-statistical checks pass and statistical tests remain diagnostic because coverage is conservatively above nominal in a very large OOT sample
- **Source**: `models/conformal_policy_status.json`

### 3.4 Fairness Audit
- **Metrics**: Demographic Parity Difference, Equalized Odds Gap, Disparate Impact Ratio
- **Attributes**: home_ownership, annual_inc quartile, verification_status
- **Thresholds**: DPD < 0.10, EO gap < 0.10, DIR > 0.80
- **Decision semantics**: `outcome_mode=approval`
- **Source**: `models/fairness_audit_status.json`

---

## 4. Model Governance

### 4.1 Roles and Responsibilities

| Role | Responsibility |
|------|---------------|
| **Developer** | Model training, testing, documentation |
| **Validator** | Independent validation, stress testing |
| **Model Owner** | Approval, deployment decisions, risk acceptance |

### 4.2 Change Management
- All model changes tracked via Git (DagsHub mirror)
- Data and model artifacts versioned with DVC
- MLflow experiments logged for reproducibility
- Decision history in `docs/DECISION_CHANGES_AND_LEARNINGS.md`

### 4.3 Documentation Lineage
- `CLAUDE.md` — project conventions and standards
- `docs/PROJECT_JUSTIFICATION.md` — methodology justification
- `configs/pd_model.yaml` — model hyperparameters
- `configs/conformal_policy.yaml` — conformal prediction policy
- `configs/fairness_policy.yaml` — fairness audit thresholds

---

## 5. Model Use and Limitations

### Approved Uses
- Academic thesis demonstration of predict-then-optimize pipeline
- Portfolio-level risk assessment and optimization
- IFRS9 Expected Credit Loss estimation under multiple scenarios
- Conformal prediction research and methodology validation

### Known Limitations
1. Model trained on US consumer lending data (2007-2020) — may not generalize to other geographies or time periods
2. Conformal coverage guarantee holds under exchangeability — distribution shifts void the guarantee
3. No causal interpretation of features — model is predictive, not causal (causal analysis in NB07 is separate)
4. LGD fixed at 0.45 in optimization — a simplification; two-stage LGD model exists but is not integrated into the optimizer
5. Portfolio optimization assumes linear programming — real-world constraints (regulatory capital, liquidity) are more complex

---

## 6. Ongoing Monitoring Plan

### Monitoring Metrics

| Metric | Threshold | Frequency | Source |
|--------|-----------|-----------|--------|
| AUC degradation | < 0.03 vs baseline | Quarterly | Pipeline summary |
| Conformal coverage | > 0.88 (at 0.90 target) | Quarterly | Conformal policy |
| Fairness (DIR) | > 0.80 | Quarterly | Fairness audit |
| Conformal statistical validity | Kupiec/Christoffersen (diagnostic only; `methodological_justification_pass` is the operational gate) | Quarterly | Conformal policy v2 |
| Drift governance | KS/CvM/C2ST policy pass | Quarterly | Governance status |
| PD validation interpretation | `warning` or better | Quarterly | PD validation interpretation status |
| Bootstrap validation | Large-`N` calibration gap uncertainty by aggregate and slice | Quarterly | Bootstrap validation status |
| Calibration mapping sidecar | Shadow remap/intercept review for cohort persistence; currently closed as keep-current-calibrator | Quarterly | Calibration mapping status + shadow impact status |
| IFRS9 diagnostics | Diagnostic review required when recursive stability / ADF power deteriorate | Quarterly | IFRS9 diagnostics status |
| Monotonicity audit | `overall_pass = true` expected for the promoted monotonic champion | Quarterly | Monotonicity audit status |
| Encoding stability | `overall_pass = true` expected unless bins/WOE become unstable | Quarterly | Encoding stability status |
| Model-shift posture | Structural shift vs predictive degradation distinction | Quarterly | Model shift status |

### Retraining Triggers
- PSI exceeds 0.25 on any monitored feature
- AUC drops more than 0.03 below the champion baseline
- Conformal coverage drops more than 0.02 below target
- See `configs/mrm_policy.yaml` for machine-readable thresholds

### Escalation
- Automated: JSON status files (`conformal_policy_status.json`, `fairness_audit_status.json`, `governance_status.json`) gate deployment
- Manual: quarterly review of monitoring dashboard (Streamlit → Model Governance page)
- Canonical single-write status artifacts are enforced for conformal/fairness/governance

---

## 7. Champion/Challenger Framework

### Current Champion
- **Model**: `models/pd_canonical.cbm` (monotonic CatBoost + probabilistic calibration; current artifact is Venn-Abers)
- **Calibrator**: `models/pd_canonical_calibrator.pkl`
- **Contract**: `models/pd_model_contract.json` (42 champion features)
- **Registry source**: `models/champion_registry.json`

### Challenger Criteria
A challenger model must demonstrate:
- AUC improvement ≥ 0.005 over champion on OOT test set
- ECE improvement ≥ 0.002 (better calibration)
- No degradation in conformal coverage or fairness metrics
- Monotonic constraints aligned with domain priors when the challenger lane is explicitly monotonic
- Feature-selection evidence package (`data/processed/challenger_feature_selection.parquet`)
- Explicit policy: **no SMOTE** in challenger or champion training flows

### Promotion Gate
All of the following must pass:
1. Conformal policy gate (current canonical artifact passes operationally via `methodological_justification_pass = true`; strict statistical diagnostics remain documented but non-blocking)
2. Fairness audit (all attributes pass thresholds)
3. Governance checks (drift, robustness, slicing)
4. Independent validation review

### Retirement Policy
- Superseded models archived with version tag in DVC
- MLflow experiment history preserved for audit trail
- Minimum 90-day parallel run before champion retirement
