# Model Risk Management Document — SR 11-7

## 1. Model Identification & Purpose

| Field | Value |
|-------|-------|
| **Model Name** | CorePDCanonical |
| **Model Type** | Probability of Default (PD) — Binary Classification |
| **Algorithm** | CatBoost Gradient Boosting + Platt Sigmoid Calibration |
| **Uncertainty Quantification** | Mondrian Conformal Prediction (MAPIE 1.3) |
| **Owner** | Carlos Vergara |
| **Version** | See `models/pd_training_record.pkl` for current version |
| **Champion Artifact** | `models/pd_canonical.cbm` |
| **Calibrator Artifact** | `models/pd_canonical_calibrator.pkl` |
| **Feature Contract** | `models/pd_model_contract.json` (44 features) |

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

**Feature Engineering**: 44 features including WOE-encoded categoricals (OptBinning), financial ratios, and temporal features. Schema enforced by Pandera (`src/features/schemas.py`).

### 2.2 Methodology

**Architecture**: Predict → Calibrate → Conformalize → Optimize

1. **Logistic Regression baseline** — regulatory interpretability benchmark
2. **CatBoost default** — gradient boosting with default hyperparameters
3. **CatBoost tuned** — Optuna HPO (when enabled by config)
4. **Calibration selection** — temporal multi-metric policy selects Platt Sigmoid or Isotonic Regression
5. **Mondrian Conformal Prediction** — group-conditional coverage by grade using MAPIE 1.3 SplitConformalRegressor
6. **Robust Portfolio Optimization** — Pyomo + HiGHS with box uncertainty sets from conformal intervals

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

### 3.2 Benchmarking
- **Comparison**: Logistic Regression vs CatBoost default vs CatBoost tuned
- **Source**: `data/processed/model_comparison.json`
- Logistic Regression serves as the mandatory interpretable baseline

### 3.3 Sensitivity and Robustness
- **Feature perturbation**: Noise injection (2%, 5%) on key features
- **AUC degradation threshold**: < 0.04 drop under perturbation
- **Source**: `data/processed/modeva_governance_robustness.parquet`

### 3.4 Conformal Coverage
- **Target**: 90% coverage (alpha = 0.10)
- **Mondrian groups**: By grade (A-G) for group-conditional coverage
- **Policy gate**: 7/7 checks must pass
- **Source**: `models/conformal_policy_status.json`

### 3.5 Fairness Audit
- **Metrics**: Demographic Parity Difference, Equalized Odds Gap, Disparate Impact Ratio
- **Attributes**: home_ownership, annual_inc quartile, verification_status
- **Thresholds**: DPD < 0.10, EO gap < 0.10, DIR > 0.80
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
| PSI (feature drift) | < 0.25 | Quarterly | Modeva governance |
| KS drift | < 0.20 | Quarterly | Modeva governance |
| AUC degradation | < 0.03 vs baseline | Quarterly | Pipeline summary |
| Conformal coverage | > 0.88 (at 0.90 target) | Quarterly | Conformal policy |
| Fairness (DIR) | > 0.80 | Quarterly | Fairness audit |
| Robustness (AUC drop under noise) | < 0.04 | Quarterly | Modeva robustness |

### Retraining Triggers
- PSI exceeds 0.25 on any monitored feature
- AUC drops more than 0.03 below the champion baseline
- Conformal coverage drops more than 0.02 below target
- See `configs/mrm_policy.yaml` for machine-readable thresholds

### Escalation
- Automated: JSON status files (`conformal_policy_status.json`, `fairness_audit_status.json`, `modeva_governance_status.json`) gate deployment
- Manual: quarterly review of monitoring dashboard (Streamlit → Model Governance page)

---

## 7. Champion/Challenger Framework

### Current Champion
- **Model**: `models/pd_canonical.cbm` (CatBoost + Platt calibration)
- **Calibrator**: `models/pd_canonical_calibrator.pkl`
- **Contract**: `models/pd_model_contract.json` (44 features)

### Challenger Criteria
A challenger model must demonstrate:
- AUC improvement ≥ 0.005 over champion on OOT test set
- ECE improvement ≥ 0.002 (better calibration)
- No degradation in conformal coverage or fairness metrics

### Promotion Gate
All of the following must pass:
1. Conformal policy gate (7/7 checks)
2. Fairness audit (all attributes pass thresholds)
3. Governance checks (drift, robustness, slicing)
4. Independent validation review

### Retirement Policy
- Superseded models archived with version tag in DVC
- MLflow experiment history preserved for audit trail
- Minimum 90-day parallel run before champion retirement
