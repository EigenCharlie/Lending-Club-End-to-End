# Project Justification — Design Rationale
Version: 2026-02-20

This document explains the **current official** technical design decisions.
For runtime metrics and latest run outputs, use artifact files (not hardcoded doc snapshots).

Official-vs-history split:
- Official current standards: `CLAUDE.md` and this file.
- Decision changes, errors, learnings: `docs/DECISION_CHANGES_AND_LEARNINGS.md`.

---

## 1) Problem and Thesis Contribution

The project builds a full credit risk decision stack:

```
Data → Feature Engineering → PD Modeling → Conformal Uncertainty →
  IFRS9 ECL Staging → Portfolio Optimization
```

The core thesis value is **decision quality under uncertainty**, not only predictive accuracy. The design explicitly connects calibrated risk estimation, finite-sample uncertainty quantification, and constrained optimization.

---

## 2) Architecture Rationale

The repository separation is intentional:
- `src/`: reusable analytical logic
- `scripts/`: executable entry points
- `notebooks/`: diagnostics and narrative analysis
- `configs/`: parameter control
- `data/` and `models/`: reproducible assets
- `api/`, `streamlit_app/`: delivery layer (implemented, Streamlit-first in thesis mode)

This supports both research-speed iteration and a clean migration path if productionization is needed later.

---

## 3) Method Justification

### 3.1 PD and Calibration
- `Logistic Regression` is the mandatory baseline for interpretability, governance, and regulatory auditability.
- `CatBoost` is the final-model family for stronger tabular discrimination (nonlinearities/interactions, native categorical + NaN handling).
- Calibration is selected by temporal multi-metric policy (Platt/Isotonic) under OOT-oriented constraints.
- Lending decisions, IFRS9, and pricing require probability quality, not only ranking quality.

### 3.2 Conformal Uncertainty
- Global split conformal as baseline, then Mondrian (grade-conditional) extension.
- Formal acceptance policy (7 checks) and temporal backtesting layer.
- Marginal guarantees are valuable but insufficient for portfolio decisions. Segment-conditional diagnostics reduce hidden subgroup risk distortions.

### 3.3 Time Series
- Portfolio-level default rate dynamics for IFRS9 forward-looking component.
- Statistical baselines (AutoARIMA, AutoETS) + ML (LightGBM with conformal intervals via mlforecast).

### 3.4 Survival Analysis
- Time-to-default modeling for IFRS9 lifetime PD curves.
- Cox PH for interpretable hazard ratios + Random Survival Forest for non-parametric flexibility.

### 3.5 Causal Inference
- Intervention-level effects beyond correlation.
- DoWhy DAG + EconML Double ML for debiased ATE/CATE estimation.
- Refutation tests validate causal identification.

### 3.6 Portfolio Optimization
- Converts predictive outputs into constrained capital allocation decisions.
- Pyomo LP/MILP with HiGHS solver. Robust mode uses conformal PD_high as worst-case constraint.
- Price of robustness quantifies the economic cost of conservative decisions.

---

## 4) Why Each Component Matters

### Decision Significance
- Better interval efficiency directly improves capital deployment feasibility.
- Overly wide intervals → conservative constraints → under-utilized budget.
- Segment-level uncertainty diagnostics prevent hidden concentration of model risk.

### IFRS9 and Governance
- Uncertainty-aware ECL ranges are more informative for reserve planning than point estimates.
- End-to-end traceability improves model risk governance and auditability.
- Conformal interval width as SICR signal adds an uncertainty-based staging dimension.

### Strategic Policy
- Causal and survival modules complement point PD by adding intervention and horizon perspectives.
- They improve policy interpretation and stress governance beyond pure prediction.

---

## 5) Remaining Technical Priorities

1. Keep artifact narratives fully dynamic in Streamlit (no stale hardcoded metrics/claims).
2. Continue benchmark refresh against Kaggle/public literature with temporal-validation comparability checks.
3. Tighten config semantics where current runtime behavior differs from legacy config wording.
4. Preserve API as optional support layer; Streamlit remains primary thesis interface.
