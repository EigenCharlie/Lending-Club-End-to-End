# Project Justification — Design Rationale
Version: 2026-02-14

This document explains the technical design decisions. For current metrics and state, see `SESSION_STATE.md`.

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
- CatBoost for robust tabular performance with native categorical and NaN handling.
- Platt sigmoid calibration selected empirically (ECE=0.0128).
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

1. Deepen thesis storytelling pages in Streamlit (method, trade-offs, governance narrative).
2. Expand governance visual layer: dbt lineage/docs + Feast feature-service explanation in UI.
3. Harden optional NL->SQL assistant workflow (prompting, SQL safety, fallback behavior).
4. CI/CD pipeline (pytest + ruff + Streamlit import smoke tests).
5. Keep API optional and focused on reusable endpoints, not as mandatory dashboard backend.
