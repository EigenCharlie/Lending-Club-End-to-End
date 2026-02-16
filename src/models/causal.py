"""Causal Machine Learning: Double ML and CATE estimation.

Uses EconML (DML, CausalForestDML) and DoWhy (DAG, refutation).
Key causal questions:
  - What is the causal effect of interest rate on default?
  - Does income verification causally reduce default probability?
  - Optimal rate/limit per customer segment (policy learning).

API notes (2025-2026 versions):
  - DoWhy 0.12 + networkx 3.6: d_separated renamed to d_separation.is_d_separator
  - EconML 0.16: CausalForestDML uses GBM nuisance models, shap_values returns nested dict
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# Fix networkx 3.6 / DoWhy 0.12 incompatibility
import networkx.algorithms as _nxa
if not hasattr(_nxa, "d_separated"):
    from networkx.algorithms.d_separation import is_d_separator as _is_d_sep
    _nxa.d_separated = lambda G, x, y, z: _is_d_sep(G, x, y, z)


def specify_causal_graph() -> str:
    """Specify the causal DAG for credit risk.

    Returns DOT string for DoWhy.
    """
    return """
    digraph {
        grade -> int_rate;
        grade -> default;
        dti -> default;
        annual_inc -> default;
        annual_inc -> loan_amnt;
        loan_amnt -> default;
        int_rate -> default;
        purpose -> default;
        home_ownership -> default;
        emp_length -> annual_inc;
        credit_history -> grade;
    }
    """


def estimate_ate_dowhy(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: list[str],
    graph: str | None = None,
) -> dict[str, Any]:
    """Estimate Average Treatment Effect using DoWhy.

    Identifies causal effect, estimates, and runs refutation tests.
    """
    import dowhy

    model = dowhy.CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes,
        graph=graph,
    )

    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.econml.dml.CausalForestDML",
        method_params={
            "init_params": {"n_estimators": 200, "random_state": 42},
            "fit_params": {},
        },
    )

    logger.info(f"ATE of {treatment} on {outcome}: {estimate.value:.6f}")
    return {
        "ate": estimate.value,
        "estimate_object": estimate,
        "identified_estimand": identified,
    }


def estimate_cate(
    Y: pd.Series,
    T: pd.Series,
    X: pd.DataFrame,
    W: pd.DataFrame | None = None,
    n_estimators: int = 200,
) -> tuple[Any, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Estimate Conditional Average Treatment Effect using CausalForestDML.

    Args:
        Y: Outcome (default_flag).
        T: Treatment (int_rate, verification_status, etc.).
        X: Effect modifiers (customer features for heterogeneity).
        W: Confounders to control for.

    Returns:
        Tuple of (fitted_model, cate_estimates, (lower_bound, upper_bound)).
    """
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor

    est = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        n_estimators=n_estimators,
        random_state=42,
        cv=3,
    )
    est.fit(Y=Y, T=T, X=X, W=W)

    cate = est.effect(X)
    lb, ub = est.effect_interval(X, alpha=0.05)

    logger.info(
        f"CATE estimated: mean={cate.mean():.6f}, "
        f"std={cate.std():.6f}, "
        f"range=[{cate.min():.6f}, {cate.max():.6f}]"
    )
    return est, cate, (lb, ub)


def run_refutation_tests(
    model,
    identified_estimand,
    estimate,
    n_tests: int = 3,
) -> list[dict[str, Any]]:
    """Run DoWhy refutation tests to validate causal estimates."""
    refutations = []

    # Placebo treatment
    ref_placebo = model.refute_estimate(
        identified_estimand, estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
    )
    refutations.append({"test": "placebo_treatment", "result": str(ref_placebo)})

    # Random common cause
    ref_random = model.refute_estimate(
        identified_estimand, estimate,
        method_name="random_common_cause",
    )
    refutations.append({"test": "random_common_cause", "result": str(ref_random)})

    # Data subset
    ref_subset = model.refute_estimate(
        identified_estimand, estimate,
        method_name="data_subset_refuter",
        subset_fraction=0.8,
    )
    refutations.append({"test": "data_subset", "result": str(ref_subset)})

    for r in refutations:
        logger.info(f"Refutation [{r['test']}]: {r['result'][:100]}...")

    return refutations
