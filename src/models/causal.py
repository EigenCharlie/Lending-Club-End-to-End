"""Causal Machine Learning helpers for the official causal stack.

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

from collections.abc import Iterable
from typing import Any

# Fix networkx 3.6 / DoWhy 0.12 incompatibility
import networkx.algorithms as _nxa
import numpy as np
import pandas as pd
from loguru import logger

if not hasattr(_nxa, "d_separated"):
    from networkx.algorithms.d_separation import is_d_separator as _is_d_sep

    _nxa.d_separated = lambda G, x, y, z: _is_d_sep(G, x, y, z)


def _require_dowhy() -> None:
    """Ensure DoWhy is installed before calling DoWhy-backed routines."""
    try:
        import dowhy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "DoWhy is optional in the main environment. Use the dedicated causal env "
            "(`.venv-causal`) created via `bash scripts/causal/setup_causal_env.sh .venv-causal`."
        ) from exc


def _require_econml() -> None:
    """Ensure EconML is installed before calling EconML-backed estimators."""
    try:
        import econml  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "EconML is optional in the main environment. Create a dedicated causal env (e.g., "
            "`.venv-causal`) with the project stack first, then overlay EconML. Example: "
            "`UV_PROJECT_ENVIRONMENT=.venv-causal uv sync --python 3.12 --extra dev --extra platform` "
            "and `uv pip install --python .venv-causal/bin/python -r requirements/causal-econml.txt`."
        ) from exc


def specify_causal_graph(
    treatment: str = "int_rate",
    outcome: str = "default_flag",
) -> str:
    """Specify the official observed-variable DAG for credit risk.

    Returns DOT string for DoWhy.
    """
    return f"""
    digraph {{
        grade_woe -> {treatment};
        grade_woe -> {outcome};
        purpose_woe -> {treatment};
        purpose_woe -> {outcome};
        home_ownership_woe -> {treatment};
        home_ownership_woe -> {outcome};
        dti -> {treatment};
        dti -> {outcome};
        annual_inc -> {treatment};
        annual_inc -> {outcome};
        annual_inc -> loan_amnt;
        loan_amnt -> {treatment};
        loan_amnt -> {outcome};
        fico_range_low -> {treatment};
        fico_range_low -> {outcome};
        {treatment} -> {outcome};
    }}
    """


def default_effect_modifiers() -> list[str]:
    """Official effect-modifier set used for heterogeneous treatment effects."""
    return ["loan_amnt", "annual_inc", "dti", "fico_range_low"]


def default_confounders() -> list[str]:
    """Official confounder set used for backdoor adjustment."""
    return ["grade_woe", "purpose_woe", "home_ownership_woe"]


def build_overlap_diagnostics(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    segment_columns: Iterable[str] | None = None,
    min_segment_size: int = 50,
) -> pd.DataFrame:
    """Summarize treatment support by categorical segment.

    The goal is operational traceability of positivity/overlap assumptions. This is
    intentionally simple: the artifact is an auditable diagnostic, not a formal test.
    """
    if treatment not in df.columns or outcome not in df.columns:
        return pd.DataFrame()

    candidates = list(segment_columns or ["grade", "purpose", "home_ownership"])
    available = [col for col in candidates if col in df.columns]
    if not available:
        payload = {
            "segment_type": ["all"],
            "segment_value": ["all"],
            "n_obs": [int(len(df))],
            "treatment_min": [float(df[treatment].min())],
            "treatment_p05": [float(df[treatment].quantile(0.05))],
            "treatment_median": [float(df[treatment].median())],
            "treatment_p95": [float(df[treatment].quantile(0.95))],
            "treatment_max": [float(df[treatment].max())],
            "treatment_std": [float(df[treatment].std(ddof=0))],
            "outcome_rate": [float(df[outcome].mean())],
            "support_ok": [bool(len(df) >= min_segment_size and df[treatment].std(ddof=0) > 0)],
        }
        return pd.DataFrame(payload)

    rows: list[dict[str, Any]] = []
    for segment_col in available:
        segment_series = df[segment_col].astype(str).fillna("UNKNOWN")
        grouped = df.assign(_segment_value=segment_series).groupby(
            "_segment_value", observed=True, dropna=False
        )
        for segment_value, grp in grouped:
            treatment_series = pd.to_numeric(grp[treatment], errors="coerce").dropna()
            if treatment_series.empty:
                continue
            rows.append(
                {
                    "segment_type": segment_col,
                    "segment_value": str(segment_value),
                    "n_obs": int(len(grp)),
                    "treatment_min": float(treatment_series.min()),
                    "treatment_p05": float(treatment_series.quantile(0.05)),
                    "treatment_median": float(treatment_series.median()),
                    "treatment_p95": float(treatment_series.quantile(0.95)),
                    "treatment_max": float(treatment_series.max()),
                    "treatment_std": float(treatment_series.std(ddof=0)),
                    "treatment_iqr": float(
                        treatment_series.quantile(0.75) - treatment_series.quantile(0.25)
                    ),
                    "outcome_rate": float(pd.to_numeric(grp[outcome], errors="coerce").mean()),
                    "support_ok": bool(
                        len(grp) >= min_segment_size and treatment_series.std(ddof=0) > 0
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["segment_type", "segment_value"], ignore_index=True)


def _extract_ate_ci(estimate: Any) -> list[float | None]:
    """Best-effort CI extraction from a DoWhy estimate object."""
    candidate = None
    if hasattr(estimate, "get_confidence_intervals"):
        try:
            candidate = estimate.get_confidence_intervals()
        except Exception:
            candidate = None
    if candidate is None:
        candidate = getattr(estimate, "confidence_intervals", None)
    if candidate is None:
        return [None, None]

    try:
        arr = np.asarray(candidate, dtype=float).reshape(-1)
    except Exception:
        return [None, None]
    if arr.size < 2:
        return [None, None]
    return [float(arr[0]), float(arr[1])]


def summarize_refutation(test_name: str, refutation: Any) -> dict[str, Any]:
    """Serialize the most useful parts of a DoWhy refutation result."""
    p_value = getattr(refutation, "p_value", None)
    try:
        p_value = float(p_value) if p_value is not None else None
    except Exception:
        p_value = None

    new_effect = getattr(refutation, "new_effect", None)
    try:
        new_effect = float(new_effect) if new_effect is not None else None
    except Exception:
        new_effect = None

    estimated_effect = getattr(refutation, "estimated_effect", None)
    try:
        estimated_effect = float(estimated_effect) if estimated_effect is not None else None
    except Exception:
        estimated_effect = None

    return {
        "test": str(test_name),
        "estimated_effect": estimated_effect,
        "new_effect": new_effect,
        "p_value": p_value,
        "result": str(refutation),
    }


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
    _require_dowhy()
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
        method_name="backdoor.linear_regression",
    )
    ate_ci = _extract_ate_ci(estimate)
    refutations = run_refutation_tests(model, identified, estimate)

    ate_value = getattr(estimate, "value", None)
    try:
        ate_value = float(ate_value) if ate_value is not None else None
    except Exception:
        ate_value = None
    logger.info(
        f"ATE of {treatment} on {outcome}: {ate_value:.6f}"
        if ate_value is not None
        else f"ATE of {treatment} on {outcome}: unavailable"
    )
    return {
        "ate": ate_value,
        "ate_ci": ate_ci,
        "estimate_object": estimate,
        "identified_estimand": identified,
        "identification_strategy": "backdoor",
        "model": model,
        "refutation_summary": refutations,
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
    _require_econml()
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

    refuter_specs = [
        (
            "placebo_treatment",
            {"method_name": "placebo_treatment_refuter", "placebo_type": "permute"},
        ),
        (
            "random_common_cause",
            {"method_name": "random_common_cause"},
        ),
        (
            "data_subset",
            {"method_name": "data_subset_refuter", "subset_fraction": 0.8},
        ),
    ]
    for test_name, params in refuter_specs[: max(int(n_tests), 0)]:
        try:
            refutation = model.refute_estimate(identified_estimand, estimate, **params)
            refutations.append(summarize_refutation(test_name, refutation))
        except Exception as exc:
            refutations.append(
                {
                    "test": str(test_name),
                    "estimated_effect": None,
                    "new_effect": None,
                    "p_value": None,
                    "result": f"refutation_unavailable: {exc}",
                }
            )

    for r in refutations:
        logger.info(f"Refutation [{r['test']}]: {r['result'][:100]}...")

    return refutations
