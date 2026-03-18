"""Survival analysis: time-to-default modeling.

Cox PH (lifelines), Random Survival Forest (scikit-survival).
Handles right-censored loans and generates lifetime PD curves for IFRS9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def train_cox_ph(
    df: pd.DataFrame,
    duration_col: str = "time_to_event",
    event_col: str = "event_observed",
    feature_cols: list[str] | None = None,
) -> tuple:
    """Train Cox Proportional Hazards model."""
    from lifelines import CoxPHFitter

    cph = CoxPHFitter(penalizer=0.01)
    cols = feature_cols + [duration_col, event_col] if feature_cols else df.columns.tolist()
    cph.fit(df[cols], duration_col=duration_col, event_col=event_col)

    logger.info(f"Cox PH — Concordance: {cph.concordance_index_:.4f}")
    cph.print_summary()
    return cph, {"concordance_index": cph.concordance_index_}


def check_cox_assumptions(cph, df: pd.DataFrame) -> None:
    """Check proportional hazards assumption."""
    cph.check_assumptions(df, show_plots=True)


def train_random_survival_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_estimators: int = 500,
    min_samples_leaf: int = 5,
    min_samples_split: int = 10,
    max_features: str | int | float = "sqrt",
    max_depth: int | None = None,
    max_samples: int | float | None = None,
    n_jobs: int = -1,
) -> tuple:
    """Train Random Survival Forest.

    Args:
        y_train/y_test: Structured array with fields (event, time).
            Create with: np.array([(bool, float), ...], dtype=[('event', bool), ('time', float)])
    """
    from sksurv.ensemble import RandomSurvivalForest

    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_depth=max_depth,
        max_samples=max_samples,
        random_state=42,
        n_jobs=n_jobs,
    )
    rsf.fit(X_train, y_train)

    c_index = rsf.score(X_test, y_test)
    logger.info(f"Random Survival Forest — C-index: {c_index:.4f}")
    return rsf, {"c_index": c_index}


def make_survival_target(
    df: pd.DataFrame,
    event_col: str = "default_flag",
    time_col: str = "time_to_event",
) -> np.ndarray:
    """Create structured array for scikit-survival."""
    return np.array(
        list(zip(df[event_col].astype(bool), df[time_col].astype(float), strict=False)),
        dtype=[("event", bool), ("time", float)],
    )


def generate_lifetime_pd_curve(
    model,
    X: pd.DataFrame,
    times: np.ndarray | None = None,
) -> pd.DataFrame:
    """Generate lifetime PD curves (1 - survival function) at given times.

    Useful for IFRS9 Stage 2/3 lifetime ECL calculation.
    """
    if times is None:
        times = np.arange(30, 1830, 30)  # Every month for 5 years

    surv_funcs = model.predict_survival_function(X)
    pd_curves = []
    for fn in surv_funcs:
        pds = 1 - fn(times)
        pd_curves.append(pds)

    result = pd.DataFrame(pd_curves, columns=[f"PD_t{t}" for t in times])
    logger.info(f"Generated lifetime PD curves: {result.shape}")
    return result


def compute_cif(
    df: pd.DataFrame,
    time_col: str = "lgd_months_since_issue",
    status_col: str = "loan_status",
    default_statuses: tuple[str, ...] = ("Charged Off", "Default"),
    prepay_statuses: tuple[str, ...] = ("Fully Paid",),
    group_col: str | None = "grade",
    conf_level: float = 0.95,
) -> dict[str, object]:
    """Compute Cumulative Incidence Functions for competing risks.

    Treats default (Charged Off) and prepayment (Fully Paid) as competing events,
    with ongoing loans as right-censored. Uses the Aalen-Johansen estimator via
    sksurv.nonparametric.cumulative_incidence_competing_risks.

    Args:
        df: DataFrame with loan records.
        time_col: Column with time-to-event (months since issue).
        status_col: Column indicating final loan status.
        default_statuses: Status labels that correspond to the default event.
        prepay_statuses: Status labels that correspond to prepayment event.
        group_col: Optional grouping column (e.g., grade) for stratified CIF.
        conf_level: Confidence level for CIF confidence intervals.

    Returns:
        Dict with keys 'overall', 'by_group' (if group_col given).
        Each entry has 'times', 'cif_default', 'cif_prepay', and optional CI arrays.
    """
    from sksurv.nonparametric import cumulative_incidence_competing_risks

    def _encode_event(status: pd.Series) -> np.ndarray:
        """Encode: 0=censored, 1=default, 2=prepayment."""
        encoded = np.zeros(len(status), dtype=int)
        encoded[status.isin(default_statuses)] = 1
        encoded[status.isin(prepay_statuses)] = 2
        return encoded

    def _compute_group_cif(sub: pd.DataFrame) -> dict[str, object] | None:
        time = pd.to_numeric(sub[time_col], errors="coerce").to_numpy(dtype=float)
        status_encoded = _encode_event(sub[status_col])

        # Drop rows with missing time or zero-time
        valid = (np.isfinite(time)) & (time > 0)
        time = time[valid]
        status_encoded = status_encoded[valid]

        if len(time) < 50:
            return None

        # Need both event types present
        if not (np.any(status_encoded == 1) and np.any(status_encoded == 2)):
            logger.warning("Group lacks both event types — skipping CIF estimation")
            return None

        try:
            result = cumulative_incidence_competing_risks(
                status_encoded, time, conf_level=conf_level, conf_type="log-log"
            )
        except Exception as exc:
            logger.warning("CIF estimation failed: {}", exc)
            return None

        # result is (times, cif_matrix[n_events+1, n_times], ci_matrix[n_events+1, 2, n_times])
        # cif_matrix[0] = overall, cif_matrix[1] = default, cif_matrix[2] = prepay
        times_arr = result[0]  # shape (n_times,)
        cif_matrix = result[1]  # shape (n_events+1, n_times)
        ci_matrix = result[2] if len(result) > 2 else None  # shape (n_events+1, 2, n_times)

        cif_default_arr = cif_matrix[1]  # event=1 (default)
        cif_prepay_arr = cif_matrix[2]  # event=2 (prepayment)

        cif_d_final = float(cif_default_arr[-1]) if len(cif_default_arr) > 0 else float("nan")
        cif_p_final = float(cif_prepay_arr[-1]) if len(cif_prepay_arr) > 0 else float("nan")

        return {
            "n": int(len(time)),
            "n_default": int((status_encoded == 1).sum()),
            "n_prepay": int((status_encoded == 2).sum()),
            "n_censored": int((status_encoded == 0).sum()),
            "times": times_arr.tolist(),
            "cif_default": cif_default_arr.tolist(),
            "cif_prepay": cif_prepay_arr.tolist(),
            "cif_default_lo": ci_matrix[1, 0, :].tolist() if ci_matrix is not None else None,
            "cif_default_hi": ci_matrix[1, 1, :].tolist() if ci_matrix is not None else None,
            "cause_specific_hazard_ratio": float(cif_d_final / max(cif_p_final, 1e-9)),
        }

    output: dict[str, object] = {}

    # Overall CIF
    logger.info("Computing overall CIF (n={:,})", len(df))
    overall = _compute_group_cif(df)
    if overall:
        output["overall"] = overall
        logger.info(
            "CIF overall: n_default={:,}, n_prepay={:,}, "
            "CIF_default(final)={:.4f}, CIF_prepay(final)={:.4f}",
            overall["n_default"],
            overall["n_prepay"],
            overall["cif_default"][-1] if overall["cif_default"] else float("nan"),
            overall["cif_prepay"][-1] if overall["cif_prepay"] else float("nan"),
        )

    # Stratified CIF
    if group_col and group_col in df.columns:
        by_group: dict[str, object] = {}
        for grp, sub in df.groupby(group_col):
            logger.info("  Computing CIF for group '{}' (n={:,})", grp, len(sub))
            grp_result = _compute_group_cif(sub)
            if grp_result:
                by_group[str(grp)] = grp_result
        output["by_group"] = by_group
        logger.info("CIF by grade: {} groups computed", len(by_group))

    return output
