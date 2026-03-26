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


def compute_lifetime_pd_conformal_intervals(
    survival_model,
    X_cal: pd.DataFrame,
    y_cal_event: np.ndarray,
    y_cal_time: np.ndarray,
    X_test: pd.DataFrame,
    horizon_months: int = 60,
    confidence_level: float = 0.90,
) -> dict[str, np.ndarray]:
    """Compute conformal prediction intervals on lifetime PD from a survival model.

    Uses split conformal on the residuals of lifetime PD predictions:
    residual = actual_default_indicator - predicted_lifetime_PD.

    For censored observations (didn't default within horizon), the actual default
    indicator is 0 only if observation time >= horizon; otherwise excluded.

    Args:
        survival_model: Fitted survival model with predict_survival_function().
        X_cal: Calibration features.
        y_cal_event: Event indicator (True = default observed).
        y_cal_time: Time to event/censoring.
        X_test: Test features for interval prediction.
        horizon_months: Prediction horizon in months.
        confidence_level: Conformal coverage target (default 0.90).

    Returns:
        Dict with 'pd_lifetime_point', 'pd_lifetime_low', 'pd_lifetime_high'.
    """
    horizon_days = horizon_months * 30

    # Calibration: compute residuals on usable observations
    y_event = np.asarray(y_cal_event, dtype=bool)
    y_time = np.asarray(y_cal_time, dtype=float)

    # Usable obs: defaulted within horizon OR survived past horizon
    usable = y_event | (y_time >= horizon_days)
    if usable.sum() < 20:
        logger.warning(
            f"Only {usable.sum()} usable calibration obs for lifetime PD conformal; "
            "returning point estimates without intervals."
        )
        try:
            surv_funcs = survival_model.predict_survival_function(X_test)
            point = np.clip(np.array([1.0 - fn(horizon_days) for fn in surv_funcs]), 0, 1)
        except Exception:
            point = np.full(len(X_test), np.nan)
        return {"pd_lifetime_point": point, "pd_lifetime_low": point, "pd_lifetime_high": point}

    X_cal_use = X_cal.iloc[np.where(usable)[0]].reset_index(drop=True)
    y_actual = np.where(y_event[usable] & (y_time[usable] <= horizon_days), 1.0, 0.0)

    # Predict lifetime PD on calibration set
    try:
        cal_surv = survival_model.predict_survival_function(X_cal_use)
        cal_pred = np.clip(np.array([1.0 - fn(horizon_days) for fn in cal_surv]), 0, 1)
    except Exception:
        logger.warning("Survival prediction failed on calibration set; no intervals.")
        point = np.full(len(X_test), np.nan)
        return {"pd_lifetime_point": point, "pd_lifetime_low": point, "pd_lifetime_high": point}

    # Conformity scores (absolute residuals)
    scores = np.abs(y_actual - cal_pred)

    # Conformal quantile
    alpha = 1.0 - confidence_level
    n_cal = len(scores)
    q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    q_hat = float(np.quantile(scores, q_level))

    # Test predictions
    test_surv = survival_model.predict_survival_function(X_test)
    test_pred = np.clip(np.array([1.0 - fn(horizon_days) for fn in test_surv]), 0, 1)

    pd_low = np.clip(test_pred - q_hat, 0, 1)
    pd_high = np.clip(test_pred + q_hat, 0, 1)

    logger.info(
        f"Lifetime PD conformal intervals: n_cal={n_cal}, q_hat={q_hat:.4f}, "
        f"mean_width={np.mean(pd_high - pd_low):.4f}, "
        f"mean_point={test_pred.mean():.4f}"
    )

    return {
        "pd_lifetime_point": test_pred,
        "pd_lifetime_low": pd_low,
        "pd_lifetime_high": pd_high,
    }


def fit_kaplan_meier(
    df: pd.DataFrame,
    duration_col: str = "time_to_event",
    event_col: str = "event_observed",
    group_col: str | None = "grade",
) -> dict[str, object]:
    """Fit Kaplan-Meier estimators, optionally stratified by group.

    Returns:
        Dict with 'overall' KaplanMeierFitter and optional 'by_group' dict,
        plus 'logrank_results' if stratified.
    """
    from lifelines import KaplanMeierFitter

    output: dict[str, object] = {}

    # Overall KM
    kmf = KaplanMeierFitter()
    kmf.fit(df[duration_col], event_observed=df[event_col], label="Overall")
    output["overall"] = kmf
    logger.info(
        "KM overall: median survival={:.1f}, n={}",
        float(kmf.median_survival_time_),
        len(df),
    )

    # Stratified KM + log-rank tests
    if group_col and group_col in df.columns:
        by_group: dict[str, object] = {}
        groups = sorted(df[group_col].dropna().unique())
        for grp in groups:
            mask = df[group_col] == grp
            if mask.sum() < 20:
                continue
            grp_kmf = KaplanMeierFitter()
            grp_kmf.fit(
                df.loc[mask, duration_col],
                event_observed=df.loc[mask, event_col],
                label=str(grp),
            )
            by_group[str(grp)] = grp_kmf
        output["by_group"] = by_group

        # Pairwise log-rank tests between adjacent groups
        try:
            from lifelines.statistics import logrank_test

            logrank_results: list[dict[str, object]] = []
            for i in range(len(groups) - 1):
                g1, g2 = groups[i], groups[i + 1]
                m1 = df[group_col] == g1
                m2 = df[group_col] == g2
                if m1.sum() < 20 or m2.sum() < 20:
                    continue
                result = logrank_test(
                    df.loc[m1, duration_col],
                    df.loc[m2, duration_col],
                    event_observed_A=df.loc[m1, event_col],
                    event_observed_B=df.loc[m2, event_col],
                )
                logrank_results.append(
                    {
                        "group_a": str(g1),
                        "group_b": str(g2),
                        "test_statistic": float(result.test_statistic),
                        "p_value": float(result.p_value),
                        "significant_005": bool(result.p_value < 0.05),
                    }
                )
            output["logrank_results"] = logrank_results
            logger.info("Log-rank tests: {} pairwise comparisons", len(logrank_results))
        except ImportError:
            logger.warning("lifelines.statistics not available for log-rank tests")

    return output


def export_km_curves(
    km_output: dict[str, object],
    output_dir: str = "reports/figures/survival",
) -> list[str]:
    """Export Kaplan-Meier survival curves as CSV for downstream plotting.

    Args:
        km_output: Result from fit_kaplan_meier().
        output_dir: Directory for output files.

    Returns:
        List of exported file paths.
    """
    from pathlib import Path as _Path

    out = _Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    overall = km_output.get("overall")
    if overall is not None and hasattr(overall, "survival_function_"):
        path = out / "km_overall.csv"
        overall.survival_function_.to_csv(str(path))
        saved.append(str(path))

    by_group = km_output.get("by_group", {})
    if isinstance(by_group, dict):
        frames = []
        for grp_name, grp_kmf in by_group.items():
            if hasattr(grp_kmf, "survival_function_"):
                sf = grp_kmf.survival_function_.copy()
                sf.columns = [grp_name]
                frames.append(sf)
        if frames:
            combined = pd.concat(frames, axis=1)
            path = out / "km_by_grade.csv"
            combined.to_csv(str(path))
            saved.append(str(path))

    logrank = km_output.get("logrank_results")
    if logrank:
        path = out / "logrank_pairwise.csv"
        pd.DataFrame(logrank).to_csv(str(path), index=False)
        saved.append(str(path))

    logger.info("Exported {} KM curve files to {}", len(saved), out)
    return saved


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
