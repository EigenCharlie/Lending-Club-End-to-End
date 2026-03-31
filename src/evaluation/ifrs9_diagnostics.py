"""ADSFCR-inspired IFRS9 diagnostics for temporal stability and stress coherence."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller
except Exception:  # pragma: no cover - defensive fallback for stripped envs
    adfuller = None


def _safe_numeric_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def _ols_fit(
    frame: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: list[str],
    weight_col: str | None = None,
) -> tuple[float, dict[str, float]]:
    selected_cols = [target_col, *feature_cols, *([weight_col] if weight_col else [])]
    data = frame.loc[:, selected_cols].copy()
    data = data.loc[:, ~data.columns.duplicated()]
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) <= len(feature_cols) + 2:
        raise ValueError("Insufficient rows for recursive regression fit.")

    X = data[feature_cols].astype(float)
    y = data[target_col].astype(float).to_numpy(dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0).replace(0.0, 1.0)
    X_std = ((X - means) / stds).to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X_std)), X_std])

    if weight_col:
        weights = np.sqrt(
            np.clip(data[weight_col].astype(float).to_numpy(dtype=float), 1e-12, None)
        )
        X_design = X_design * weights[:, None]
        y = y * weights

    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    intercept = float(beta[0])
    coefs = {feature: float(beta[idx + 1]) for idx, feature in enumerate(feature_cols)}
    return intercept, coefs


def recursive_regression_paths(
    frame: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    feature_cols: Iterable[str],
    min_window: int = 36,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Fit expanding-window regressions and return coefficient paths."""
    feature_list = [str(col) for col in feature_cols]
    ordered = frame[
        [time_col, target_col, *feature_list, *([weight_col] if weight_col else [])]
    ].copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col], errors="coerce")
    ordered = ordered.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    rows: list[dict[str, float | int | str]] = []
    if len(ordered) < max(min_window, len(feature_list) + 5):
        return pd.DataFrame(
            columns=["window_end", "n_obs", "feature", "coefficient", "intercept", "window_end_ts"]
        )

    for end_idx in range(min_window, len(ordered) + 1):
        window = ordered.iloc[:end_idx].copy()
        intercept, coefs = _ols_fit(
            window,
            target_col=target_col,
            feature_cols=feature_list,
            weight_col=weight_col,
        )
        end_ts = window[time_col].iloc[-1]
        for feature, coef in coefs.items():
            rows.append(
                {
                    "window_end": int(end_idx),
                    "n_obs": int(len(window)),
                    "feature": feature,
                    "coefficient": float(coef),
                    "intercept": intercept,
                    "window_end_ts": pd.Timestamp(end_ts),
                }
            )
    return pd.DataFrame(rows)


def recursive_regression_status(
    path_df: pd.DataFrame, *, min_sign_match_share: float = 0.8
) -> dict:
    """Summarize recursive regression stability."""
    if path_df.empty:
        return {
            "overall_pass": False,
            "n_features": 0,
            "min_sign_match_share": 0.0,
            "max_sign_flips": 0,
            "max_relative_range": float("inf"),
        }

    rows: list[dict[str, float | str | int | bool]] = []
    for feature, grp in path_df.groupby("feature", observed=True):
        coef = grp["coefficient"].astype(float).to_numpy(dtype=float)
        full_coef = float(coef[-1])
        full_sign = 0 if np.isclose(full_coef, 0.0) else int(np.sign(full_coef))
        signs = np.sign(coef)
        sign_match_share = 1.0 if full_sign == 0 else float(np.mean(signs == full_sign))
        sign_flips = int(np.sum(np.diff(np.signbit(coef)) != 0))
        rel_range = float((np.max(coef) - np.min(coef)) / max(abs(full_coef), 1e-6))
        rows.append(
            {
                "feature": str(feature),
                "full_sample_coef": full_coef,
                "sign_match_share": sign_match_share,
                "sign_flips": sign_flips,
                "relative_range_vs_final": rel_range,
                "overall_pass": bool(sign_match_share >= min_sign_match_share and sign_flips <= 2),
            }
        )

    detail = pd.DataFrame(rows).sort_values(
        ["overall_pass", "sign_match_share", "relative_range_vs_final"],
        ascending=[True, True, False],
    )
    return {
        "overall_pass": bool(detail["overall_pass"].all()),
        "n_features": int(len(detail)),
        "min_sign_match_share": float(detail["sign_match_share"].min()),
        "max_sign_flips": int(detail["sign_flips"].max()),
        "max_relative_range": float(detail["relative_range_vs_final"].max()),
        "detail": detail.reset_index(drop=True),
    }


def adf_power_diagnostic(
    series: pd.Series,
    *,
    n_simulations: int = 200,
    alpha: float = 0.05,
    candidate_phis: Iterable[float] = (0.8, 0.9, 0.95, 0.98),
    random_state: int = 42,
) -> dict[str, float | int | bool | dict[str, float] | None]:
    """Estimate ADF power at the observed sample length using AR(1) simulations."""
    values = _safe_numeric_series(series).to_numpy(dtype=float)
    if len(values) < 24 or adfuller is None:
        return {
            "available": False,
            "n_obs": int(len(values)),
            "adf_pvalue_level": None,
            "adf_pvalue_diff1": None,
            "power_by_phi": {},
            "near_unit_root_power": None,
        }

    centered = values - float(np.mean(values))
    diff1 = np.diff(centered)
    try:
        adf_pvalue = float(adfuller(centered, regression="c", autolag="AIC")[1])
    except Exception:
        adf_pvalue = float("nan")
    try:
        adf_diff_pvalue = (
            float(adfuller(diff1, regression="c", autolag="AIC")[1])
            if len(diff1) >= 12
            else float("nan")
        )
    except Exception:
        adf_diff_pvalue = float("nan")

    rng = np.random.default_rng(random_state)
    power_by_phi: dict[str, float] = {}
    n_obs = len(centered)
    burn_in = 100
    for phi in candidate_phis:
        rejections = 0
        for _ in range(int(n_simulations)):
            eps = rng.normal(0.0, 1.0, size=n_obs + burn_in)
            sim = np.zeros(n_obs + burn_in, dtype=float)
            for idx in range(1, len(sim)):
                sim[idx] = float(phi) * sim[idx - 1] + eps[idx]
            test_series = sim[burn_in:]
            try:
                p_value = float(adfuller(test_series, regression="c", autolag="AIC")[1])
            except Exception:
                p_value = 1.0
            rejections += int(p_value < alpha)
        power_by_phi[f"{float(phi):.2f}"] = float(rejections / max(int(n_simulations), 1))

    near_unit_root_power = power_by_phi.get("0.95")
    return {
        "available": True,
        "n_obs": int(n_obs),
        "adf_pvalue_level": None if np.isnan(adf_pvalue) else adf_pvalue,
        "adf_pvalue_diff1": None if np.isnan(adf_diff_pvalue) else adf_diff_pvalue,
        "power_by_phi": power_by_phi,
        "near_unit_root_power": near_unit_root_power,
        "adequate_near_unit_root_power": bool(
            near_unit_root_power is not None and float(near_unit_root_power) >= 0.50
        ),
    }


def scenario_sign_coherence(
    summary_df: pd.DataFrame,
    *,
    scenario_order: Iterable[str] = ("baseline", "mild_stress", "adverse", "severe"),
    monotone_metrics: Iterable[str] = (
        "pd_mult",
        "stage2_share",
        "stage3_share",
        "total_ecl",
        "total_ecl_high",
    ),
) -> pd.DataFrame:
    """Check that stress scenarios preserve expected ordering."""
    if summary_df.empty:
        return pd.DataFrame(columns=["metric", "overall_pass", "values"])

    order_lookup = {name: idx for idx, name in enumerate(scenario_order)}
    ordered = summary_df[summary_df["scenario"].isin(order_lookup)].copy()
    ordered["scenario_order"] = ordered["scenario"].map(order_lookup)
    ordered = ordered.sort_values("scenario_order")

    rows = []
    for metric in monotone_metrics:
        if metric not in ordered.columns:
            continue
        values = pd.to_numeric(ordered[metric], errors="coerce").to_numpy(dtype=float)
        passed = False if np.isnan(values).any() else bool(np.all(np.diff(values) >= -1e-12))
        rows.append(
            {
                "metric": str(metric),
                "overall_pass": passed,
                "values": [float(v) for v in values if np.isfinite(v)],
            }
        )
    return pd.DataFrame(rows)


def scenario_interval_uncertainty(scenarios_df: pd.DataFrame) -> dict[str, float | bool]:
    """Summarize horizon uncertainty from IFRS9 scenario forecasts."""
    if scenarios_df.empty:
        return {
            "available": False,
            "mean_width_90": 0.0,
            "mean_width_95": 0.0,
            "mean_relative_width_90": 0.0,
            "width_90_non_decreasing_over_horizon": False,
        }

    frame = scenarios_df.copy()
    width_90 = pd.to_numeric(frame["adverse_90"], errors="coerce") - pd.to_numeric(
        frame["optimistic_90"], errors="coerce"
    )
    width_95 = pd.to_numeric(frame["adverse_95"], errors="coerce") - pd.to_numeric(
        frame["optimistic_95"], errors="coerce"
    )
    point = pd.to_numeric(frame["point_forecast"], errors="coerce").abs().clip(lower=1e-6)
    rel_width_90 = width_90 / point
    return {
        "available": True,
        "mean_width_90": float(width_90.mean()),
        "mean_width_95": float(width_95.mean()),
        "max_width_90": float(width_90.max()),
        "mean_relative_width_90": float(rel_width_90.mean()),
        "width_90_non_decreasing_over_horizon": bool(
            np.all(np.diff(width_90.to_numpy(dtype=float)) >= -1e-12)
        ),
    }


def sensitivity_surface_summary(grid_df: pd.DataFrame) -> dict[str, float | str | bool]:
    """Rank IFRS9 levers by how strongly they move total ECL in the sensitivity grid."""
    if grid_df.empty:
        return {
            "available": False,
            "dominant_driver": "unknown",
            "pd_mult_slope": 0.0,
            "lgd_mult_slope": 0.0,
            "discount_rate_slope": 0.0,
        }

    data = grid_df[["pd_mult", "lgd_mult", "discount_rate", "total_ecl"]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 12:
        return {
            "available": False,
            "dominant_driver": "unknown",
            "pd_mult_slope": 0.0,
            "lgd_mult_slope": 0.0,
            "discount_rate_slope": 0.0,
        }

    X = data[["pd_mult", "lgd_mult", "discount_rate"]].astype(float)
    y = data["total_ecl"].astype(float).to_numpy(dtype=float)
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0).replace(0.0, 1.0)
    beta, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(len(X_std)), X_std.to_numpy(dtype=float)]),
        y,
        rcond=None,
    )
    slopes = {
        "pd_mult_slope": float(beta[1]),
        "lgd_mult_slope": float(beta[2]),
        "discount_rate_slope": float(beta[3]),
    }
    dominant_driver = max(slopes, key=lambda key: abs(slopes[key])).replace("_slope", "")
    return {
        "available": True,
        "dominant_driver": dominant_driver,
        **slopes,
    }
