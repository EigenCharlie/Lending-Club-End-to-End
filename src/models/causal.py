"""Causal modeling helpers for the pricing-intervention research lane."""

from __future__ import annotations

import re
from collections.abc import Iterable
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import networkx.algorithms as _nxa
import numpy as np
import pandas as pd
import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version

if not hasattr(_nxa, "d_separated"):
    from networkx.algorithms.d_separation import is_d_separator as _is_d_sep

    _nxa.d_separated = lambda G, x, y, z: _is_d_sep(G, x, y, z)


DEFAULT_CAUSAL_CONFIG_PATH = Path("configs/causal_lane.yaml")
CAUSAL_ENV_COMPATIBILITY: dict[str, str] = {
    "dowhy": ">=0.14,<0.15",
    "econml": ">=0.16,<0.17",
    "statsmodels": ">=0.14,<0.15",
    "scikit-learn": ">=1.0,<1.7",
    "shap": ">=0.38.1,<0.49",
}


def _require_dowhy() -> None:
    try:
        import dowhy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "DoWhy is optional in the main environment. Use the dedicated causal env "
            "(`.venv-causal`) created via `bash scripts/causal/setup_causal_env.sh .venv-causal`."
        ) from exc


def _require_econml() -> None:
    try:
        import econml  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "EconML is optional in the main environment. Create a dedicated causal env (e.g., "
            "`.venv-causal`) with the project stack first, then overlay EconML."
        ) from exc


def _package_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def load_causal_config(config_path: str | Path = DEFAULT_CAUSAL_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists() and not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.exists():
        raise FileNotFoundError(f"Causal config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Causal config must be a mapping: {path}")
    return payload


def inspect_causal_environment() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    compatible = True
    for package, spec in CAUSAL_ENV_COMPATIBILITY.items():
        installed = _package_version(package)
        package_ok = False
        if installed is not None:
            try:
                package_ok = Version(installed) in SpecifierSet(spec)
            except Exception:
                package_ok = False
        packages[package] = {
            "installed": installed,
            "expected": spec,
            "compatible": package_ok,
        }
        compatible = compatible and package_ok
    return {
        "environment": "causal_lane",
        "compatible": compatible,
        "packages": packages,
    }


def validate_causal_environment(*, raise_on_incompatible: bool = True) -> dict[str, Any]:
    payload = inspect_causal_environment()
    if raise_on_incompatible and not payload.get("compatible", False):
        incompatible = [
            f"{name}={meta.get('installed')} expected {meta.get('expected')}"
            for name, meta in payload.get("packages", {}).items()
            if not meta.get("compatible", False)
        ]
        raise RuntimeError(
            "Causal environment is incompatible with the official EconML lane: "
            + ", ".join(incompatible)
        )
    return payload


def specify_causal_graph(
    treatment: str = "int_rate",
    outcome: str = "default_flag",
) -> str:
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
    return ["loan_amnt", "annual_inc", "dti", "fico_range_low"]


def default_confounders() -> list[str]:
    return ["grade_woe", "purpose_woe", "home_ownership_woe"]


def required_causal_columns(
    *,
    treatment: str = "int_rate",
    outcome: str = "default_flag",
) -> list[str]:
    return [treatment, outcome, *default_effect_modifiers(), *default_confounders()]


def sanitize_causal_dataframe(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariate_columns: list[str],
    max_covariate_missing_rate: float = 0.05,
    max_row_drop_rate: float = 0.02,
    impute_covariates: str = "median",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = df.copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)

    required = [treatment, outcome, *covariate_columns]
    missing_cols = [col for col in required if col not in frame.columns]
    if missing_cols:
        raise ValueError(
            "Missing required causal columns for the official DAG/contract: "
            + ", ".join(sorted(missing_cols))
        )

    numeric_cols = list(dict.fromkeys(required))
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    n_rows_input = int(len(frame))
    covariate_missing = {
        col: float(frame[col].isna().mean()) for col in covariate_columns if col in frame.columns
    }
    high_missing = {
        col: rate
        for col, rate in covariate_missing.items()
        if rate > float(max_covariate_missing_rate)
    }
    if high_missing:
        formatted = ", ".join(f"{col}={rate:.4f}" for col, rate in sorted(high_missing.items()))
        raise ValueError(
            "Causal covariates exceeded max missing rate before sanitization: " + formatted
        )

    before_drop = len(frame)
    frame = frame.dropna(subset=[treatment, outcome]).copy()
    n_rows_dropped_nonfinite = int(before_drop - len(frame))
    drop_rate = float(n_rows_dropped_nonfinite / max(n_rows_input, 1))
    if drop_rate > float(max_row_drop_rate):
        raise ValueError(
            f"Causal row drop rate {drop_rate:.4%} exceeded max_row_drop_rate={max_row_drop_rate:.4%}"
        )

    imputation_values: dict[str, float] = {}
    n_imputed_cells = 0
    if str(impute_covariates).lower() == "median":
        for col in covariate_columns:
            missing_mask = frame[col].isna()
            if not missing_mask.any():
                continue
            median = float(frame[col].median())
            if np.isnan(median):
                raise ValueError(f"Causal covariate {col} is entirely missing after sanitization.")
            frame.loc[missing_mask, col] = median
            imputation_values[col] = median
            n_imputed_cells += int(missing_mask.sum())
    elif str(impute_covariates).lower() not in {"none", "drop"}:
        raise ValueError(f"Unsupported causal imputation strategy: {impute_covariates}")

    remaining_missing = [col for col in required if frame[col].isna().any()]
    if remaining_missing:
        raise ValueError(
            "Causal sanitization left missing values in required columns: "
            + ", ".join(sorted(remaining_missing))
        )

    stats = {
        "n_rows_input": n_rows_input,
        "n_rows_after_sanitization": int(len(frame)),
        "n_rows_dropped_nonfinite": n_rows_dropped_nonfinite,
        "drop_rate": drop_rate,
        "n_imputed_cells": int(n_imputed_cells),
        "imputation_strategy": str(impute_covariates),
        "imputation_values": imputation_values,
        "covariate_missing_rate": covariate_missing,
    }
    return frame.reset_index(drop=True), stats


def build_overlap_diagnostics(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    segment_columns: Iterable[str] | None = None,
    min_segment_size: int = 50,
) -> pd.DataFrame:
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


def evaluate_overlap_status(
    overlap: pd.DataFrame,
    *,
    min_support_ok_share: float = 0.80,
) -> dict[str, Any]:
    if overlap.empty:
        return {
            "overlap_pass": False,
            "support_ok_share": 0.0,
            "failing_segments": [],
        }
    support_ok_share = float(overlap["support_ok"].mean())
    failing = overlap.loc[~overlap["support_ok"], ["segment_type", "segment_value", "n_obs"]]
    return {
        "overlap_pass": bool(support_ok_share >= float(min_support_ok_share)),
        "support_ok_share": support_ok_share,
        "failing_segments": failing.to_dict(orient="records"),
    }


def _extract_ate_ci(estimate: Any) -> list[float | None]:
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
    p_value = getattr(refutation, "p_value", None)
    try:
        p_value = float(p_value) if p_value is not None else None
    except Exception:
        p_value = None
    result_text = str(refutation)
    if p_value is None:
        match = re.search(
            r"p\s*value\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            result_text,
            flags=re.IGNORECASE,
        )
        if match:
            try:
                p_value = float(match.group(1))
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
        "result": result_text,
    }


def estimate_ate_dowhy(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: list[str],
    graph: str | None = None,
) -> dict[str, Any]:
    _require_dowhy()
    import dowhy

    model = dowhy.CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes,
        graph=graph,
    )

    identified = model.identify_effect(proceed_when_unidentifiable=False)
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
    return {
        "ate": ate_value,
        "ate_ci": ate_ci,
        "estimate_object": estimate,
        "identified_estimand": identified,
        "identification_strategy": "backdoor",
        "model": model,
        "refutation_summary": refutations,
    }


def _nuisance_regressor(random_state: int = 42):
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=random_state)


def estimate_ate_linear_dml(
    *,
    Y: pd.Series,
    T: pd.Series,
    X: pd.DataFrame,
    W: pd.DataFrame | None = None,
    cv: int = 3,
    mc_iters: int = 1,
    random_state: int = 42,
) -> dict[str, Any]:
    _require_econml()
    from econml.dml import LinearDML

    est = LinearDML(
        model_y=_nuisance_regressor(random_state),
        model_t=_nuisance_regressor(random_state),
        cv=max(2, int(cv)),
        mc_iters=max(1, int(mc_iters)),
        random_state=random_state,
    )
    est.fit(Y=np.asarray(Y, dtype=float), T=np.asarray(T, dtype=float), X=X, W=W)
    cate = np.asarray(est.const_marginal_effect(X), dtype=float).reshape(-1)
    lb, ub = est.const_marginal_effect_interval(X, alpha=0.05)
    return {
        "estimator": est,
        "ate": float(np.mean(cate)),
        "ate_ci": [
            float(np.mean(np.asarray(lb, dtype=float))),
            float(np.mean(np.asarray(ub, dtype=float))),
        ],
        "cate": cate,
        "cate_lb": np.asarray(lb, dtype=float).reshape(-1),
        "cate_ub": np.asarray(ub, dtype=float).reshape(-1),
        "estimator_family": "linear_dml",
    }


def estimate_cate_candidates(
    *,
    Y: pd.Series,
    T: pd.Series,
    X: pd.DataFrame,
    W: pd.DataFrame | None = None,
    candidate_names: list[str],
    random_state: int = 42,
    causal_forest_cfg: dict[str, Any] | None = None,
    linear_dml_cfg: dict[str, Any] | None = None,
    selector: str = "rscorer",
) -> dict[str, Any]:
    _require_econml()
    from econml.dml import CausalForestDML, LinearDML

    successful: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    candidate_names = list(dict.fromkeys(candidate_names))
    causal_forest_cfg = causal_forest_cfg or {}
    linear_dml_cfg = linear_dml_cfg or {}

    for name in candidate_names:
        try:
            if name == "causal_forest_dml":
                est = CausalForestDML(
                    model_y=_nuisance_regressor(random_state),
                    model_t=_nuisance_regressor(random_state),
                    n_estimators=int(causal_forest_cfg.get("n_estimators", 200)),
                    cv=max(2, int(causal_forest_cfg.get("cv", 3))),
                    mc_iters=max(1, int(causal_forest_cfg.get("mc_iters", 1))),
                    criterion=str(causal_forest_cfg.get("criterion", "mse")),
                    min_balancedness_tol=float(causal_forest_cfg.get("min_balancedness_tol", 0.45)),
                    honest=bool(causal_forest_cfg.get("honest", True)),
                    max_samples=float(causal_forest_cfg.get("max_samples", 0.45)),
                    n_jobs=int(causal_forest_cfg.get("n_jobs", -1)),
                    random_state=random_state,
                )
            elif name == "linear_dml":
                est = LinearDML(
                    model_y=_nuisance_regressor(random_state),
                    model_t=_nuisance_regressor(random_state),
                    cv=max(2, int(linear_dml_cfg.get("cv", 3))),
                    mc_iters=max(1, int(linear_dml_cfg.get("mc_iters", 1))),
                    random_state=random_state,
                )
            else:
                raise ValueError(f"Unsupported causal estimator candidate: {name}")

            est.fit(Y=np.asarray(Y, dtype=float), T=np.asarray(T, dtype=float), X=X, W=W)
            cate = np.asarray(est.const_marginal_effect(X), dtype=float).reshape(-1)
            lb, ub = est.const_marginal_effect_interval(X, alpha=0.05)
            successful[name] = {
                "estimator": est,
                "cate": cate,
                "cate_lb": np.asarray(lb, dtype=float).reshape(-1),
                "cate_ub": np.asarray(ub, dtype=float).reshape(-1),
                "cate_mean": float(np.mean(cate)),
                "cate_std": float(np.std(cate)),
                "estimator_family": name,
                "selection_score": None,
            }
        except Exception as exc:
            failures[name] = str(exc)

    if not successful:
        raise RuntimeError(
            "No causal estimator candidate fitted successfully: "
            + "; ".join(f"{name}: {err}" for name, err in failures.items())
        )

    selection_reason = "first_successful"
    selected_name = next(iter(successful))
    if selector == "rscorer" and len(successful) > 1:
        try:
            from econml.score import RScorer

            scorer = RScorer(
                model_y=_nuisance_regressor(random_state),
                model_t=_nuisance_regressor(random_state),
                cv=max(
                    2,
                    int(
                        max(
                            causal_forest_cfg.get("cv", 3),
                            linear_dml_cfg.get("cv", 3),
                        )
                    ),
                ),
                random_state=random_state,
            )
            scorer.fit(np.asarray(Y, dtype=float), np.asarray(T, dtype=float), X=X, W=W)
            best_score = None
            for name, payload in successful.items():
                score = float(scorer.score(payload["estimator"]))
                payload["selection_score"] = score
                if best_score is None or score > best_score:
                    best_score = score
                    selected_name = name
            selection_reason = "rscorer"
        except Exception as exc:
            selection_reason = f"rscorer_unavailable: {exc}"

    return {
        "selected_name": selected_name,
        "selected": successful[selected_name],
        "candidates": successful,
        "failures": failures,
        "selection_reason": selection_reason,
    }


def build_sensitivity_status(
    estimator: Any,
    *,
    min_robustness_value: float = 0.05,
    alpha: float = 0.05,
    c_y: float = 0.05,
    c_t: float = 0.05,
    rho: float = 1.0,
) -> dict[str, Any]:
    payload = {
        "sensitivity_supported": False,
        "sensitivity_pass": False,
        "robustness_value": None,
        "sensitivity_interval": [None, None],
        "sensitivity_summary": None,
    }
    if estimator is None or not hasattr(estimator, "robustness_value"):
        return payload
    try:
        robustness = float(estimator.robustness_value(alpha=alpha))
    except Exception:
        robustness = None
    try:
        interval = estimator.sensitivity_interval(alpha=alpha, c_y=c_y, c_t=c_t, rho=rho)
        interval_arr = np.asarray(interval, dtype=float).reshape(-1)
        sensitivity_interval = [
            float(interval_arr[0]) if interval_arr.size > 0 else None,
            float(interval_arr[1]) if interval_arr.size > 1 else None,
        ]
    except Exception:
        sensitivity_interval = [None, None]
    try:
        summary = str(
            estimator.sensitivity_summary(alpha=alpha, c_y=c_y, c_t=c_t, rho=rho, decimals=3)
        )
    except Exception:
        summary = None
    payload.update(
        {
            "sensitivity_supported": True,
            "sensitivity_pass": bool(
                robustness is not None and robustness >= float(min_robustness_value)
            ),
            "robustness_value": robustness,
            "sensitivity_interval": sensitivity_interval,
            "sensitivity_summary": summary,
        }
    )
    return payload


def run_refutation_tests(
    model,
    identified_estimand,
    estimate,
    n_tests: int = 3,
) -> list[dict[str, Any]]:
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
    return refutations
