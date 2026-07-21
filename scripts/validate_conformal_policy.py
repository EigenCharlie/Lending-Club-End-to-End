"""Validate conformal artifacts against explicit acceptance policy.

Usage:
    uv run python scripts/validate_conformal_policy.py --config configs/conformal_policy.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

import src.evaluation.backtesting as _bt

try:
    from mapie.metrics.regression import regression_mwi_score as _mapie_mwi_score

    _MAPIE_MWI_AVAILABLE = True
except ImportError:
    _MAPIE_MWI_AVAILABLE = False
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag
from src.utils.baseline_registry import resolve_official_baseline_run_tag


def _fallback_interval_violations(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return (np.logical_or(y_true < lower, y_true > upper)).astype(int)


def _fallback_kupiec_pof_test(
    violations: np.ndarray,
    *,
    nominal_alpha: float | None = None,
    alpha: float | None = None,
) -> dict[str, float | bool]:
    violations = np.asarray(violations, dtype=float)
    n = int(violations.size)
    n_fail = int(np.nansum(violations))
    fail_rate = float(n_fail / n) if n > 0 else float("nan")
    alpha_value = (
        float(alpha)
        if alpha is not None
        else float(nominal_alpha)
        if nominal_alpha is not None
        else float("nan")
    )
    return {
        "lr_pof": float("nan"),
        "p_value": float("nan"),
        "reject_h0": False,
        "n": n,
        "n_fail": n_fail,
        "fail_rate": fail_rate,
        "nominal_alpha": alpha_value,
    }


def _fallback_christoffersen_test(
    _violations: np.ndarray,
    *,
    alpha: float | None = None,
) -> dict[str, float | bool]:
    return {
        "lr_uc": float("nan"),
        "p_uc": float("nan"),
        "lr_ind": float("nan"),
        "p_ind": float("nan"),
        "lr_cc": float("nan"),
        "p_cc": float("nan"),
        "reject_cc": False,
    }


def _fallback_winkler_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = np.maximum(upper - lower, 0.0)
    below = np.maximum(lower - y_true, 0.0)
    above = np.maximum(y_true - upper, 0.0)
    penalty = (2.0 / max(float(alpha), 1e-12)) * (below + above)
    return width + penalty


interval_violations = getattr(_bt, "interval_violations", _fallback_interval_violations)
kupiec_pof_test = getattr(_bt, "kupiec_pof_test", _fallback_kupiec_pof_test)
christoffersen_test = getattr(_bt, "christoffersen_test", _fallback_christoffersen_test)
winkler_interval_score = getattr(_bt, "winkler_interval_score", _fallback_winkler_interval_score)

SCHEMA_VERSION = "2026-03-01.1"


def _check(
    metric_name: str, value: float, threshold: float, comparator: str, scope: str
) -> dict[str, object]:
    if comparator == ">=":
        passed = value >= threshold
    elif comparator == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"Unsupported comparator: {comparator}")
    return {
        "scope": scope,
        "metric": metric_name,
        "value": float(value),
        "threshold": float(threshold),
        "comparator": comparator,
        "passed": bool(passed),
    }


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_artifact_namespace(
    cfg: dict[str, object],
    artifact_namespace: str | None,
) -> dict[str, object]:
    if not artifact_namespace:
        return cfg
    ns = str(artifact_namespace).strip().replace("/", "_")
    data_dir = Path("data/processed/conformal_gap") / ns
    models_dir = Path("models/conformal_gap") / ns
    updated = dict(cfg)
    updated["artifacts"] = dict(updated.get("artifacts", {}) or {})
    updated["output"] = dict(updated.get("output", {}) or {})
    updated["artifacts"]["conformal_results_path"] = str(
        models_dir / "conformal_results_mondrian.pkl"
    )
    updated["artifacts"]["group_metrics_path"] = str(
        data_dir / "conformal_group_metrics_mondrian.parquet"
    )
    updated["artifacts"]["backtest_monthly_path"] = str(
        data_dir / "conformal_backtest_monthly.parquet"
    )
    updated["artifacts"]["backtest_alerts_path"] = str(
        data_dir / "conformal_backtest_alerts.parquet"
    )
    updated["artifacts"]["intervals_path"] = str(data_dir / "conformal_intervals_mondrian.parquet")
    updated["output"]["policy_status_json"] = str(models_dir / "conformal_policy_status.json")
    updated["output"]["policy_checks_parquet"] = str(data_dir / "conformal_policy_checks.parquet")
    return updated


def main(
    config_path: str = "configs/conformal_policy.yaml",
    run_tag: str | None = None,
    sensitivity_config_path: str | None = None,
    artifact_namespace: str | None = None,
) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if sensitivity_config_path is not None:
        with open(sensitivity_config_path, encoding="utf-8") as f:
            sens_cfg = yaml.safe_load(f)
        if "policy_sensitivity" in sens_cfg:
            cfg["policy_sensitivity"] = sens_cfg["policy_sensitivity"]
            logger.info(
                f"Overriding policy_sensitivity from {sensitivity_config_path}: "
                f"{cfg['policy_sensitivity']}"
            )

    cfg = _apply_artifact_namespace(cfg, artifact_namespace)

    policy = cfg["policy"]
    artifacts = cfg["artifacts"]
    output = cfg["output"]
    resolved_run_tag = resolve_run_tag(
        run_tag,
        fallback_candidates=[resolve_official_baseline_run_tag()],
        require_explicit=True,
    )

    with open(artifacts["conformal_results_path"], "rb") as f:
        results = pickle.load(f)
    group_metrics = pd.read_parquet(artifacts["group_metrics_path"])
    backtest_monthly = pd.read_parquet(artifacts["backtest_monthly_path"])
    alerts_path = Path(artifacts["backtest_alerts_path"])
    alerts = (
        pd.read_parquet(alerts_path) if alerts_path.exists() else pd.DataFrame(columns=["severity"])
    )
    intervals_path = Path(
        artifacts.get("intervals_path", "data/processed/conformal_intervals_mondrian.parquet")
    )
    intervals_df = pd.read_parquet(intervals_path)
    lgd_ead_status_path = Path("models/conformal_lgd_ead_status.json")
    lgd_ead_status = (
        json.loads(lgd_ead_status_path.read_text(encoding="utf-8"))
        if lgd_ead_status_path.exists()
        else {"available": False, "reason": "missing_status_artifact"}
    )

    metrics_90 = results.get("metrics_90", {})
    metrics_95 = results.get("metrics_95", {})

    coverage_90 = float(metrics_90.get("empirical_coverage", 0.0))
    coverage_95 = float(metrics_95.get("empirical_coverage", 0.0))
    avg_width_90 = float(metrics_90.get("avg_interval_width", 999.0))
    min_group_coverage_90 = float(group_metrics.get("coverage_90", pd.Series([0.0])).min())
    critical_alerts = int((alerts.get("severity", pd.Series([], dtype=str)) == "critical").sum())
    warning_alerts = int((alerts.get("severity", pd.Series([], dtype=str)) == "warning").sum())
    total_alerts = int(len(alerts))

    # Conformal quality/statistical checks (v2)
    if {"y_true", "pd_low_90", "pd_high_90"}.issubset(intervals_df.columns):
        y_true = pd.to_numeric(intervals_df["y_true"], errors="coerce").to_numpy(dtype=float)
        low_90 = pd.to_numeric(intervals_df["pd_low_90"], errors="coerce").to_numpy(dtype=float)
        high_90 = pd.to_numeric(intervals_df["pd_high_90"], errors="coerce").to_numpy(dtype=float)
        valid_90 = np.isfinite(y_true) & np.isfinite(low_90) & np.isfinite(high_90)
        y90 = y_true[valid_90]
        lo90 = low_90[valid_90]
        hi90 = high_90[valid_90]
    else:
        y90 = np.array([], dtype=float)
        lo90 = np.array([], dtype=float)
        hi90 = np.array([], dtype=float)

    if {"y_true", "pd_low_95", "pd_high_95"}.issubset(intervals_df.columns):
        y_true_95 = pd.to_numeric(intervals_df["y_true"], errors="coerce").to_numpy(dtype=float)
        low_95 = pd.to_numeric(intervals_df["pd_low_95"], errors="coerce").to_numpy(dtype=float)
        high_95 = pd.to_numeric(intervals_df["pd_high_95"], errors="coerce").to_numpy(dtype=float)
        valid_95 = np.isfinite(y_true_95) & np.isfinite(low_95) & np.isfinite(high_95)
        y95 = y_true_95[valid_95]
        lo95 = low_95[valid_95]
        hi95 = high_95[valid_95]
    else:
        y95 = np.array([], dtype=float)
        lo95 = np.array([], dtype=float)
        hi95 = np.array([], dtype=float)

    winkler_90 = (
        float(np.mean(winkler_interval_score(y90, lo90, hi90, alpha=0.10)))
        if y90.size
        else float("inf")
    )
    winkler_95 = (
        float(np.mean(winkler_interval_score(y95, lo95, hi95, alpha=0.05)))
        if y95.size
        else float("inf")
    )

    # Cross-check: MAPIE native regression_mwi_score (should match manual Winkler)
    mapie_mwi_90: float | None = None
    if _MAPIE_MWI_AVAILABLE and y90.size:
        try:
            mapie_mwi_90 = float(np.mean(_mapie_mwi_score(y90, lo90, hi90, alpha_=0.10)))
            delta = abs(mapie_mwi_90 - winkler_90)
            if delta > 0.01:
                logger.warning(
                    f"MAPIE MWI ({mapie_mwi_90:.4f}) deviates from manual Winkler "
                    f"({winkler_90:.4f}) by {delta:.4f} — check score definition."
                )
            else:
                logger.info(f"MAPIE MWI cross-check OK: {mapie_mwi_90:.4f} ≈ {winkler_90:.4f}")
        except Exception as exc:
            logger.warning(f"MAPIE MWI cross-check failed: {exc}")
    violations_90 = interval_violations(y90, lo90, hi90) if y90.size else np.array([], dtype=float)
    violations_95 = interval_violations(y95, lo95, hi95) if y95.size else np.array([], dtype=float)
    kupiec_90 = kupiec_pof_test(violations_90, alpha=0.10)
    kupiec_95 = kupiec_pof_test(violations_95, alpha=0.05)
    christ_90 = christoffersen_test(violations_90, alpha=0.10)
    christ_95 = christoffersen_test(violations_95, alpha=0.05)

    max_winkler_90 = float(policy.get("max_winkler_90", float("inf")))
    max_winkler_95 = float(policy.get("max_winkler_95", float("inf")))
    enable_compensated_winkler_90 = bool(policy.get("enable_compensated_winkler_90", False))
    compensated_winkler_90_max = float(policy.get("compensated_winkler_90_max", max_winkler_90))
    compensated_min_coverage_90 = float(
        policy.get("compensated_min_coverage_90", policy["target_coverage_90_min"])
    )
    compensated_min_group_coverage_90 = float(
        policy.get(
            "compensated_min_group_coverage_90",
            policy["min_group_coverage_90_min"],
        )
    )
    compensated_max_avg_width_90 = float(
        policy.get("compensated_max_avg_width_90", policy["max_avg_width_90"])
    )
    min_kupiec_p90 = float(policy.get("min_kupiec_pvalue_90", 0.0))
    min_kupiec_p95 = float(policy.get("min_kupiec_pvalue_95", 0.0))
    min_christ_p90 = float(policy.get("min_christoffersen_pvalue_90", 0.0))
    min_christ_p95 = float(policy.get("min_christoffersen_pvalue_95", 0.0))
    allow_methodological_justification = bool(
        policy.get("allow_methodological_justification", False)
    )
    statistical_tests_role = str(policy.get("statistical_tests_role", "strict_blocking"))
    max_cov_dev_90 = float(policy.get("max_coverage_deviation_for_statistical_warning_90", 0.0))
    max_cov_dev_95 = float(policy.get("max_coverage_deviation_for_statistical_warning_95", 0.0))
    min_ind_p90 = float(policy.get("min_christoffersen_independence_pvalue_90", 0.0))
    min_ind_p95 = float(policy.get("min_christoffersen_independence_pvalue_95", 0.0))

    winkler_90_raw_pass = bool(winkler_90 <= max_winkler_90)
    winkler_90_compensated_pass = bool(
        enable_compensated_winkler_90
        and (not winkler_90_raw_pass)
        and winkler_90 <= compensated_winkler_90_max
        and coverage_90 >= compensated_min_coverage_90
        and min_group_coverage_90 >= compensated_min_group_coverage_90
        and avg_width_90 <= compensated_max_avg_width_90
        and critical_alerts <= float(policy["max_critical_alerts"])
    )
    winkler_90_policy_pass = bool(winkler_90_raw_pass or winkler_90_compensated_pass)
    winkler_90_policy_mode = (
        "strict"
        if winkler_90_raw_pass
        else "compensated_band"
        if winkler_90_compensated_pass
        else "strict"
    )
    winkler_90_check = _check("winkler_90", winkler_90, max_winkler_90, "<=", "quality")
    winkler_90_check["passed"] = bool(winkler_90_policy_pass)
    winkler_90_check["policy_mode"] = str(winkler_90_policy_mode)
    winkler_90_check["raw_threshold"] = float(max_winkler_90)
    winkler_90_check["raw_passed"] = bool(winkler_90_raw_pass)
    winkler_90_check["compensated_band_enabled"] = bool(enable_compensated_winkler_90)
    winkler_90_check["compensated_threshold"] = float(compensated_winkler_90_max)
    winkler_90_check["compensated_passed"] = bool(winkler_90_compensated_pass)

    checks = [
        _check(
            "coverage_90", coverage_90, float(policy["target_coverage_90_min"]), ">=", "portfolio"
        ),
        _check(
            "coverage_95", coverage_95, float(policy["target_coverage_95_min"]), ">=", "portfolio"
        ),
        _check(
            "min_group_coverage_90",
            min_group_coverage_90,
            float(policy["min_group_coverage_90_min"]),
            ">=",
            "group",
        ),
        _check("avg_width_90", avg_width_90, float(policy["max_avg_width_90"]), "<=", "portfolio"),
        _check(
            "critical_alerts",
            float(critical_alerts),
            float(policy["max_critical_alerts"]),
            "<=",
            "monitoring",
        ),
        _check(
            "total_alerts",
            float(total_alerts),
            float(policy["max_total_alerts"]),
            "<=",
            "monitoring",
        ),
        _check(
            "warning_alerts",
            float(warning_alerts),
            float(policy["max_warning_alerts"]),
            "<=",
            "monitoring",
        ),
        winkler_90_check,
        _check("winkler_95", winkler_95, max_winkler_95, "<=", "quality"),
        _check(
            "kupiec_pvalue_90",
            float(kupiec_90["p_value"]),
            min_kupiec_p90,
            ">=",
            "statistical_coverage",
        ),
        _check(
            "kupiec_pvalue_95",
            float(kupiec_95["p_value"]),
            min_kupiec_p95,
            ">=",
            "statistical_coverage",
        ),
        _check(
            "christoffersen_pvalue_90",
            float(christ_90["p_cc"]),
            min_christ_p90,
            ">=",
            "statistical_coverage",
        ),
        _check(
            "christoffersen_pvalue_95",
            float(christ_95["p_cc"]),
            min_christ_p95,
            ">=",
            "statistical_coverage",
        ),
    ]
    checks_df = pd.DataFrame(checks)
    statistical_mask = checks_df["scope"].eq("statistical_coverage")
    coverage_deviation_90 = abs(coverage_90 - float(policy["target_coverage_90_min"]))
    coverage_deviation_95 = abs(coverage_95 - float(policy["target_coverage_95_min"]))
    christ_p_ind_90 = _safe_float(christ_90.get("p_ind"))
    christ_p_ind_95 = _safe_float(christ_95.get("p_ind"))
    independence_ok_90 = (not np.isfinite(christ_p_ind_90)) or christ_p_ind_90 >= min_ind_p90
    independence_ok_95 = (not np.isfinite(christ_p_ind_95)) or christ_p_ind_95 >= min_ind_p95
    coverage_materiality_ok = bool(
        coverage_deviation_90 <= max_cov_dev_90 and coverage_deviation_95 <= max_cov_dev_95
    )
    coverage_undercoverage_materiality_ok = bool(
        coverage_90 >= float(policy["target_coverage_90_min"]) - max_cov_dev_90
        and coverage_95 >= float(policy["target_coverage_95_min"]) - max_cov_dev_95
    )
    statistical_tests_diagnostic_only = statistical_tests_role in {
        "diagnostic_informational",
        "diagnostic_warning",
        "strict_diagnostics",
        "warning_only",
        "non_blocking",
    }

    def _evaluate_check_frame(frame: pd.DataFrame) -> dict[str, object]:
        strict = bool(frame["passed"].all())
        non_stat_pass = bool(frame.loc[~statistical_mask, "passed"].all())
        failing_all = frame.loc[~frame["passed"], "metric"].astype(str).tolist()
        failing_stats = (
            frame.loc[statistical_mask & ~frame["passed"], "metric"].astype(str).tolist()
        )
        failing_non_stats = (
            frame.loc[~statistical_mask & ~frame["passed"], "metric"].astype(str).tolist()
        )
        only_stats = bool((not strict) and len(failing_stats) > 0 and len(failing_non_stats) == 0)
        if statistical_tests_diagnostic_only:
            materiality_ok_for_role = coverage_undercoverage_materiality_ok
        else:
            materiality_ok_for_role = coverage_materiality_ok
        methodological_pass = bool(
            allow_methodological_justification
            and only_stats
            and materiality_ok_for_role
            and independence_ok_90
            and independence_ok_95
        )
        if strict:
            methodological_status = "not_needed_strict_pass"
        elif methodological_pass:
            methodological_status = "eligible_statistical_warning_only"
        elif not allow_methodological_justification:
            methodological_status = "disabled"
        elif len(failing_non_stats) > 0:
            methodological_status = "blocked_non_statistical_failures"
        elif not materiality_ok_for_role:
            methodological_status = (
                "blocked_undercoverage_materiality"
                if statistical_tests_diagnostic_only
                else "blocked_materiality"
            )
        else:
            methodological_status = "blocked_statistical_pattern"
        return {
            "strict_overall_pass": strict,
            "non_statistical_checks_pass": non_stat_pass,
            "failing_checks": failing_all,
            "failing_statistical_checks": failing_stats,
            "failing_non_statistical_checks": failing_non_stats,
            "methodological_justification_pass": methodological_pass,
            "methodological_justification_status": methodological_status,
        }

    evaluation = _evaluate_check_frame(checks_df)
    strict_overall_pass = bool(evaluation["strict_overall_pass"])
    non_statistical_checks_pass = bool(evaluation["non_statistical_checks_pass"])
    failing_checks = list(evaluation["failing_checks"])
    failing_statistical_checks = list(evaluation["failing_statistical_checks"])
    failing_non_statistical_checks = list(evaluation["failing_non_statistical_checks"])
    methodological_justification_pass = bool(evaluation["methodological_justification_pass"])
    methodological_status = str(evaluation["methodological_justification_status"])
    overall_pass = strict_overall_pass or methodological_justification_pass

    latest_month = (
        backtest_monthly.sort_values("month").iloc[-1]["month"]
        if not backtest_monthly.empty
        else None
    )

    out_status = {
        "overall_pass": overall_pass,
        "strict_overall_pass": strict_overall_pass,
        "non_statistical_checks_pass": non_statistical_checks_pass,
        "methodological_justification_pass": methodological_justification_pass,
        "methodological_justification_status": methodological_status,
        "statistical_tests_role": statistical_tests_role,
        "checks_passed": int(checks_df["passed"].sum()),
        "checks_total": int(len(checks_df)),
        "failing_checks": failing_checks,
        "failing_statistical_checks": failing_statistical_checks,
        "failing_non_statistical_checks": failing_non_statistical_checks,
        "coverage_90": coverage_90,
        "coverage_95": coverage_95,
        "avg_width_90": avg_width_90,
        "min_group_coverage_90": min_group_coverage_90,
        "critical_alerts": critical_alerts,
        "warning_alerts": warning_alerts,
        "total_alerts": total_alerts,
        "winkler_90": winkler_90,
        "winkler_90_raw_pass": bool(winkler_90_raw_pass),
        "winkler_90_policy_pass": bool(winkler_90_policy_pass),
        "winkler_90_policy_mode": str(winkler_90_policy_mode),
        "winkler_90_compensated_pass": bool(winkler_90_compensated_pass),
        "winkler_90_compensated_threshold": float(compensated_winkler_90_max),
        "winkler_95": winkler_95,
        "mapie_mwi_90": mapie_mwi_90,
        "kupiec_pvalue_90": float(kupiec_90["p_value"]),
        "kupiec_pvalue_95": float(kupiec_95["p_value"]),
        "christoffersen_pvalue_90": float(christ_90["p_cc"]),
        "christoffersen_pvalue_95": float(christ_95["p_cc"]),
        "statistical_tests": {
            "kupiec_90": kupiec_90,
            "kupiec_95": kupiec_95,
            "christoffersen_90": christ_90,
            "christoffersen_95": christ_95,
        },
        "sample_size_context": {
            "n_total_90": int(y90.size),
            "n_total_95": int(y95.size),
            "violation_rate_90": _safe_float(
                kupiec_90.get("violation_rate", kupiec_90.get("fail_rate"))
            ),
            "violation_rate_95": _safe_float(
                kupiec_95.get("violation_rate", kupiec_95.get("fail_rate"))
            ),
            "nominal_alpha_90": _safe_float(kupiec_90.get("nominal_alpha")),
            "nominal_alpha_95": _safe_float(kupiec_95.get("nominal_alpha")),
            "christoffersen_independence_pvalue_90": christ_p_ind_90,
            "christoffersen_independence_pvalue_95": christ_p_ind_95,
        },
        "methodological_justification": {
            "allowed": allow_methodological_justification,
            "only_statistical_failures": bool(
                (not strict_overall_pass)
                and len(failing_statistical_checks) > 0
                and len(failing_non_statistical_checks) == 0
            ),
            "coverage_materiality_ok": coverage_materiality_ok,
            "coverage_undercoverage_materiality_ok": coverage_undercoverage_materiality_ok,
            "coverage_deviation_90": coverage_deviation_90,
            "coverage_deviation_95": coverage_deviation_95,
            "max_coverage_deviation_for_statistical_warning_90": max_cov_dev_90,
            "max_coverage_deviation_for_statistical_warning_95": max_cov_dev_95,
            "independence_ok_90": bool(independence_ok_90),
            "independence_ok_95": bool(independence_ok_95),
            "min_christoffersen_independence_pvalue_90": min_ind_p90,
            "min_christoffersen_independence_pvalue_95": min_ind_p95,
            "winkler_90_policy_mode": str(winkler_90_policy_mode),
            "winkler_90_raw_pass": bool(winkler_90_raw_pass),
            "winkler_90_compensated_pass": bool(winkler_90_compensated_pass),
            "winkler_90_compensated_threshold": float(compensated_winkler_90_max),
            "decision": bool(methodological_justification_pass),
            "justification_role": (
                "diagnostic_warning_not_blocking_for_promotion"
                if statistical_tests_diagnostic_only
                else "strict_blocking"
            ),
        },
        "latest_backtest_month": str(latest_month) if latest_month is not None else None,
        "intervals_path": str(intervals_path),
        "artifact_namespace": artifact_namespace or "",
        "policy_config": config_path,
        "lgd_ead_conformal_status_path": str(lgd_ead_status_path),
        "lgd_ead_conformal_status": lgd_ead_status,
        **build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=resolved_run_tag,
            require_explicit=True,
        ),
    }

    sensitivity_cfg = cfg.get("policy_sensitivity", {}) or {}
    max_winkler_values = [
        float(x) for x in sensitivity_cfg.get("max_winkler_90_values", []) or [] if x is not None
    ]
    if max_winkler_values:
        sensitivity_rows: list[dict[str, object]] = []
        for threshold in max_winkler_values:
            checks_sens = checks_df.copy()
            mask = checks_sens["metric"].eq("winkler_90")
            checks_sens.loc[mask, "threshold"] = float(threshold)
            checks_sens.loc[mask, "passed"] = float(winkler_90) <= float(threshold)
            sens_eval = _evaluate_check_frame(checks_sens)
            sensitivity_rows.append(
                {
                    "max_winkler_90": float(threshold),
                    "strict_overall_pass": bool(sens_eval["strict_overall_pass"]),
                    "non_statistical_checks_pass": bool(sens_eval["non_statistical_checks_pass"]),
                    "methodological_justification_pass": bool(
                        sens_eval["methodological_justification_pass"]
                    ),
                    "failing_non_statistical_checks": list(
                        sens_eval["failing_non_statistical_checks"]
                    ),
                    "failing_statistical_checks": list(sens_eval["failing_statistical_checks"]),
                }
            )
        out_status["policy_sensitivity"] = {
            "metric": "winkler_90",
            "results": sensitivity_rows,
        }

    checks_path = Path(output["policy_checks_parquet"])
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    checks_df.to_parquet(checks_path, index=False)

    status_path = Path(output["policy_status_json"])
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(out_status, f, indent=2)

    logger.info(f"Policy checks saved: {checks_path}")
    logger.info(f"Policy status saved: {status_path}")
    logger.info(
        "Conformal policy strict_pass="
        f"{strict_overall_pass} ({out_status['checks_passed']}/{out_status['checks_total']}); "
        f"methodological_justification_pass={methodological_justification_pass}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/conformal_policy.yaml")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument(
        "--sensitivity-config",
        default=None,
        help="Optional YAML with policy_sensitivity overrides (e.g. configs/conformal_policy_sensitivity.yaml)",
    )
    parser.add_argument("--artifact-namespace", default=None)
    args = parser.parse_args()
    main(
        args.config,
        run_tag=args.run_tag,
        sensitivity_config_path=args.sensitivity_config,
        artifact_namespace=args.artifact_namespace,
    )
