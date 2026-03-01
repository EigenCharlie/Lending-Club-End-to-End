"""Validate conformal artifacts against explicit acceptance policy.

Usage:
    uv run python scripts/validate_conformal_policy.py --config configs/conformal_policy.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

import src.evaluation.backtesting as _bt


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


def main(config_path: str = "configs/conformal_policy.yaml", run_tag: str | None = None):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    policy = cfg["policy"]
    artifacts = cfg["artifacts"]
    output = cfg["output"]
    resolved_run_tag = str(run_tag or "").strip() or str(os.environ.get("PIPELINE_RUN_TAG", "")).strip()
    if not resolved_run_tag:
        resolved_run_tag = "untracked"

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
    violations_90 = interval_violations(y90, lo90, hi90) if y90.size else np.array([], dtype=float)
    violations_95 = interval_violations(y95, lo95, hi95) if y95.size else np.array([], dtype=float)
    kupiec_90 = kupiec_pof_test(violations_90, alpha=0.10)
    kupiec_95 = kupiec_pof_test(violations_95, alpha=0.05)
    christ_90 = christoffersen_test(violations_90, alpha=0.10)
    christ_95 = christoffersen_test(violations_95, alpha=0.05)

    max_winkler_90 = float(policy.get("max_winkler_90", float("inf")))
    max_winkler_95 = float(policy.get("max_winkler_95", float("inf")))
    min_kupiec_p90 = float(policy.get("min_kupiec_pvalue_90", 0.0))
    min_kupiec_p95 = float(policy.get("min_kupiec_pvalue_95", 0.0))
    min_christ_p90 = float(policy.get("min_christoffersen_pvalue_90", 0.0))
    min_christ_p95 = float(policy.get("min_christoffersen_pvalue_95", 0.0))

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
        _check("winkler_90", winkler_90, max_winkler_90, "<=", "quality"),
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
    overall_pass = bool(checks_df["passed"].all())

    latest_month = (
        backtest_monthly.sort_values("month").iloc[-1]["month"]
        if not backtest_monthly.empty
        else None
    )

    out_status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_tag": resolved_run_tag,
        "overall_pass": overall_pass,
        "checks_passed": int(checks_df["passed"].sum()),
        "checks_total": int(len(checks_df)),
        "coverage_90": coverage_90,
        "coverage_95": coverage_95,
        "avg_width_90": avg_width_90,
        "min_group_coverage_90": min_group_coverage_90,
        "critical_alerts": critical_alerts,
        "warning_alerts": warning_alerts,
        "total_alerts": total_alerts,
        "winkler_90": winkler_90,
        "winkler_95": winkler_95,
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
        "latest_backtest_month": str(latest_month) if latest_month is not None else None,
        "intervals_path": str(intervals_path),
        "policy_config": config_path,
        "lgd_ead_conformal_status_path": str(lgd_ead_status_path),
        "lgd_ead_conformal_status": lgd_ead_status,
    }

    checks_path = Path(output["policy_checks_parquet"])
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    checks_df.to_parquet(checks_path, index=False)

    status_path = Path(output["policy_status_json"])
    status_v2_path = Path(
        output.get("policy_status_json_v2", "models/conformal_policy_status_v2.json")
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(out_status, f, indent=2)
    status_v2_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_v2_path, "w", encoding="utf-8") as f:
        json.dump(out_status, f, indent=2)

    logger.info(f"Policy checks saved: {checks_path}")
    logger.info(f"Policy status saved: {status_path}")
    logger.info(f"Policy status v2 saved: {status_v2_path}")
    logger.info(
        f"Conformal policy pass={overall_pass} ({out_status['checks_passed']}/{out_status['checks_total']})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/conformal_policy.yaml")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    main(args.config, run_tag=args.run_tag)
