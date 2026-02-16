"""Validate conformal artifacts against explicit acceptance policy.

Usage:
    uv run python scripts/validate_conformal_policy.py --config configs/conformal_policy.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger


def _check(metric_name: str, value: float, threshold: float, comparator: str, scope: str) -> dict[str, object]:
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


def main(config_path: str = "configs/conformal_policy.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    policy = cfg["policy"]
    artifacts = cfg["artifacts"]
    output = cfg["output"]

    with open(artifacts["conformal_results_path"], "rb") as f:
        results = pickle.load(f)
    group_metrics = pd.read_parquet(artifacts["group_metrics_path"])
    backtest_monthly = pd.read_parquet(artifacts["backtest_monthly_path"])
    alerts_path = Path(artifacts["backtest_alerts_path"])
    alerts = pd.read_parquet(alerts_path) if alerts_path.exists() else pd.DataFrame(columns=["severity"])

    metrics_90 = results.get("metrics_90", {})
    metrics_95 = results.get("metrics_95", {})

    coverage_90 = float(metrics_90.get("empirical_coverage", 0.0))
    coverage_95 = float(metrics_95.get("empirical_coverage", 0.0))
    avg_width_90 = float(metrics_90.get("avg_interval_width", 999.0))
    min_group_coverage_90 = float(group_metrics.get("coverage_90", pd.Series([0.0])).min())
    critical_alerts = int((alerts.get("severity", pd.Series([], dtype=str)) == "critical").sum())
    warning_alerts = int((alerts.get("severity", pd.Series([], dtype=str)) == "warning").sum())
    total_alerts = int(len(alerts))

    checks = [
        _check("coverage_90", coverage_90, float(policy["target_coverage_90_min"]), ">=", "portfolio"),
        _check("coverage_95", coverage_95, float(policy["target_coverage_95_min"]), ">=", "portfolio"),
        _check("min_group_coverage_90", min_group_coverage_90, float(policy["min_group_coverage_90_min"]), ">=", "group"),
        _check("avg_width_90", avg_width_90, float(policy["max_avg_width_90"]), "<=", "portfolio"),
        _check("critical_alerts", float(critical_alerts), float(policy["max_critical_alerts"]), "<=", "monitoring"),
        _check("total_alerts", float(total_alerts), float(policy["max_total_alerts"]), "<=", "monitoring"),
        _check("warning_alerts", float(warning_alerts), float(policy["max_warning_alerts"]), "<=", "monitoring"),
    ]
    checks_df = pd.DataFrame(checks)
    overall_pass = bool(checks_df["passed"].all())

    latest_month = (
        backtest_monthly.sort_values("month").iloc[-1]["month"]
        if not backtest_monthly.empty
        else None
    )

    out_status = {
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
        "latest_backtest_month": str(latest_month) if latest_month is not None else None,
        "policy_config": config_path,
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
    logger.info(f"Conformal policy pass={overall_pass} ({out_status['checks_passed']}/{out_status['checks_total']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/conformal_policy.yaml")
    args = parser.parse_args()
    main(args.config)
