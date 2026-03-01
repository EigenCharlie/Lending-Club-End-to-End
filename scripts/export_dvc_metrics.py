"""Export canonical KPI summaries for DVC metrics/plots."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

try:
    from sklearn.metrics import d2_brier_score
except ImportError:  # sklearn < 1.8
    d2_brier_score = None

ROOT = Path(__file__).resolve().parents[1]


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    total = len(y_true)
    if total == 0:
        return 0.0

    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        n = int(mask.sum())
        if n == 0:
            continue
        frac = n / total
        ece += frac * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def _pd_metrics() -> dict[str, float]:
    preds = pd.read_parquet(ROOT / "data/processed/test_predictions.parquet")
    y_true = preds["y_true"].astype(int).to_numpy()
    score_col = "pd_calibrated" if "pd_calibrated" in preds.columns else "y_prob_final"
    y_prob = preds[score_col].astype(float).to_numpy()

    auc = float(roc_auc_score(y_true, y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = float(np.max(tpr - fpr))
    metrics = {
        "pd.auc": auc,
        "pd.gini": 2.0 * auc - 1.0,
        "pd.ks": ks,
        "pd.brier": float(brier_score_loss(y_true, y_prob)),
        "pd.ece": _ece(y_true, y_prob),
    }
    if d2_brier_score is not None:
        metrics["pd.d2_brier"] = float(d2_brier_score(y_true, y_prob))
    return metrics


def _conformal_metrics() -> dict[str, float]:
    with open(ROOT / "models/conformal_policy_status.json", encoding="utf-8") as f:
        status = json.load(f)
    return {
        "conformal.coverage90": float(status.get("coverage_90", 0.0)),
        "conformal.coverage95": float(status.get("coverage_95", 0.0)),
        "conformal.avg_width90": float(status.get("avg_width_90", 0.0)),
        "conformal.min_group_coverage90": float(status.get("min_group_coverage_90", 0.0)),
        "conformal.overall_pass": float(int(bool(status.get("overall_pass", False)))),
    }


def _ifrs9_metrics() -> dict[str, float]:
    df = pd.read_parquet(ROOT / "data/processed/ifrs9_scenario_summary.parquet")
    by_scenario = {
        str(row["scenario"]): float(row["total_ecl"])
        for _, row in df[["scenario", "total_ecl"]].iterrows()
    }
    baseline = by_scenario.get("baseline", 0.0)
    severe = by_scenario.get("severe", 0.0)
    return {
        "ifrs9.ecl_baseline": baseline,
        "ifrs9.ecl_severe": severe,
        "ifrs9.severe_uplift_pct": ((severe / baseline) - 1.0) * 100.0 if baseline else 0.0,
    }


def _optimization_metrics() -> dict[str, float]:
    pipeline = _load_pickle(ROOT / "models/pipeline_results.pkl")
    if not isinstance(pipeline, dict):
        raise TypeError("models/pipeline_results.pkl must contain a dict")
    return {
        "optimization.robust_return": float(pipeline.get("robust_return", 0.0)),
        "optimization.nonrobust_return": float(pipeline.get("nonrobust_return", 0.0)),
        "optimization.price_of_robustness": float(pipeline.get("price_of_robustness", 0.0)),
        "optimization.robust_funded": float(pipeline.get("robust_funded", 0.0)),
        "optimization.nonrobust_funded": float(pipeline.get("nonrobust_funded", 0.0)),
    }


def _write_conformal_backtest_plot(out_path: Path) -> None:
    df = pd.read_parquet(ROOT / "data/processed/conformal_backtest_monthly.parquet").copy()
    keep = [
        "month",
        "n",
        "coverage_90",
        "target_90",
        "coverage_95",
        "target_95",
        "avg_width_90",
        "coverage_90_roll3",
        "coverage_95_roll3",
        "avg_width_90_roll3",
    ]
    df = df[keep].sort_values("month")
    df["month"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m-%d")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _write_robustness_frontier_plot(out_path: Path) -> None:
    df = pd.read_parquet(ROOT / "data/processed/portfolio_robustness_frontier.parquet").copy()
    keep = [
        "policy",
        "risk_tolerance",
        "uncertainty_aversion",
        "price_of_robustness",
        "price_of_robustness_pct",
        "expected_return_net_point",
        "worst_case_loss",
        "worst_case_pd",
        "point_pd",
        "n_funded",
        "solver_status",
    ]
    df = (
        df[keep]
        .sort_values(["risk_tolerance", "policy", "uncertainty_aversion"])
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    out_dir = ROOT / "reports/dvc"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    metrics.update(_pd_metrics())
    metrics.update(_conformal_metrics())
    metrics.update(_ifrs9_metrics())
    metrics.update(_optimization_metrics())

    invalid = [k for k, v in metrics.items() if not np.isfinite(float(v))]
    if invalid:
        raise ValueError(f"Non-finite values found in DVC metrics export: {sorted(invalid)}")

    metrics_path = out_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")

    _write_conformal_backtest_plot(out_dir / "conformal_coverage_backtest.csv")
    _write_robustness_frontier_plot(out_dir / "robustness_frontier.csv")

    logger.info(f"Wrote DVC metrics summary: {metrics_path}")
    logger.info(f"Metrics exported: {len(metrics)} keys")


if __name__ == "__main__":
    main()
