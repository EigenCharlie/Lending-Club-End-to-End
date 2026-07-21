"""Build canonical pipeline_results artifact from stage outputs.

Reads canonical artifacts (predictions, conformal intervals, IFRS9 scenarios,
portfolio robustness) and assembles a pipeline_results.pkl summary consumed
by export_streamlit_artifacts.py to produce pipeline_summary.json.

Usage:
    uv run python scripts/build_pipeline_results.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")


def _persist_pipeline_results() -> None:
    """Build a pipeline summary artifact from canonical outputs."""
    results: dict[str, float | int | dict[str, int]] = {
        "batch_size": 0,
        "pd_mean": 0.0,
        "pd_auc": 0.0,
        "interval_width_mean": 0.0,
        "stages": {"S1": 0, "S2": 0, "S3": 0},
        "ecl_expected": 0.0,
        "ecl_conservative": 0.0,
        "ecl_range": 0.0,
        "robust_return": 0.0,
        "robust_funded": 0,
        "nonrobust_return": 0.0,
        "nonrobust_funded": 0,
        "price_of_robustness": 0.0,
        "pipeline_time_s": 0.0,
    }

    preds_path = DATA_DIR / "test_predictions.parquet"
    if preds_path.exists():
        preds = pd.read_parquet(preds_path)
        if "y_prob_final" in preds.columns:
            results["batch_size"] = int(len(preds))
            results["pd_mean"] = float(preds["y_prob_final"].mean())
        if {"y_true", "y_prob_final"}.issubset(preds.columns):
            results["pd_auc"] = float(roc_auc_score(preds["y_true"], preds["y_prob_final"]))

    intervals_path = DATA_DIR / "conformal_intervals_mondrian.parquet"
    if intervals_path.exists():
        ints = pd.read_parquet(intervals_path)
        if "interval_width" in ints.columns:
            results["interval_width_mean"] = float(ints["interval_width"].mean())
        elif {"pd_low_90", "pd_high_90"}.issubset(ints.columns):
            results["interval_width_mean"] = float((ints["pd_high_90"] - ints["pd_low_90"]).mean())
        elif {"pd_low", "pd_high"}.issubset(ints.columns):
            results["interval_width_mean"] = float((ints["pd_high"] - ints["pd_low"]).mean())

    ifrs9_path = DATA_DIR / "ifrs9_scenario_summary.parquet"
    if ifrs9_path.exists():
        ifrs9 = pd.read_parquet(ifrs9_path)
        baseline = ifrs9[ifrs9["scenario"] == "baseline"]
        severe = ifrs9[ifrs9["scenario"] == "severe"]
        if not baseline.empty:
            row = baseline.iloc[0]
            n_loans = int(row.get("n_loans", 0))
            results["stages"] = {
                "S1": int(round(float(row.get("stage1_share", 0.0)) * n_loans)),
                "S2": int(round(float(row.get("stage2_share", 0.0)) * n_loans)),
                "S3": int(round(float(row.get("stage3_share", 0.0)) * n_loans)),
            }
            results["ecl_expected"] = float(row.get("total_ecl", 0.0))
        if not severe.empty:
            results["ecl_conservative"] = float(severe.iloc[0].get("total_ecl", 0.0))
            results["ecl_range"] = max(0.0, results["ecl_conservative"] - results["ecl_expected"])

    robust_path = DATA_DIR / "portfolio_robustness_summary.parquet"
    if robust_path.exists():
        robust = pd.read_parquet(robust_path)
        if not robust.empty:
            if "risk_tolerance" in robust.columns:
                robust = robust.assign(_dist=(robust["risk_tolerance"] - 0.10).abs())
                row = robust.sort_values("_dist").iloc[0]
            else:
                row = robust.iloc[0]
            results["robust_return"] = float(row.get("best_robust_return", 0.0))
            results["nonrobust_return"] = float(row.get("baseline_nonrobust_return", 0.0))
            results["price_of_robustness"] = float(row.get("price_of_robustness", 0.0))
            results["robust_funded"] = int(row.get("best_robust_funded", 0))
            results["nonrobust_funded"] = int(
                row.get("baseline_nonrobust_funded", row.get("best_robust_funded", 0))
            )

    status_path = MODEL_DIR / "pipeline_run_status.pkl"
    if status_path.exists():
        with open(status_path, "rb") as f:
            status = pickle.load(f)  # noqa: S301
        try:
            results["pipeline_time_s"] = float(status.get("pipeline_time_s", 0.0))
        except Exception:
            results["pipeline_time_s"] = 0.0

    out = MODEL_DIR / "pipeline_results.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved pipeline results to {out}")


def main() -> None:
    _persist_pipeline_results()


if __name__ == "__main__":
    main()
