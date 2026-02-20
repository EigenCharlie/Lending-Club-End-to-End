"""End-to-end pipeline orchestrator.

Pipeline:
Data -> Features -> PD -> Conformal -> Optimization -> IFRS9 artifacts

Usage:
    uv run python scripts/end_to_end_pipeline.py --run_name v2
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score


def _persist_status(status: dict[str, str], elapsed: float) -> None:
    status["pipeline_time_s"] = f"{elapsed:.2f}"
    out_path = Path("models/pipeline_run_status.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(status, f)


def _persist_pipeline_results() -> None:
    """Build a pipeline summary artifact from canonical outputs."""
    data_dir = Path("data/processed")
    model_dir = Path("models")
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

    preds_path = data_dir / "test_predictions.parquet"
    if preds_path.exists():
        preds = pd.read_parquet(preds_path)
        if "y_prob_final" in preds.columns:
            results["batch_size"] = int(len(preds))
            results["pd_mean"] = float(preds["y_prob_final"].mean())
        if {"y_true", "y_prob_final"}.issubset(preds.columns):
            results["pd_auc"] = float(roc_auc_score(preds["y_true"], preds["y_prob_final"]))

    intervals_path = data_dir / "conformal_intervals_mondrian.parquet"
    if intervals_path.exists():
        ints = pd.read_parquet(intervals_path)
        if "interval_width" in ints.columns:
            results["interval_width_mean"] = float(ints["interval_width"].mean())
        elif {"pd_low_90", "pd_high_90"}.issubset(ints.columns):
            results["interval_width_mean"] = float((ints["pd_high_90"] - ints["pd_low_90"]).mean())
        elif {"pd_low", "pd_high"}.issubset(ints.columns):
            results["interval_width_mean"] = float((ints["pd_high"] - ints["pd_low"]).mean())

    ifrs9_path = data_dir / "ifrs9_scenario_summary.parquet"
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

    robust_path = data_dir / "portfolio_robustness_summary.parquet"
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

    status_path = model_dir / "pipeline_run_status.pkl"
    if status_path.exists():
        with open(status_path, "rb") as f:
            status = pickle.load(f)
        try:
            results["pipeline_time_s"] = float(status.get("pipeline_time_s", 0.0))
        except Exception:
            results["pipeline_time_s"] = 0.0

    out = model_dir / "pipeline_results.pkl"
    with open(out, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved pipeline results to {out}")


def _run_step(
    step_name: str,
    step_fn: Callable[[], None],
    status: dict[str, str],
    continue_on_error: bool,
) -> None:
    try:
        step_fn()
        status[step_name] = "ok"
    except Exception as exc:
        status[step_name] = f"error: {exc}"
        logger.exception(f"Step '{step_name}' failed")
        if not continue_on_error:
            raise


def _step_data() -> None:
    from src.data.build_datasets import main as build_datasets
    from src.data.make_dataset import main as make_dataset
    from src.data.prepare_dataset import main as prepare_dataset

    make_dataset()
    prepare_dataset()
    build_datasets()


def _step_pd_model() -> None:
    from scripts.train_pd_model import main as train_pd

    train_pd("configs/pd_model.yaml", sample_size=300_000)


def _step_time_series() -> None:
    from scripts.forecast_default_rates import main as forecast_ts

    forecast_ts(horizon=12)


def _step_conformal() -> None:
    from scripts.generate_conformal_intervals import main as generate_cp

    generate_cp()


def _step_causal() -> None:
    from scripts.estimate_causal_effects import main as estimate_causal

    estimate_causal("int_rate", sample_size=200_000)


def _step_survival() -> None:
    from scripts.run_survival_analysis import main as run_survival

    run_survival(sample_size=100_000, rsf_n_estimators=200)


def _step_ifrs9() -> None:
    from scripts.run_ifrs9_sensitivity import main as run_ifrs9

    run_ifrs9()


def _step_optimization() -> None:
    from scripts.optimize_portfolio import main as optimize_portfolio

    optimize_portfolio("configs/optimization.yaml", 0.10)


def _step_optimization_tradeoff() -> None:
    from scripts.optimize_portfolio_tradeoff import main as optimize_portfolio_tradeoff

    optimize_portfolio_tradeoff(config_path="configs/optimization.yaml")


def _step_modeva_governance() -> None:
    from scripts.side_projects.run_modeva_governance_checks import (
        main as run_modeva_governance,
    )

    run_modeva_governance("configs/modeva_governance.yaml")


def main(
    run_name: str = "v2",
    include_modeva_side_task: bool = False,
    continue_on_error: bool = False,
    skip_make_dataset: bool = False,
) -> int:
    t0 = time.time()
    logger.info(
        "Starting end-to-end pipeline: "
        f"{run_name} (continue_on_error={continue_on_error}, skip_make_dataset={skip_make_dataset})"
    )

    status: dict[str, str] = {}
    failed = False

    try:
        if skip_make_dataset:
            from src.data.build_datasets import main as build_datasets
            from src.data.prepare_dataset import main as prepare_dataset

            def _step_data_skip_make() -> None:
                prepare_dataset()
                build_datasets()

            _run_step("data", _step_data_skip_make, status, continue_on_error)
        else:
            _run_step("data", _step_data, status, continue_on_error)
        _run_step("pd_model", _step_pd_model, status, continue_on_error)
        _run_step("time_series", _step_time_series, status, continue_on_error)
        _run_step("conformal_mondrian", _step_conformal, status, continue_on_error)
        _run_step("causal", _step_causal, status, continue_on_error)
        _run_step("survival", _step_survival, status, continue_on_error)
        _run_step("ifrs9", _step_ifrs9, status, continue_on_error)
        _run_step("optimization", _step_optimization, status, continue_on_error)
        _run_step("optimization_tradeoff", _step_optimization_tradeoff, status, continue_on_error)

        if include_modeva_side_task:
            _run_step(
                "modeva_governance_side_task",
                _step_modeva_governance,
                status,
                continue_on_error,
            )
        else:
            status["modeva_governance_side_task"] = "skipped"
    except Exception:
        failed = True
    finally:
        elapsed = time.time() - t0
        _persist_status(status, elapsed)
        _persist_pipeline_results()
        logger.info(f"Pipeline {run_name} finished in {elapsed:.2f}s")
        logger.info(f"Pipeline status: {status}")

    if failed:
        logger.error("Pipeline finished with errors (fail-fast mode).")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="v2")
    parser.add_argument("--include_modeva_side_task", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-make-dataset", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        main(
            run_name=args.run_name,
            include_modeva_side_task=args.include_modeva_side_task,
            continue_on_error=args.continue_on_error,
            skip_make_dataset=args.skip_make_dataset,
        )
    )
