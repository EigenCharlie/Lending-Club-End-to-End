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
from pathlib import Path

from loguru import logger


def main(run_name: str = "v2", include_modeva_side_task: bool = False):
    t0 = time.time()
    logger.info(f"Starting end-to-end pipeline: {run_name}")

    status: dict[str, str] = {}

    # Step 1: Data preparation
    try:
        from src.data.prepare_dataset import main as prepare_dataset
        from src.data.build_datasets import main as build_datasets

        prepare_dataset()
        build_datasets()
        status["data"] = "ok"
    except Exception as e:
        status["data"] = f"error: {e}"
        logger.exception("Data step failed")

    # Step 2: PD training + conformal artifacts
    try:
        from scripts.train_pd_model import main as train_pd

        train_pd("configs/pd_model.yaml", sample_size=300_000)
        status["pd_model"] = "ok"
    except Exception as e:
        status["pd_model"] = f"error: {e}"
        logger.exception("PD step failed")

    # Step 3: Time series forecasts
    try:
        from scripts.forecast_default_rates import main as forecast_ts

        forecast_ts(horizon=12)
        status["time_series"] = "ok"
    except Exception as e:
        status["time_series"] = f"error: {e}"
        logger.exception("Time-series step failed")

    # Step 4: Mondrian conformal intervals
    try:
        from scripts.generate_conformal_intervals import main as generate_cp

        generate_cp()
        status["conformal_mondrian"] = "ok"
    except Exception as e:
        status["conformal_mondrian"] = f"error: {e}"
        logger.exception("Mondrian conformal step failed")

    # Step 5: Causal
    try:
        from scripts.estimate_causal_effects import main as estimate_causal

        estimate_causal("int_rate", sample_size=200_000)
        status["causal"] = "ok"
    except Exception as e:
        status["causal"] = f"error: {e}"
        logger.exception("Causal step failed")

    # Step 6: Optimization
    try:
        from scripts.optimize_portfolio import main as optimize_portfolio

        optimize_portfolio("configs/optimization.yaml", 0.10)
        status["optimization"] = "ok"
    except Exception as e:
        status["optimization"] = f"error: {e}"
        logger.exception("Optimization step failed")

    # Step 7 (optional): Modeva governance side-task on canonical PD
    if include_modeva_side_task:
        try:
            from scripts.side_projects.run_modeva_governance_checks import main as run_modeva_governance

            run_modeva_governance("configs/modeva_governance.yaml")
            status["modeva_governance_side_task"] = "ok"
        except Exception as e:
            status["modeva_governance_side_task"] = f"error: {e}"
            logger.exception("Modeva governance side-task failed")
    else:
        status["modeva_governance_side_task"] = "skipped"

    elapsed = time.time() - t0
    status["pipeline_time_s"] = f"{elapsed:.2f}"

    out_path = Path("models/pipeline_run_status.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(status, f)

    logger.info(f"Pipeline {run_name} complete in {elapsed:.2f}s")
    logger.info(f"Pipeline status: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="v2")
    parser.add_argument("--include_modeva_side_task", action="store_true")
    args = parser.parse_args()
    main(args.run_name, args.include_modeva_side_task)
