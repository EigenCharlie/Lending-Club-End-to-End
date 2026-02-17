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

from loguru import logger


def _persist_status(status: dict[str, str], elapsed: float) -> None:
    status["pipeline_time_s"] = f"{elapsed:.2f}"
    out_path = Path("models/pipeline_run_status.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(status, f)


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
    from src.data.prepare_dataset import main as prepare_dataset

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


def _step_optimization() -> None:
    from scripts.optimize_portfolio import main as optimize_portfolio

    optimize_portfolio("configs/optimization.yaml", 0.10)


def _step_modeva_governance() -> None:
    from scripts.side_projects.run_modeva_governance_checks import (
        main as run_modeva_governance,
    )

    run_modeva_governance("configs/modeva_governance.yaml")


def main(
    run_name: str = "v2",
    include_modeva_side_task: bool = False,
    continue_on_error: bool = False,
) -> int:
    t0 = time.time()
    logger.info(f"Starting end-to-end pipeline: {run_name} (continue_on_error={continue_on_error})")

    status: dict[str, str] = {}
    failed = False

    try:
        _run_step("data", _step_data, status, continue_on_error)
        _run_step("pd_model", _step_pd_model, status, continue_on_error)
        _run_step("time_series", _step_time_series, status, continue_on_error)
        _run_step("conformal_mondrian", _step_conformal, status, continue_on_error)
        _run_step("causal", _step_causal, status, continue_on_error)
        _run_step("optimization", _step_optimization, status, continue_on_error)

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
    args = parser.parse_args()
    raise SystemExit(
        main(
            run_name=args.run_name,
            include_modeva_side_task=args.include_modeva_side_task,
            continue_on_error=args.continue_on_error,
        )
    )
