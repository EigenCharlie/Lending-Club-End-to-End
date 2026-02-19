"""Run portfolio optimization with conformal prediction uncertainty sets.

Usage:
    uv run python scripts/optimize_portfolio.py --risk_tolerance 0.05 --uncertainty_aversion 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.models.conformal_artifacts import load_conformal_intervals
from src.optimization.portfolio_model import build_portfolio_model, solve_portfolio
from src.optimization.robust_opt import scenario_analysis


def _parse_percent_series(series: pd.Series) -> np.ndarray:
    """Parse Lending Club style percentages into decimals."""
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    else:
        values = (
            series.astype(str)
            .str.strip()
            .str.rstrip("%")
            .pipe(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
    values = np.nan_to_num(values, nan=12.0)  # default annual rate (%)
    return values / 100.0


def _load_candidates() -> pd.DataFrame:
    """Load portfolio candidate pool, preferring feature-engineered data."""
    fe_path = Path("data/processed/test_fe.parquet")
    raw_path = Path("data/processed/test.parquet")
    if fe_path.exists():
        return pd.read_parquet(fe_path)
    return pd.read_parquet(raw_path)


def _load_interval_artifact() -> pd.DataFrame:
    """Load conformal interval artifact (canonical first)."""
    intervals, intervals_path, is_legacy = load_conformal_intervals(allow_legacy_fallback=True)
    logger.info(
        f"Loaded conformal intervals from {intervals_path} "
        f"(legacy={is_legacy}, rows={len(intervals):,})"
    )
    return intervals.reset_index(drop=True)


def _resolve_interval_columns(intervals: pd.DataFrame) -> tuple[str, str, str]:
    col_point = "y_pred" if "y_pred" in intervals.columns else "pd_point"
    col_low = "pd_low_90" if "pd_low_90" in intervals.columns else "pd_low"
    col_high = "pd_high_90" if "pd_high_90" in intervals.columns else "pd_high"
    return col_point, col_low, col_high


def _align_candidates_and_intervals(
    candidates: pd.DataFrame,
    intervals: pd.DataFrame,
    max_candidates: int = 5_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Align loans with PD intervals using ID when available, with legacy fallback."""
    col_point, col_low, col_high = _resolve_interval_columns(intervals)
    base_n = min(len(candidates), len(intervals), max_candidates)

    if "id" in candidates.columns and "id" in intervals.columns:
        cand = candidates.copy()
        ints = intervals.copy()
        cand["_id_join"] = cand["id"].astype(str)
        ints["_id_join"] = ints["id"].astype(str)
        ints = ints.drop_duplicates(subset="_id_join", keep="first")
        merged = cand.merge(
            ints[["_id_join", col_point, col_low, col_high]],
            on="_id_join",
            how="inner",
        )
        if merged.empty:
            raise ValueError("ID-based merge between candidates and intervals returned zero rows.")
        n = min(len(merged), max_candidates)
        if len(merged) > n:
            rng = np.random.default_rng(random_state)
            idx = np.sort(rng.choice(np.arange(len(merged)), size=n, replace=False))
            merged = merged.iloc[idx].reset_index(drop=True)
        else:
            merged = merged.reset_index(drop=True)

        aligned_loans = merged[candidates.columns].copy()
        pd_point = merged[col_point].to_numpy(dtype=float)
        pd_low = merged[col_low].to_numpy(dtype=float)
        pd_high = merged[col_high].to_numpy(dtype=float)
        logger.info(
            f"Aligned candidates and intervals by id: n={len(aligned_loans):,} "
            f"(candidate_rows={len(candidates):,}, interval_rows={len(intervals):,})"
        )
        return aligned_loans, pd_point, pd_low, pd_high

    # Legacy fallback where interval artifact has no stable key.
    logger.warning(
        "Conformal interval artifact has no id alignment key; using positional fallback for optimization."
    )
    aligned_loans = candidates.head(base_n).reset_index(drop=True).copy()
    ints = intervals.head(base_n).reset_index(drop=True)
    pd_point = ints[col_point].to_numpy(dtype=float)
    pd_low = ints[col_low].to_numpy(dtype=float)
    pd_high = ints[col_high].to_numpy(dtype=float)
    return aligned_loans, pd_point, pd_low, pd_high


def main(
    config_path: str = "configs/optimization.yaml",
    risk_tolerance: float = 0.10,
    uncertainty_aversion: float = 0.0,
    min_budget_utilization: float = 0.0,
    pd_cap_slack_penalty: float = 0.0,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    test = _load_candidates()
    intervals = _load_interval_artifact()
    test_sample, pd_point, pd_low, pd_high = _align_candidates_and_intervals(
        test, intervals, max_candidates=5_000, random_state=42
    )
    n = len(test_sample)

    lgd = np.full(n, 0.45)
    if "int_rate" in test_sample.columns:
        int_rates = _parse_percent_series(test_sample["int_rate"])
    else:
        int_rates = np.full(n, 0.12)

    model = build_portfolio_model(
        loans=test_sample,
        pd_point=pd_point,
        pd_low=pd_low,
        pd_high=pd_high,
        lgd=lgd,
        int_rates=int_rates,
        total_budget=config["portfolio"]["total_budget"],
        max_concentration=config["portfolio"]["max_concentration"],
        max_portfolio_pd=risk_tolerance,
        robust=config["portfolio"]["robust"],
        uncertainty_aversion=uncertainty_aversion,
        min_budget_utilization=min_budget_utilization,
        pd_cap_slack_penalty=pd_cap_slack_penalty,
    )

    solution = solve_portfolio(
        model,
        time_limit=config["optimization"]["time_limit"],
        threads=config["optimization"]["threads"],
    )

    allocation = np.array([solution["allocation"][i] for i in range(n)], dtype=float)
    loan_amounts = (
        test_sample["loan_amnt"].to_numpy(dtype=float)
        if "loan_amnt" in test_sample.columns
        else np.ones(n, dtype=float) * 10_000
    )
    scenarios = scenario_analysis(allocation, loan_amounts, pd_low, pd_point, pd_high, lgd)

    logger.info(f"Optimization complete: {solution['solver_status']}")
    logger.info(f"Objective value: {solution['objective_value']:,.2f}")
    logger.info(f"Scenarios:\n{scenarios}")

    # Persist artifacts for downstream reporting.
    model_dir = Path("models")
    data_dir = Path("data/processed")
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "portfolio_results.pkl", "wb") as f:
        import pickle

        pickle.dump(
            {
                "solution": solution,
                "scenario_analysis": scenarios.to_dict(orient="records")[0],
                "n_candidates": n,
                "risk_tolerance": risk_tolerance,
                "uncertainty_aversion": uncertainty_aversion,
                "min_budget_utilization": min_budget_utilization,
                "pd_cap_slack_penalty": pd_cap_slack_penalty,
            },
            f,
        )

    alloc_df = pd.DataFrame(
        {
            "loan_idx": np.arange(n),
            "alloc": allocation,
            "loan_amnt": loan_amounts,
            "pd_point": pd_point,
            "pd_low": pd_low,
            "pd_high": pd_high,
            "int_rate": int_rates,
        }
    )
    alloc_df.to_parquet(data_dir / "portfolio_allocations.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/optimization.yaml")
    parser.add_argument("--risk_tolerance", type=float, default=0.10)
    parser.add_argument("--uncertainty_aversion", type=float, default=0.0)
    parser.add_argument("--min_budget_utilization", type=float, default=0.0)
    parser.add_argument("--pd_cap_slack_penalty", type=float, default=0.0)
    args = parser.parse_args()
    main(
        args.config,
        args.risk_tolerance,
        args.uncertainty_aversion,
        args.min_budget_utilization,
        args.pd_cap_slack_penalty,
    )
