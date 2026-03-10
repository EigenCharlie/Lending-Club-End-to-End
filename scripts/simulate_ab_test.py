"""Simulate A/B test: robust vs non-robust portfolio on OOT test set.

Retroactively applies two portfolio strategies to the OOT test set
and compares realized outcomes using actual default_flag as ground truth.

Strategy A (control): non-robust portfolio (pd_point for PD constraint)
Strategy B (treatment): robust portfolio (pd_high for PD constraint)

Usage:
    uv run python scripts/simulate_ab_test.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.ab_testing import ab_summary, compare_strategies
from src.optimization.portfolio_model import (
    compute_effective_pd,
    optimize_portfolio_allocation,
)

SCHEMA_VERSION = "2026-03-01.1"


def _artifact_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    root = str(os.environ.get("GPU_REPLAY_ARTIFACT_ROOT", "")).strip()
    return (Path(root) / path) if root else path


def _compute_realized_return(
    allocation: dict[int, float],
    loan_amnt: np.ndarray,
    int_rates: np.ndarray,
    default_flag: np.ndarray,
    lgd: float = 0.45,
) -> np.ndarray:
    """Compute per-loan realized return given actual defaults.

    For funded loans: return = alloc * loan_amnt * (rate*(1-default) - default*lgd)
    For unfunded loans: return = 0
    """
    n = len(loan_amnt)
    returns = np.zeros(n)
    for i in range(n):
        alloc = allocation.get(i, 0.0)
        if alloc > 0.01:
            if default_flag[i] == 1:
                returns[i] = alloc * loan_amnt[i] * (-lgd)
            else:
                returns[i] = alloc * loan_amnt[i] * int_rates[i]
    return returns


def _parse_percent_series(s: pd.Series, default: float = 0.12) -> np.ndarray:
    """Convert percent column to decimal."""
    if pd.api.types.is_numeric_dtype(s):
        arr = s.to_numpy(dtype=float)
        if np.nanmedian(arr) > 1:
            arr = arr / 100.0
        return np.nan_to_num(arr, nan=default)
    return (
        s.astype(str)
        .str.strip()
        .str.rstrip("%")
        .pipe(pd.to_numeric, errors="coerce")
        .div(100)
        .fillna(default)
        .to_numpy(dtype=float)
    )


def _resolve_robust_policy(
    *,
    max_portfolio_pd: float,
    policy_selector: str = "promotion_first",
    summary_path: str = "data/processed/portfolio_robustness_summary.parquet",
    champion_policy_path: str = "models/champion_portfolio_policy.json",
) -> dict[str, float | str]:
    """Resolve robust strategy parameters from tradeoff summary, with fallback defaults."""
    default = {
        "source": "fallback_default",
        "risk_tolerance": float(max_portfolio_pd),
        "uncertainty_aversion": 0.0,
        "min_budget_utilization": 0.0,
        "pd_cap_slack_penalty": 0.0,
        "policy_mode": "hard_worst_case",
        "gamma": 1.0,
    }
    champion_path = _artifact_path(champion_policy_path)
    if champion_path.exists():
        try:
            payload = json.loads(champion_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                if policy_selector == "robustness_aware":
                    selected = payload.get("selected_policy_robustness_aware") or payload.get(
                        "selected_policy", {}
                    )
                elif policy_selector == "balanced_robustness":
                    selected = (
                        payload.get("selected_policy_balanced_robustness")
                        or payload.get("selected_policy_robustness_aware")
                        or payload.get("selected_policy", {})
                    )
                else:
                    selected = payload.get("selected_policy", {})
            else:
                selected = {}
            policy = {
                "source": f"champion_policy_artifact::{policy_selector}",
                "risk_tolerance": float(selected.get("risk_tolerance", max_portfolio_pd)),
                "uncertainty_aversion": float(selected.get("uncertainty_aversion", 0.0)),
                "min_budget_utilization": float(selected.get("min_budget_utilization", 0.0)),
                "pd_cap_slack_penalty": float(selected.get("pd_cap_slack_penalty", 0.0)),
                "policy_mode": str(selected.get("policy_mode", "hard_worst_case")),
                "gamma": float(selected.get("gamma", 1.0)),
            }
            logger.info(
                "Resolved robust policy from champion artifact: "
                f"risk_tolerance={policy['risk_tolerance']:.4f}, "
                f"policy_mode={policy['policy_mode']}, gamma={policy['gamma']:.2f}"
            )
            return policy
        except Exception as exc:
            logger.warning(
                f"Could not parse champion portfolio policy ({champion_path}): {exc}. "
                "Falling back to summary-based policy."
            )

    path = _artifact_path(summary_path)
    if not path.exists():
        logger.warning(f"Robustness summary not found ({path}); using fallback robust policy.")
        return default

    try:
        summary = pd.read_parquet(path)
    except Exception as exc:
        logger.warning(f"Could not read robustness summary ({path}): {exc}")
        return default

    required_cols = {
        "risk_tolerance",
        "best_robust_lambda",
        "best_robust_min_budget_utilization",
        "best_robust_pd_cap_slack_penalty",
    }
    if summary.empty or not required_cols.issubset(set(summary.columns)):
        missing = sorted(required_cols - set(summary.columns))
        logger.warning(
            "Robustness summary missing required columns or empty; "
            f"missing={missing}. Using fallback robust policy."
        )
        return default

    work = summary.copy()
    for col in required_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=list(required_cols)).reset_index(drop=True)
    if work.empty:
        logger.warning("No valid numeric robust summary rows; using fallback policy.")
        return default

    target = float(max_portfolio_pd)
    lower_eq = work.loc[work["risk_tolerance"] <= target + 1e-12].copy()
    candidate_pool = lower_eq if not lower_eq.empty else work
    candidate_pool["_distance"] = (candidate_pool["risk_tolerance"] - target).abs()
    if "best_robust_return" in candidate_pool.columns:
        candidate_pool["best_robust_return"] = pd.to_numeric(
            candidate_pool["best_robust_return"], errors="coerce"
        ).fillna(float("-inf"))
        row = candidate_pool.sort_values(
            by=["_distance", "best_robust_return"],
            ascending=[True, False],
        ).iloc[0]
    else:
        row = candidate_pool.sort_values(by=["_distance"], ascending=[True]).iloc[0]

    policy = {
        "source": "portfolio_robustness_summary",
        "risk_tolerance": float(row["risk_tolerance"]),
        "uncertainty_aversion": float(row["best_robust_lambda"]),
        "min_budget_utilization": float(row["best_robust_min_budget_utilization"]),
        "pd_cap_slack_penalty": float(row["best_robust_pd_cap_slack_penalty"]),
        "policy_mode": str(row.get("best_robust_policy_mode", "hard_worst_case")),
        "gamma": float(row.get("best_robust_gamma", 1.0)),
    }
    logger.info(
        "Resolved robust policy from summary: "
        f"risk_tolerance={policy['risk_tolerance']:.4f}, "
        f"uncertainty_aversion={policy['uncertainty_aversion']:.4f}, "
        f"min_budget_utilization={policy['min_budget_utilization']:.4f}, "
        f"pd_cap_slack_penalty={policy['pd_cap_slack_penalty']:.4f}"
    )
    return policy


def _apply_candidate_universe(
    test_df: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    candidate_universe_path: str,
    max_candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    path = _artifact_path(candidate_universe_path)
    max_candidates_norm = None if int(max_candidates) <= 0 else int(max_candidates)
    if path.exists() and "id" in test_df.columns and "id" in intervals.columns:
        universe = pd.read_parquet(path)
        if "id" in universe.columns and not universe.empty:
            ordered_ids = universe["id"].astype(str)
            if max_candidates_norm is not None:
                ordered_ids = ordered_ids.iloc[:max_candidates_norm]
            order_df = pd.DataFrame(
                {
                    "_id_join": ordered_ids.values,
                    "_sample_order": np.arange(len(ordered_ids), dtype=int),
                }
            )
            test_work = test_df.copy()
            ints_work = intervals.copy()
            test_work["_id_join"] = test_work["id"].astype(str)
            ints_work["_id_join"] = ints_work["id"].astype(str)
            test_work = test_work.merge(order_df, on="_id_join", how="inner")
            ints_work = ints_work.merge(order_df, on="_id_join", how="inner")
            test_work = test_work.sort_values("_sample_order").drop_duplicates("_id_join")
            ints_work = ints_work.sort_values("_sample_order").drop_duplicates("_id_join")
            merged_n = min(len(test_work), len(ints_work))
            test_out = test_work.iloc[:merged_n].drop(columns=["_id_join", "_sample_order"])
            ints_out = ints_work.iloc[:merged_n].drop(columns=["_id_join", "_sample_order"])
            if merged_n > 0:
                logger.info(
                    "Using champion candidate universe from {} with n={}",
                    path,
                    merged_n,
                )
                return test_out.reset_index(drop=True), ints_out.reset_index(drop=True), str(path)

    n = min(len(test_df), len(intervals))
    if max_candidates_norm is not None:
        n = min(n, max_candidates_norm)
    logger.info(
        "Using positional candidate cohort with n={} (no shared universe artifact).",
        n,
    )
    return (
        test_df.iloc[:n].reset_index(drop=True),
        intervals.iloc[:n].reset_index(drop=True),
        "",
    )


def main(
    total_budget: float = 1_000_000,
    max_portfolio_pd: float = 0.10,
    max_candidates: int = 5_000,
    n_boot: int = 1000,
    seed: int = 42,
    no_regression_tolerance_pct: float = 0.05,
    robust_policy_summary_path: str = "data/processed/portfolio_robustness_summary.parquet",
    champion_policy_path: str = "models/champion_portfolio_policy.json",
    candidate_universe_path: str = "data/processed/champion_candidate_universe.parquet",
    results_path: str = "data/processed/ab_simulation_results.parquet",
    summary_path: str = "data/processed/ab_simulation_summary.parquet",
    status_path: str = "models/ab_simulation_status.json",
    run_tag: str | None = None,
    solver_backend: str = "highs",
    policy_selector: str = "promotion_first",
) -> None:
    """Run the A/B simulation."""
    data_dir = Path("data/processed")
    test_path = data_dir / "test_fe.parquet"
    intervals_path = data_dir / "conformal_intervals_mondrian.parquet"

    for p in [test_path, intervals_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    test_df = pd.read_parquet(test_path)
    intervals = pd.read_parquet(intervals_path)
    resolved_run_tag = (
        str(run_tag or "").strip() or str(os.environ.get("PIPELINE_RUN_TAG", "")).strip()
    )
    if not resolved_run_tag:
        resolved_run_tag = "untracked"

    max_candidates_norm = None if int(max_candidates) <= 0 else int(max_candidates)
    test_df, intervals, universe_source = _apply_candidate_universe(
        test_df,
        intervals,
        candidate_universe_path=candidate_universe_path,
        max_candidates=max_candidates,
    )
    n = min(len(test_df), len(intervals))
    logger.info(
        f"Using {n} candidates "
        f"(max_candidates={'full' if max_candidates_norm is None else max_candidates_norm})"
    )

    # Extract arrays
    # Map column names: conformal intervals use y_pred, pd_low_90, pd_high_90
    pd_col = next(
        (c for c in ["pd_calibrated", "y_pred"] if c in intervals.columns), intervals.columns[0]
    )
    low_col = next((c for c in ["pd_low", "pd_low_90"] if c in intervals.columns), None)
    high_col = next((c for c in ["pd_high", "pd_high_90"] if c in intervals.columns), None)
    pd_point = intervals[pd_col].values
    pd_low = intervals[low_col].values if low_col else pd_point * 0.8
    pd_high = intervals[high_col].values if high_col else pd_point * 1.3
    lgd_val = 0.45
    lgd = np.full(n, lgd_val)
    int_rates = (
        _parse_percent_series(test_df["int_rate"])
        if "int_rate" in test_df.columns
        else np.full(n, 0.12)
    )
    default_flag = (
        test_df["default_flag"].values if "default_flag" in test_df.columns else np.zeros(n)
    )

    loan_amnt = (
        test_df["loan_amnt"].values if "loan_amnt" in test_df.columns else np.full(n, 10000.0)
    )

    robust_policy = _resolve_robust_policy(
        max_portfolio_pd=float(max_portfolio_pd),
        policy_selector=str(policy_selector),
        summary_path=str(robust_policy_summary_path),
        champion_policy_path=str(champion_policy_path),
    )
    effective_max_portfolio_pd = float(robust_policy.get("risk_tolerance", max_portfolio_pd))

    common = {
        "loans": test_df,
        "pd_point": pd_point,
        "pd_low": pd_low,
        "pd_high": pd_high,
        "lgd": lgd,
        "int_rates": int_rates,
        "total_budget": total_budget,
        "max_portfolio_pd": effective_max_portfolio_pd,
    }

    # Strategy A: non-robust
    logger.info("Strategy A (control): non-robust portfolio")
    sol_a = optimize_portfolio_allocation(
        robust=False,
        solver_backend=solver_backend,
        **common,
    )

    # Strategy B: robust
    logger.info("Strategy B (treatment): robust portfolio")
    effective_pd_b = compute_effective_pd(
        pd_point=pd_point,
        pd_high=pd_high,
        policy_mode=str(robust_policy.get("policy_mode", "hard_worst_case")),
        gamma=float(robust_policy.get("gamma", 1.0)),
    )
    sol_b = optimize_portfolio_allocation(
        robust=True,
        uncertainty_aversion=float(robust_policy.get("uncertainty_aversion", 0.0)),
        min_budget_utilization=float(robust_policy.get("min_budget_utilization", 0.0)),
        pd_cap_slack_penalty=float(robust_policy.get("pd_cap_slack_penalty", 0.0)),
        pd_constraint_override=effective_pd_b,
        solver_backend=solver_backend,
        **common,
    )

    # Compute realized returns
    returns_a = _compute_realized_return(
        sol_a["allocation"], loan_amnt, int_rates, default_flag, lgd_val
    )
    returns_b = _compute_realized_return(
        sol_b["allocation"], loan_amnt, int_rates, default_flag, lgd_val
    )

    # Statistical comparison
    comparison = compare_strategies(
        returns_a, returns_b, method="bootstrap", n_boot=n_boot, seed=seed
    )

    # Aggregate metrics
    metrics_a = {
        "total_return": float(returns_a.sum()),
        "n_funded": sol_a["n_funded"],
        "total_allocated": sol_a["total_allocated"],
        "avg_return_per_funded": float(returns_a[returns_a != 0].mean())
        if (returns_a != 0).any()
        else 0.0,
    }
    metrics_b = {
        "total_return": float(returns_b.sum()),
        "n_funded": sol_b["n_funded"],
        "total_allocated": sol_b["total_allocated"],
        "avg_return_per_funded": float(returns_b[returns_b != 0].mean())
        if (returns_b != 0).any()
        else 0.0,
    }

    summary = ab_summary(metrics_a, metrics_b)

    # Save results
    results_df = pd.DataFrame(
        [
            {
                "strategy_a_return": metrics_a["total_return"],
                "strategy_b_return": metrics_b["total_return"],
                "diff": comparison["diff"],
                "ci_low": comparison["ci_low"],
                "ci_high": comparison["ci_high"],
                "p_value": comparison["p_value"],
                "significant": comparison["significant"],
                "n_funded_a": sol_a["n_funded"],
                "n_funded_b": sol_b["n_funded"],
            }
        ]
    )
    results_out = _artifact_path(results_path)
    results_out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(results_out, index=False)
    logger.info(f"Saved results: {results_out}")

    summary_out = _artifact_path(summary_path)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(summary_out, index=False)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_tag": resolved_run_tag,
        "strategy_a": "non_robust",
        "strategy_b": "robust_selected_for_champion",
        "comparison": comparison,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "n_candidates_available": int(min(len(test_df), len(intervals))),
        "n_candidates_used": int(n),
        "max_candidates_requested": None if max_candidates_norm is None else max_candidates_norm,
        "dataset_scope": "full_candidates" if max_candidates_norm is None else "sampled_candidates",
        "solver_backend": str(solver_backend),
        "policy_selector": str(policy_selector),
        "max_portfolio_pd_requested": float(max_portfolio_pd),
        "max_portfolio_pd_effective": float(effective_max_portfolio_pd),
        "robust_policy": robust_policy,
        "champion_policy_path": str(champion_policy_path),
        "candidate_universe_path": universe_source or str(candidate_universe_path),
        "gate_contract": {
            "gate": "no_regression",
            "significance_role": "diagnostic",
        },
        "diagnostics": {
            "p_value": float(comparison["p_value"]),
            "significant": bool(comparison["significant"]),
            "n_boot": int(n_boot),
            "seed": int(seed),
        },
    }
    diff_total_return = float(metrics_b["total_return"] - metrics_a["total_return"])
    tolerance_total_return = abs(float(metrics_a["total_return"])) * float(
        no_regression_tolerance_pct
    )
    no_regression_pass = bool(diff_total_return >= -tolerance_total_return)
    status["no_regression"] = {
        "diff_total_return": diff_total_return,
        "tolerance_total_return": tolerance_total_return,
        "tolerance_pct_of_control": float(no_regression_tolerance_pct),
        "passed": no_regression_pass,
    }
    status_out = _artifact_path(status_path)
    status_out.parent.mkdir(parents=True, exist_ok=True)
    with open(status_out, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)
    logger.info(f"Saved status: {status_out}")

    logger.info(
        f"A/B result: A(non-robust)={metrics_a['total_return']:,.2f}, "
        f"B(robust)={metrics_b['total_return']:,.2f}, "
        f"diff={comparison['diff']:,.2f}, p={comparison['p_value']:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B simulation: robust vs non-robust")
    parser.add_argument("--total_budget", type=float, default=1_000_000)
    parser.add_argument("--max_portfolio_pd", type=float, default=0.10)
    parser.add_argument("--max_candidates", type=int, default=5_000)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_regression_tolerance_pct", type=float, default=0.05)
    parser.add_argument(
        "--robust_policy_summary_path",
        default="data/processed/portfolio_robustness_summary.parquet",
    )
    parser.add_argument(
        "--champion_policy_path",
        default="models/champion_portfolio_policy.json",
    )
    parser.add_argument(
        "--candidate_universe_path",
        default="data/processed/champion_candidate_universe.parquet",
    )
    parser.add_argument("--results_path", default="data/processed/ab_simulation_results.parquet")
    parser.add_argument("--summary_path", default="data/processed/ab_simulation_summary.parquet")
    parser.add_argument("--status_path", default="models/ab_simulation_status.json")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--solver_backend", choices=["highs", "cuopt"], default="highs")
    parser.add_argument(
        "--policy_selector",
        choices=["promotion_first", "robustness_aware", "balanced_robustness"],
        default="promotion_first",
    )
    args = parser.parse_args()
    main(
        total_budget=args.total_budget,
        max_portfolio_pd=args.max_portfolio_pd,
        max_candidates=args.max_candidates,
        n_boot=args.n_boot,
        seed=args.seed,
        no_regression_tolerance_pct=args.no_regression_tolerance_pct,
        robust_policy_summary_path=args.robust_policy_summary_path,
        champion_policy_path=args.champion_policy_path,
        candidate_universe_path=args.candidate_universe_path,
        results_path=args.results_path,
        summary_path=args.summary_path,
        status_path=args.status_path,
        run_tag=args.run_tag,
        solver_backend=args.solver_backend,
        policy_selector=args.policy_selector,
    )
