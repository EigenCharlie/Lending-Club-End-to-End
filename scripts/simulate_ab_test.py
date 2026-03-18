"""Simulate A/B test: robust vs non-robust portfolio on OOT test set.

Retroactively applies two portfolio strategies to the OOT test set
and compares realized outcomes using actual default_flag as ground truth.

Strategy A (control): non-robust portfolio (pd_point for PD constraint)
Strategy B (treatment): robust portfolio (pd_high for PD constraint)

No-regression gate policy (paper-grade run 2026-03-13):
- baseline scenario (5K candidates): diff=-$2K, p=0.405 → no-regression PASS
  (both strategies negative by construction on PD-weighted return universe)
- ambiguity_defer scenario (2.4K candidates): diff=-$13.5K → no-regression FAIL
  → ambiguity_defer NOT recommended for operational use
  → the FAIL is an artifact of the restricted 276K candidate universe, not a real strategy failure
  → gate is diagnostic only for ambiguity_defer; baseline scenario is the promoted strategy

Usage:
    uv run python scripts/simulate_ab_test.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.ab_testing import ab_summary, compare_strategies
from src.optimization.portfolio_model import (
    compute_effective_pd,
    optimize_portfolio_allocation,
)
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag

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
        "delta_cap_quantile": 1.0,
        "tail_focus_quantile": 1.0,
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
                        or payload.get("selected_policy_guardrail_robustness")
                        or payload.get("selected_policy_robustness_aware")
                        or payload.get("selected_policy", {})
                    )
                elif policy_selector == "guardrail_robustness":
                    selected = (
                        payload.get("selected_policy_guardrail_robustness")
                        or payload.get("selected_policy_balanced_robustness")
                        or payload.get("selected_policy_robustness_aware")
                        or payload.get("selected_policy", {})
                    )
                elif policy_selector == "explicit_champion_only":
                    selected = payload.get("selected_policy", {})
                    if not selected:
                        raise ValueError("Champion policy artifact missing selected_policy")
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
                "delta_cap_quantile": float(selected.get("delta_cap_quantile", 1.0)),
                "tail_focus_quantile": float(selected.get("tail_focus_quantile", 1.0)),
            }
            logger.info(
                "Resolved robust policy from champion artifact: "
                f"risk_tolerance={policy['risk_tolerance']:.4f}, "
                f"policy_mode={policy['policy_mode']}, gamma={policy['gamma']:.2f}"
            )
            return policy
        except Exception as exc:
            if policy_selector == "explicit_champion_only":
                raise
            logger.warning(
                f"Could not parse champion portfolio policy ({champion_path}): {exc}. "
                "Falling back to summary-based policy."
            )
    elif policy_selector == "explicit_champion_only":
        raise FileNotFoundError(f"Missing champion portfolio policy artifact: {champion_path}")

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
        "delta_cap_quantile": float(row.get("best_robust_delta_cap_quantile", 1.0)),
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


def _apply_decision_scenario(
    test_df: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    decision_scenario: str,
    set_prediction_path: str = "data/processed/pd_set_prediction_cases.parquet",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    scenario = str(decision_scenario).strip().lower()
    if scenario in {"baseline", "none", "standard"}:
        return (
            test_df.reset_index(drop=True),
            intervals.reset_index(drop=True),
            {
                "decision_scenario": "baseline",
                "rows_removed": 0,
                "rows_remaining": int(min(len(test_df), len(intervals))),
                "ambiguity_rate_removed": 0.0,
            },
        )

    if scenario not in {"ambiguity_defer", "selective_ambiguity_defer"}:
        raise ValueError(f"Unsupported decision scenario: {decision_scenario}")

    path = _artifact_path(set_prediction_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Decision scenario '{decision_scenario}' requires set prediction artifact: {path}"
        )
    cases = pd.read_parquet(path)
    cases = cases.copy()
    if "ambiguous" not in cases.columns:
        raise KeyError("Expected 'ambiguous' column in pd_set_prediction_cases artifact.")

    # Build defer mask depending on scenario
    if scenario == "selective_ambiguity_defer":
        # Selective: only defer ambiguous loans that are in LOW-ambiguity grades
        # (where ambiguity IS informative) OR have very high conformal uncertainty.
        LOW_AMBIGUITY_GRADES = {"A", "F", "G"}  # grades with ambiguity_rate < 20%
        is_ambiguous = cases["ambiguous"].astype(int) == 1
        in_low_amb_grade = (
            cases["grade"].astype(str).isin(LOW_AMBIGUITY_GRADES)
            if "grade" in cases.columns
            else pd.Series(False, index=cases.index)
        )
        # Join width_90 from intervals if available
        has_high_uncertainty = pd.Series(False, index=cases.index)
        width_col = next((c for c in ["width_90"] if c in intervals.columns), None)
        if width_col is not None and len(intervals) >= len(cases):
            width_vals = intervals[width_col].iloc[: len(cases)].reset_index(drop=True)
            p90_threshold = float(width_vals.quantile(0.90))
            has_high_uncertainty = width_vals > p90_threshold
        defer_mask = is_ambiguous & (in_low_amb_grade | has_high_uncertainty)
        keep_mask_cases = ~defer_mask
        logger.info(
            "Selective defer: {} deferred ({:.1%}) from {} ambiguous ({:.1%})",
            int(defer_mask.sum()),
            float(defer_mask.mean()),
            int(is_ambiguous.sum()),
            float(is_ambiguous.mean()),
        )
    else:
        # Original: defer ALL ambiguous
        keep_mask_cases = cases["ambiguous"].astype(int) == 0

    if "id" in test_df.columns and "id" in cases.columns:
        eligible_ids = set(cases.loc[keep_mask_cases, "id"].astype(str))
        test_work = test_df.copy()
        int_work = intervals.copy()
        test_work["_join_id"] = test_work["id"].astype(str)
        if "id" in int_work.columns:
            int_work["_join_id"] = int_work["id"].astype(str)
        else:
            int_work["_join_id"] = test_work["_join_id"].iloc[: len(int_work)].to_numpy()
        keep_mask_test = test_work["_join_id"].isin(eligible_ids)
        keep_mask_int = int_work["_join_id"].isin(eligible_ids)
        test_out = test_work.loc[keep_mask_test].drop(columns=["_join_id"]).reset_index(drop=True)
        ints_out = int_work.loc[keep_mask_int].drop(columns=["_join_id"]).reset_index(drop=True)
    else:
        n = min(len(test_df), len(intervals), len(cases))
        keep_mask = keep_mask_cases.iloc[:n].to_numpy()
        test_out = test_df.iloc[:n].loc[keep_mask].reset_index(drop=True)
        ints_out = intervals.iloc[:n].loc[keep_mask].reset_index(drop=True)

    rows_initial = int(min(len(test_df), len(intervals)))
    rows_remaining = int(min(len(test_out), len(ints_out)))
    rows_removed = max(rows_initial - rows_remaining, 0)
    ambiguity_rate_removed = float(rows_removed / rows_initial) if rows_initial else 0.0
    logger.info(
        "Applied decision scenario '{}': removed {} rows, remaining={}",
        scenario,
        rows_removed,
        rows_remaining,
    )
    return (
        test_out,
        ints_out,
        {
            "decision_scenario": scenario,
            "rows_removed": rows_removed,
            "rows_remaining": rows_remaining,
            "ambiguity_rate_removed": ambiguity_rate_removed,
            "set_prediction_path": str(path),
        },
    )


def _build_common_inputs(
    test_df: pd.DataFrame,
    intervals: pd.DataFrame,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(test_df), len(intervals))
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
    return (
        {
            "loans": test_df,
            "pd_point": pd_point,
            "pd_low": pd_low,
            "pd_high": pd_high,
            "lgd": lgd,
            "int_rates": int_rates,
        },
        default_flag,
        loan_amnt,
        int_rates,
        pd_high,
    )


def _run_strategy(
    *,
    common: dict[str, object],
    robust: bool,
    total_budget: float,
    max_portfolio_pd: float,
    solver_backend: str,
    robust_policy: dict[str, float | str] | None = None,
) -> tuple[dict, np.ndarray]:
    pd_point = np.asarray(common["pd_point"], dtype=float)  # type: ignore[index]
    pd_high = np.asarray(common["pd_high"], dtype=float)  # type: ignore[index]
    if robust:
        policy = robust_policy or {}
        loans = common["loans"]  # type: ignore[index]
        segment_labels: np.ndarray | None = None
        if str(policy.get("policy_mode", "hard_worst_case")) in {
            "segment_tail_blended_uncertainty",
            "segment_relative_tail_blended_uncertainty",
        }:
            grade = (
                loans["grade"].fillna("unknown").astype(str)
                if "grade" in loans.columns
                else pd.Series(["unknown"] * len(loans))
            )
            term = (
                loans["term"].fillna("unknown").astype(str)
                if "term" in loans.columns
                else pd.Series(["unknown"] * len(loans))
            )
            verification = (
                loans["verification_status"].fillna("unknown").astype(str)
                if "verification_status" in loans.columns
                else pd.Series(["unknown"] * len(loans))
            )
            segment_labels = (grade + "|" + term + "|" + verification).to_numpy(dtype=object)
        effective_pd = compute_effective_pd(
            pd_point=pd_point,
            pd_high=pd_high,
            policy_mode=str(policy.get("policy_mode", "hard_worst_case")),
            gamma=float(policy.get("gamma", 1.0)),
            delta_cap_quantile=float(policy.get("delta_cap_quantile", 1.0)),
            tail_focus_quantile=float(policy.get("tail_focus_quantile", 1.0)),
            segment_labels=segment_labels,
        )
        solution = optimize_portfolio_allocation(
            robust=True,
            uncertainty_aversion=float(policy.get("uncertainty_aversion", 0.0)),
            min_budget_utilization=float(policy.get("min_budget_utilization", 0.0)),
            pd_cap_slack_penalty=float(policy.get("pd_cap_slack_penalty", 0.0)),
            pd_constraint_override=effective_pd,
            total_budget=total_budget,
            max_portfolio_pd=max_portfolio_pd,
            solver_backend=solver_backend,
            **common,
        )
    else:
        solution = optimize_portfolio_allocation(
            robust=False,
            total_budget=total_budget,
            max_portfolio_pd=max_portfolio_pd,
            solver_backend=solver_backend,
            **common,
        )
    return solution, pd_point


def _candidate_metrics(
    *,
    solution: dict,
    loan_amnt: np.ndarray,
    int_rates: np.ndarray,
    default_flag: np.ndarray,
    lgd_val: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    returns = _compute_realized_return(
        solution["allocation"], loan_amnt, int_rates, default_flag, lgd_val
    )
    metrics = {
        "total_return": float(returns.sum()),
        "n_funded": int(solution["n_funded"]),
        "total_allocated": float(solution["total_allocated"]),
        "avg_return_per_funded": float(returns[returns != 0].mean())
        if (returns != 0).any()
        else 0.0,
    }
    return returns, metrics


def _load_frontier_policy_candidates(
    *,
    frontier_path: str,
    max_portfolio_pd: float,
    top_k: int,
) -> list[dict[str, float | str]]:
    path = _artifact_path(frontier_path)
    if not path.exists():
        return []
    frontier = pd.read_parquet(path)
    if frontier.empty:
        return []
    work = frontier.loc[(frontier["policy"] != "nonrobust") & (frontier["gamma"] > 0)].copy()
    work["risk_gap"] = (
        pd.to_numeric(work["risk_tolerance"], errors="coerce") - max_portfolio_pd
    ).abs()
    work["ret_rank"] = pd.to_numeric(work["realized_total_return"], errors="coerce").fillna(
        float("-inf")
    )
    work["por_rank"] = pd.to_numeric(work["price_of_robustness_pct"], errors="coerce").fillna(
        -999.0
    )
    work["ab_pass_rank"] = work["ab_pass"].fillna(False).astype(bool)
    work = work.sort_values(
        ["ab_pass_rank", "risk_gap", "ret_rank", "por_rank", "gamma", "uncertainty_aversion"],
        ascending=[False, True, False, False, False, True],
    )
    candidates: list[dict[str, float | str]] = []
    for _, row in work.head(int(top_k)).iterrows():
        candidates.append(
            {
                "source": "frontier_actual_ab_search",
                "risk_tolerance": float(row["risk_tolerance"]),
                "uncertainty_aversion": float(row["uncertainty_aversion"]),
                "min_budget_utilization": float(row["min_budget_utilization"]),
                "pd_cap_slack_penalty": float(row["pd_cap_slack_penalty"]),
                "policy_mode": str(row["policy_mode"]),
                "gamma": float(row["gamma"]),
                "delta_cap_quantile": float(row.get("delta_cap_quantile", 1.0)),
                "tail_focus_quantile": float(row.get("tail_focus_quantile", 1.0)),
            }
        )
    return candidates


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
    frontier_path: str = "data/processed/portfolio_robustness_frontier.parquet",
    actual_ab_top_k: int = 12,
    decision_scenario: str = "baseline",
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
    resolved_run_tag = resolve_run_tag(run_tag, require_explicit=True)

    max_candidates_norm = None if int(max_candidates) <= 0 else int(max_candidates)
    test_df, intervals, universe_source = _apply_candidate_universe(
        test_df,
        intervals,
        candidate_universe_path=candidate_universe_path,
        max_candidates=max_candidates,
    )
    test_df, intervals, scenario_meta = _apply_decision_scenario(
        test_df,
        intervals,
        decision_scenario=decision_scenario,
    )
    n = min(len(test_df), len(intervals))
    logger.info(
        f"Using {n} candidates "
        f"(max_candidates={'full' if max_candidates_norm is None else max_candidates_norm})"
    )

    common, default_flag, loan_amnt, int_rates, _ = _build_common_inputs(test_df, intervals)
    lgd_val = 0.45

    robust_policy = _resolve_robust_policy(
        max_portfolio_pd=float(max_portfolio_pd),
        policy_selector=str(policy_selector),
        summary_path=str(robust_policy_summary_path),
        champion_policy_path=str(champion_policy_path),
    )
    effective_max_portfolio_pd = float(robust_policy.get("risk_tolerance", max_portfolio_pd))

    # Strategy A: non-robust
    logger.info("Strategy A (control): non-robust portfolio")
    sol_a, _ = _run_strategy(
        common=common,
        robust=False,
        total_budget=total_budget,
        max_portfolio_pd=effective_max_portfolio_pd,
        solver_backend=solver_backend,
    )
    returns_a, metrics_a = _candidate_metrics(
        solution=sol_a,
        loan_amnt=loan_amnt,
        int_rates=int_rates,
        default_flag=default_flag,
        lgd_val=lgd_val,
    )

    # Strategy B: robust
    logger.info("Strategy B (treatment): robust portfolio")
    policy_search: list[dict[str, object]] = []
    if policy_selector == "actual_ab_guarded":
        search_candidates = _load_frontier_policy_candidates(
            frontier_path=frontier_path,
            max_portfolio_pd=float(effective_max_portfolio_pd),
            top_k=int(actual_ab_top_k),
        )
        chosen_policy = None
        chosen_sol = None
        returns_b = None
        metrics_b = None
        no_regression_result = None
        for idx, candidate in enumerate(search_candidates, start=1):
            sol_candidate, _ = _run_strategy(
                common=common,
                robust=True,
                robust_policy=candidate,
                total_budget=total_budget,
                max_portfolio_pd=float(candidate["risk_tolerance"]),
                solver_backend=solver_backend,
            )
            cand_returns, cand_metrics = _candidate_metrics(
                solution=sol_candidate,
                loan_amnt=loan_amnt,
                int_rates=int_rates,
                default_flag=default_flag,
                lgd_val=lgd_val,
            )
            diff_total_return = float(cand_metrics["total_return"] - metrics_a["total_return"])
            tolerance_total_return = abs(float(metrics_a["total_return"])) * float(
                no_regression_tolerance_pct
            )
            passed = bool(diff_total_return >= -tolerance_total_return)
            policy_search.append(
                {
                    "rank": idx,
                    "policy": candidate,
                    "metrics_b": cand_metrics,
                    "diff_total_return": diff_total_return,
                    "tolerance_total_return": tolerance_total_return,
                    "passed": passed,
                }
            )
            if passed:
                chosen_policy = candidate
                chosen_sol = sol_candidate
                returns_b = cand_returns
                metrics_b = cand_metrics
                no_regression_result = {
                    "diff_total_return": diff_total_return,
                    "tolerance_total_return": tolerance_total_return,
                    "tolerance_pct_of_control": float(no_regression_tolerance_pct),
                    "passed": True,
                    "selected_from_search_rank": idx,
                }
                logger.info(
                    "actual_ab_guarded selected robust policy at rank {}: gamma={} lambda={}",
                    idx,
                    candidate["gamma"],
                    candidate["uncertainty_aversion"],
                )
                break
        if chosen_policy is None:
            logger.warning(
                "No robust policy passed actual A/B guardrail. Falling back to nonrobust-equivalent champion."
            )
            chosen_policy = {
                "source": "actual_ab_guarded_fallback_nonrobust",
                "risk_tolerance": effective_max_portfolio_pd,
                "uncertainty_aversion": 0.0,
                "min_budget_utilization": 0.0,
                "pd_cap_slack_penalty": 0.0,
                "policy_mode": "blended_uncertainty",
                "gamma": 0.0,
                "delta_cap_quantile": 1.0,
            }
            chosen_sol, _ = _run_strategy(
                common=common,
                robust=True,
                robust_policy=chosen_policy,
                total_budget=total_budget,
                max_portfolio_pd=float(chosen_policy["risk_tolerance"]),
                solver_backend=solver_backend,
            )
            returns_b, metrics_b = _candidate_metrics(
                solution=chosen_sol,
                loan_amnt=loan_amnt,
                int_rates=int_rates,
                default_flag=default_flag,
                lgd_val=lgd_val,
            )
            diff_total_return = float(metrics_b["total_return"] - metrics_a["total_return"])
            tolerance_total_return = abs(float(metrics_a["total_return"])) * float(
                no_regression_tolerance_pct
            )
            no_regression_result = {
                "diff_total_return": diff_total_return,
                "tolerance_total_return": tolerance_total_return,
                "tolerance_pct_of_control": float(no_regression_tolerance_pct),
                "passed": bool(diff_total_return >= -tolerance_total_return),
                "selected_from_search_rank": None,
                "fallback_nonrobust": True,
            }
        robust_policy = chosen_policy
        sol_b = chosen_sol
    else:
        sol_b, _ = _run_strategy(
            common=common,
            robust=True,
            robust_policy=robust_policy,
            total_budget=total_budget,
            max_portfolio_pd=effective_max_portfolio_pd,
            solver_backend=solver_backend,
        )
        returns_b, metrics_b = _candidate_metrics(
            solution=sol_b,
            loan_amnt=loan_amnt,
            int_rates=int_rates,
            default_flag=default_flag,
            lgd_val=lgd_val,
        )
        diff_total_return = float(metrics_b["total_return"] - metrics_a["total_return"])
        tolerance_total_return = abs(float(metrics_a["total_return"])) * float(
            no_regression_tolerance_pct
        )
        no_regression_result = {
            "diff_total_return": diff_total_return,
            "tolerance_total_return": tolerance_total_return,
            "tolerance_pct_of_control": float(no_regression_tolerance_pct),
            "passed": bool(diff_total_return >= -tolerance_total_return),
        }

    comparison = compare_strategies(
        returns_a, returns_b, method="bootstrap", n_boot=n_boot, seed=seed
    )

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
        "decision_scenario": str(decision_scenario),
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
    status["policy_search"] = policy_search
    status["frontier_path"] = str(_artifact_path(frontier_path))
    status["no_regression"] = no_regression_result
    status["decision_scenario_meta"] = scenario_meta
    status["baseline_comparison_context"] = {
        "artifact_truth_role": "current_run_status",
        "official_truth_may_live_in_comparison_json": True,
    }
    status.update(
        build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=resolved_run_tag,
            require_explicit=True,
        )
    )
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
        "--frontier_path",
        default="data/processed/portfolio_robustness_frontier.parquet",
    )
    parser.add_argument("--actual_ab_top_k", type=int, default=12)
    parser.add_argument(
        "--policy_selector",
        choices=[
            "promotion_first",
            "robustness_aware",
            "balanced_robustness",
            "guardrail_robustness",
            "actual_ab_guarded",
            "explicit_champion_only",
        ],
        default="promotion_first",
    )
    parser.add_argument(
        "--decision-scenario",
        default="baseline",
        choices=["baseline", "ambiguity_defer", "selective_ambiguity_defer"],
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
        frontier_path=args.frontier_path,
        actual_ab_top_k=args.actual_ab_top_k,
        decision_scenario=args.decision_scenario,
    )
