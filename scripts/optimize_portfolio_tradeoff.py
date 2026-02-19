"""Portfolio robustness trade-off analysis.

Evaluates the explicit cost of robustness across:
- portfolio PD caps (risk tolerance grid)
- uncertainty aversion penalties in the objective

Usage:
    uv run python scripts/optimize_portfolio_tradeoff.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.models.conformal_artifacts import load_conformal_intervals
from src.optimization.portfolio_model import build_portfolio_model, solve_portfolio


def _parse_percent_series(series: pd.Series) -> np.ndarray:
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
    values = np.nan_to_num(values, nan=12.0)
    return values / 100.0


def _load_candidates() -> pd.DataFrame:
    fe_path = Path("data/processed/test_fe.parquet")
    raw_path = Path("data/processed/test.parquet")
    return pd.read_parquet(fe_path if fe_path.exists() else raw_path)


def _load_intervals() -> pd.DataFrame:
    intervals, path, is_legacy = load_conformal_intervals(allow_legacy_fallback=True)
    logger.info(
        f"Loaded conformal intervals from {path} (legacy={is_legacy}, rows={len(intervals):,})"
    )
    return intervals


def _resolve_interval_columns(intervals: pd.DataFrame) -> tuple[str, str, str]:
    col_point = "y_pred" if "y_pred" in intervals.columns else "pd_point"
    col_low = "pd_low_90" if "pd_low_90" in intervals.columns else "pd_low"
    col_high = "pd_high_90" if "pd_high_90" in intervals.columns else "pd_high"
    return col_point, col_low, col_high


def _align_loans_and_intervals(
    candidates: pd.DataFrame,
    intervals: pd.DataFrame,
    max_candidates: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align candidate loans with interval rows by id where possible."""
    if "id" in candidates.columns and "id" in intervals.columns:
        cand = candidates.copy()
        ints = intervals.copy()
        cand["_id_join"] = cand["id"].astype(str)
        ints["_id_join"] = ints["id"].astype(str)
        ints = ints.drop_duplicates(subset="_id_join", keep="first")
        merged = cand.merge(ints, on="_id_join", how="inner", suffixes=("", "_int"))
        if merged.empty:
            raise ValueError("ID-based merge between candidates and intervals returned zero rows.")

        n = min(len(merged), max_candidates)
        if len(merged) > n:
            idx = np.random.default_rng(random_state).choice(np.arange(len(merged)), size=n, replace=False)
            idx = np.sort(idx)
            merged = merged.iloc[idx].reset_index(drop=True)
        else:
            merged = merged.reset_index(drop=True)

        loans = merged[candidates.columns].copy()
        interval_cols = [c for c in intervals.columns if c in merged.columns]
        ints_aligned = merged[interval_cols].copy()
        logger.info(
            f"Aligned tradeoff candidates and intervals by id: n={len(loans):,} "
            f"(candidate_rows={len(candidates):,}, interval_rows={len(intervals):,})"
        )
        return loans, ints_aligned

    logger.warning(
        "Conformal interval artifact has no id alignment key; using positional fallback in tradeoff analysis."
    )
    n = min(len(candidates), len(intervals), max_candidates)
    idx = np.random.default_rng(random_state).choice(np.arange(n), size=n, replace=False)
    idx = np.sort(idx)
    loans = candidates.iloc[idx].reset_index(drop=True).copy()
    ints_aligned = intervals.iloc[idx].reset_index(drop=True).copy()
    return loans, ints_aligned


def _parse_float_grid(raw: str) -> list[float]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("Grid cannot be empty.")
    return sorted(set(vals))


def _solve_single(
    loans: pd.DataFrame,
    pd_point: np.ndarray,
    pd_low: np.ndarray,
    pd_high: np.ndarray,
    lgd: np.ndarray,
    int_rates: np.ndarray,
    total_budget: float,
    max_concentration: float,
    risk_tolerance: float,
    robust: bool,
    uncertainty_aversion: float,
    min_budget_utilization: float,
    pd_cap_slack_penalty: float,
    time_limit: int,
    threads: int,
) -> dict[str, float | int | str]:
    model = build_portfolio_model(
        loans=loans,
        pd_point=pd_point,
        pd_low=pd_low,
        pd_high=pd_high,
        lgd=lgd,
        int_rates=int_rates,
        total_budget=total_budget,
        max_concentration=max_concentration,
        max_portfolio_pd=risk_tolerance,
        robust=robust,
        uncertainty_aversion=uncertainty_aversion,
        min_budget_utilization=min_budget_utilization,
        pd_cap_slack_penalty=pd_cap_slack_penalty,
    )
    solution = solve_portfolio(model, time_limit=time_limit, threads=threads)

    n = len(loans)
    allocation = np.array([solution["allocation"][i] for i in range(n)], dtype=float)
    loan_amounts = (
        loans["loan_amnt"].to_numpy(dtype=float)
        if "loan_amnt" in loans.columns
        else np.ones(n) * 10_000
    )
    total_allocated = float(np.sum(allocation * loan_amounts))

    expected_loss = float(np.sum(allocation * loan_amounts * pd_point * lgd))
    worst_loss = float(np.sum(allocation * loan_amounts * pd_high * lgd))
    expected_return = float(np.sum(allocation * loan_amounts * int_rates))
    economic_return = expected_return - expected_loss
    uncertainty_cost = float(
        uncertainty_aversion
        * np.sum(allocation * loan_amounts * np.clip(pd_high - pd_point, 0.0, 1.0) * lgd)
    )
    worst_pd = float(np.sum(allocation * loan_amounts * pd_high) / (total_allocated + 1e-6))
    point_pd = float(np.sum(allocation * loan_amounts * pd_point) / (total_allocated + 1e-6))

    return {
        "solver_status": str(solution["solver_status"]),
        "objective_value": float(solution["objective_value"]),
        "n_funded": int(solution["n_funded"]),
        "total_allocated": total_allocated,
        "expected_return_gross": expected_return,
        "expected_loss_point": expected_loss,
        "expected_return_net_point": economic_return,
        "worst_case_loss": worst_loss,
        "uncertainty_penalty_cost": uncertainty_cost,
        "pd_cap_slack": float(solution.get("pd_cap_slack", 0.0)),
        "worst_case_pd": worst_pd,
        "point_pd": point_pd,
    }


def main(
    config_path: str = "configs/optimization.yaml",
    risk_grid: str = "0.06,0.08,0.10,0.12",
    aversion_grid: str = "0.0,0.5,1.0,2.0",
    max_candidates: int = 3000,
    random_state: int = 42,
    robust_min_budget_utilization: float = 0.05,
    strict_risk_threshold: float = 0.12,
    robust_pd_slack_penalty: float = 1.5,
):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    risk_values = _parse_float_grid(risk_grid)
    aversion_values = _parse_float_grid(aversion_grid)

    candidates = _load_candidates().reset_index(drop=True)
    intervals = _load_intervals().reset_index(drop=True)
    loans, ints = _align_loans_and_intervals(
        candidates=candidates,
        intervals=intervals,
        max_candidates=max_candidates,
        random_state=random_state,
    )
    n = len(loans)

    col_point, col_low, col_high = _resolve_interval_columns(ints)
    pd_point = ints[col_point].to_numpy(dtype=float)
    pd_low = ints[col_low].to_numpy(dtype=float)
    pd_high = ints[col_high].to_numpy(dtype=float)
    lgd = np.full(n, 0.45, dtype=float)
    int_rates = (
        _parse_percent_series(loans["int_rate"])
        if "int_rate" in loans.columns
        else np.full(n, 0.12)
    )

    rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    logger.info(
        f"Starting robustness trade-off optimization on n={n:,}, "
        f"risk_grid={risk_values}, aversion_grid={aversion_values}"
    )

    for risk_tol in risk_values:
        baseline = _solve_single(
            loans=loans,
            pd_point=pd_point,
            pd_low=pd_low,
            pd_high=pd_high,
            lgd=lgd,
            int_rates=int_rates,
            total_budget=float(config["portfolio"]["total_budget"]),
            max_concentration=float(config["portfolio"]["max_concentration"]),
            risk_tolerance=float(risk_tol),
            robust=False,
            uncertainty_aversion=0.0,
            min_budget_utilization=0.0,
            pd_cap_slack_penalty=0.0,
            time_limit=int(config["optimization"]["time_limit"]),
            threads=int(config["optimization"]["threads"]),
        )
        rows.append(
            {
                "policy": "nonrobust",
                "risk_tolerance": float(risk_tol),
                "uncertainty_aversion": 0.0,
                "min_budget_utilization": 0.0,
                "pd_cap_slack_penalty": 0.0,
                "price_of_robustness": 0.0,
                "price_of_robustness_pct": 0.0,
                **baseline,
            }
        )
        baseline_ret = float(baseline["expected_return_net_point"])

        robust_candidates = []
        for lam in aversion_values:
            enforce_floor = float(risk_tol) <= float(strict_risk_threshold)
            min_util = float(robust_min_budget_utilization) if enforce_floor else 0.0
            slack_penalty = float(robust_pd_slack_penalty) if enforce_floor else 0.0
            robust_run = _solve_single(
                loans=loans,
                pd_point=pd_point,
                pd_low=pd_low,
                pd_high=pd_high,
                lgd=lgd,
                int_rates=int_rates,
                total_budget=float(config["portfolio"]["total_budget"]),
                max_concentration=float(config["portfolio"]["max_concentration"]),
                risk_tolerance=float(risk_tol),
                robust=True,
                uncertainty_aversion=float(lam),
                min_budget_utilization=min_util,
                pd_cap_slack_penalty=slack_penalty,
                time_limit=int(config["optimization"]["time_limit"]),
                threads=int(config["optimization"]["threads"]),
            )
            robust_ret = float(robust_run["expected_return_net_point"])
            por = baseline_ret - robust_ret
            por_pct = por / (abs(baseline_ret) + 1e-6) * 100.0
            row = {
                "policy": "robust",
                "risk_tolerance": float(risk_tol),
                "uncertainty_aversion": float(lam),
                "min_budget_utilization": min_util,
                "pd_cap_slack_penalty": slack_penalty,
                "price_of_robustness": float(por),
                "price_of_robustness_pct": float(por_pct),
                **robust_run,
            }
            rows.append(row)
            robust_candidates.append(row)

        best_robust = sorted(
            robust_candidates,
            key=lambda r: (r["expected_return_net_point"], -r["worst_case_pd"]),
            reverse=True,
        )[0]
        summary_rows.append(
            {
                "risk_tolerance": float(risk_tol),
                "baseline_nonrobust_return": baseline_ret,
                "best_robust_return": float(best_robust["expected_return_net_point"]),
                "best_robust_lambda": float(best_robust["uncertainty_aversion"]),
                "best_robust_min_budget_utilization": float(best_robust["min_budget_utilization"]),
                "best_robust_pd_cap_slack_penalty": float(best_robust["pd_cap_slack_penalty"]),
                "best_robust_pd_cap_slack": float(best_robust["pd_cap_slack"]),
                "best_robust_worst_pd": float(best_robust["worst_case_pd"]),
                "best_robust_funded": int(best_robust["n_funded"]),
                "baseline_nonrobust_funded": int(baseline["n_funded"]),
                "price_of_robustness": float(best_robust["price_of_robustness"]),
                "price_of_robustness_pct": float(best_robust["price_of_robustness_pct"]),
            }
        )

    frontier = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows).sort_values("risk_tolerance")

    data_dir = Path("data/processed")
    model_dir = Path("models")
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    frontier_path = data_dir / "portfolio_robustness_frontier.parquet"
    summary_path = data_dir / "portfolio_robustness_summary.parquet"
    frontier.to_parquet(frontier_path, index=False)
    summary.to_parquet(summary_path, index=False)

    payload = {
        "risk_grid": risk_values,
        "aversion_grid": aversion_values,
        "n_candidates": int(n),
        "frontier_path": str(frontier_path),
        "summary_path": str(summary_path),
        "summary_rows": summary.to_dict(orient="records"),
    }
    with open(model_dir / "portfolio_robustness_results.pkl", "wb") as f:
        pickle.dump(payload, f)

    logger.info(f"Saved robustness frontier: {frontier_path} ({len(frontier):,} rows)")
    logger.info(f"Saved robustness summary: {summary_path} ({len(summary):,} rows)")
    logger.info("Best robust policy per risk tolerance:")
    logger.info(f"\n{summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/optimization.yaml")
    parser.add_argument("--risk_grid", default="0.06,0.08,0.10,0.12")
    parser.add_argument("--aversion_grid", default="0.0,0.5,1.0,2.0")
    parser.add_argument("--max_candidates", type=int, default=3000)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--robust_min_budget_utilization", type=float, default=0.05)
    parser.add_argument("--strict_risk_threshold", type=float, default=0.12)
    parser.add_argument("--robust_pd_slack_penalty", type=float, default=1.5)
    args = parser.parse_args()
    main(
        config_path=args.config,
        risk_grid=args.risk_grid,
        aversion_grid=args.aversion_grid,
        max_candidates=args.max_candidates,
        random_state=args.random_state,
        robust_min_budget_utilization=args.robust_min_budget_utilization,
        strict_risk_threshold=args.strict_risk_threshold,
        robust_pd_slack_penalty=args.robust_pd_slack_penalty,
    )
