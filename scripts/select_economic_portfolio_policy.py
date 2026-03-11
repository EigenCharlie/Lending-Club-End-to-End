"""Select canonical portfolio policy using actual A/B economics on the real universe."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from scripts.optimize_portfolio_tradeoff import _allocation_similarity
from scripts.simulate_ab_test import (
    _apply_candidate_universe,
    _build_common_inputs,
    _candidate_metrics,
    _run_strategy,
)

SCHEMA_VERSION = "2026-03-10.1"


def _artifact_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    root = str(os.environ.get("GPU_REPLAY_ARTIFACT_ROOT", "")).strip()
    return (Path(root) / path) if root else path


def _policy_key(row: pd.Series) -> tuple[object, ...]:
    return (
        str(row.get("policy_mode", "")),
        float(row.get("gamma", 0.0)),
        float(row.get("risk_tolerance", 0.0)),
        float(row.get("delta_cap_quantile", 1.0)),
        float(row.get("uncertainty_aversion", 0.0)),
        float(row.get("min_budget_utilization", 0.0)),
        float(row.get("pd_cap_slack_penalty", 0.0)),
    )


def _policy_from_row(row: pd.Series, source: str) -> dict[str, float | str]:
    return {
        "source": source,
        "risk_tolerance": float(row["risk_tolerance"]),
        "uncertainty_aversion": float(row["uncertainty_aversion"]),
        "min_budget_utilization": float(row["min_budget_utilization"]),
        "pd_cap_slack_penalty": float(row["pd_cap_slack_penalty"]),
        "policy_mode": str(row["policy_mode"]),
        "gamma": float(row["gamma"]),
        "delta_cap_quantile": float(row.get("delta_cap_quantile", 1.0)),
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe_candidates(rows: list[pd.Series]) -> list[pd.Series]:
    out: list[pd.Series] = []
    seen: set[tuple[object, ...]] = set()
    for row in rows:
        key = _policy_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _select_candidate_rows(frontier: pd.DataFrame, top_k: int) -> list[pd.Series]:
    work = frontier.copy()
    work = work.loc[
        (work["policy"] != "nonrobust")
        & work["eligible_for_canonical_selection"].fillna(False).astype(bool)
    ].copy()
    if work.empty:
        return []

    work["realized_total_return"] = pd.to_numeric(work["realized_total_return"], errors="coerce")
    candidates: list[pd.Series] = []
    bucket_cols = ["policy_mode", "gamma", "risk_tolerance", "delta_cap_quantile"]
    for _, bucket in work.groupby(bucket_cols, dropna=False):
        top = bucket.sort_values("realized_total_return", ascending=False).head(int(top_k))
        candidates.extend(top.to_dict(orient="records"))

    flag_cols = [
        "selected_for_champion",
        "selected_for_balanced_robustness",
        "selected_for_guardrail_robustness",
    ]
    for col in flag_cols:
        if col in work.columns:
            flagged = work.loc[work[col].fillna(False).astype(bool)]
            candidates.extend(flagged.to_dict(orient="records"))
    return _dedupe_candidates([pd.Series(r) for r in candidates])


def _control_metrics_by_risk(
    *,
    common: dict[str, object],
    default_flag: np.ndarray,
    loan_amnt: np.ndarray,
    int_rates: np.ndarray,
    risk_values: list[float],
    total_budget: float,
    solver_backend: str,
) -> dict[float, dict[str, object]]:
    controls: dict[float, dict[str, object]] = {}
    for risk_tol in sorted({float(x) for x in risk_values}):
        sol, _ = _run_strategy(
            common=common,
            robust=False,
            total_budget=total_budget,
            max_portfolio_pd=risk_tol,
            solver_backend=solver_backend,
        )
        returns, metrics = _candidate_metrics(
            solution=sol,
            loan_amnt=loan_amnt,
            int_rates=int_rates,
            default_flag=default_flag,
            lgd_val=0.45,
        )
        alloc = np.array([sol["allocation"][i] for i in range(len(loan_amnt))], dtype=float)
        controls[risk_tol] = {
            "solution": sol,
            "returns": returns,
            "metrics": metrics,
            "allocation": alloc,
            "worst_case_pd": float(
                np.sum(
                    alloc * loan_amnt * np.asarray(common["pd_high"], dtype=float)  # type: ignore[index]
                )
                / (float(sol["total_allocated"]) + 1e-6)
            ),
        }
    return controls


def main(
    config_path: str = "configs/optimization.yaml",
    frontier_path: str = "data/processed/portfolio_robustness_frontier.parquet",
    research_policy_path: str = "models/portfolio_research_policy.json",
    champion_policy_path: str = "models/champion_portfolio_policy.json",
    status_path: str = "models/champion_policy_selection_status.json",
    candidate_universe_path: str = "data/processed/champion_candidate_universe.parquet",
    run_tag: str | None = None,
    solver_backend: str = "highs",
) -> None:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    selection_cfg = dict(config.get("portfolio_selection", {}) or {})
    top_k = int(selection_cfg.get("actual_ab_top_k", 20))
    min_funded_ratio = float(selection_cfg.get("min_funded_ratio", 0.95))
    max_por_pct = float(selection_cfg.get("max_price_of_robustness_pct", -15.0))
    canonical_modes = {
        str(x) for x in selection_cfg.get("canonical_policy_modes", ["blended_uncertainty"])
    }

    frontier = pd.read_parquet(_artifact_path(frontier_path))
    if frontier.empty:
        raise ValueError("portfolio_robustness_frontier.parquet is empty")

    test_df = pd.read_parquet("data/processed/test_fe.parquet")
    intervals = pd.read_parquet("data/processed/conformal_intervals_mondrian.parquet")
    test_df, intervals, universe_source = _apply_candidate_universe(
        test_df,
        intervals,
        candidate_universe_path=candidate_universe_path,
        max_candidates=0,
    )
    common, default_flag, loan_amnt, int_rates, pd_high = _build_common_inputs(test_df, intervals)
    total_budget = float(config["portfolio"]["total_budget"])
    controls = _control_metrics_by_risk(
        common=common,
        default_flag=default_flag,
        loan_amnt=loan_amnt,
        int_rates=int_rates,
        risk_values=frontier["risk_tolerance"].tolist(),
        total_budget=total_budget,
        solver_backend=solver_backend,
    )

    candidate_rows = _select_candidate_rows(frontier, top_k=top_k)
    if not candidate_rows:
        raise ValueError("No eligible canonical candidates found in frontier")

    evaluated: list[dict[str, object]] = []
    for row in candidate_rows:
        policy = _policy_from_row(row, source="economic_actual_ab_v1")
        risk_tol = float(policy["risk_tolerance"])
        control = controls[risk_tol]
        sol_b, _ = _run_strategy(
            common=common,
            robust=True,
            robust_policy=policy,
            total_budget=total_budget,
            max_portfolio_pd=risk_tol,
            solver_backend=solver_backend,
        )
        _, metrics_b = _candidate_metrics(
            solution=sol_b,
            loan_amnt=loan_amnt,
            int_rates=int_rates,
            default_flag=default_flag,
            lgd_val=0.45,
        )
        control_metrics = control["metrics"]
        diff_total_return = float(
            metrics_b["total_return"] - float(control_metrics["total_return"])
        )
        tolerance_total_return = abs(float(control_metrics["total_return"])) * 0.05
        passed_no_regression = bool(diff_total_return >= -tolerance_total_return)
        funded_ratio = float(metrics_b["n_funded"] / max(float(control_metrics["n_funded"]), 1.0))
        alloc_b = np.array([sol_b["allocation"][i] for i in range(len(loan_amnt))], dtype=float)
        cand_worst_pd = float(
            np.sum(alloc_b * loan_amnt * pd_high) / (float(sol_b["total_allocated"]) + 1e-6)
        )
        evaluated.append(
            {
                "policy": policy,
                "risk_tolerance": risk_tol,
                "passed_no_regression": passed_no_regression,
                "diff_total_return": diff_total_return,
                "tolerance_total_return": tolerance_total_return,
                "funded_ratio": funded_ratio,
                "worst_case_pd_reduction_bps": float(
                    (float(control["worst_case_pd"]) - cand_worst_pd) * 1e4
                ),
                "price_of_robustness_pct": float(row.get("price_of_robustness_pct", 0.0)),
                "return_per_funded_delta": float(
                    metrics_b["avg_return_per_funded"]
                    - float(control_metrics["avg_return_per_funded"])
                ),
                "allocation_similarity": _allocation_similarity(control["allocation"], alloc_b),
                "n_funded_candidate": int(metrics_b["n_funded"]),
                "n_funded_control": int(control_metrics["n_funded"]),
                "total_return_candidate": float(metrics_b["total_return"]),
                "total_return_control": float(control_metrics["total_return"]),
                "eligible_hard_filters": False,
            }
        )

    for item in evaluated:
        item["eligible_hard_filters"] = bool(
            item["passed_no_regression"]
            and float(item["funded_ratio"]) >= min_funded_ratio
            and float(item["price_of_robustness_pct"]) >= max_por_pct
            and str(item["policy"]["policy_mode"]) in canonical_modes
        )

    eligible = [x for x in evaluated if bool(x["eligible_hard_filters"])]
    robust_eligible = [x for x in eligible if float(x["policy"]["gamma"]) > 0.0]
    candidate_pool = robust_eligible if robust_eligible else eligible

    fallback_applied = False
    fallback_reason = None
    selector_outcome = "robust_selected"
    if candidate_pool:
        selected = sorted(
            candidate_pool,
            key=lambda x: (
                float(x["worst_case_pd_reduction_bps"]),
                float(x["diff_total_return"]),
                -abs(float(x["price_of_robustness_pct"])),
                float(x["funded_ratio"]),
            ),
            reverse=True,
        )[0]
        if float(selected["policy"]["gamma"]) <= 0.0:
            selector_outcome = "fallback_nonrobust"
            fallback_applied = True
            fallback_reason = "no_economically_viable_robust_policy"
    else:
        fallback_applied = True
        selector_outcome = "fallback_nonrobust"
        fallback_reason = "no_economically_viable_robust_policy"
        fallback_row = frontier.loc[frontier["selected_for_champion"].fillna(False).astype(bool)]
        selected_row = fallback_row.iloc[0] if not fallback_row.empty else frontier.iloc[0]
        selected = {
            "policy": {
                **_policy_from_row(selected_row, source="economic_actual_ab_v1_fallback"),
                "gamma": 0.0,
                "policy_mode": "blended_uncertainty",
                "delta_cap_quantile": 1.0,
                "uncertainty_aversion": 0.0,
                "min_budget_utilization": 0.0,
                "pd_cap_slack_penalty": 0.0,
            },
            "risk_tolerance": float(selected_row["risk_tolerance"]),
            "passed_no_regression": True,
            "diff_total_return": 0.0,
            "tolerance_total_return": abs(
                float(controls[float(selected_row["risk_tolerance"])]["metrics"]["total_return"])
            )
            * 0.05,
            "funded_ratio": 1.0,
            "worst_case_pd_reduction_bps": 0.0,
            "price_of_robustness_pct": 0.0,
            "return_per_funded_delta": 0.0,
            "allocation_similarity": 1.0,
            "n_funded_candidate": int(
                controls[float(selected_row["risk_tolerance"])]["metrics"]["n_funded"]
            ),
            "n_funded_control": int(
                controls[float(selected_row["risk_tolerance"])]["metrics"]["n_funded"]
            ),
            "total_return_candidate": float(
                controls[float(selected_row["risk_tolerance"])]["metrics"]["total_return"]
            ),
            "total_return_control": float(
                controls[float(selected_row["risk_tolerance"])]["metrics"]["total_return"]
            ),
            "eligible_hard_filters": False,
        }

    resolved_run_tag = (
        str(run_tag or "").strip()
        or str(os.environ.get("PIPELINE_RUN_TAG", "")).strip()
        or "untracked"
    )
    research_policy = _load_json(_artifact_path(research_policy_path))
    champion_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_tag": resolved_run_tag,
        "selection_stage": "economic_actual_ab_v1",
        "selection_universe_path": universe_source or str(_artifact_path(candidate_universe_path)),
        "selection_outcome": selector_outcome,
        "selected_policy": selected["policy"],
        "economic_metrics": {
            "diff_total_return": float(selected["diff_total_return"]),
            "passed_no_regression": bool(selected["passed_no_regression"]),
            "funded_ratio": float(selected["funded_ratio"]),
            "return_per_funded_delta": float(selected["return_per_funded_delta"]),
        },
        "robustness_metrics": {
            "worst_case_pd_reduction_bps": float(selected["worst_case_pd_reduction_bps"]),
            "price_of_robustness_pct": float(selected["price_of_robustness_pct"]),
            "allocation_similarity": float(selected["allocation_similarity"]),
        },
        "research_alternatives": {
            "promotion_first": research_policy.get("selected_policy"),
            "robustness_aware": research_policy.get("selected_policy_robustness_aware"),
            "balanced_robustness": research_policy.get("selected_policy_balanced_robustness"),
            "guardrail_robustness": research_policy.get("selected_policy_guardrail_robustness"),
        },
    }
    status_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_tag": resolved_run_tag,
        "selector_name": "economic_actual_ab_v1",
        "universe_path": universe_source or str(_artifact_path(candidate_universe_path)),
        "control_metrics": {str(k): dict(v["metrics"]) for k, v in controls.items()},
        "evaluated_candidates": evaluated,
        "selected_candidate": selected,
        "selector_outcome": selector_outcome,
        "fallback_applied": fallback_applied,
        "fallback_reason": fallback_reason,
    }

    champion_out = _artifact_path(champion_policy_path)
    status_out = _artifact_path(status_path)
    champion_out.parent.mkdir(parents=True, exist_ok=True)
    status_out.parent.mkdir(parents=True, exist_ok=True)
    champion_out.write_text(json.dumps(champion_payload, indent=2), encoding="utf-8")
    status_out.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
    logger.info("Saved champion portfolio policy: {}", champion_out)
    logger.info("Saved champion policy selection status: {}", status_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/optimization.yaml")
    parser.add_argument(
        "--frontier_path", default="data/processed/portfolio_robustness_frontier.parquet"
    )
    parser.add_argument("--research_policy_path", default="models/portfolio_research_policy.json")
    parser.add_argument("--champion_policy_path", default="models/champion_portfolio_policy.json")
    parser.add_argument("--status_path", default="models/champion_policy_selection_status.json")
    parser.add_argument(
        "--candidate_universe_path", default="data/processed/champion_candidate_universe.parquet"
    )
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--solver_backend", choices=["highs", "cuopt"], default="highs")
    args = parser.parse_args()
    main(
        config_path=args.config,
        frontier_path=args.frontier_path,
        research_policy_path=args.research_policy_path,
        champion_policy_path=args.champion_policy_path,
        status_path=args.status_path,
        candidate_universe_path=args.candidate_universe_path,
        run_tag=args.run_tag,
        solver_backend=args.solver_backend,
    )
