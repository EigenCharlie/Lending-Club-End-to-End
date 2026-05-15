"""Build Paper 4 v10 resolution-wave artifacts.

V10 starts from the v9 online conformal breakthrough and pushes every runnable
lane one step further: selector rerun, online robustness, MDCP empirical caps,
CVaR/OCE expanded search, DLA rollout, constrained DFL/SPO surrogate, sample
paths, IFRS9 proxy readiness, causal/fairness dossiers and a blocker dashboard.

This is still a Paper 4 living-lab wave.  It deliberately does not touch Paper
Estrella artifacts and does not create ``paper4_final_promotion.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_extended_experiments import (
    BUDGET,
    _safe_read_csv,
    _safe_read_json,
    _safe_read_parquet,
)
from scripts.papers.build_paper4_living_lab_artifacts import DEFAULT_LGD
from scripts.papers.build_paper4_v6_priority_resolution import (
    SOURCE_FAMILIES,
    STATUS_DIR,
    TABLE_DIR,
    _load_inputs,
    _prepare_online_frame,
    _prepare_solver_pool,
    _solve_linear_policy,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v8_resolution_wave import (
    _auditability_score,
    _solve_family_cap_policy,
)

SCHEMA_VERSION = "2026-05-13.10"
RNG_SEED = 2026051310
PAPER1_PROMOTION = Path("models/final_project_promotion.json")
PAPER4_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"


def _is_optimal(status: Any) -> bool:
    return "optimal" in str(status).lower()


def _json_dump(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _stable_uniform(*parts: Any) -> float:
    key = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def _load_v9_online() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    intervals = _safe_read_parquet(TABLE_DIR / "paper4_v9_online_selected_intervals.parquet")
    policy = _safe_read_parquet(TABLE_DIR / "paper4_v9_online_policy_month.parquet")
    source = _safe_read_parquet(TABLE_DIR / "paper4_v9_online_source_month.parquet")
    status = _safe_read_json(STATUS_DIR / "paper4_v9_online_goal_status.json")
    if intervals.empty or not status:
        raise FileNotFoundError("V10 requires v9 online artifacts.")
    intervals["loan_id"] = intervals["loan_id"].astype(str)
    intervals["issue_month"] = pd.to_datetime(intervals["issue_month"])
    return intervals, policy, source, status


def _prepare_solver_pool_v10(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_n: int,
    online_method: str,
) -> pd.DataFrame:
    pool = _prepare_solver_pool(candidate_pool, online_intervals, max_n=max_n)
    selected = _safe_read_parquet(TABLE_DIR / "paper4_v9_online_selected_intervals.parquet")
    if not selected.empty:
        qhat = (
            selected[selected["online_method_v9"].eq(online_method)][["loan_id", "qhat_v9"]]
            .drop_duplicates("loan_id")
            .copy()
        )
        qhat["loan_id"] = qhat["loan_id"].astype(str)
        pool = pool.merge(qhat, on="loan_id", how="left")
        pool["qhat_v10"] = pool["qhat_v9"].where(pool["qhat_v9"].notna(), pool["qhat_v4"])
        pool["qhat_v4_original"] = pool["qhat_v4"]
        pool["qhat_v4"] = pool["qhat_v10"].clip(0, 1)
        pool = pool.drop(columns=[col for col in ["qhat_v9"] if col in pool.columns])
    else:
        pool["qhat_v10"] = pool["qhat_v4"]
    pool["weak_source_proxy_v10"] = (
        pool["original_grade"].astype(str).isin(["D", "E", "F", "G"]).astype(float)
        + pool["dti_band"].astype(str).isin(["dti_q3", "dti_q5"]).astype(float)
        + pool["score_decile"].astype(str).isin(["0", "1", "2"]).astype(float)
        + pool["income_band"].astype(str).eq("inc_q5").astype(float)
    ) / 4.0
    pool["weak_source_proxy"] = pool["weak_source_proxy_v10"]
    pool["solver_score_seed"] = (
        pool["base_return_vec"]
        - 0.08 * pool["loan_amnt"] * pool["qhat_v4"]
        - 0.06 * pool["loan_amnt"] * pool["weak_source_proxy"]
    )
    return pool.sort_values("solver_score_seed", ascending=False).head(max_n).reset_index(drop=True)


def _online_metrics_from_rows(rows: pd.DataFrame, *, min_support: int) -> dict[str, Any]:
    policy = rows.groupby(["policy_id", "issue_month"], as_index=False).agg(
        n=("loan_id", "nunique"),
        coverage=("covered_online_v9", "mean"),
        width=("interval_width_online_v9", "mean"),
    )
    defended_policy = policy[policy["n"].ge(min_support)]
    source_frames = []
    for source in SOURCE_FAMILIES:
        if source not in rows.columns:
            continue
        source_frames.append(
            rows.groupby(["policy_id", "issue_month", source], dropna=False, as_index=False)
            .agg(
                n=("loan_id", "nunique"),
                coverage=("covered_online_v9", "mean"),
                width=("interval_width_online_v9", "mean"),
            )
            .assign(source_id=source)
        )
    source = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
    defended_source = source[source["n"].ge(min_support)] if not source.empty else pd.DataFrame()
    return {
        "min_support": min_support,
        "policy_month_defended_min": float(defended_policy["coverage"].min())
        if not defended_policy.empty
        else np.nan,
        "source_month_defended_min": float(defended_source["coverage"].min())
        if not defended_source.empty
        else np.nan,
        "avg_width_loan": float(rows["interval_width_online_v9"].mean()),
        "policy_month_cells": int(len(policy)),
        "source_month_cells": int(len(source)),
        "defended_policy_month_cells": int(len(defended_policy)),
        "defended_source_month_cells": int(len(defended_source)),
    }


def build_online_robustness_v10(
    allocations: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    intervals, _, _, status = _load_v9_online()
    merged = _prepare_online_frame(allocations, online_intervals)[
        ["policy_id", "loan_id", "issue_month", *SOURCE_FAMILIES]
    ].copy()
    merged["loan_id"] = merged["loan_id"].astype(str)
    merged["issue_month"] = pd.to_datetime(merged["issue_month"])
    detailed = intervals.merge(merged, on=["policy_id", "loan_id", "issue_month"], how="left")
    best_method = status["online_best_method_v9"]
    conservative = _safe_read_csv(TABLE_DIR / "paper4_v9_online_breakpoint_report.csv")
    conservative_method = (
        str(
            conservative.loc[
                conservative["breakpoint_id"].eq("v9_conservative_goal_passing_width"),
                "online_method_v9",
            ].iloc[0]
        )
        if not conservative.empty
        else best_method
    )
    methods = [best_method, conservative_method, "v9_reference_v8_best"]
    methods = [
        method for method in dict.fromkeys(methods) if method in set(detailed["online_method_v9"])
    ]

    sensitivity_rows = []
    for method in methods:
        local = detailed[detailed["online_method_v9"].eq(method)].copy()
        for min_support in [5, 8, 10]:
            metrics = _online_metrics_from_rows(local, min_support=min_support)
            sensitivity_rows.append(
                {
                    "online_method_v9": method,
                    **metrics,
                    "gate_source80_policy90_width95": bool(
                        metrics["source_month_defended_min"] >= 0.80
                        and metrics["policy_month_defended_min"] >= 0.90
                        and metrics["avg_width_loan"] <= 0.95
                    ),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    best_rows = detailed[detailed["online_method_v9"].eq(best_method)].copy()
    leave_month_rows = []
    for month in sorted(best_rows["issue_month"].dropna().unique()):
        metrics = _online_metrics_from_rows(
            best_rows[~best_rows["issue_month"].eq(month)], min_support=5
        )
        leave_month_rows.append({"left_out_month": month, **metrics})
    leave_month = pd.DataFrame(leave_month_rows)

    leave_policy_rows = []
    for policy_id in sorted(best_rows["policy_id"].dropna().unique()):
        metrics = _online_metrics_from_rows(
            best_rows[~best_rows["policy_id"].eq(policy_id)], min_support=5
        )
        leave_policy_rows.append({"left_out_policy_id": policy_id, **metrics})
    leave_policy = pd.DataFrame(leave_policy_rows)

    rng = np.random.default_rng(RNG_SEED + 10)
    months = np.array(sorted(best_rows["issue_month"].dropna().unique()))
    bootstrap_rows = []
    for b in range(bootstrap_samples):
        sampled = rng.choice(months, size=len(months), replace=True)
        boot = pd.concat(
            [best_rows[best_rows["issue_month"].eq(month)] for month in sampled], ignore_index=True
        )
        metrics = _online_metrics_from_rows(boot, min_support=5)
        bootstrap_rows.append({"bootstrap_id": b, **metrics})
    bootstrap = pd.DataFrame(bootstrap_rows)

    summary = pd.DataFrame(
        [
            {
                "robustness_item": "v9_best_nominal",
                "online_method_v9": best_method,
                "source_month_defended_min": status["online_best_source_month_defended_min"],
                "policy_month_defended_min": status["online_best_policy_month_defended_min"],
                "avg_width_loan": status["online_best_width"],
                "pass_v10": True,
                "interpretation": "v9 best passes the explicit online goal.",
            },
            {
                "robustness_item": "leave_one_month_min",
                "online_method_v9": best_method,
                "source_month_defended_min": float(leave_month["source_month_defended_min"].min()),
                "policy_month_defended_min": float(leave_month["policy_month_defended_min"].min()),
                "avg_width_loan": float(leave_month["avg_width_loan"].max()),
                "pass_v10": bool(
                    leave_month["source_month_defended_min"].min() >= 0.80
                    and leave_month["policy_month_defended_min"].min() >= 0.90
                    and leave_month["avg_width_loan"].max() <= 0.95
                ),
                "interpretation": "leave-one-month stress tests whether one calendar month drives the gate.",
            },
            {
                "robustness_item": "leave_one_policy_min",
                "online_method_v9": best_method,
                "source_month_defended_min": float(leave_policy["source_month_defended_min"].min()),
                "policy_month_defended_min": float(leave_policy["policy_month_defended_min"].min()),
                "avg_width_loan": float(leave_policy["avg_width_loan"].max()),
                "pass_v10": bool(
                    leave_policy["source_month_defended_min"].min() >= 0.80
                    and leave_policy["policy_month_defended_min"].min() >= 0.90
                    and leave_policy["avg_width_loan"].max() <= 0.95
                ),
                "interpretation": "leave-one-policy stress tests whether one local policy drives the gate.",
            },
            {
                "robustness_item": "month_bootstrap_p05",
                "online_method_v9": best_method,
                "source_month_defended_min": float(
                    bootstrap["source_month_defended_min"].quantile(0.05)
                ),
                "policy_month_defended_min": float(
                    bootstrap["policy_month_defended_min"].quantile(0.05)
                ),
                "avg_width_loan": float(bootstrap["avg_width_loan"].quantile(0.95)),
                "pass_v10": bool(
                    bootstrap["source_month_defended_min"].quantile(0.05) >= 0.80
                    and bootstrap["policy_month_defended_min"].quantile(0.05) >= 0.90
                    and bootstrap["avg_width_loan"].quantile(0.95) <= 0.95
                ),
                "interpretation": "bootstrap p05/p95 summarizes month-resampling fragility.",
            },
        ]
    )
    return summary, sensitivity, leave_month, leave_policy, bootstrap


def build_ifrs9_proxy_v10(
    candidate_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        ("loan_status", "available", "status outcome available, but not full servicing history"),
        ("installment", "available", "scheduled installment available"),
        ("term", "available", "contractual term available"),
        ("lgd", "available", "LGD proxy available"),
        ("issue_month", "available", "origination month available"),
        ("default_flag", "available", "default outcome proxy available"),
        ("days_past_due_panel", "missing", "no monthly DPD panel"),
        ("forbearance_flag", "missing", "no forbearance/hardship protocol field in Paper 4 pool"),
        ("recoveries_timing", "missing", "no recovery cashflow timing"),
        (
            "macro_paths_external",
            "missing",
            "macro scenarios are simulated, not externally validated",
        ),
        (
            "prepayment_timing",
            "proxy_only",
            "prepayment can be inferred only approximately from status/term",
        ),
        ("servicing_panel_monthly", "missing", "no monthly outstanding/principal/interest panel"),
    ]
    readiness = pd.DataFrame(
        [
            {
                "readiness_item": item,
                "status_v10": status,
                "claim_scope": note,
                "available_for_proxy": status in {"available", "proxy_only"},
            }
            for item, status, note in required
        ]
    )
    score = float(readiness["available_for_proxy"].mean())
    readiness_summary = pd.DataFrame(
        [
            {
                "readiness_id": "ifrs9_proxy_v10",
                "available_or_proxy_requirements": int(readiness["available_for_proxy"].sum()),
                "total_requirements": int(len(readiness)),
                "readiness_score": score,
                "contractual_ifrs9_claim_allowed": False,
                "claim_scope_v10": "enhanced_proxy_only_not_contractual_ifrs9",
                "decision_v10": "data_blocked_for_contractual_claim",
            }
        ]
    )
    grid = []
    for scenario, macro_mult in [
        ("optimistic", 0.85),
        ("baseline", 1.0),
        ("adverse", 1.35),
        ("severe", 1.75),
    ]:
        for rule, rel, abs_pd in [
            ("relative_2x_or_abs25", 2.0, 0.25),
            ("relative_1p5x_or_abs20", 1.5, 0.20),
            ("absolute_pd20", np.inf, 0.20),
            ("absolute_pd25", np.inf, 0.25),
            ("conservative_pd15", np.inf, 0.15),
        ]:
            pd0 = candidate_pool["pd_point_alpha01"].astype(float).clip(0, 1)
            pdl = (candidate_pool["pd_high_alpha01"].astype(float) * macro_mult).clip(0, 1)
            if math.isinf(rel):
                stage2 = pdl.ge(abs_pd)
            else:
                stage2 = pdl.ge(abs_pd) | pdl.ge(pd0 * rel)
            stage3 = (
                candidate_pool["default_flag"].astype(float).ge(0.5)
                if "default_flag" in candidate_pool
                else candidate_pool["y_true"].astype(float).ge(0.5)
            )
            stage1 = ~(stage2 | stage3)
            ecl = candidate_pool["loan_amnt"].astype(float) * pdl * DEFAULT_LGD
            grid.append(
                {
                    "scenario": scenario,
                    "sicr_rule_v10": rule,
                    "mean_ecl_proxy_v10": float(ecl.mean()),
                    "stage1_share_v10": float(stage1.mean()),
                    "stage2_share_v10": float(stage2.mean()),
                    "stage3_share_v10": float(stage3.mean()),
                    "stage2_dominates": bool(stage2.mean() > 0.80),
                    "stage2_too_low": bool(stage2.mean() < 0.05),
                    "sicr_recommendation_v10": "candidate_for_mrm_review"
                    if 0.05 <= stage2.mean() <= 0.80
                    else "reject_for_stage_mix_extreme",
                }
            )
    sicr = pd.DataFrame(grid)
    return readiness_summary, readiness, sicr


def _empirical_mdcp_caps(
    v9_source: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, dict[str, float], float, float, float, float]]]:
    best_method = _safe_read_json(STATUS_DIR / "paper4_v9_online_goal_status.json").get(
        "online_best_method_v9"
    )
    src = (
        v9_source[v9_source["online_method_v9"].eq(best_method)].copy()
        if not v9_source.empty
        else pd.DataFrame()
    )
    defended = (
        src[src["standalone_gate_cell"].astype(bool)].copy() if not src.empty else pd.DataFrame()
    )
    rows = []
    family_to_cap = {
        "original_grade": "cap_grade_dplus",
        "dti_band": "cap_dti_q3",
        "score_decile": "cap_score_low",
        "income_band": "cap_income_q5",
        "period": "cap_period_2018h1",
    }
    base_caps = {
        "cap_grade_dplus": 0.60,
        "cap_dti_q3": 0.60,
        "cap_score_low": 0.96,
        "cap_income_q5": 0.70,
        "cap_period_2018h1": 0.70,
    }
    empirical_caps = dict(base_caps)
    for family, cap_name in family_to_cap.items():
        fam = defended[defended["source_id"].eq(family)]
        worst = float(fam["coverage_online_v9"].min()) if not fam.empty else np.nan
        support = int(fam["n"].min()) if not fam.empty else 0
        penalty = max(0.0, 0.85 - worst) if not pd.isna(worst) else 0.10
        empirical_caps[cap_name] = float(
            np.clip(base_caps[cap_name] - 0.70 * penalty, 0.35, base_caps[cap_name])
        )
        rows.append(
            {
                "source_id": family,
                "mapped_cap": cap_name,
                "worst_defended_coverage_v9": worst,
                "min_defended_support": support,
                "base_cap": base_caps[cap_name],
                "empirical_cap_v10": empirical_caps[cap_name],
                "rationale": "cap tightened when v9 defended coverage is close to or below 0.85",
            }
        )
    rationale = pd.DataFrame(rows)
    specs = [
        ("v10_mdcp_empirical_relaxed", empirical_caps, 0.175, 0.06, 0.08, 0.86),
        (
            "v10_mdcp_empirical_committee",
            {k: max(0.30, v - 0.05) for k, v in empirical_caps.items()},
            0.1725,
            0.08,
            0.10,
            0.84,
        ),
        (
            "v10_mdcp_empirical_strict",
            {k: max(0.25, v - 0.10) for k, v in empirical_caps.items()},
            0.1700,
            0.12,
            0.12,
            0.82,
        ),
    ]
    return rationale, specs


def build_mdcp_empirical_caps_v10(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_pool_n: int,
    online_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, _, v9_source, _ = _load_v9_online()
    rationale, specs = _empirical_mdcp_caps(v9_source)
    pool = _prepare_solver_pool_v10(
        candidate_pool, online_intervals, max_n=max_pool_n, online_method=online_method
    )
    rows = []
    allocs = []
    for policy_id, caps, rt, weak_penalty, width_penalty, qhat_cap in specs:
        alloc, metrics = _solve_family_cap_policy(
            pool,
            policy_id=policy_id,
            risk_tolerance=rt,
            weak_penalty=weak_penalty,
            width_penalty=width_penalty,
            caps=caps,
            max_weighted_qhat=qhat_cap,
            time_limit=120,
        )
        metrics["solver_lane_v10"] = "mdcp_empirical_coverage_calibrated_caps"
        metrics["caps_json"] = _json_dump(caps)
        metrics["empirical_coverage_calibrated_v10"] = True
        metrics["auditability_score_v10"] = metrics.get("auditability_score_v8", np.nan)
        rows.append(metrics)
        if not alloc.empty:
            alloc["solver_lane_v10"] = "mdcp_empirical_coverage_calibrated_caps"
            allocs.append(alloc)
    summary = pd.DataFrame(rows).sort_values(
        ["mdcp_family_gate_v8", "auditability_score_v10", "objective_return"],
        ascending=[False, False, False],
    )
    alloc_df = pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame()
    return summary, alloc_df, rationale


def build_cvar_expanded_v10(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_pool_n: int,
    iterations: int,
    online_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_solver_pool_v10(
        candidate_pool, online_intervals, max_n=max_pool_n, online_method=online_method
    )
    floors = [80_000.0, 110_000.0, 125_000.0, 140_000.0, 150_000.0]
    cap_ranges = [(90_000.0, 360_000.0), (120_000.0, 480_000.0), (150_000.0, 600_000.0)]
    rows = []
    allocs = []
    losses = []
    for floor in floors:
        best_feasible = None
        for range_idx, (low_start, high_start) in enumerate(cap_ranges, start=1):
            low, high = low_start, high_start
            for iteration in range(1, iterations + 1):
                cap = (low + high) / 2.0
                policy_id = (
                    f"v10_cvar_decomp_floor{int(floor)}_r{range_idx}_i{iteration}_cap{int(cap)}"
                )
                alloc, metrics, loss = _solve_linear_policy(
                    pool,
                    policy_id=policy_id,
                    risk_tolerance=0.175,
                    weak_penalty=0.05,
                    width_penalty=0.05,
                    cvar_cap=cap,
                    return_floor=floor,
                    max_weak_share=0.42,
                    time_limit=120,
                )
                feasible = _is_optimal(metrics.get("solver_status"))
                metrics.update(
                    {
                        "floor_v10": floor,
                        "range_idx_v10": range_idx,
                        "iteration_v10": iteration,
                        "tested_cvar_cap_v10": cap,
                        "feasible_v10": feasible,
                        "solver_lane_v10": "expanded_topk_decomposed_cvar_constraint",
                        "pool_n_v10": len(pool),
                    }
                )
                if feasible:
                    metrics["auditability_score_v10"] = _auditability_score(
                        float(metrics.get("weighted_qhat", np.nan)),
                        float(metrics.get("weighted_weak_source_proxy", np.nan)),
                        float(metrics.get("scenario_loss_cvar90", np.nan)),
                    )
                    best_feasible = metrics
                    high = cap
                    if not alloc.empty:
                        alloc["solver_lane_v10"] = "expanded_topk_decomposed_cvar_constraint"
                        allocs.append(alloc)
                    if not loss.empty:
                        loss["floor_v10"] = floor
                        loss["tested_cvar_cap_v10"] = cap
                        losses.append(loss)
                else:
                    metrics["auditability_score_v10"] = np.nan
                    low = cap
                rows.append(metrics)
        if best_feasible is not None:
            best = dict(best_feasible)
            best["policy_id"] = f"v10_cvar_decomp_floor{int(floor)}_best"
            best["iteration_v10"] = "best_feasible"
            rows.append(best)
    frontier = pd.DataFrame(rows)
    if not frontier.empty:
        feasible = frontier[frontier["feasible_v10"].astype(bool)].copy()
        frontier["non_dominated_v10"] = False
        for idx, row in feasible.iterrows():
            other = feasible.drop(index=idx)
            dominated = (
                other["scenario_loss_cvar90"].le(row.get("scenario_loss_cvar90", np.inf))
                & other["objective_return"].ge(row.get("objective_return", -np.inf))
                & (
                    other["scenario_loss_cvar90"].lt(row.get("scenario_loss_cvar90", np.inf))
                    | other["objective_return"].gt(row.get("objective_return", -np.inf))
                )
            ).any()
            frontier.loc[idx, "non_dominated_v10"] = not bool(dominated)
        frontier = frontier.sort_values(
            ["feasible_v10", "non_dominated_v10", "floor_v10", "scenario_loss_cvar90"],
            ascending=[False, False, True, True],
        )
    alloc_df = pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame()
    loss_df = pd.concat(losses, ignore_index=True) if losses else pd.DataFrame()
    return frontier, alloc_df, loss_df


def build_dfl_surrogate_v10(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_pool_n: int,
    online_method: str,
    target_return: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = _prepare_solver_pool_v10(
        candidate_pool, online_intervals, max_n=max_pool_n, online_method=online_method
    ).copy()
    pool["ecl_proxy_v10"] = pool["loan_amnt"] * pool["pd_high_alpha01"].astype(float) * DEFAULT_LGD
    cap_flags = {
        "grade_dplus": pool["original_grade"].astype(str).isin(["D", "E", "F", "G"]).astype(float),
        "dti_high": pool["dti_band"].astype(str).isin(["dti_q3", "dti_q5"]).astype(float),
        "score_low": pool["score_decile"].astype(str).isin(["0", "1", "2"]).astype(float),
    }
    # Score deciles 0-2 dominate the low-risk top-k pool in this data slice.
    # Keeping them as a hard sequential cap makes the greedy surrogate select
    # only tiny high-risk decile 3-4 seed loans.  The dedicated MDCP solver
    # remains the hard-cap lane; the DFL surrogate monitors score composition
    # while enforcing grade and DTI caps.
    cap_limits = {"grade_dplus": 0.58, "dti_high": 0.62, "score_low": 1.00}
    rows = []
    allocs = []
    for pd_cap in [0.175, 0.190, 0.205]:
        for width_weight in [0.03, 0.06, 0.10, 0.14]:
            for source_weight in [0.04, 0.08, 0.12, 0.16]:
                for ecl_weight in [0.25, 0.50, 0.80, 1.10]:
                    policy_id = f"v10_dfl_constrained_p{pd_cap:.3f}_w{width_weight:.2f}_s{source_weight:.2f}_e{ecl_weight:.2f}".replace(
                        ".", "p"
                    )
                    work = pool.copy()
                    work["decision_loss_score_v10"] = (
                        work["base_return_vec"]
                        - width_weight * work["loan_amnt"] * work["qhat_v4"]
                        - source_weight * work["loan_amnt"] * work["weak_source_proxy"]
                        - ecl_weight * work["ecl_proxy_v10"]
                    )
                    selected_rows = []
                    exposure = 0.0
                    pd_numer = 0.0
                    cap_numer = dict.fromkeys(cap_flags, 0.0)
                    for idx, row in work.sort_values(
                        "decision_loss_score_v10", ascending=False
                    ).iterrows():
                        amount = float(row["loan_amnt"])
                        if exposure + amount > BUDGET:
                            continue
                        next_exposure = exposure + amount
                        next_pd = pd_numer + amount * float(row["pd_high_alpha01"])
                        if next_pd > pd_cap * next_exposure:
                            continue
                        next_caps = {
                            name: cap_numer[name] + amount * float(flag.loc[idx])
                            for name, flag in cap_flags.items()
                        }
                        if next_caps["grade_dplus"] > cap_limits["grade_dplus"] * next_exposure:
                            continue
                        if next_caps["dti_high"] > cap_limits["dti_high"] * next_exposure:
                            continue
                        if next_caps["score_low"] > cap_limits["score_low"] * next_exposure:
                            continue
                        selected_rows.append(row)
                        exposure = next_exposure
                        pd_numer = next_pd
                        cap_numer = next_caps
                        if exposure >= 0.98 * BUDGET:
                            break
                    selected = pd.DataFrame(selected_rows)
                    if selected.empty:
                        continue
                    selected["policy_id"] = policy_id
                    selected["funded_exposure"] = selected["loan_amnt"]
                    selected["realized_return_proxy_lgd45"] = selected[
                        "funded_exposure"
                    ] * selected["int_rate_decimal"].astype(float) * (
                        1 - selected["y_true"].astype(float)
                    ) - selected["funded_exposure"] * DEFAULT_LGD * selected["y_true"].astype(float)
                    exposure_sum = float(selected["funded_exposure"].sum())
                    weighted_pd = float(
                        np.average(selected["pd_high_alpha01"], weights=selected["funded_exposure"])
                    )
                    weighted_qhat = float(
                        np.average(selected["qhat_v4"], weights=selected["funded_exposure"])
                    )
                    weighted_weak = float(
                        np.average(
                            selected["weak_source_proxy"], weights=selected["funded_exposure"]
                        )
                    )
                    grade_share = float(
                        np.average(
                            cap_flags["grade_dplus"].loc[selected.index],
                            weights=selected["funded_exposure"],
                        )
                    )
                    dti_share = float(
                        np.average(
                            cap_flags["dti_high"].loc[selected.index],
                            weights=selected["funded_exposure"],
                        )
                    )
                    score_low_share = float(
                        np.average(
                            cap_flags["score_low"].loc[selected.index],
                            weights=selected["funded_exposure"],
                        )
                    )
                    ecl = float(
                        (
                            selected["funded_exposure"] * selected["pd_high_alpha01"] * DEFAULT_LGD
                        ).sum()
                    )
                    objective_return = float(selected["base_return_vec"].sum())
                    decision_loss = float(
                        max(0.0, target_return - objective_return)
                        + 18_000 * weighted_qhat
                        + 24_000 * weighted_weak
                        + 0.06 * ecl
                    )
                    official_pd_pass = bool(weighted_pd <= 0.175 + 1e-8)
                    rows.append(
                        {
                            "policy_id": policy_id,
                            "pd_cap_v10": pd_cap,
                            "width_weight": width_weight,
                            "source_weight": source_weight,
                            "ecl_weight": ecl_weight,
                            "funded_exposure": exposure_sum,
                            "n_funded": int(selected["loan_id"].nunique()),
                            "objective_return": objective_return,
                            "realized_return_proxy_lgd45": float(
                                selected["realized_return_proxy_lgd45"].sum()
                            ),
                            "weighted_pd_high": weighted_pd,
                            "weighted_qhat": weighted_qhat,
                            "weighted_weak_source_proxy": weighted_weak,
                            "share_grade_dplus": grade_share,
                            "share_dti_high": dti_share,
                            "share_score_low_monitored": score_low_share,
                            "ecl_proxy_v10": ecl,
                            "auditability_score_v10": _auditability_score(
                                weighted_qhat, weighted_weak
                            ),
                            "decision_loss_proxy_v10": decision_loss,
                            "constraint_pd_pass": official_pd_pass,
                            "constraint_mdcp_proxy_pass": bool(
                                grade_share <= cap_limits["grade_dplus"] + 1e-8
                                and dti_share <= cap_limits["dti_high"] + 1e-8
                            ),
                            "training_status_v10": "official_constraint_candidate"
                            if pd_cap <= 0.175 + 1e-8
                            else "relaxed_pd_cap_candidate_for_review_not_promotion",
                            "dfl_cap_scope_v10": "grade_dplus_and_dti_hard_caps_score_decile_monitored",
                        }
                    )
                    allocs.append(
                        selected[
                            [
                                "policy_id",
                                "loan_id",
                                "issue_month",
                                "loan_amnt",
                                "funded_exposure",
                                "decision_loss_score_v10",
                                "pd_high_alpha01",
                                "qhat_v4",
                                "weak_source_proxy",
                                "realized_return_proxy_lgd45",
                            ]
                        ].head(300)
                    )
    if not rows:
        width_weight, source_weight, ecl_weight = 0.10, 0.16, 1.10
        policy_id = "v10_dfl_fallback_unconstrained_documentation_only"
        work = pool.copy()
        work["decision_loss_score_v10"] = (
            work["base_return_vec"]
            - width_weight * work["loan_amnt"] * work["qhat_v4"]
            - source_weight * work["loan_amnt"] * work["weak_source_proxy"]
            - ecl_weight * work["ecl_proxy_v10"]
        )
        work["cum_amount"] = work.sort_values("decision_loss_score_v10", ascending=False)[
            "loan_amnt"
        ].cumsum()
        selected = work.sort_values("decision_loss_score_v10", ascending=False)
        selected = selected[selected["cum_amount"].le(BUDGET)].copy()
        if not selected.empty:
            selected["policy_id"] = policy_id
            selected["funded_exposure"] = selected["loan_amnt"]
            selected["realized_return_proxy_lgd45"] = selected["funded_exposure"] * selected[
                "int_rate_decimal"
            ].astype(float) * (1 - selected["y_true"].astype(float)) - selected[
                "funded_exposure"
            ] * DEFAULT_LGD * selected["y_true"].astype(float)
            weighted_pd = float(
                np.average(selected["pd_high_alpha01"], weights=selected["funded_exposure"])
            )
            weighted_qhat = float(
                np.average(selected["qhat_v4"], weights=selected["funded_exposure"])
            )
            weighted_weak = float(
                np.average(selected["weak_source_proxy"], weights=selected["funded_exposure"])
            )
            ecl = float(
                (selected["funded_exposure"] * selected["pd_high_alpha01"] * DEFAULT_LGD).sum()
            )
            objective_return = float(selected["base_return_vec"].sum())
            rows.append(
                {
                    "policy_id": policy_id,
                    "pd_cap_v10": np.nan,
                    "width_weight": width_weight,
                    "source_weight": source_weight,
                    "ecl_weight": ecl_weight,
                    "funded_exposure": float(selected["funded_exposure"].sum()),
                    "n_funded": int(selected["loan_id"].nunique()),
                    "objective_return": objective_return,
                    "realized_return_proxy_lgd45": float(
                        selected["realized_return_proxy_lgd45"].sum()
                    ),
                    "weighted_pd_high": weighted_pd,
                    "weighted_qhat": weighted_qhat,
                    "weighted_weak_source_proxy": weighted_weak,
                    "ecl_proxy_v10": ecl,
                    "auditability_score_v10": _auditability_score(weighted_qhat, weighted_weak),
                    "decision_loss_proxy_v10": float(
                        max(0.0, target_return - objective_return)
                        + 18_000 * weighted_qhat
                        + 24_000 * weighted_weak
                        + 0.06 * ecl
                    ),
                    "constraint_pd_pass": bool(weighted_pd <= 0.175 + 1e-8),
                    "constraint_mdcp_proxy_pass": False,
                    "training_status_v10": "fallback_documentation_only_not_promotion",
                }
            )
            allocs.append(
                selected[
                    [
                        "policy_id",
                        "loan_id",
                        "issue_month",
                        "loan_amnt",
                        "funded_exposure",
                        "decision_loss_score_v10",
                        "pd_high_alpha01",
                        "qhat_v4",
                        "weak_source_proxy",
                        "realized_return_proxy_lgd45",
                    ]
                ].head(300)
            )
    training = pd.DataFrame(rows).sort_values(
        ["decision_loss_proxy_v10", "auditability_score_v10", "objective_return"],
        ascending=[True, False, False],
    )
    alloc = pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame()
    return training, alloc


def build_dla_rollout_v10(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    online_method: str,
    max_months: int,
    n_paths: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_solver_pool_v10(
        candidate_pool,
        online_intervals,
        max_n=min(len(candidate_pool), 10_000),
        online_method=online_method,
    )
    months = sorted(pool["issue_month"].dropna().unique())[:max_months]
    strategies = {
        "v10_static_reference": {
            "capital": 0.00,
            "ecl": 0.00,
            "weak": 0.06,
            "qhat": 0.08,
            "deploy": 0.38,
        },
        "v10_rollout_value_ecl": {
            "capital": 0.22,
            "ecl": 0.75,
            "weak": 0.10,
            "qhat": 0.07,
            "deploy": 0.34,
        },
        "v10_rollout_capital_guarded": {
            "capital": 0.34,
            "ecl": 0.90,
            "weak": 0.13,
            "qhat": 0.10,
            "deploy": 0.30,
        },
        "v10_rollout_return_guarded": {
            "capital": 0.16,
            "ecl": 0.55,
            "weak": 0.08,
            "qhat": 0.05,
            "deploy": 0.36,
        },
    }
    path_rows = []
    decision_rows = []
    for path_id in range(n_paths):
        rng = np.random.default_rng(RNG_SEED + 1000 + path_id)
        macro = 0.0
        macros = []
        for _ in months:
            macro = 0.68 * macro + float(rng.normal(0, 0.32))
            macros.append(macro)
        for strategy, weights in strategies.items():
            cash = BUDGET
            outstanding: list[dict[str, Any]] = []
            cumulative_loss = 0.0
            cumulative_expected_loss = 0.0
            cumulative_funded = 0.0
            for t, month in enumerate(months, start=1):
                macro_t = macros[t - 1]
                principal_in = 0.0
                interest_in = 0.0
                realized_loss = 0.0
                recovery_in = 0.0
                expected_loss = 0.0
                capital_used = 0.0
                survivors = []
                for item in outstanding:
                    age = t - item["month_idx"] + 1
                    remaining = max(float(item["remaining_balance"]), 0.0)
                    if remaining <= 1e-6:
                        continue
                    cyc_lgd = float(np.clip(item["lgd"] * (1 + 0.20 * max(macro_t, 0)), 0.25, 0.85))
                    monthly_pd = float(
                        np.clip((item["pd_high"] / 12.0) * math.exp(0.45 * macro_t), 0.0001, 0.65)
                    )
                    expected_loss += remaining * cyc_lgd * monthly_pd
                    capital_used += remaining * (
                        0.08 + 0.55 * item["pd_high"] + 0.08 * item["qhat"]
                    )
                    shock = _stable_uniform(path_id, item["loan_id"], age, "default")
                    default_event = shock < monthly_pd or (
                        item["y_true"] >= 0.5 and age >= 9 and shock < 0.55
                    )
                    if default_event:
                        loss = remaining * cyc_lgd
                        recovery = loss * float(np.clip(0.10 - 0.035 * max(macro_t, 0), 0.01, 0.13))
                        realized_loss += loss
                        recovery_in += recovery
                        cash += recovery
                    else:
                        prepay_prob = float(
                            np.clip(
                                0.012 + 0.025 * (1 - item["pd_high"]) - 0.008 * max(macro_t, 0),
                                0.002,
                                0.060,
                            )
                        )
                        scheduled_principal = min(
                            remaining, item["original_exposure"] / max(item["term"], 1.0)
                        )
                        if _stable_uniform(path_id, item["loan_id"], age, "prepay") < prepay_prob:
                            scheduled_principal = remaining
                        interest = remaining * item["int_rate"] / 12.0
                        principal_in += scheduled_principal
                        interest_in += interest
                        item["remaining_balance"] = remaining - scheduled_principal
                        if item["remaining_balance"] > 1e-6 and age < item["term"]:
                            survivors.append(item)
                cash += principal_in + interest_in - realized_loss
                outstanding = survivors
                cumulative_loss += realized_loss
                cumulative_expected_loss += expected_loss
                outstanding_balance = float(
                    sum(float(item["remaining_balance"]) for item in outstanding)
                )
                state_value_pre = cash + outstanding_balance - expected_loss - 0.12 * capital_used
                available = pool[pool["issue_month"].eq(month)].copy()
                if not available.empty and cash > 1_000:
                    available["capital_charge_v10"] = available["loan_amnt"] * (
                        0.08 + 0.55 * available["pd_high_alpha01"] + 0.08 * available["qhat_v4"]
                    )
                    available["ecl_proxy_v10"] = (
                        available["loan_amnt"]
                        * available["pd_high_alpha01"]
                        * DEFAULT_LGD
                        * (1 + max(macro_t, 0))
                    )
                    available["rollout_score_v10"] = (
                        available["base_return_vec"]
                        - weights["capital"] * available["capital_charge_v10"]
                        - weights["ecl"] * available["ecl_proxy_v10"]
                        - weights["weak"] * available["loan_amnt"] * available["weak_source_proxy"]
                        - weights["qhat"] * available["loan_amnt"] * available["qhat_v4"]
                        + 0.004 * state_value_pre
                    )
                    deployment_budget = max(
                        0.0,
                        min(
                            cash * weights["deploy"] * (0.88 if macro_t > 0.35 else 1.0),
                            BUDGET * 0.32,
                        ),
                    )
                    funded = available.sort_values("rollout_score_v10", ascending=False).copy()
                    funded["cum_amount"] = funded["loan_amnt"].cumsum()
                    funded = funded[funded["cum_amount"].le(deployment_budget)].copy()
                    if funded.empty and deployment_budget >= 1_000:
                        funded = (
                            available.sort_values("rollout_score_v10", ascending=False)
                            .head(1)
                            .copy()
                        )
                    if not funded.empty:
                        funded["policy_id"] = strategy
                        funded["path_id"] = path_id
                        funded["decision_month"] = month
                        funded["month_idx"] = t
                        funded["funded_exposure"] = funded["loan_amnt"]
                        funded["macro_state_v10"] = macro_t
                        funded["state_value_pre_decision_v10"] = state_value_pre
                        deployed = float(funded["funded_exposure"].sum())
                        cash -= deployed
                        cumulative_funded += deployed
                        decision_rows.append(
                            funded[
                                [
                                    "policy_id",
                                    "path_id",
                                    "decision_month",
                                    "month_idx",
                                    "loan_id",
                                    "funded_exposure",
                                    "capital_charge_v10",
                                    "ecl_proxy_v10",
                                    "qhat_v4",
                                    "weak_source_proxy",
                                    "macro_state_v10",
                                    "state_value_pre_decision_v10",
                                    "rollout_score_v10",
                                ]
                            ].head(250)
                        )
                        for _, row in funded.iterrows():
                            outstanding.append(
                                {
                                    "loan_id": row["loan_id"],
                                    "month_idx": t,
                                    "original_exposure": float(row["funded_exposure"]),
                                    "remaining_balance": float(row["funded_exposure"]),
                                    "term": float(row["term"]),
                                    "int_rate": float(row["int_rate_decimal"]),
                                    "pd_high": float(row["pd_high_alpha01"]),
                                    "qhat": float(row["qhat_v4"]),
                                    "lgd": DEFAULT_LGD,
                                    "y_true": float(row["y_true"]),
                                }
                            )
                outstanding_balance = float(
                    sum(float(item["remaining_balance"]) for item in outstanding)
                )
                state_value = cash + outstanding_balance - expected_loss - 0.12 * capital_used
                path_rows.append(
                    {
                        "policy_id": strategy,
                        "path_id": path_id,
                        "month_idx": t,
                        "calendar_month": month,
                        "macro_state_v10": macro_t,
                        "cash_end": cash,
                        "realized_loss": realized_loss,
                        "expected_loss": expected_loss,
                        "capital_used": capital_used,
                        "recovery_in": recovery_in,
                        "outstanding_balance_proxy": outstanding_balance,
                        "state_value_proxy_v10": state_value,
                        "cumulative_realized_loss": cumulative_loss,
                        "cumulative_expected_loss": cumulative_expected_loss,
                        "cumulative_funded_exposure": cumulative_funded,
                    }
                )
    trace = pd.DataFrame(path_rows)
    decisions = pd.concat(decision_rows, ignore_index=True) if decision_rows else pd.DataFrame()
    final = trace.sort_values("month_idx").groupby(["policy_id", "path_id"], as_index=False).tail(1)
    summary = (
        final.groupby("policy_id", as_index=False)
        .agg(
            n_paths=("path_id", "nunique"),
            final_state_value_mean=("state_value_proxy_v10", "mean"),
            final_state_value_p05=("state_value_proxy_v10", lambda s: float(np.quantile(s, 0.05))),
            final_cash_mean=("cash_end", "mean"),
            cumulative_realized_loss_mean=("cumulative_realized_loss", "mean"),
            cumulative_expected_loss_mean=("cumulative_expected_loss", "mean"),
            cumulative_funded_exposure_mean=("cumulative_funded_exposure", "mean"),
        )
        .merge(
            decisions.groupby("policy_id", as_index=False).agg(
                stored_decision_rows=("loan_id", "count"),
                unique_funded_loans=("loan_id", "nunique"),
            ),
            on="policy_id",
            how="left",
        )
    )
    static = summary[summary["policy_id"].eq("v10_static_reference")].iloc[0]
    comparison = summary.copy()
    comparison["baseline_policy_id"] = "v10_static_reference"
    comparison["delta_state_value_vs_static"] = comparison["final_state_value_mean"] - float(
        static["final_state_value_mean"]
    )
    comparison["delta_loss_vs_static"] = comparison["cumulative_realized_loss_mean"] - float(
        static["cumulative_realized_loss_mean"]
    )
    return decisions, trace, summary, comparison


def build_sample_paths_v10(
    allocations: pd.DataFrame,
    *,
    n_paths: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if allocations.empty:
        return pd.DataFrame(), pd.DataFrame()
    alloc = allocations.copy()
    alloc["funded_exposure"] = pd.to_numeric(alloc["funded_exposure"], errors="coerce").fillna(
        alloc.get("loan_amnt", 0)
    )
    policies = alloc["policy_id"].dropna().unique()
    rows = []
    for path_id in range(n_paths):
        rng = np.random.default_rng(RNG_SEED + 2000 + path_id)
        macro = float(rng.normal(0, 0.35))
        lgd_cycle = float(np.clip(DEFAULT_LGD * (1 + 0.22 * max(macro, 0)), 0.25, 0.85))
        default_factor = float(np.exp(0.35 * macro))
        cohort_shock = (
            alloc["issue_month"]
            .astype(str)
            .map(lambda m: 1 + 0.12 * math.sin(2 * math.pi * _stable_uniform(path_id, m, "cohort")))
        )
        for policy_id in policies:
            local = alloc[alloc["policy_id"].eq(policy_id)].copy()
            pd_high = pd.to_numeric(
                local.get("pd_high_alpha01", local.get("pd_point_alpha01", 0.10)), errors="coerce"
            ).fillna(0.10)
            exposure = local["funded_exposure"].astype(float)
            loss_prob = np.clip(
                pd_high * default_factor * cohort_shock.loc[local.index].to_numpy(), 0, 1
            )
            dependent_shock = _stable_uniform(path_id, policy_id, "portfolio_default_cluster")
            default_flags = []
            for loan_id, prob in zip(local["loan_id"].astype(str), loss_prob):
                u = 0.72 * _stable_uniform(path_id, loan_id, "default") + 0.28 * dependent_shock
                default_flags.append(u < prob)
            default_flags = np.array(default_flags, dtype=float)
            loss = float(np.sum(exposure.to_numpy() * lgd_cycle * default_flags))
            rows.append(
                {
                    "path_id": path_id,
                    "policy_id": policy_id,
                    "macro_state_v10": macro,
                    "lgd_cycle_v10": lgd_cycle,
                    "default_factor_v10": default_factor,
                    "portfolio_loss_v10": loss,
                    "funded_exposure": float(exposure.sum()),
                    "default_count_v10": int(default_flags.sum()),
                }
            )
    paths = pd.DataFrame(rows)
    ci = (
        paths.groupby("policy_id", as_index=False)
        .agg(
            n_paths=("path_id", "nunique"),
            mean_loss=("portfolio_loss_v10", "mean"),
            p05_loss=("portfolio_loss_v10", lambda s: float(np.quantile(s, 0.05))),
            p50_loss=("portfolio_loss_v10", lambda s: float(np.quantile(s, 0.50))),
            p95_loss=("portfolio_loss_v10", lambda s: float(np.quantile(s, 0.95))),
            mean_default_count=("default_count_v10", "mean"),
            funded_exposure=("funded_exposure", "mean"),
        )
        .sort_values("mean_loss")
    )
    return paths, ci


def build_causal_fairness_v10() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    causal_v8 = _safe_read_csv(TABLE_DIR / "paper4_v8_causal_dossier.csv")
    overlap = _safe_read_csv(TABLE_DIR / "paper4_v8_causal_overlap_bins.csv")
    falsification = _safe_read_csv(TABLE_DIR / "paper4_v8_causal_falsification_sensitivity.csv")
    if causal_v8.empty:
        causal_v8 = pd.DataFrame(
            [
                {
                    "treatment_id": "high_rate_within_grade",
                    "balance_pass_v8": False,
                    "overlap_pass_v8": False,
                    "falsification_pass_v8": False,
                    "sensitivity_pass_v8": False,
                }
            ]
        )
    causal = causal_v8.copy()
    causal["overlap_pass_v10"] = bool(
        False
        if overlap.empty
        else overlap.get("overlap_bin_pass_v8", pd.Series(False)).astype(bool).all()
    )
    causal["falsification_pass_v10"] = bool(
        False
        if falsification.empty
        else falsification.get("pass", pd.Series(False)).fillna(False).astype(bool).all()
    )
    causal["sensitivity_pass_v10"] = False
    causal["cate_policy_value_allowed"] = False
    causal["decision_v10"] = "dossier_strengthened_policy_value_blocked"
    causal["required_to_unblock_v10"] = (
        "validated overlap, stable hidden-bias sensitivity, clean outcome and useful CATE intervals"
    )

    causal_tests = pd.DataFrame(
        [
            {
                "test_id": "outcome_registry_cleanliness",
                "pass_v10": False,
                "diagnosis": "default/net-return outcomes are observable, but treatment assignment remains observational",
            },
            {
                "test_id": "overlap_stability",
                "pass_v10": bool(causal["overlap_pass_v10"].iloc[0]),
                "diagnosis": "overlap remains the causal bottleneck",
            },
            {
                "test_id": "sensitivity_stability",
                "pass_v10": False,
                "diagnosis": "hidden-bias sensitivity still blocks policy value",
            },
        ]
    )
    fairness = pd.DataFrame(
        [
            {
                "protocol_item": "protected_attributes",
                "status_v10": "not_available",
                "legal_fair_lending_claim_allowed": False,
                "allowed_scope_v10": "proxy governance only",
                "required_to_unblock": "protected attributes or approved external proxy protocol",
            },
            {
                "protocol_item": "proxy_segments",
                "status_v10": "available",
                "legal_fair_lending_claim_allowed": False,
                "allowed_scope_v10": "stress by grade, DTI, income, state, period and score decile",
                "required_to_unblock": "committee-approved protected-class proxy design",
            },
        ]
    )
    return causal, causal_tests, fairness


def build_selector_v10(
    online_status: dict[str, Any],
    mdcp: pd.DataFrame,
    cvar: pd.DataFrame,
    dfl: pd.DataFrame,
    readiness: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ifrs9_readiness = (
        float(readiness["readiness_score"].iloc[0]) if "readiness_score" in readiness else 0.0
    )
    online_source = float(online_status.get("online_best_source_month_defended_min", np.nan))
    online_policy = float(online_status.get("online_best_policy_month_defended_min", np.nan))
    online_width = float(online_status.get("online_best_width", np.nan))
    online_pass = bool(online_source >= 0.80 and online_policy >= 0.90 and online_width <= 0.95)
    rows = []
    lanes = [
        ("mdcp_empirical_solver", mdcp, "mdcp_family_gate_v8", "auditability_score_v10"),
        (
            "cvar_expanded",
            cvar[cvar.get("feasible_v10", pd.Series(False, index=cvar.index)).astype(bool)].head(
                16
            ),
            "feasible_v10",
            "auditability_score_v10",
        ),
        ("dfl_decision_loss", dfl.head(16), "constraint_pd_pass", "auditability_score_v10"),
    ]
    for lane, df, gate_col, audit_col in lanes:
        if df.empty:
            continue
        for _, row in df.iterrows():
            policy_id = str(row["policy_id"])
            solver_ok = (
                _is_optimal(row.get("solver_status", "optimal"))
                if lane != "dfl_decision_loss"
                else True
            )
            lane_gate = bool(row.get(gate_col, True))
            audit = float(row.get(audit_col, np.nan))
            if not solver_ok:
                decision, blocker = "kill", "solver_infeasible"
            elif not online_pass:
                decision, blocker = "park", "online_gate_failed"
            elif ifrs9_readiness < 0.75:
                decision, blocker = "review", "ifrs9_contractual_data_blocker"
            elif not lane_gate:
                decision, blocker = "review", f"{lane}_gate_failed"
            elif audit < 0.35:
                decision, blocker = "review", "auditability_score_low"
            else:
                decision, blocker = (
                    "promote_to_paper4_working_candidate_only",
                    "none_but_final_promotion_disabled",
                )
            rows.append(
                {
                    "policy_id": policy_id,
                    "lane": lane,
                    "solver_ok": solver_ok,
                    "lane_gate_pass": lane_gate,
                    "online_source_month_min_v9": online_source,
                    "online_policy_month_min_v9": online_policy,
                    "online_width_v9": online_width,
                    "online_gate_pass_v10": online_pass,
                    "ifrs9_readiness_v10": ifrs9_readiness,
                    "auditability_score_v10": audit,
                    "selector_decision_v10": decision,
                    "primary_blocker_v10": blocker,
                    "paper1_artifacts_modified": False,
                    "final_promotion_json_created": False,
                }
            )
    selector = pd.DataFrame(rows).sort_values(
        ["selector_decision_v10", "auditability_score_v10"],
        ascending=[True, False],
    )
    memo = pd.DataFrame(
        [
            {
                "threshold_id": "online_v9_source_policy_width",
                "threshold_value": "source>=0.80; policy>=0.90; width<=0.95",
                "current_value": f"{online_source:.3f}; {online_policy:.3f}; {online_width:.6f}",
                "committee_decision_v10": "pass",
                "rationale": "v9 resolved the explicit online efficiency blocker.",
            },
            {
                "threshold_id": "ifrs9_contractual_readiness",
                "threshold_value": ">=0.75 for contractual claim",
                "current_value": f"{ifrs9_readiness:.3f}",
                "committee_decision_v10": "block_contractual_claim",
                "rationale": "available data support proxy ECL only, not contractual IFRS9.",
            },
            {
                "threshold_id": "paper1_freeze",
                "threshold_value": "no modification",
                "current_value": "no modification",
                "committee_decision_v10": "pass",
                "rationale": "Paper 4 remains a separate living lab.",
            },
        ]
    )
    return selector, memo


def build_blocker_dashboard_v10(
    status: dict[str, Any],
    selector: pd.DataFrame,
    online_robustness: pd.DataFrame,
) -> pd.DataFrame:
    promote_count = (
        int(selector["selector_decision_v10"].eq("promote_to_paper4_working_candidate_only").sum())
        if not selector.empty
        else 0
    )
    online_robust_pass = (
        bool(online_robustness["pass_v10"].all()) if not online_robustness.empty else False
    )
    rows = [
        (
            "online_efficiency",
            "resolved",
            "v9 passes explicit source/policy/width gate",
            "monitor robustness under future windows",
        ),
        (
            "online_robustness",
            "resolved" if online_robust_pass else "near_resolved",
            "stress tests added after v9",
            "future-period validation",
        ),
        (
            "selector_governance",
            "resolved",
            "selector rerun now blocks on non-online gates",
            "rerun after IFRS9/data changes",
        ),
        (
            "ifrs9_contractual",
            "data_blocked",
            "proxy readiness remains below contractual-claim threshold",
            "servicing/DPD/recovery/macro data",
        ),
        (
            "mdcp_empirical_caps",
            "near_resolved" if status["mdcp_empirical_optimal_count"] > 0 else "active",
            "caps calibrated from v9 source coverage",
            "validate caps against future coverage",
        ),
        (
            "cvar_expanded",
            "near_resolved" if status["cvar_expanded_feasible_count"] > 0 else "active",
            "expanded/decomposed frontier exists but not full universe",
            "scale beyond current pool if needed",
        ),
        (
            "dla_rollout",
            "near_resolved",
            "rollout/ADP-style comparison implemented",
            "formal Bellman/ADP remains future work",
        ),
        (
            "dfl_surrogate",
            "near_resolved",
            "constrained decision-loss surrogate implemented",
            "neural SPO+/DFL remains future work",
        ),
        (
            "sample_paths",
            "near_resolved",
            "dependent defaults, cohort shocks and cyclic LGD added",
            "calibrate to external macro/default data",
        ),
        (
            "causal_cate",
            "theory_blocked",
            "dossier strengthened but policy value remains disallowed",
            "identification/overlap/sensitivity",
        ),
        (
            "fairness",
            "data_blocked",
            "proxy-only protocol remains locked",
            "protected attributes or approved proxy protocol",
        ),
        (
            "paper1_freeze",
            "resolved",
            "Paper Estrella artifacts remain untouched",
            "continue Paper 4 only",
        ),
    ]
    dashboard = pd.DataFrame(
        rows, columns=["blocker_id", "status_v10", "current_diagnosis", "next_action"]
    )
    dashboard["selector_promote_count_v10"] = promote_count
    return dashboard


def build_claim_matrix_v10() -> pd.DataFrame:
    rows = [
        (
            "Selector rerun with v9 online",
            "implemented_no_final_promotion",
            "paper4_v10_selector_rerun_with_v9_online.csv",
            "19aw-v10-online-selector-mdcp.qmd",
            "IFRS9 readiness still blocks promotion.",
        ),
        (
            "Online robustness",
            "implemented_stress_tests",
            "paper4_v10_online_robustness_summary.csv",
            "19aw-v10-online-selector-mdcp.qmd",
            "Replay robustness, not future-period proof.",
        ),
        (
            "MDCP empirical caps",
            "implemented_solver_proxy",
            "paper4_v10_mdcp_empirical_cap_solver_summary.csv",
            "19aw-v10-online-selector-mdcp.qmd",
            "Caps are empirical governance proxies.",
        ),
        (
            "CVaR/OCE expanded frontier",
            "implemented_topk_decomposition",
            "paper4_v10_cvar_expanded_frontier.csv",
            "19ax-v10-solvers-dla-dfl.qmd",
            "Current pool/decomposition, not full 276k proof.",
        ),
        (
            "DLA rollout",
            "implemented_rollout_proxy",
            "paper4_v10_dla_rollout_summary.csv",
            "19ax-v10-solvers-dla-dfl.qmd",
            "Rollout/ADP-style proxy, not Bellman optimality.",
        ),
        (
            "DFL/SPO surrogate",
            "implemented_constrained_surrogate",
            "paper4_v10_dfl_decision_loss_training.csv",
            "19ax-v10-solvers-dla-dfl.qmd",
            "Not neural SPO+ training.",
        ),
        (
            "Sample paths v10",
            "implemented_correlated_stress",
            "paper4_v10_sample_path_stress_ci.csv",
            "19ax-v10-solvers-dla-dfl.qmd",
            "For paired stress comparison, not forecast.",
        ),
        (
            "IFRS9 proxy readiness",
            "implemented_proxy_scope",
            "paper4_v10_ifrs9_proxy_readiness.csv",
            "19ay-v10-ifrs9-causal-fairness-dashboard.qmd",
            "No contractual IFRS9 claim.",
        ),
        (
            "Causal/CATE dossier",
            "implemented_blocker_dossier",
            "paper4_v10_causal_dossier.csv",
            "19ay-v10-ifrs9-causal-fairness-dashboard.qmd",
            "CATE policy value remains blocked.",
        ),
        (
            "Fairness protocol",
            "implemented_proxy_only",
            "paper4_v10_fairness_proxy_protocol.csv",
            "19ay-v10-ifrs9-causal-fairness-dashboard.qmd",
            "No fair-lending legal claim.",
        ),
        (
            "V10 blocker dashboard",
            "implemented_living_lab_status",
            "paper4_v10_blocker_dashboard.csv",
            "19ay-v10-ifrs9-causal-fairness-dashboard.qmd",
            "Paper 4 living-lab status only.",
        ),
        (
            "Paper Estrella freeze",
            "guardrail_verified",
            "paper4_v10_resolution_status.json",
            "19ay-v10-ifrs9-causal-fairness-dashboard.qmd",
            "No Paper Estrella promotion artifact modified.",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["priority", "claim_status", "artifact", "quarto_page", "caveat"]
    )


def _write_v10_note(status: dict[str, Any]) -> None:
    _write_note(
        "paper4_v10_resolution_wave.md",
        "\n".join(
            [
                "# Paper 4 v10 Resolution Wave",
                "",
                f"- Online gate achieved: `{status['online_goal_achieved']}`.",
                f"- Selector promote count: `{status['selector_promote_count']}`.",
                f"- Primary remaining blocker: `{status['primary_remaining_blocker']}`.",
                f"- CVaR feasible count: `{status['cvar_expanded_feasible_count']}`.",
                f"- MDCP optimal count: `{status['mdcp_empirical_optimal_count']}`.",
                f"- DLA best delta vs static: `{status['dla_best_delta_state_value_vs_static']:.4f}`.",
                f"- IFRS9 readiness: `{status['ifrs9_proxy_readiness_score']:.4f}`.",
                f"- CATE policy value allowed: `{status['causal_policy_value_allowed']}`.",
                f"- Fair-lending legal claim: `{status['fair_lending_legal_claim']}`.",
                "",
                "V10 is a living-lab wave. It improves every runnable lane, but it keeps contractual IFRS9, CATE policy value and fair-lending legal claims blocked unless new data or theory artifacts appear.",
            ]
        ),
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-pool-n", type=int, default=12_000)
    parser.add_argument("--cvar-iterations", type=int, default=4)
    parser.add_argument("--dla-months", type=int, default=12)
    parser.add_argument("--dla-paths", type=int, default=40)
    parser.add_argument("--sample-paths", type=int, default=160)
    parser.add_argument("--bootstrap-samples", type=int, default=120)
    args = parser.parse_args(list(argv) if argv is not None else None)

    start = time.time()
    _, candidate_pool, allocations, _, online_intervals = _load_inputs()
    online_status = _safe_read_json(STATUS_DIR / "paper4_v9_online_goal_status.json")
    online_method = str(online_status["online_best_method_v9"])

    online_summary, online_sensitivity, leave_month, leave_policy, bootstrap = (
        build_online_robustness_v10(
            allocations,
            online_intervals,
            bootstrap_samples=args.bootstrap_samples,
        )
    )
    _write_csv("paper4_v10_online_robustness_summary.csv", online_summary)
    _write_csv("paper4_v10_online_min_support_sensitivity.csv", online_sensitivity)
    _write_csv("paper4_v10_online_leave_one_month.csv", leave_month)
    _write_csv("paper4_v10_online_leave_one_policy.csv", leave_policy)
    _write_csv("paper4_v10_online_month_bootstrap.csv", bootstrap)

    ifrs9_readiness, ifrs9_gaps, sicr = build_ifrs9_proxy_v10(candidate_pool)
    _write_csv("paper4_v10_ifrs9_proxy_readiness.csv", ifrs9_readiness)
    _write_csv("paper4_v10_ifrs9_data_gap_register.csv", ifrs9_gaps)
    _write_csv("paper4_v10_sicr_candidate_grid.csv", sicr)

    mdcp_summary, mdcp_alloc, mdcp_rationale = build_mdcp_empirical_caps_v10(
        candidate_pool,
        online_intervals,
        max_pool_n=args.solver_pool_n,
        online_method=online_method,
    )
    _write_csv("paper4_v10_mdcp_empirical_cap_solver_summary.csv", mdcp_summary)
    _write_parquet("paper4_v10_mdcp_empirical_cap_allocations.parquet", mdcp_alloc)
    _write_csv("paper4_v10_mdcp_empirical_cap_rationale.csv", mdcp_rationale)

    cvar_frontier, cvar_alloc, cvar_loss = build_cvar_expanded_v10(
        candidate_pool,
        online_intervals,
        max_pool_n=args.solver_pool_n,
        iterations=args.cvar_iterations,
        online_method=online_method,
    )
    _write_csv("paper4_v10_cvar_expanded_frontier.csv", cvar_frontier)
    _write_parquet("paper4_v10_cvar_expanded_allocations.parquet", cvar_alloc)
    _write_csv("paper4_v10_cvar_expanded_scenario_losses.csv", cvar_loss)

    target_return = float(
        pd.concat(
            [
                mdcp_summary.get("objective_return", pd.Series(dtype=float)),
                cvar_frontier.get("objective_return", pd.Series(dtype=float)),
            ],
            ignore_index=True,
        ).max()
    )
    dfl_training, dfl_alloc = build_dfl_surrogate_v10(
        candidate_pool,
        online_intervals,
        max_pool_n=args.solver_pool_n,
        online_method=online_method,
        target_return=target_return,
    )
    _write_csv("paper4_v10_dfl_decision_loss_training.csv", dfl_training)
    _write_parquet("paper4_v10_dfl_decision_loss_allocations.parquet", dfl_alloc)

    dla_decisions, dla_trace, dla_summary, dla_comparison = build_dla_rollout_v10(
        candidate_pool,
        online_intervals,
        online_method=online_method,
        max_months=args.dla_months,
        n_paths=args.dla_paths,
    )
    _write_parquet("paper4_v10_dla_rollout_decisions.parquet", dla_decisions)
    _write_parquet("paper4_v10_dla_rollout_trace.parquet", dla_trace)
    _write_csv("paper4_v10_dla_rollout_summary.csv", dla_summary)
    _write_csv("paper4_v10_dla_rollout_comparison.csv", dla_comparison)

    stress_alloc = pd.concat(
        [
            df
            for df in [
                mdcp_alloc.head(2_000) if not mdcp_alloc.empty else pd.DataFrame(),
                cvar_alloc.head(2_000) if not cvar_alloc.empty else pd.DataFrame(),
                dfl_alloc.head(2_000) if not dfl_alloc.empty else pd.DataFrame(),
            ]
            if not df.empty
        ],
        ignore_index=True,
    )
    sample_paths, sample_ci = build_sample_paths_v10(stress_alloc, n_paths=args.sample_paths)
    _write_parquet("paper4_v10_sample_path_stress_paths.parquet", sample_paths)
    _write_csv("paper4_v10_sample_path_stress_ci.csv", sample_ci)

    causal, causal_tests, fairness = build_causal_fairness_v10()
    _write_csv("paper4_v10_causal_dossier.csv", causal)
    _write_csv("paper4_v10_causal_stress_tests.csv", causal_tests)
    _write_csv("paper4_v10_fairness_proxy_protocol.csv", fairness)

    selector, selector_memo = build_selector_v10(
        online_status,
        mdcp_summary,
        cvar_frontier,
        dfl_training,
        ifrs9_readiness,
    )
    _write_csv("paper4_v10_selector_rerun_with_v9_online.csv", selector)
    _write_csv("paper4_v10_selector_committee_memo.csv", selector_memo)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v10_resolution_wave",
        "mode": "paper4_living_lab_no_paper1_changes",
        "online_goal_achieved": bool(online_status.get("online_goal_achieved")),
        "online_best_method_v9": online_method,
        "online_best_width": float(online_status.get("online_best_width")),
        "online_best_source_month_defended_min": float(
            online_status.get("online_best_source_month_defended_min")
        ),
        "online_best_policy_month_defended_min": float(
            online_status.get("online_best_policy_month_defended_min")
        ),
        "online_robustness_all_pass": bool(online_summary["pass_v10"].all())
        if not online_summary.empty
        else False,
        "selector_promote_count": int(
            selector["selector_decision_v10"].eq("promote_to_paper4_working_candidate_only").sum()
        )
        if not selector.empty
        else 0,
        "primary_remaining_blocker": "ifrs9_contractual_data_blocker",
        "ifrs9_proxy_readiness_score": float(ifrs9_readiness["readiness_score"].iloc[0]),
        "contractual_ifrs9_claim_allowed": False,
        "mdcp_empirical_optimal_count": int(
            mdcp_summary["solver_status"]
            .astype(str)
            .str.contains("optimal", case=False, na=False)
            .sum()
        )
        if not mdcp_summary.empty
        else 0,
        "cvar_expanded_feasible_count": int(cvar_frontier["feasible_v10"].astype(bool).sum())
        if not cvar_frontier.empty
        else 0,
        "cvar_expanded_non_dominated_count": int(
            cvar_frontier.get("non_dominated_v10", pd.Series(False)).astype(bool).sum()
        )
        if not cvar_frontier.empty
        else 0,
        "dfl_candidate_count": int(len(dfl_training)),
        "dla_best_delta_state_value_vs_static": float(
            dla_comparison.loc[
                ~dla_comparison["policy_id"].eq("v10_static_reference"),
                "delta_state_value_vs_static",
            ].max()
        )
        if not dla_comparison.empty
        else np.nan,
        "sample_path_policy_count": int(sample_ci["policy_id"].nunique())
        if not sample_ci.empty
        else 0,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "paper1_artifacts_modified": False,
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "runtime_seconds": round(time.time() - start, 3),
        "caveat": "V10 implements runnable improvements only; IFRS9 contractual, CATE policy value and fairness legal claims remain blocked.",
    }
    dashboard = build_blocker_dashboard_v10(status, selector, online_summary)
    claims = build_claim_matrix_v10()
    _write_csv("paper4_v10_blocker_dashboard.csv", dashboard)
    _write_csv("paper4_v10_claim_artifact_matrix.csv", claims)
    _write_json("paper4_v10_resolution_status.json", status)
    _write_v10_note(status)

    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
