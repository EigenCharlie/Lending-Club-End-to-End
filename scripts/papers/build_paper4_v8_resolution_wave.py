"""Build Paper 4 v8 artifacts for the current resolution wave.

V8 is a living-lab implementation pass over the ten "resolvable now" items:

1. micro-iterate deployable online conformal around the v7 efficiency blocker;
2. create an adaptive CVaR bisection frontier with active caps;
3. put MDCP caps by source family inside the solver;
4. create selector governance v8 with a committee memo;
5. compare a DLA value-function proxy against a static monthly reference;
6. strengthen the causal dossier without allowing CATE policy value;
7. lock fairness as a permanent proxy-governance protocol unless protected
   attributes or an approved external proxy protocol appear;
8. train a simple SPO+/DFL-style surrogate against LP-like targets;
9. create a live blocker dashboard;
10. automate a v6/v7/v8 change report.

This script deliberately writes no ``paper4_final_promotion.json`` and does not
touch any Paper Estrella promotion artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs

from scripts.papers.build_paper4_extended_experiments import (
    BUDGET,
    _safe_read_csv,
    _safe_read_json,
)
from scripts.papers.build_paper4_living_lab_artifacts import DEFAULT_LGD
from scripts.papers.build_paper4_v4_open_priorities import FROZEN_PAPER1_CHAMPION
from scripts.papers.build_paper4_v6_priority_resolution import (
    OUT_ROOT,
    ROOT,
    SOURCE_FAMILIES,
    STATUS_DIR,
    TABLE_DIR,
    _coverage,
    _interval_width,
    _load_inputs,
    _prepare_online_frame,
    _prepare_solver_pool,
    _solve_linear_policy,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v7_resolution_loop import (
    _auditability_score,
    _historical_source_borrowing_mask,
    _targeted_structural_masks,
)

SCHEMA_VERSION = "2026-05-13.8"
RNG_SEED = 202605138


def _is_optimal(status: Any) -> bool:
    return "optimal" in str(status).lower()


def _json_dump(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def build_literature_registry_v8() -> pd.DataFrame:
    """Primary-source map from method idea to implemented artifact."""

    rows = [
        {
            "source_id": "gibbs_candes_aci_2021",
            "primary_source_url": "https://arxiv.org/abs/2106.00170",
            "concept": "adaptive conformal inference under distribution shift",
            "v8_use": "online alpha/width adaptation is treated as a sequential decision over a calibration knob",
            "implemented_artifact": "paper4_v8_online_efficiency_frontier.csv",
            "caveat": "Paper 4 still uses a credit-risk replay and source-month gates, not a theorem for all conditional cells.",
        },
        {
            "source_id": "angelopoulos_bates_fisch_lei_schuster_crc_2024",
            "primary_source_url": "https://arxiv.org/abs/2208.02814",
            "concept": "conformal risk control for monotone losses",
            "v8_use": "coverage, interval width and source-cell risk are recorded as controllable risk objects",
            "implemented_artifact": "paper4_v8_claim_artifact_matrix.csv",
            "caveat": "The current source-month gate is an empirical governance gate, not a formal CRC guarantee.",
        },
        {
            "source_id": "rockafellar_uryasev_cvar_2000",
            "primary_source_url": "https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf",
            "concept": "CVaR scenario optimization via auxiliary variables",
            "v8_use": "CVaR is optimized as a linear constraint and searched by bisection over active caps",
            "implemented_artifact": "paper4_v8_cvar_bisection_frontier.csv",
            "caveat": "The v8 solve is top-k/local-pool, not a decomposed 276k-loan full-universe solve.",
        },
        {
            "source_id": "elmachtoub_grigas_spo_2017",
            "primary_source_url": "https://arxiv.org/abs/1710.08005",
            "concept": "Smart Predict-then-Optimize and SPO+ decision-loss surrogate",
            "v8_use": "a simple decision-loss surrogate grid is trained against LP-like return/width/source/ECL targets",
            "implemented_artifact": "paper4_v8_dfl_surrogate_training.csv",
            "caveat": "This is not neural SPO+ training yet; it is a transparent surrogate target search.",
        },
        {
            "source_id": "ifrs9_standard_impairment",
            "primary_source_url": "https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf",
            "concept": "ECL, significant increase in credit risk and forward-looking information",
            "v8_use": "SICR and contractual-readiness blockers are recorded as accounting-scope gates",
            "implemented_artifact": "paper4_v8_selector_committee_memo.csv",
            "caveat": "No true contractual ECL claim is allowed without servicing, DPD, recovery timing and macro paths.",
        },
        {
            "source_id": "regulation_b_ecoa_federal_reserve_cfpb",
            "primary_source_url": "https://www.federalreserve.gov/frrs/regulations/background-and-summary-of-regulation-b.htm",
            "concept": "credit decisions cannot be claimed fair-lending compliant without protected-basis governance",
            "v8_use": "fairness remains proxy governance only and no legal fair-lending claim is made",
            "implemented_artifact": "paper4_v8_fairness_permanent_proxy_protocol.csv",
            "caveat": "Source-family stress is not a substitute for race/ethnicity/sex/age or an approved proxy protocol.",
        },
    ]
    return pd.DataFrame(rows)


def _source_month_metrics_v8(
    local: pd.DataFrame, method: str, min_support: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_month = (
        local.groupby(["policy_id", "issue_month"], as_index=False)
        .agg(
            n_funded=("loan_id", "nunique"),
            coverage_online_v8=("covered_online_v8", "mean"),
            avg_width_online_v8=("interval_width_online_v8", "mean"),
        )
        .rename(columns={"issue_month": "month"})
    )
    policy_month["online_method_v8"] = method
    policy_month["standalone_gate_cell"] = policy_month["n_funded"].ge(min_support)

    frames: list[pd.DataFrame] = []
    for source in SOURCE_FAMILIES:
        if source not in local.columns:
            continue
        src = (
            local.groupby(["policy_id", "issue_month", source], dropna=False, as_index=False)
            .agg(
                n=("loan_id", "nunique"),
                coverage_online_v8=("covered_online_v8", "mean"),
                avg_width_online_v8=("interval_width_online_v8", "mean"),
            )
            .rename(columns={"issue_month": "month", source: "source_value"})
        )
        src["source_id"] = source
        src["online_method_v8"] = method
        src["standalone_gate_cell"] = src["n"].ge(min_support)
        frames.append(src)
    source_month = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not source_month.empty:
        source_month["source_value"] = source_month["source_value"].astype(str)
    return policy_month, source_month


def _evaluate_online_v8(
    merged: pd.DataFrame,
    q: pd.Series,
    *,
    method: str,
    family: str,
    min_support: int,
    params: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    local = merged.copy()
    q = q.clip(0, 1)
    local["qhat_v8"] = q
    local["covered_online_v8"] = _coverage(local["y_true"], local["y_pred"], q)
    local["interval_width_online_v8"] = _interval_width(local["y_pred"], q)
    policy_month, source_month = _source_month_metrics_v8(local, method, min_support)
    defended_policy = policy_month[policy_month["standalone_gate_cell"].astype(bool)]
    defended_source = source_month[source_month["standalone_gate_cell"].astype(bool)]
    policy_min = (
        float(defended_policy["coverage_online_v8"].min()) if not defended_policy.empty else np.nan
    )
    source_min = (
        float(defended_source["coverage_online_v8"].min()) if not defended_source.empty else np.nan
    )
    avg_width = float(local["interval_width_online_v8"].mean())
    gate80 = bool(policy_min >= 0.80 and source_min >= 0.80)
    gate90 = bool(policy_min >= 0.90 and source_min >= 0.90)
    row = {
        "online_method_v8": method,
        "method_family": family,
        "deployable_without_current_outcomes": True,
        "min_effective_sample_size": min_support,
        "coverage_policy_month_raw_min": float(policy_month["coverage_online_v8"].min()),
        "coverage_policy_month_defended_min": policy_min,
        "coverage_source_month_raw_min": float(source_month["coverage_online_v8"].min())
        if not source_month.empty
        else np.nan,
        "coverage_source_month_defended_min": source_min,
        "avg_width_loan": avg_width,
        "avg_width_policy_month": float(policy_month["avg_width_online_v8"].mean()),
        "share_rows_widened": float((q > merged["qhat_v4"] + 1e-12).mean()),
        "share_rows_shrunk": float((q < merged["qhat_v4"] - 1e-12).mean()),
        "small_policy_month_cells_pooled": int(
            (~policy_month["standalone_gate_cell"].astype(bool)).sum()
        ),
        "small_source_month_cells_pooled": int(
            (~source_month["standalone_gate_cell"].astype(bool)).sum()
        )
        if not source_month.empty
        else 0,
        "gate_pass_80_defended": gate80,
        "gate_pass_90_defended": gate90,
        "efficiency_gate_width_95": bool(avg_width <= 0.95),
        "efficiency_gate_width_98": bool(avg_width <= 0.98),
        "promotion_eligible": bool(gate80 and avg_width <= 0.95),
        "width_gap_to_0p95": float(avg_width - 0.95),
        "parameters_json": _json_dump(params),
    }
    return row, policy_month, source_month


def _online_candidate_masks(merged: pd.DataFrame) -> dict[str, pd.Series]:
    masks = _targeted_structural_masks(merged)
    score_decile = merged["score_decile"].astype(str).str.replace(".0", "", regex=False)
    grade_dplus = merged["original_grade"].astype(str).isin(["D", "E", "F", "G"])
    grade_d = merged["original_grade"].astype(str).eq("D")
    dti_q3 = merged["dti_band"].astype(str).eq("dti_q3")
    score_low = score_decile.isin(["0", "1", "2"])
    income_q5 = merged["income_band"].astype(str).eq("inc_q5")
    period_2018h1 = merged["period"].astype(str).eq("2018H1")
    risk_score = (
        0.38 * grade_dplus.astype(float)
        + 0.25 * dti_q3.astype(float)
        + 0.22 * score_low.astype(float)
        + 0.10 * income_q5.astype(float)
        + 0.05 * period_2018h1.astype(float)
    )
    masks.update(
        {
            "gradeD_only": grade_d,
            "gradeDplus_only": grade_dplus,
            "dti_q3_only": dti_q3,
            "score_low_only": score_low,
            "gradeDplus_or_dti_q3": grade_dplus | dti_q3,
            "gradeD_and_score_low_or_dti": (grade_d & score_low) | dti_q3,
            "gradeDplus_score_low_or_income": (grade_dplus & score_low) | income_q5,
            "risk_score_ge_0p35": risk_score.ge(0.35),
            "risk_score_ge_0p45": risk_score.ge(0.45),
            "risk_score_ge_0p55": risk_score.ge(0.55),
            "risk_score_ge_0p65": risk_score.ge(0.65),
        }
    )
    return {name: mask.fillna(False).astype(bool) for name, mask in masks.items()}


def build_online_v8(
    allocations: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    min_support: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = _prepare_online_frame(allocations, online_intervals)
    masks = _online_candidate_masks(merged)
    historical = _historical_source_borrowing_mask(
        merged, prior_support=20, prior_coverage_floor=0.84
    )

    rows: list[dict[str, Any]] = []
    policy_maps: dict[str, pd.DataFrame] = {}
    source_maps: dict[str, pd.DataFrame] = {}
    q_maps: dict[str, pd.Series] = {}

    def add_candidate(method: str, family: str, q: pd.Series, params: dict[str, Any]) -> None:
        row, policy, source = _evaluate_online_v8(
            merged,
            q,
            method=method,
            family=family,
            min_support=min_support,
            params=params,
        )
        rows.append(row)
        policy_maps[method] = policy
        source_maps[method] = source
        q_maps[method] = q.clip(0, 1)

    add_candidate(
        "v8_reference_source_aware_guarded",
        "reference_v4_source_aware_guarded",
        merged["qhat_v4"].clip(0, 1),
        {"delta": 0.0, "base_multiplier": 1.0, "mask_name": "none"},
    )

    delta_grid = [round(x, 3) for x in np.arange(0.070, 0.084, 0.001)]
    multiplier_grid = [0.990, 0.995, 1.000]
    for mask_name, mask in masks.items():
        for multiplier in multiplier_grid:
            for delta in delta_grid:
                method = f"v8_{mask_name}_d{delta:.3f}_m{multiplier:.3f}"
                q = (merged["qhat_v4"] * multiplier + delta * mask.astype(float)).clip(0, 1)
                add_candidate(
                    method,
                    "deployable_mask_width_penalized_microgrid",
                    q,
                    {"mask_name": mask_name, "delta": delta, "base_multiplier": multiplier},
                )

    for multiplier in [0.990, 0.995, 1.000]:
        for delta in [0.040, 0.050, 0.060, 0.070, 0.080]:
            method = f"v8_prior_source_borrowing_d{delta:.3f}_m{multiplier:.3f}"
            q = (merged["qhat_v4"] * multiplier + delta * historical.astype(float)).clip(0, 1)
            add_candidate(
                method,
                "deployable_prior_source_month_borrowing_width_penalized",
                q,
                {
                    "mask_name": "prior_source_borrowing",
                    "delta": delta,
                    "base_multiplier": multiplier,
                    "prior_support": 20,
                    "prior_coverage_floor": 0.84,
                },
            )

    search = pd.DataFrame(rows).sort_values(
        ["promotion_eligible", "gate_pass_80_defended", "avg_width_loan"],
        ascending=[False, False, True],
    )
    passing = search[search["gate_pass_80_defended"].astype(bool)].copy()
    if passing.empty:
        best_method = search.sort_values(
            [
                "coverage_source_month_defended_min",
                "coverage_policy_month_defended_min",
                "avg_width_loan",
            ],
            ascending=[False, False, True],
        )["online_method_v8"].iloc[0]
    else:
        best_method = passing.sort_values("avg_width_loan")["online_method_v8"].iloc[0]
    width95 = passing[passing["avg_width_loan"].le(0.95)].sort_values("avg_width_loan")
    selected = list(
        dict.fromkeys(
            [
                "v8_reference_source_aware_guarded",
                best_method,
                width95["online_method_v8"].iloc[0] if not width95.empty else best_method,
            ]
        )
    )

    interval_frames = []
    for method in selected:
        local = merged.copy()
        q = q_maps[method]
        local["online_method_v8"] = method
        local["qhat_v8"] = q
        local["pd_low_online_v8"] = (local["y_pred"] - q).clip(0, 1)
        local["pd_high_online_v8"] = (local["y_pred"] + q).clip(0, 1)
        local["covered_online_v8"] = _coverage(local["y_true"], local["y_pred"], q)
        local["interval_width_online_v8"] = _interval_width(local["y_pred"], q)
        interval_frames.append(
            local[
                [
                    "policy_id",
                    "loan_id",
                    "issue_month",
                    "online_method_v8",
                    "qhat_v8",
                    "pd_low_online_v8",
                    "pd_high_online_v8",
                    "covered_online_v8",
                    "interval_width_online_v8",
                ]
            ]
        )
    intervals = pd.concat(interval_frames, ignore_index=True)
    policy = pd.concat([policy_maps[m] for m in selected], ignore_index=True)
    source = pd.concat([source_maps[m] for m in selected], ignore_index=True)

    best_row = search.loc[search["online_method_v8"].eq(best_method)].iloc[0]
    below_gate = search[~search["gate_pass_80_defended"].astype(bool)].copy()
    under95 = search[search["avg_width_loan"].le(0.95)].sort_values(
        ["coverage_source_month_defended_min", "coverage_policy_month_defended_min"],
        ascending=[False, False],
    )
    breakpoint = pd.DataFrame(
        [
            {
                "breakpoint_id": "v8_best_passing_source80",
                "online_method_v8": best_method,
                "coverage_source_month_defended_min": float(
                    best_row["coverage_source_month_defended_min"]
                ),
                "coverage_policy_month_defended_min": float(
                    best_row["coverage_policy_month_defended_min"]
                ),
                "avg_width_loan": float(best_row["avg_width_loan"]),
                "width_gap_to_0p95": float(best_row["width_gap_to_0p95"]),
                "interpretation": "best deployable method that passes defended 0.80 source/policy-month gate",
            },
            {
                "breakpoint_id": "v8_best_under_width95",
                "online_method_v8": under95["online_method_v8"].iloc[0]
                if not under95.empty
                else "",
                "coverage_source_month_defended_min": float(
                    under95["coverage_source_month_defended_min"].iloc[0]
                )
                if not under95.empty
                else np.nan,
                "coverage_policy_month_defended_min": float(
                    under95["coverage_policy_month_defended_min"].iloc[0]
                )
                if not under95.empty
                else np.nan,
                "avg_width_loan": float(under95["avg_width_loan"].iloc[0])
                if not under95.empty
                else np.nan,
                "width_gap_to_0p95": float(under95["width_gap_to_0p95"].iloc[0])
                if not under95.empty
                else np.nan,
                "interpretation": "best coverage method under width 0.95, whether or not it passes the source gate",
            },
            {
                "breakpoint_id": "v8_nearest_failing_neighbor",
                "online_method_v8": below_gate.sort_values("avg_width_loan")[
                    "online_method_v8"
                ].iloc[0]
                if not below_gate.empty
                else "",
                "coverage_source_month_defended_min": float(
                    below_gate.sort_values("avg_width_loan")[
                        "coverage_source_month_defended_min"
                    ].iloc[0]
                )
                if not below_gate.empty
                else np.nan,
                "coverage_policy_month_defended_min": float(
                    below_gate.sort_values("avg_width_loan")[
                        "coverage_policy_month_defended_min"
                    ].iloc[0]
                )
                if not below_gate.empty
                else np.nan,
                "avg_width_loan": float(
                    below_gate.sort_values("avg_width_loan")["avg_width_loan"].iloc[0]
                )
                if not below_gate.empty
                else np.nan,
                "width_gap_to_0p95": float(
                    below_gate.sort_values("avg_width_loan")["width_gap_to_0p95"].iloc[0]
                )
                if not below_gate.empty
                else np.nan,
                "interpretation": "lowest-width candidate that still fails defended 0.80 gate",
            },
        ]
    )
    return search, intervals, policy, source, breakpoint


def build_cvar_bisection_v8(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_pool_n: int,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_solver_pool(candidate_pool, online_intervals, max_pool_n)
    floors = [80_000.0, 110_000.0, 125_000.0, 140_000.0]
    rows: list[dict[str, Any]] = []
    allocs: list[pd.DataFrame] = []
    losses: list[pd.DataFrame] = []
    for floor in floors:
        low = 120_000.0
        high = 600_000.0
        best_feasible: dict[str, Any] | None = None
        for iteration in range(1, iterations + 1):
            cap = (low + high) / 2.0
            policy_id = f"v8_cvar_bisect_floor{int(floor)}_iter{iteration}_cap{int(cap)}"
            alloc, metrics, loss = _solve_linear_policy(
                pool,
                policy_id=policy_id,
                risk_tolerance=0.175,
                weak_penalty=0.04,
                width_penalty=0.06,
                cvar_cap=cap,
                return_floor=floor,
                max_weak_share=0.45,
                time_limit=120,
            )
            feasible = _is_optimal(metrics.get("solver_status"))
            metrics.update(
                {
                    "bisection_floor": floor,
                    "bisection_iteration": iteration,
                    "tested_cvar_cap": cap,
                    "search_low_before": low,
                    "search_high_before": high,
                    "feasible_v8": feasible,
                    "solver_lane_v8": "adaptive_cvar_bisection_constraint",
                }
            )
            if feasible:
                metrics["auditability_score_v8"] = _auditability_score(
                    float(metrics.get("weighted_qhat", np.nan)),
                    float(metrics.get("weighted_weak_source_proxy", np.nan)),
                    float(metrics.get("scenario_loss_cvar90", np.nan)),
                )
                best_feasible = metrics
                high = cap
                if not alloc.empty:
                    alloc["solver_lane_v8"] = "adaptive_cvar_bisection_constraint"
                    allocs.append(alloc)
                if not loss.empty:
                    loss["bisection_floor"] = floor
                    loss["tested_cvar_cap"] = cap
                    losses.append(loss)
            else:
                metrics["auditability_score_v8"] = np.nan
                low = cap
            rows.append(metrics)
        if best_feasible is not None:
            best = dict(best_feasible)
            best["policy_id"] = f"v8_cvar_bisect_floor{int(floor)}_best"
            best["bisection_iteration"] = "best_feasible"
            rows.append(best)
    frontier = pd.DataFrame(rows)
    if not frontier.empty:
        feasible = frontier[frontier["feasible_v8"].astype(bool)].copy()
        frontier["non_dominated_v8"] = False
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
            frontier.loc[idx, "non_dominated_v8"] = not bool(dominated)
        frontier = frontier.sort_values(
            ["feasible_v8", "non_dominated_v8", "bisection_floor", "scenario_loss_cvar90"],
            ascending=[False, False, True, True],
        )
    alloc = pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame()
    loss_df = pd.concat(losses, ignore_index=True) if losses else pd.DataFrame()
    return frontier, alloc, loss_df


def _solve_family_cap_policy(
    pool: pd.DataFrame,
    *,
    policy_id: str,
    risk_tolerance: float,
    weak_penalty: float,
    width_penalty: float,
    caps: dict[str, float],
    max_weighted_qhat: float | None,
    time_limit: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = len(pool)
    loan = pool["loan_amnt"].to_numpy(dtype=float)
    pd_high = pool["pd_high_alpha01"].to_numpy(dtype=float)
    base_return = pool["base_return_vec"].to_numpy(dtype=float)
    weak = pool["weak_source_proxy"].to_numpy(dtype=float)
    qhat = pool["qhat_v4"].to_numpy(dtype=float)
    flags = {
        "cap_grade_dplus": pool["original_grade"]
        .astype(str)
        .isin(["D", "E", "F", "G"])
        .to_numpy(dtype=float),
        "cap_dti_q3": pool["dti_band"].astype(str).eq("dti_q3").to_numpy(dtype=float),
        "cap_score_low": pool["score_decile"]
        .astype(str)
        .isin(["0", "1", "2"])
        .to_numpy(dtype=float),
        "cap_income_q5": pool["income_band"].astype(str).eq("inc_q5").to_numpy(dtype=float),
        "cap_period_2018h1": pool["period"].astype(str).eq("2018H1").to_numpy(dtype=float),
    }
    obj_vec = base_return - weak_penalty * loan * weak - width_penalty * loan * qhat
    model = pyo.ConcreteModel(policy_id)
    model.I = pyo.RangeSet(0, n - 1)
    model.x = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=(0, 1))
    exposure = sum(model.x[i] * loan[i] for i in model.I)
    model.budget = pyo.Constraint(expr=exposure <= BUDGET)
    model.min_budget = pyo.Constraint(expr=exposure >= 0.85 * BUDGET)
    model.pd_cap = pyo.Constraint(
        expr=sum(model.x[i] * loan[i] * pd_high[i] for i in model.I)
        <= risk_tolerance * (exposure + 1e-6)
    )
    for cap_name, cap_value in caps.items():
        flag = flags[cap_name]
        setattr(
            model,
            cap_name,
            pyo.Constraint(
                expr=sum(model.x[i] * loan[i] * flag[i] for i in model.I)
                <= cap_value * (exposure + 1e-6)
            ),
        )
    if max_weighted_qhat is not None:
        model.qhat_cap = pyo.Constraint(
            expr=sum(model.x[i] * loan[i] * qhat[i] for i in model.I)
            <= max_weighted_qhat * (exposure + 1e-6)
        )
    model.obj = pyo.Objective(
        expr=sum(model.x[i] * obj_vec[i] for i in model.I), sense=pyo.maximize
    )
    solver = Highs()
    solver.config.time_limit = time_limit
    t0 = time.perf_counter()
    try:
        results = solver.solve(model)
        status = str(getattr(results, "termination_condition", "unknown"))
    except RuntimeError as exc:
        elapsed = time.perf_counter() - t0
        return (
            pd.DataFrame(),
            {
                "policy_id": policy_id,
                "solver_status": f"infeasible_or_no_solution: {str(exc).splitlines()[0]}",
                "elapsed_seconds": elapsed,
                "risk_tolerance": risk_tolerance,
                "weak_penalty": weak_penalty,
                "width_penalty": width_penalty,
                "max_weighted_qhat": max_weighted_qhat if max_weighted_qhat is not None else np.nan,
                "funded_exposure": 0.0,
                "n_funded": 0,
                "objective_return": np.nan,
                "realized_return_proxy_lgd45": np.nan,
                "weighted_pd_high": np.nan,
                "weighted_qhat": np.nan,
                "weighted_weak_source_proxy": np.nan,
                "mdcp_family_gate_v8": False,
                "auditability_score_v8": np.nan,
                "caps_json": _json_dump(caps),
                **{cap_name.replace("cap_", "share_"): np.nan for cap_name in caps},
                **{f"{cap_name}_limit": cap_value for cap_name, cap_value in caps.items()},
            },
        )
    allocation = np.array([float(pyo.value(model.x[i])) for i in model.I])
    mask = allocation > 1e-8
    funded = pool.loc[mask].copy()
    funded["policy_id"] = policy_id
    funded["allocation_fraction"] = allocation[mask]
    funded["funded_exposure"] = funded["allocation_fraction"] * funded["loan_amnt"]
    funded["realized_return_proxy_lgd45"] = funded["funded_exposure"] * funded[
        "int_rate_decimal"
    ].astype(float) * (1 - funded["y_true"].astype(float)) - funded[
        "funded_exposure"
    ] * DEFAULT_LGD * funded["y_true"].astype(float)
    exposure_sum = float(funded["funded_exposure"].sum())
    metrics: dict[str, Any] = {
        "policy_id": policy_id,
        "solver_status": status,
        "elapsed_seconds": time.perf_counter() - t0,
        "risk_tolerance": risk_tolerance,
        "weak_penalty": weak_penalty,
        "width_penalty": width_penalty,
        "max_weighted_qhat": max_weighted_qhat if max_weighted_qhat is not None else np.nan,
        "n_funded": int(funded["loan_id"].nunique()),
        "funded_exposure": exposure_sum,
        "objective_return": float(np.dot(allocation, base_return)),
        "realized_return_proxy_lgd45": float(funded["realized_return_proxy_lgd45"].sum())
        if exposure_sum
        else np.nan,
        "weighted_pd_high": float(
            np.average(funded["pd_high_alpha01"], weights=funded["funded_exposure"])
        )
        if exposure_sum
        else np.nan,
        "weighted_qhat": float(np.average(funded["qhat_v4"], weights=funded["funded_exposure"]))
        if exposure_sum
        else np.nan,
        "weighted_weak_source_proxy": float(
            np.average(funded["weak_source_proxy"], weights=funded["funded_exposure"])
        )
        if exposure_sum
        else np.nan,
    }
    for cap_name, cap_value in caps.items():
        column = cap_name.replace("cap_", "share_")
        if exposure_sum:
            source_col = {
                "cap_grade_dplus": funded["original_grade"].astype(str).isin(["D", "E", "F", "G"]),
                "cap_dti_q3": funded["dti_band"].astype(str).eq("dti_q3"),
                "cap_score_low": funded["score_decile"].astype(str).isin(["0", "1", "2"]),
                "cap_income_q5": funded["income_band"].astype(str).eq("inc_q5"),
                "cap_period_2018h1": funded["period"].astype(str).eq("2018H1"),
            }[cap_name].astype(float)
            metrics[column] = float(np.average(source_col, weights=funded["funded_exposure"]))
        else:
            metrics[column] = np.nan
        metrics[f"{cap_name}_limit"] = cap_value
    metrics["mdcp_family_gate_v8"] = bool(
        _is_optimal(status)
        and exposure_sum > 0
        and all(
            (
                pd.isna(metrics.get(cap_name.replace("cap_", "share_")))
                or metrics[cap_name.replace("cap_", "share_")] <= cap_value + 1e-6
            )
            for cap_name, cap_value in caps.items()
        )
        and (
            max_weighted_qhat is None
            or metrics.get("weighted_qhat", np.inf) <= max_weighted_qhat + 1e-6
        )
    )
    metrics["auditability_score_v8"] = _auditability_score(
        float(metrics.get("weighted_qhat", np.nan)),
        float(metrics.get("weighted_weak_source_proxy", np.nan)),
    )
    return funded, metrics


def build_mdcp_family_caps_v8(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_pool_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = _prepare_solver_pool(candidate_pool, online_intervals, max_pool_n)
    specs = [
        (
            "v8_mdcp_family_relaxed",
            0.175,
            0.04,
            0.06,
            {
                "cap_grade_dplus": 0.70,
                "cap_dti_q3": 0.70,
                "cap_score_low": 1.00,
                "cap_income_q5": 0.75,
                "cap_period_2018h1": 0.75,
            },
            0.90,
        ),
        (
            "v8_mdcp_family_balanced",
            0.175,
            0.06,
            0.08,
            {
                "cap_grade_dplus": 0.60,
                "cap_dti_q3": 0.60,
                "cap_score_low": 0.98,
                "cap_income_q5": 0.70,
                "cap_period_2018h1": 0.70,
            },
            0.86,
        ),
        (
            "v8_mdcp_family_committee",
            0.1725,
            0.08,
            0.10,
            {
                "cap_grade_dplus": 0.55,
                "cap_dti_q3": 0.55,
                "cap_score_low": 0.95,
                "cap_income_q5": 0.68,
                "cap_period_2018h1": 0.65,
            },
            0.84,
        ),
        (
            "v8_mdcp_family_strict_review",
            0.170,
            0.12,
            0.12,
            {
                "cap_grade_dplus": 0.45,
                "cap_dti_q3": 0.45,
                "cap_score_low": 0.93,
                "cap_income_q5": 0.66,
                "cap_period_2018h1": 0.60,
            },
            0.82,
        ),
    ]
    allocs: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for policy_id, rt, weak_penalty, width_penalty, caps, qhat_cap in specs:
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
        metrics["solver_lane_v8"] = "mdcp_family_caps_inside_lp"
        metrics["caps_json"] = _json_dump(caps)
        rows.append(metrics)
        if not alloc.empty:
            alloc["solver_lane_v8"] = "mdcp_family_caps_inside_lp"
            allocs.append(alloc)
    summary = pd.DataFrame(rows).sort_values(
        ["mdcp_family_gate_v8", "auditability_score_v8", "objective_return"],
        ascending=[False, False, False],
    )
    alloc_df = pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame()
    return summary, alloc_df


def build_dla_value_function_v8(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_solver_pool(
        candidate_pool, online_intervals, max_n=min(len(candidate_pool), 10_000)
    )
    months = sorted(pool["issue_month"].dropna().unique())[:max_months]
    macro_rng = np.random.default_rng(RNG_SEED + 80)
    macro_series = []
    macro = 0.0
    for _ in months:
        macro = 0.72 * macro + float(macro_rng.normal(0, 0.28))
        macro_series.append(macro)

    def simulate(strategy: str, *, value_proxy: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(RNG_SEED + (91 if value_proxy else 92))
        cash = BUDGET
        outstanding: list[dict[str, Any]] = []
        states: list[dict[str, Any]] = []
        decisions: list[pd.DataFrame] = []
        for t, month in enumerate(months, start=1):
            macro_t = macro_series[t - 1]
            principal_in = 0.0
            interest_in = 0.0
            realized_loss = 0.0
            recovery_in = 0.0
            expected_loss = 0.0
            capital_used = 0.0
            next_outstanding: list[dict[str, Any]] = []
            for item in outstanding:
                age = t - item["funded_month_idx"] + 1
                remaining = max(item["remaining_balance"], 0.0)
                if remaining <= 0:
                    continue
                monthly_pd = np.clip((item["pd_high"] / 12.0) * math.exp(macro_t), 0.0001, 0.65)
                expected_loss += remaining * item["lgd"] * monthly_pd
                capital_used += remaining * (0.08 + 0.55 * item["pd_high"] + 0.08 * item["qhat"])
                default_event = bool(
                    rng.uniform() < monthly_pd or (item["y_true"] >= 0.5 and age >= 9)
                )
                if default_event:
                    loss = remaining * item["lgd"]
                    recovery = loss * float(np.clip(0.11 - 0.03 * max(macro_t, 0), 0.02, 0.14))
                    realized_loss += loss
                    recovery_in += recovery
                    cash += recovery
                    continue
                scheduled_principal = min(
                    remaining, item["original_exposure"] / max(item["term"], 1.0)
                )
                if rng.uniform() < np.clip(
                    0.014 + 0.030 * (1 - item["pd_high"]) - 0.010 * max(macro_t, 0), 0.002, 0.055
                ):
                    scheduled_principal = remaining
                interest = remaining * item["int_rate"] / 12.0
                principal_in += scheduled_principal
                interest_in += interest
                item["remaining_balance"] = remaining - scheduled_principal
                if item["remaining_balance"] > 1e-6 and age < item["term"]:
                    next_outstanding.append(item)
            cash += principal_in + interest_in - realized_loss
            outstanding_balance_pre = float(
                sum(item["remaining_balance"] for item in next_outstanding)
            )
            state_value_pre = cash + outstanding_balance_pre - expected_loss - 0.12 * capital_used
            coverage_multiplier = 0.78 if macro_t > 0.55 else 0.92 if macro_t > 0.20 else 1.0
            deployment_budget = max(
                0.0,
                min(cash * (0.34 if value_proxy else 0.38) * coverage_multiplier, BUDGET * 0.32),
            )
            available = pool[pool["issue_month"].eq(month)].copy()
            if not available.empty and deployment_budget >= 1_000:
                available["capital_charge_v8"] = available["loan_amnt"] * (
                    0.08 + 0.55 * available["pd_high_alpha01"] + 0.08 * available["qhat_v4"]
                )
                available["ecl_proxy_v8"] = (
                    available["loan_amnt"]
                    * available["pd_high_alpha01"]
                    * DEFAULT_LGD
                    * (1 + max(macro_t, 0))
                )
                available["continuation_value_proxy_v8"] = (
                    0.025 * cash
                    - 0.10 * outstanding_balance_pre
                    - 0.20 * expected_loss
                    - 0.04 * capital_used
                )
                if value_proxy:
                    available["decision_score_v8"] = (
                        available["base_return_vec"]
                        - 0.24 * available["capital_charge_v8"]
                        - 0.70 * available["ecl_proxy_v8"]
                        - 0.10 * available["loan_amnt"] * available["weak_source_proxy"]
                        - 0.07 * available["loan_amnt"] * available["qhat_v4"]
                        + 0.01 * available["continuation_value_proxy_v8"]
                    )
                    action = "fund_by_value_function_proxy"
                else:
                    available["decision_score_v8"] = available["solver_score_seed"]
                    action = "fund_by_static_reference_score"
                local = available.sort_values("decision_score_v8", ascending=False).copy()
                local["cum_amount"] = local["loan_amnt"].cumsum()
                funded = local[local["cum_amount"].le(deployment_budget)].copy()
                if funded.empty:
                    funded = local.head(1).copy()
                funded["policy_id"] = strategy
                funded["decision_month"] = month
                funded["month_idx"] = t
                funded["funded_exposure"] = funded["loan_amnt"]
                funded["macro_state_v8"] = macro_t
                funded["state_value_pre_decision_v8"] = state_value_pre
                funded["action_v8"] = action
                deployed = float(funded["funded_exposure"].sum())
                cash -= deployed
                for _, row in funded.iterrows():
                    outstanding.append(
                        {
                            "loan_id": row["loan_id"],
                            "funded_month_idx": t,
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
                decisions.append(
                    funded[
                        [
                            "policy_id",
                            "decision_month",
                            "month_idx",
                            "loan_id",
                            "funded_exposure",
                            "capital_charge_v8",
                            "ecl_proxy_v8",
                            "qhat_v4",
                            "weak_source_proxy",
                            "macro_state_v8",
                            "state_value_pre_decision_v8",
                            "action_v8",
                        ]
                    ]
                )
            outstanding = next_outstanding + [
                item for item in outstanding if item["funded_month_idx"] == t
            ]
            outstanding_balance = float(sum(item["remaining_balance"] for item in outstanding))
            state_value = cash + outstanding_balance - expected_loss - 0.12 * capital_used
            states.append(
                {
                    "policy_id": strategy,
                    "month_idx": t,
                    "calendar_month": month,
                    "cash_end": cash,
                    "principal_in": principal_in,
                    "interest_in": interest_in,
                    "realized_loss": realized_loss,
                    "recovery_in": recovery_in,
                    "expected_loss": expected_loss,
                    "capital_used": capital_used,
                    "outstanding_items": len(outstanding),
                    "outstanding_balance_proxy": outstanding_balance,
                    "macro_state_v8": macro_t,
                    "state_value_proxy_v8": state_value,
                    "continuation_value_proxy_v8": state_value - cash,
                    "decision_scope": "loan_level_monthly_value_function_proxy"
                    if value_proxy
                    else "loan_level_monthly_static_reference_score",
                }
            )
        state_df = pd.DataFrame(states)
        decision_df = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()
        return decision_df, state_df

    dynamic_decisions, dynamic_state = simulate("v8_dla_value_function_proxy", value_proxy=True)
    static_decisions, static_state = simulate("v8_static_monthly_reference", value_proxy=False)
    decisions = pd.concat([dynamic_decisions, static_decisions], ignore_index=True)
    trace = pd.concat([dynamic_state, static_state], ignore_index=True)
    summary = (
        trace.groupby("policy_id", as_index=False)
        .agg(
            horizon_months=("month_idx", "max"),
            final_cash=("cash_end", "last"),
            final_state_value_proxy_v8=("state_value_proxy_v8", "last"),
            cumulative_realized_loss=("realized_loss", "sum"),
            cumulative_expected_loss=("expected_loss", "sum"),
            cumulative_capital_used=("capital_used", "sum"),
            cumulative_recovery=("recovery_in", "sum"),
            final_outstanding_balance=("outstanding_balance_proxy", "last"),
        )
        .merge(
            decisions.groupby("policy_id", as_index=False).agg(
                funded_loans=("loan_id", "nunique"),
                total_funded_exposure=("funded_exposure", "sum"),
            ),
            on="policy_id",
            how="left",
        )
    )
    dynamic = summary[summary["policy_id"].eq("v8_dla_value_function_proxy")].iloc[0]
    static = summary[summary["policy_id"].eq("v8_static_monthly_reference")].iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "comparison_id": "v8_dla_value_function_vs_static_reference",
                "dynamic_policy_id": "v8_dla_value_function_proxy",
                "static_policy_id": "v8_static_monthly_reference",
                "delta_final_state_value": float(
                    dynamic["final_state_value_proxy_v8"] - static["final_state_value_proxy_v8"]
                ),
                "delta_final_cash": float(dynamic["final_cash"] - static["final_cash"]),
                "delta_cumulative_realized_loss": float(
                    dynamic["cumulative_realized_loss"] - static["cumulative_realized_loss"]
                ),
                "delta_cumulative_expected_loss": float(
                    dynamic["cumulative_expected_loss"] - static["cumulative_expected_loss"]
                ),
                "delta_total_funded_exposure": float(
                    dynamic["total_funded_exposure"] - static["total_funded_exposure"]
                ),
                "interpretation": "positive state-value delta supports further ADP work; negative delta means the proxy is not yet useful",
            }
        ]
    )
    return decisions, trace, summary, comparison


def build_causal_dossier_v8() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dossier_v4 = _safe_read_csv(TABLE_DIR / "paper4_causal_high_rate_v4_dossier.csv")
    balance = _safe_read_csv(TABLE_DIR / "paper4_causal_high_rate_v4_balance.csv")
    overlap = _safe_read_csv(TABLE_DIR / "paper4_v5_causal_overlap_bins.csv")
    falsification = _safe_read_csv(TABLE_DIR / "paper4_v5_causal_falsification_tests.csv")
    sensitivity = _safe_read_csv(TABLE_DIR / "paper4_v5_causal_sensitivity_formal.csv")
    cate_gate = _safe_read_csv(TABLE_DIR / "paper4_v6_cate_gate.csv")

    max_smd = float(balance["smd_trimmed_ipw_att"].max()) if not balance.empty else np.nan
    overlap_min_n = int(overlap["n"].min()) if not overlap.empty else 0
    overlap_trim_issue = (
        bool((overlap["treatment_share"].lt(0.05) | overlap["treatment_share"].gt(0.95)).any())
        if not overlap.empty
        else True
    )
    falsification_pass = (
        bool(falsification["pass"].astype(bool).all()) if not falsification.empty else False
    )
    sensitivity_rows_blocked = (
        int((~sensitivity["policy_value_allowed"].astype(bool)).sum())
        if not sensitivity.empty
        else 0
    )
    cate_allowed = bool(cate_gate["policy_value_allowed"].iloc[0]) if not cate_gate.empty else False
    sensitivity_stable = (
        bool(dossier_v4["sensitivity_sign_stable_6pp"].iloc[0]) if not dossier_v4.empty else False
    )
    checks = pd.DataFrame(
        [
            {
                "check_id": "balance_trimmed_ipw",
                "metric": "max_smd_trimmed_ipw_att",
                "value": max_smd,
                "threshold": 0.10,
                "pass_v8": bool(max_smd <= 0.10) if not pd.isna(max_smd) else False,
                "interpretation": "balance is adequate for dossier use if all key SMDs are below 0.10",
            },
            {
                "check_id": "overlap_grade_period",
                "metric": "min_grade_period_n",
                "value": overlap_min_n,
                "threshold": 100,
                "pass_v8": bool(overlap_min_n >= 100 and not overlap_trim_issue),
                "interpretation": "overlap bins must avoid tiny cells and near-deterministic treatment shares",
            },
            {
                "check_id": "falsification_tests",
                "metric": "all_placebos_pass",
                "value": float(falsification_pass),
                "threshold": 1.0,
                "pass_v8": falsification_pass,
                "interpretation": "leakage/placebo tests are necessary but not sufficient for causal policy value",
            },
            {
                "check_id": "hidden_bias_sensitivity",
                "metric": "sensitivity_sign_stable_6pp",
                "value": float(sensitivity_stable),
                "threshold": 1.0,
                "pass_v8": sensitivity_stable,
                "interpretation": "policy value remains blocked unless sign/magnitude survives hidden-bias stress",
            },
            {
                "check_id": "cate_policy_value_gate",
                "metric": "cate_policy_value_allowed",
                "value": float(cate_allowed),
                "threshold": 1.0,
                "pass_v8": cate_allowed,
                "interpretation": "CATE cannot enter the solver while intervals remain mostly inconclusive",
            },
        ]
    )
    dossier = pd.DataFrame(
        [
            {
                "treatment_id": "high_rate_within_grade",
                "balance_pass_v8": bool(
                    checks.loc[checks["check_id"].eq("balance_trimmed_ipw"), "pass_v8"].iloc[0]
                ),
                "overlap_pass_v8": bool(
                    checks.loc[checks["check_id"].eq("overlap_grade_period"), "pass_v8"].iloc[0]
                ),
                "falsification_pass_v8": falsification_pass,
                "sensitivity_pass_v8": sensitivity_stable,
                "cate_policy_value_allowed": False,
                "sensitivity_rows_blocked": sensitivity_rows_blocked,
                "decision_v8": "dossier_usable_policy_value_blocked",
                "required_to_unblock": "stable hidden-bias sensitivity, cleaner causal outcome, and CATE intervals that do not mostly cross zero",
            }
        ]
    )
    if not overlap.empty:
        overlap_out = overlap.copy()
        overlap_out["overlap_bin_pass_v8"] = overlap_out["n"].ge(100) & overlap_out[
            "treatment_share"
        ].between(0.05, 0.95)
    else:
        overlap_out = pd.DataFrame()
    combined = pd.concat(
        [
            falsification.assign(section_v8="falsification")
            if not falsification.empty
            else pd.DataFrame(),
            sensitivity.assign(section_v8="sensitivity")
            if not sensitivity.empty
            else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    return dossier, overlap_out, combined


def build_fairness_protocol_v8() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protocol_item": "protected_attributes",
                "current_status": "not_available",
                "decision_v8": "no_fair_lending_legal_claim",
                "allowed_claim": "source/proxy governance diagnostic only",
                "blocked_claim": "race/ethnicity/sex/age protected-class compliance or disparate-impact conclusion",
                "required_to_change": "approved protected-attribute data, BISG/proxy protocol, or external legal/MRM review",
            },
            {
                "protocol_item": "geography_income_dti_grade_sources",
                "current_status": "available_as_business_sources",
                "decision_v8": "use_for_governance_stress_only",
                "allowed_claim": "worst-source and composition stress by observable business segments",
                "blocked_claim": "legal fair-lending conclusion",
                "required_to_change": "mapping to protected attributes with documented governance approval",
            },
            {
                "protocol_item": "paper4_language",
                "current_status": "locked",
                "decision_v8": "permanent_proxy_only_until_data_changes",
                "allowed_claim": "Paper 4 can report proxy gaps and source coverage gates",
                "blocked_claim": "Paper 4 cannot claim fair-lending compliance",
                "required_to_change": "new data/protocol artifact plus explicit committee approval",
            },
        ]
    )


def build_dfl_surrogate_v8(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_pool_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = _prepare_solver_pool(candidate_pool, online_intervals, max_pool_n).copy()
    pool["ecl_proxy"] = pool["loan_amnt"] * pool["pd_high_alpha01"].astype(float) * DEFAULT_LGD
    grids = []
    candidates = []
    best_target = _safe_read_csv(TABLE_DIR / "paper4_v7_mdcp_soft_penalty_solver_summary.csv")
    target_return = (
        float(best_target["objective_return"].max()) if not best_target.empty else 120_000.0
    )
    for width_weight in [0.02, 0.05, 0.08, 0.12]:
        for source_weight in [0.02, 0.06, 0.10, 0.14]:
            for ecl_weight in [0.20, 0.45, 0.70]:
                policy_id = f"v8_dfl_surrogate_w{width_weight:.2f}_s{source_weight:.2f}_e{ecl_weight:.2f}".replace(
                    ".", "p"
                )
                work = pool.copy()
                work["surrogate_score_v8"] = (
                    work["base_return_vec"]
                    - width_weight * work["loan_amnt"] * work["qhat_v4"]
                    - source_weight * work["loan_amnt"] * work["weak_source_proxy"]
                    - ecl_weight * work["ecl_proxy"]
                )
                selected_rows = []
                exposure = 0.0
                pd_numer = 0.0
                for _, row in work.sort_values("surrogate_score_v8", ascending=False).iterrows():
                    amount = float(row["loan_amnt"])
                    if exposure + amount > BUDGET:
                        continue
                    next_exposure = exposure + amount
                    next_pd = pd_numer + amount * float(row["pd_high_alpha01"])
                    if next_pd > 0.175 * next_exposure:
                        continue
                    selected_rows.append(row)
                    exposure = next_exposure
                    pd_numer = next_pd
                    if exposure >= 0.98 * BUDGET:
                        break
                selected = pd.DataFrame(selected_rows)
                if selected.empty:
                    continue
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
                objective_return = float(selected["base_return_vec"].sum())
                decision_loss_proxy = float(
                    max(0.0, target_return - objective_return)
                    + 25_000 * weighted_qhat
                    + 20_000 * weighted_weak
                )
                row = {
                    "policy_id": policy_id,
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
                    "auditability_score_v8": _auditability_score(weighted_qhat, weighted_weak),
                    "decision_loss_proxy_v8": decision_loss_proxy,
                    "training_status_v8": "transparent_surrogate_not_neural_spo_plus",
                }
                grids.append(row)
                candidates.append(
                    selected[
                        [
                            "policy_id",
                            "loan_id",
                            "issue_month",
                            "loan_amnt",
                            "funded_exposure",
                            "surrogate_score_v8",
                            "pd_high_alpha01",
                            "qhat_v4",
                            "weak_source_proxy",
                            "realized_return_proxy_lgd45",
                        ]
                    ].head(250)
                )
    training = pd.DataFrame(grids).sort_values(
        ["decision_loss_proxy_v8", "auditability_score_v8", "objective_return"],
        ascending=[True, False, False],
    )
    candidate_rows = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    return training, candidate_rows


def build_selector_governance_v8(
    online: pd.DataFrame,
    mdcp: pd.DataFrame,
    cvar: pd.DataFrame,
    dfl: pd.DataFrame,
    readiness: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ifrs9_readiness = float(readiness["readiness_score"].iloc[0]) if not readiness.empty else 0.0
    best_online = (
        online[online["gate_pass_80_defended"].astype(bool)].sort_values("avg_width_loan").head(1)
    )
    online_source_min = (
        float(best_online["coverage_source_month_defended_min"].iloc[0])
        if not best_online.empty
        else np.nan
    )
    online_width = float(best_online["avg_width_loan"].iloc[0]) if not best_online.empty else np.nan
    online_eff_pass = bool(online_width <= 0.95) if not pd.isna(online_width) else False
    online_gate_pass = bool(online_source_min >= 0.80) if not pd.isna(online_source_min) else False

    rows = []
    for lane, df, id_col in [
        ("mdcp_family_solver", mdcp, "policy_id"),
        (
            "cvar_bisection",
            cvar[cvar.get("feasible_v8", pd.Series(False, index=cvar.index)).astype(bool)].head(12)
            if not cvar.empty
            else cvar,
            "policy_id",
        ),
        ("dfl_surrogate", dfl.head(12) if not dfl.empty else dfl, "policy_id"),
    ]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            policy_id = str(row[id_col])
            solver_ok = (
                _is_optimal(row.get("solver_status", "optimal"))
                if lane != "dfl_surrogate"
                else True
            )
            audit = float(
                row.get("auditability_score_v8", row.get("auditability_score_v7", np.nan))
            )
            mdcp_ok = (
                bool(row.get("mdcp_family_gate_v8", True)) if lane == "mdcp_family_solver" else True
            )
            cvar_ok = bool(row.get("feasible_v8", True)) if lane == "cvar_bisection" else True
            if not solver_ok:
                decision = "kill"
                blocker = "solver_infeasible"
            elif not online_gate_pass:
                decision = "park"
                blocker = "online_source_month_gate"
            elif not online_eff_pass:
                decision = "review"
                blocker = "online_efficiency_width_gt_0p95"
            elif ifrs9_readiness < 0.75:
                decision = "review"
                blocker = "ifrs9_contractual_data_blocker"
            elif lane == "mdcp_family_solver" and not mdcp_ok:
                decision = "review"
                blocker = "mdcp_family_cap_failed"
            elif lane == "cvar_bisection" and not cvar_ok:
                decision = "park"
                blocker = "cvar_infeasible"
            elif audit < 0.35:
                decision = "review"
                blocker = "auditability_score_low"
            else:
                decision = "promote_to_paper4_working_candidate_only"
                blocker = "none_but_final_promotion_disabled"
            rows.append(
                {
                    "policy_id": policy_id,
                    "lane": lane,
                    "solver_ok": solver_ok,
                    "online_source_month_min": online_source_min,
                    "online_width": online_width,
                    "online_gate_pass": online_gate_pass,
                    "online_efficiency_pass_0p95": online_eff_pass,
                    "ifrs9_readiness": ifrs9_readiness,
                    "auditability_score_v8": audit,
                    "mdcp_gate_v8": mdcp_ok,
                    "cvar_gate_v8": cvar_ok,
                    "selector_decision_v8": decision,
                    "primary_blocker_v8": blocker,
                    "paper1_artifacts_modified": False,
                    "final_promotion_json_created": False,
                }
            )
    results = pd.DataFrame(rows).sort_values(
        ["selector_decision_v8", "auditability_score_v8"], ascending=[True, False]
    )
    memo = pd.DataFrame(
        [
            {
                "threshold_id": "online_source_month_min",
                "threshold_value": 0.80,
                "committee_rationale": "Minimum defended source-month coverage required before any Paper 4 working candidate can be considered.",
                "current_value": online_source_min,
                "decision_effect": "blocks promotion if below 0.80",
            },
            {
                "threshold_id": "online_width_efficiency",
                "threshold_value": 0.95,
                "committee_rationale": "Coverage must not be bought with materially wider intervals than the current operational gate.",
                "current_value": online_width,
                "decision_effect": "review if source gate passes but width remains above 0.95",
            },
            {
                "threshold_id": "ifrs9_contractual_readiness",
                "threshold_value": 0.75,
                "committee_rationale": "A contractual IFRS9 claim needs most core servicing/macro components present.",
                "current_value": ifrs9_readiness,
                "decision_effect": "review/data-blocked below 0.75",
            },
            {
                "threshold_id": "fair_lending_claim",
                "threshold_value": 0.00,
                "committee_rationale": "No protected attributes or approved proxy protocol means legal claim remains disallowed.",
                "current_value": 0.0,
                "decision_effect": "proxy governance only",
            },
        ]
    )
    return results, memo


def build_blocker_dashboard_v8(status: dict[str, Any], selector: pd.DataFrame) -> pd.DataFrame:
    promote_count = (
        int(selector["selector_decision_v8"].eq("promote_to_paper4_working_candidate_only").sum())
        if not selector.empty
        else 0
    )
    rows = [
        (
            "online_efficiency",
            "near_resolved" if status["online_best_width"] <= 0.96 else "active",
            "coverage gate passes but width must reach <=0.95",
            "micro-iterate weak-cell predictor",
        ),
        (
            "cvar_bisection",
            "resolved" if status["cvar_bisection_feasible_count"] > 0 else "active",
            "active-cap frontier exists but not full-universe",
            "scale/decompose",
        ),
        (
            "mdcp_family_caps",
            "resolved" if status["mdcp_family_cap_optimal_count"] > 0 else "active",
            "family caps inside LP now solve for at least one spec",
            "link caps to empirical coverage",
        ),
        (
            "selector_governance",
            "resolved" if not selector.empty else "active",
            "committee memo exists and blocks final promotion",
            "rerun after online/IFRS9 changes",
        ),
        (
            "dla_value_function",
            "near_resolved",
            "value-function proxy exists, not Bellman/ADP optimal",
            "rollout/ADP and common paths",
        ),
        (
            "causal_cate",
            "theory_blocked",
            "dossier improved but CATE policy value remains disallowed",
            "formal identification and sensitivity",
        ),
        (
            "fairness",
            "data_blocked",
            "proxy-only protocol locked",
            "protected attributes or approved proxy",
        ),
        (
            "dfl_surrogate",
            "near_resolved",
            "transparent surrogate exists, not full SPO+ training",
            "train real decision-loss model",
        ),
        (
            "ifrs9_contractual",
            "data_blocked",
            "readiness remains below contractual-claim threshold",
            "servicing/DPD/recovery/macro data",
        ),
        (
            "paper1_freeze",
            "resolved",
            "Paper Estrella artifacts remain untouched",
            "continue Paper 4 only",
        ),
    ]
    dashboard = pd.DataFrame(
        rows, columns=["blocker_id", "status_v8", "current_diagnosis", "next_action"]
    )
    dashboard["selector_promote_count_v8"] = promote_count
    return dashboard


def build_change_report_v8(status_v8: dict[str, Any]) -> pd.DataFrame:
    v6 = _safe_read_json(STATUS_DIR / "paper4_v6_priority_resolution_status.json")
    v7 = _safe_read_json(STATUS_DIR / "paper4_v7_resolution_status.json")
    rows = [
        {
            "metric": "online_best_deployable_width",
            "v6_value": v6.get("online_best_deployable_width"),
            "v7_value": v7.get("online_best_width"),
            "v8_value": status_v8.get("online_best_width"),
            "interpretation": "v8 tests whether the v7 source gate can be kept while reducing width toward 0.95",
        },
        {
            "metric": "online_source_month_gate",
            "v6_value": v6.get("online_best_deployable_source_min"),
            "v7_value": v7.get("online_best_source_month_defended_min"),
            "v8_value": status_v8.get("online_best_source_month_defended_min"),
            "interpretation": "source-month coverage is the non-negotiable promotion gate",
        },
        {
            "metric": "mdcp_solver_success_count",
            "v6_value": v6.get("mdcp_solver_optimal_count"),
            "v7_value": v7.get("mdcp_soft_optimal_count"),
            "v8_value": status_v8.get("mdcp_family_cap_optimal_count"),
            "interpretation": "hard global caps failed; soft penalties and family caps are the viable path",
        },
        {
            "metric": "cvar_frontier_feasible_count",
            "v6_value": v6.get("cvar_solver_optimal_count"),
            "v7_value": v7.get("cvar_frontier_feasible_count"),
            "v8_value": status_v8.get("cvar_bisection_feasible_count"),
            "interpretation": "v8 moves from arbitrary caps to bisection around active feasibility boundaries",
        },
        {
            "metric": "contractual_ifrs9_readiness_score",
            "v6_value": v6.get("contractual_ifrs9_readiness_score"),
            "v7_value": v7.get("contractual_ifrs9_readiness_score"),
            "v8_value": status_v8.get("contractual_ifrs9_readiness_score"),
            "interpretation": "the readiness blocker is stable and data-bound",
        },
        {
            "metric": "fair_lending_legal_claim",
            "v6_value": v6.get("fair_lending_legal_claim"),
            "v7_value": v7.get("fair_lending_legal_claim"),
            "v8_value": status_v8.get("fair_lending_legal_claim"),
            "interpretation": "fairness remains proxy governance only",
        },
    ]
    return pd.DataFrame(rows)


def build_claim_matrix_v8() -> pd.DataFrame:
    rows = [
        (
            "Literature registry",
            "implemented_primary_source_mapping",
            "paper4_v8_literature_to_method_registry.csv",
            "19ar-v8-method-foundations-and-online.qmd",
            "primary sources inform method design, not publication claims",
        ),
        (
            "Online efficiency frontier",
            "implemented_micro_iteration",
            "paper4_v8_online_efficiency_frontier.csv",
            "19ar-v8-method-foundations-and-online.qmd",
            "promotion still blocked if width > 0.95",
        ),
        (
            "Online breakpoint report",
            "implemented_breakpoint_diagnostic",
            "paper4_v8_online_breakpoint_report.csv",
            "19ar-v8-method-foundations-and-online.qmd",
            "shows closest pass/fail boundary",
        ),
        (
            "CVaR bisection frontier",
            "implemented_active_cap_search",
            "paper4_v8_cvar_bisection_frontier.csv",
            "19as-v8-solvers-selector-dla.qmd",
            "top-k/local-pool, not full-universe decomposition",
        ),
        (
            "MDCP family-cap solver",
            "implemented_family_caps_inside_lp",
            "paper4_v8_mdcp_family_cap_solver_summary.csv",
            "19as-v8-solvers-selector-dla.qmd",
            "family caps are governance proxies",
        ),
        (
            "Selector governance v8",
            "implemented_committee_memo",
            "paper4_v8_selector_governance_results.csv",
            "19as-v8-solvers-selector-dla.qmd",
            "no final promotion JSON",
        ),
        (
            "DLA value-function proxy",
            "implemented_dynamic_vs_static_comparison",
            "paper4_v8_dla_value_function_summary.csv",
            "19as-v8-solvers-selector-dla.qmd",
            "not Bellman-optimal ADP",
        ),
        (
            "Causal dossier v8",
            "implemented_stronger_blocker_dossier",
            "paper4_v8_causal_dossier.csv",
            "19at-v8-causal-fairness-hybrids-dashboard.qmd",
            "CATE policy value remains blocked",
        ),
        (
            "Fairness protocol v8",
            "implemented_permanent_proxy_only_protocol",
            "paper4_v8_fairness_permanent_proxy_protocol.csv",
            "19at-v8-causal-fairness-hybrids-dashboard.qmd",
            "no fair-lending legal claim",
        ),
        (
            "DFL surrogate v8",
            "implemented_transparent_decision_loss_surrogate",
            "paper4_v8_dfl_surrogate_training.csv",
            "19at-v8-causal-fairness-hybrids-dashboard.qmd",
            "not neural SPO+/DFL",
        ),
        (
            "Blocker dashboard",
            "implemented_live_blocker_dashboard",
            "paper4_v8_blocker_dashboard.csv",
            "19au-v8-results-and-pending.qmd",
            "status is living-lab, not final publication decision",
        ),
        (
            "Change report v6-v7-v8",
            "implemented_automated_change_report",
            "paper4_v8_v6_v7_change_report.csv",
            "19au-v8-results-and-pending.qmd",
            "compares status JSONs only",
        ),
        (
            "V8 status",
            "implemented_living_lab_status",
            "paper4_v8_resolution_status.json",
            "19au-v8-results-and-pending.qmd",
            "Paper Estrella frozen",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["priority", "claim_status", "artifact", "quarto_page", "caveat"]
    )


def update_manifest_v8(claims: pd.DataFrame) -> None:
    manifest_path = TABLE_DIR / "paper4_table0_source_manifest.csv"
    manifest = _safe_read_csv(manifest_path)
    if manifest.empty:
        return
    rows = []
    for _, row in claims.iterrows():
        base_dir = "status" if str(row["artifact"]).endswith(".json") else "tables"
        artifact_path = OUT_ROOT / base_dir / row["artifact"]
        rows.append(
            {
                "artifact": str(artifact_path.relative_to(ROOT)),
                "source_paper": "Paper 4 v8",
                "role": row["priority"],
                "status": row["claim_status"],
                "run_tag": "paper4_v8_resolution_wave_2026-05-13",
                "caveat": row["caveat"],
                "path_exists": artifact_path.exists(),
            }
        )
    new = pd.DataFrame(rows)
    manifest = manifest[~manifest["artifact"].isin(set(new["artifact"]))]
    manifest = pd.concat([manifest, new], ignore_index=True)
    manifest["path_exists"] = manifest["artifact"].map(lambda p: (ROOT / p).exists())
    manifest.to_csv(manifest_path, index=False)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-pool-n", type=int, default=12_000)
    parser.add_argument("--cvar-iterations", type=int, default=5)
    parser.add_argument("--dla-months", type=int, default=18)
    parser.add_argument("--min-support", type=int, default=5)
    args = parser.parse_args(list(argv) if argv is not None else None)

    _, candidate_pool, allocations, _, online_intervals = _load_inputs()
    literature = build_literature_registry_v8()
    _write_csv("paper4_v8_literature_to_method_registry.csv", literature)

    online_search, online_selected, online_policy, online_source, breakpoint = build_online_v8(
        allocations, online_intervals, min_support=args.min_support
    )
    _write_csv("paper4_v8_online_efficiency_frontier.csv", online_search)
    _write_parquet("paper4_v8_online_selected_intervals.parquet", online_selected)
    _write_parquet("paper4_v8_online_policy_month.parquet", online_policy)
    _write_parquet("paper4_v8_online_source_month.parquet", online_source)
    _write_csv("paper4_v8_online_breakpoint_report.csv", breakpoint)

    cvar_frontier, cvar_alloc, cvar_losses = build_cvar_bisection_v8(
        candidate_pool,
        online_intervals,
        max_pool_n=args.solver_pool_n,
        iterations=args.cvar_iterations,
    )
    _write_csv("paper4_v8_cvar_bisection_frontier.csv", cvar_frontier)
    _write_parquet("paper4_v8_cvar_bisection_allocations.parquet", cvar_alloc)
    _write_csv("paper4_v8_cvar_bisection_scenario_losses.csv", cvar_losses)

    mdcp_summary, mdcp_alloc = build_mdcp_family_caps_v8(
        candidate_pool, online_intervals, max_pool_n=args.solver_pool_n
    )
    _write_csv("paper4_v8_mdcp_family_cap_solver_summary.csv", mdcp_summary)
    _write_parquet("paper4_v8_mdcp_family_cap_allocations.parquet", mdcp_alloc)

    dla_decisions, dla_trace, dla_summary, dla_comparison = build_dla_value_function_v8(
        candidate_pool, online_intervals, max_months=args.dla_months
    )
    _write_parquet("paper4_v8_dla_value_function_decisions.parquet", dla_decisions)
    _write_csv("paper4_v8_dla_value_function_trace.csv", dla_trace)
    _write_csv("paper4_v8_dla_value_function_summary.csv", dla_summary)
    _write_csv("paper4_v8_dla_vs_static_path_comparison.csv", dla_comparison)

    causal_dossier, causal_overlap, causal_sensitivity = build_causal_dossier_v8()
    _write_csv("paper4_v8_causal_dossier.csv", causal_dossier)
    _write_csv("paper4_v8_causal_overlap_bins.csv", causal_overlap)
    _write_csv("paper4_v8_causal_falsification_sensitivity.csv", causal_sensitivity)
    fairness = build_fairness_protocol_v8()
    _write_csv("paper4_v8_fairness_permanent_proxy_protocol.csv", fairness)

    dfl_training, dfl_candidates = build_dfl_surrogate_v8(
        candidate_pool, online_intervals, max_pool_n=args.solver_pool_n
    )
    _write_csv("paper4_v8_dfl_surrogate_training.csv", dfl_training)
    _write_csv("paper4_v8_dfl_surrogate_candidates.csv", dfl_candidates)

    readiness = _safe_read_csv(TABLE_DIR / "paper4_v7_contractual_ifrs9_readiness.csv")
    selector, committee_memo = build_selector_governance_v8(
        online_search, mdcp_summary, cvar_frontier, dfl_training, readiness
    )
    _write_csv("paper4_v8_selector_governance_results.csv", selector)
    _write_csv("paper4_v8_selector_committee_memo.csv", committee_memo)

    best_online = (
        online_search[online_search["gate_pass_80_defended"].astype(bool)]
        .sort_values("avg_width_loan")
        .head(1)
    )
    if best_online.empty:
        best_online = online_search.sort_values(
            [
                "coverage_source_month_defended_min",
                "coverage_policy_month_defended_min",
                "avg_width_loan",
            ],
            ascending=[False, False, True],
        ).head(1)
    best = best_online.iloc[0]
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v8_resolution_wave",
        "mode": "paper4_living_lab_no_paper1_changes",
        "paper1_artifacts_modified": False,
        "paper4_final_promotion_created": False,
        "paper1_frozen_champion": FROZEN_PAPER1_CHAMPION,
        "priorities_targeted": 10,
        "online_best_method": str(best["online_method_v8"]),
        "online_best_method_family": str(best["method_family"]),
        "online_best_policy_month_defended_min": float(best["coverage_policy_month_defended_min"]),
        "online_best_source_month_defended_min": float(best["coverage_source_month_defended_min"]),
        "online_best_width": float(best["avg_width_loan"]),
        "online_gate80_defended_deployable_exists": bool(
            online_search["gate_pass_80_defended"].astype(bool).any()
        ),
        "online_gate80_width95_exists": bool(
            (
                online_search["gate_pass_80_defended"].astype(bool)
                & online_search["avg_width_loan"].le(0.95)
            ).any()
        ),
        "online_efficiency_blocker": bool(float(best["avg_width_loan"]) > 0.95),
        "online_promotion_eligible": bool(online_search["promotion_eligible"].astype(bool).any()),
        "cvar_bisection_feasible_count": int(cvar_frontier["feasible_v8"].astype(bool).sum())
        if not cvar_frontier.empty
        else 0,
        "cvar_bisection_non_dominated_count": int(
            cvar_frontier.get("non_dominated_v8", pd.Series(dtype=bool)).astype(bool).sum()
        )
        if not cvar_frontier.empty
        else 0,
        "mdcp_family_cap_optimal_count": int(
            mdcp_summary["solver_status"]
            .astype(str)
            .str.contains("optimal", case=False, na=False)
            .sum()
        )
        if not mdcp_summary.empty
        else 0,
        "mdcp_family_gate_count": int(
            mdcp_summary.get("mdcp_family_gate_v8", pd.Series(dtype=bool)).astype(bool).sum()
        )
        if not mdcp_summary.empty
        else 0,
        "selector_promote_count": int(
            selector["selector_decision_v8"].eq("promote_to_paper4_working_candidate_only").sum()
        )
        if not selector.empty
        else 0,
        "contractual_ifrs9_readiness_score": float(readiness["readiness_score"].iloc[0])
        if not readiness.empty
        else np.nan,
        "dla_value_function_implemented": not dla_summary.empty,
        "dla_value_delta_vs_static": float(dla_comparison["delta_final_state_value"].iloc[0])
        if not dla_comparison.empty
        else np.nan,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "dfl_surrogate_rows": int(len(dfl_training)),
    }

    dashboard = build_blocker_dashboard_v8(status, selector)
    _write_csv("paper4_v8_blocker_dashboard.csv", dashboard)
    change_report = build_change_report_v8(status)
    _write_csv("paper4_v8_v6_v7_change_report.csv", change_report)
    claims = build_claim_matrix_v8()
    _write_csv("paper4_v8_claim_artifact_matrix.csv", claims)
    status["generated_artifacts"] = claims["artifact"].tolist()
    _write_json("paper4_v8_resolution_status.json", status)
    update_manifest_v8(claims)
    _write_note(
        "paper4_v8_resolution_wave_memo.qmd",
        """---
title: "Paper 4 v8 Resolution Wave Memo"
format: html
---

# Paper 4 v8 Resolution Wave Memo

V8 turns the current blocker list into reproducible artifacts.  The main result
is not a final promotion; it is a sharper laboratory map:

- online conformal is micro-iterated at the source-month/width boundary;
- CVaR is searched by bisection rather than arbitrary caps;
- MDCP moves from soft penalty to family caps inside the LP;
- DLA receives a value-function proxy and a static reference comparison;
- CATE and fairness remain explicitly blocked for policy/legal claims;
- SPO+/DFL is represented as a transparent decision-loss surrogate, not a
  neural training claim.

No Paper Estrella artifact is changed and no final Paper 4 promotion JSON is
created.
""",
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
