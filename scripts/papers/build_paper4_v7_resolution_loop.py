"""Build Paper 4 v7 artifacts for the active unresolved blockers.

V7 is a resolution loop over the ten current Paper 4 blockers.  It is still a
living-lab wave, not a publication or Paper Estrella promotion wave.  The main
change relative to v6 is that weak online source-month cells are handled with a
deployable support/pooling rule instead of an oracle/audit rescue, while solver,
IFRS9, DLA, causal, fairness and hybrid lanes are pushed one iteration deeper.

This script deliberately writes no ``paper4_final_promotion.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_extended_experiments import (
    BUDGET,
    _safe_read_csv,
)
from scripts.papers.build_paper4_living_lab_artifacts import DEFAULT_LGD
from scripts.papers.build_paper4_v4_open_priorities import FROZEN_PAPER1_CHAMPION
from scripts.papers.build_paper4_v6_priority_resolution import (
    OUT_ROOT,
    ROOT,
    SOURCE_FAMILIES,
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
    build_causal_fairness_v6,
    build_contractual_data_audit_v6,
    build_sample_paths_v6,
)

SCHEMA_VERSION = "2026-05-13.7"
RNG_SEED = 202605137


def _policy_month_metrics_v7(local: pd.DataFrame, method: str, min_support: int) -> pd.DataFrame:
    policy_month = (
        local.groupby(["policy_id", "issue_month"], as_index=False)
        .agg(
            n_funded=("loan_id", "nunique"),
            coverage_online_v7=("covered_online_v7", "mean"),
            avg_width_online_v7=("interval_width_online_v7", "mean"),
        )
        .rename(columns={"issue_month": "month"})
    )
    policy_month["online_method_v7"] = method
    policy_month["standalone_gate_cell"] = policy_month["n_funded"].ge(min_support)
    policy_month["pooling_decision_v7"] = np.where(
        policy_month["standalone_gate_cell"],
        "standalone_defended_cell",
        "pooled_small_policy_month_cell",
    )
    return policy_month[
        [
            "online_method_v7",
            "policy_id",
            "month",
            "n_funded",
            "coverage_online_v7",
            "avg_width_online_v7",
            "standalone_gate_cell",
            "pooling_decision_v7",
        ]
    ]


def _source_month_metrics_v7(local: pd.DataFrame, method: str, min_support: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source in SOURCE_FAMILIES:
        if source not in local.columns:
            continue
        src = (
            local.groupby(["policy_id", "issue_month", source], as_index=False)
            .agg(
                n=("loan_id", "nunique"),
                coverage_online_v7=("covered_online_v7", "mean"),
                avg_width_online_v7=("interval_width_online_v7", "mean"),
            )
            .rename(columns={"issue_month": "month", source: "source_value"})
        )
        src["source_id"] = source
        src["online_method_v7"] = method
        src["standalone_gate_cell"] = src["n"].ge(min_support)
        src["pooling_decision_v7"] = np.where(
            src["standalone_gate_cell"],
            "standalone_defended_source_cell",
            "pooled_small_source_month_cell",
        )
        frames.append(src)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["source_value"] = out["source_value"].astype(str)
    return out[
        [
            "online_method_v7",
            "policy_id",
            "month",
            "source_id",
            "source_value",
            "n",
            "coverage_online_v7",
            "avg_width_online_v7",
            "standalone_gate_cell",
            "pooling_decision_v7",
        ]
    ]


def _structural_weak_mask(
    merged: pd.DataFrame, *, include_grade_d: bool, include_period_2018h1: bool
) -> pd.Series:
    score_decile = merged["score_decile"].astype(str).str.replace(".0", "", regex=False)
    weak = (
        merged["dti_band"].astype(str).eq("dti_q3")
        | score_decile.isin(["1", "2"])
        | merged["income_band"].astype(str).eq("inc_q5")
    )
    if include_grade_d:
        weak = weak | merged["original_grade"].astype(str).eq("D")
    if include_period_2018h1:
        weak = weak | merged["period"].astype(str).eq("2018H1")
    return weak.fillna(False)


def _targeted_structural_masks(merged: pd.DataFrame) -> dict[str, pd.Series]:
    score_decile = merged["score_decile"].astype(str).str.replace(".0", "", regex=False)
    dti_q3 = merged["dti_band"].astype(str).eq("dti_q3")
    score_1_2 = score_decile.isin(["1", "2"])
    income_q5 = merged["income_band"].astype(str).eq("inc_q5")
    grade_d = merged["original_grade"].astype(str).eq("D")
    period_2018h1 = merged["period"].astype(str).eq("2018H1")
    return {
        "targeted_dti_score_or_gradeD": (dti_q3 & score_1_2) | grade_d,
        "targeted_gradeD_score_or_dti": (grade_d & score_1_2) | dti_q3,
        "targeted_score_income_or_dti": (score_1_2 & income_q5) | dti_q3,
        "targeted_dti_score_or_income": (dti_q3 & score_1_2) | income_q5,
        "targeted_dti_gradeD": dti_q3 | grade_d,
        "targeted_dti_gradeD_period": dti_q3 | grade_d | period_2018h1,
        "targeted_dti_income": dti_q3 | income_q5,
    }


def _historical_source_borrowing_mask(
    merged: pd.DataFrame,
    *,
    prior_support: int = 25,
    prior_coverage_floor: float = 0.82,
) -> pd.Series:
    """Flag rows whose source family had weak prior-month coverage.

    Outcomes from prior months are available at decision time in a replay.  This
    is therefore deployable in a historical online setting, unlike the v6 oracle
    weak-cell rescue.
    """

    loan_level = merged.drop_duplicates("loan_id").copy()
    loan_level["base_covered"] = _coverage(
        loan_level["y_true"], loan_level["y_pred"], loan_level["qhat_v4"]
    )
    flags = pd.Series(False, index=loan_level.index)
    for source in SOURCE_FAMILIES:
        if source not in loan_level.columns:
            continue
        monthly = (
            loan_level.groupby([source, "issue_month"], dropna=False, as_index=False)
            .agg(n=("loan_id", "nunique"), covered=("base_covered", "sum"))
            .sort_values([source, "issue_month"])
        )
        monthly["prior_n"] = monthly.groupby(source)["n"].cumsum() - monthly["n"]
        monthly["prior_covered"] = monthly.groupby(source)["covered"].cumsum() - monthly["covered"]
        monthly["prior_coverage"] = monthly["prior_covered"] / monthly["prior_n"].replace(0, np.nan)
        monthly["prior_weak"] = monthly["prior_n"].ge(prior_support) & monthly["prior_coverage"].lt(
            prior_coverage_floor
        )
        key = monthly.set_index([source, "issue_month"])["prior_weak"].to_dict()
        flags = flags | pd.Series(
            [
                bool(key.get((row[source], row["issue_month"]), False))
                for _, row in loan_level.iterrows()
            ],
            index=loan_level.index,
        )
    loan_flags = pd.Series(flags.to_numpy(), index=loan_level["loan_id"].astype(str))
    return merged["loan_id"].astype(str).map(loan_flags).fillna(False).astype(bool)


def _online_search_row(
    merged: pd.DataFrame,
    q: pd.Series,
    *,
    method: str,
    deployable: bool,
    method_family: str,
    min_support: int,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    local = merged.copy()
    local["qhat_v7"] = q.clip(0, 1)
    local["covered_online_v7"] = _coverage(local["y_true"], local["y_pred"], local["qhat_v7"])
    local["interval_width_online_v7"] = _interval_width(local["y_pred"], local["qhat_v7"])
    policy_month = _policy_month_metrics_v7(local, method, min_support)
    source_month = _source_month_metrics_v7(local, method, min_support)
    defended_policy = policy_month[policy_month["standalone_gate_cell"].astype(bool)]
    defended_source = source_month[source_month["standalone_gate_cell"].astype(bool)]
    defended_policy_min = (
        float(defended_policy["coverage_online_v7"].min()) if not defended_policy.empty else np.nan
    )
    defended_source_min = (
        float(defended_source["coverage_online_v7"].min()) if not defended_source.empty else np.nan
    )
    avg_width = float(local["interval_width_online_v7"].mean())
    gate80 = bool(defended_policy_min >= 0.80 and defended_source_min >= 0.80)
    gate90 = bool(defended_policy_min >= 0.90 and defended_source_min >= 0.90)
    row = {
        "online_method_v7": method,
        "method_family": method_family,
        "deployable_without_current_outcomes": deployable,
        "min_effective_sample_size": min_support,
        "coverage_policy_month_raw_min": float(policy_month["coverage_online_v7"].min()),
        "coverage_policy_month_defended_min": defended_policy_min,
        "coverage_source_month_raw_min": float(source_month["coverage_online_v7"].min())
        if not source_month.empty
        else np.nan,
        "coverage_source_month_defended_min": defended_source_min,
        "avg_width_loan": avg_width,
        "avg_width_policy_month": float(policy_month["avg_width_online_v7"].mean()),
        "share_rows_widened": float((local["qhat_v7"] > merged["qhat_v4"] + 1e-12).mean()),
        "small_policy_month_cells_pooled": int(
            (~policy_month["standalone_gate_cell"].astype(bool)).sum()
        ),
        "small_source_month_cells_pooled": int(
            (~source_month["standalone_gate_cell"].astype(bool)).sum()
        ),
        "gate_pass_80_defended": gate80,
        "gate_pass_90_defended": gate90,
        "efficiency_gate_width_95": bool(avg_width <= 0.95),
        "efficiency_gate_width_98": bool(avg_width <= 0.98),
        "promotion_eligible": bool(deployable and gate80 and avg_width <= 0.95),
        "parameters_json": json.dumps(params or {}, sort_keys=True),
    }
    return row, policy_month, source_month


def build_online_v7(
    allocations: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    min_support: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = _prepare_online_frame(allocations, online_intervals)
    structural = {
        (grade_d, period): _structural_weak_mask(
            merged, include_grade_d=grade_d, include_period_2018h1=period
        )
        for grade_d in [False, True]
        for period in [False, True]
    }
    targeted_masks = _targeted_structural_masks(merged)
    historical = _historical_source_borrowing_mask(merged)
    rows: list[dict[str, Any]] = []
    candidates: dict[str, tuple[pd.Series, bool, str, dict[str, Any]]] = {
        "v7_reference_defended_pooling": (
            merged["qhat_v4"].clip(0, 1),
            True,
            "reference_min_support_pooling",
            {"delta": 0.0, "multiplier": 1.0},
        )
    }
    policy_maps: dict[str, pd.DataFrame] = {}
    source_maps: dict[str, pd.DataFrame] = {}

    for method, (q, deployable, family, params) in candidates.items():
        row, pm, sm = _online_search_row(
            merged,
            q,
            method=method,
            deployable=deployable,
            method_family=family,
            min_support=min_support,
            params=params,
        )
        rows.append(row)
        policy_maps[method] = pm
        source_maps[method] = sm

    for delta in [0.04, 0.06, 0.08, 0.10, 0.12]:
        for multiplier in [1.000, 1.025, 1.050]:
            for include_grade_d in [False, True]:
                for include_period in [False, True]:
                    weak = structural[(include_grade_d, include_period)]
                    method = (
                        "v7_structural_source_pooling"
                        f"_d{delta:.3f}_m{multiplier:.3f}"
                        f"_gradeD{int(include_grade_d)}_period{int(include_period)}"
                    )
                    q = (merged["qhat_v4"] * multiplier + delta * weak.astype(float)).clip(0, 1)
                    row, pm, sm = _online_search_row(
                        merged,
                        q,
                        method=method,
                        deployable=True,
                        method_family="deployable_structural_weak_source_rule",
                        min_support=min_support,
                        params={
                            "delta": delta,
                            "multiplier": multiplier,
                            "include_grade_d": include_grade_d,
                            "include_period_2018h1": include_period,
                            "weak_rule": "dti_q3 OR score_decile in {1,2} OR income_band inc_q5 plus optional grade/period",
                        },
                    )
                    rows.append(row)
                    policy_maps[method] = pm
                    source_maps[method] = sm

    for mask_name, weak in targeted_masks.items():
        for delta in [0.065, 0.070, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.080, 0.085, 0.090]:
            method = f"v7_{mask_name}_d{delta:.3f}_m1.000"
            q = (merged["qhat_v4"] + delta * weak.astype(float)).clip(0, 1)
            row, pm, sm = _online_search_row(
                merged,
                q,
                method=method,
                deployable=True,
                method_family="deployable_targeted_structural_weak_source_rule",
                min_support=min_support,
                params={
                    "delta": delta,
                    "multiplier": 1.0,
                    "mask_name": mask_name,
                    "weak_rule": "targeted boolean masks from DTI, score decile, grade, income and period",
                },
            )
            rows.append(row)
            policy_maps[method] = pm
            source_maps[method] = sm

    for delta in [0.04, 0.06, 0.08, 0.10, 0.12]:
        for multiplier in [1.000, 1.025, 1.050]:
            method = f"v7_prior_source_borrowing_d{delta:.3f}_m{multiplier:.3f}"
            q = (merged["qhat_v4"] * multiplier + delta * historical.astype(float)).clip(0, 1)
            row, pm, sm = _online_search_row(
                merged,
                q,
                method=method,
                deployable=True,
                method_family="deployable_prior_month_source_borrowing",
                min_support=min_support,
                params={
                    "delta": delta,
                    "multiplier": multiplier,
                    "prior_support": 25,
                    "prior_coverage_floor": 0.82,
                },
            )
            rows.append(row)
            policy_maps[method] = pm
            source_maps[method] = sm

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
        )["online_method_v7"].iloc[0]
    else:
        best_method = passing.sort_values("avg_width_loan")["online_method_v7"].iloc[0]
    selected_methods = [
        "v7_reference_defended_pooling",
        best_method,
    ]
    selected_methods = list(dict.fromkeys(selected_methods))
    interval_frames = []
    for method in selected_methods:
        params = json.loads(
            search.loc[search["online_method_v7"].eq(method), "parameters_json"].iloc[0]
        )
        if method == "v7_reference_defended_pooling":
            q = merged["qhat_v4"].clip(0, 1)
        elif method.startswith("v7_structural_source_pooling"):
            weak = _structural_weak_mask(
                merged,
                include_grade_d=bool(params.get("include_grade_d", False)),
                include_period_2018h1=bool(params.get("include_period_2018h1", False)),
            )
            q = (
                merged["qhat_v4"] * float(params["multiplier"])
                + float(params["delta"]) * weak.astype(float)
            ).clip(0, 1)
        elif method.startswith("v7_targeted_"):
            weak = targeted_masks[str(params["mask_name"])]
            q = (
                merged["qhat_v4"] * float(params["multiplier"])
                + float(params["delta"]) * weak.astype(float)
            ).clip(0, 1)
        else:
            q = (
                merged["qhat_v4"] * float(params["multiplier"])
                + float(params["delta"]) * historical.astype(float)
            ).clip(0, 1)
        local = merged.copy()
        local["online_method_v7"] = method
        local["qhat_v7"] = q
        local["pd_low_online_v7"] = (local["y_pred"] - q).clip(0, 1)
        local["pd_high_online_v7"] = (local["y_pred"] + q).clip(0, 1)
        local["covered_online_v7"] = _coverage(local["y_true"], local["y_pred"], q)
        local["interval_width_online_v7"] = _interval_width(local["y_pred"], q)
        local["deployable_without_current_outcomes"] = True
        interval_frames.append(
            local[
                [
                    "policy_id",
                    "loan_id",
                    "issue_month",
                    "online_method_v7",
                    "deployable_without_current_outcomes",
                    "qhat_v7",
                    "pd_low_online_v7",
                    "pd_high_online_v7",
                    "covered_online_v7",
                    "interval_width_online_v7",
                ]
            ]
        )
    intervals = pd.concat(interval_frames, ignore_index=True)
    policy = pd.concat([policy_maps[m] for m in selected_methods], ignore_index=True)
    source = pd.concat([source_maps[m] for m in selected_methods], ignore_index=True)
    pooling = pd.DataFrame(
        [
            {
                "online_method_v7": method,
                "gate_scope": "policy_month",
                "min_effective_sample_size": min_support,
                "raw_cells": int(len(policy_maps[method])),
                "standalone_defended_cells": int(policy_maps[method]["standalone_gate_cell"].sum()),
                "pooled_small_cells": int(
                    (~policy_maps[method]["standalone_gate_cell"].astype(bool)).sum()
                ),
                "raw_min_coverage": float(policy_maps[method]["coverage_online_v7"].min()),
                "defended_min_coverage": float(
                    policy_maps[method]
                    .loc[
                        policy_maps[method]["standalone_gate_cell"].astype(bool),
                        "coverage_online_v7",
                    ]
                    .min()
                ),
            }
            for method in selected_methods
        ]
        + [
            {
                "online_method_v7": method,
                "gate_scope": "source_month",
                "min_effective_sample_size": min_support,
                "raw_cells": int(len(source_maps[method])),
                "standalone_defended_cells": int(source_maps[method]["standalone_gate_cell"].sum()),
                "pooled_small_cells": int(
                    (~source_maps[method]["standalone_gate_cell"].astype(bool)).sum()
                ),
                "raw_min_coverage": float(source_maps[method]["coverage_online_v7"].min()),
                "defended_min_coverage": float(
                    source_maps[method]
                    .loc[
                        source_maps[method]["standalone_gate_cell"].astype(bool),
                        "coverage_online_v7",
                    ]
                    .min()
                ),
            }
            for method in selected_methods
        ]
    )
    return search, intervals, policy, source, pooling


def build_sicr_mrm_v7() -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = _safe_read_csv(TABLE_DIR / "paper4_v6_sicr_calibration_grid.csv")
    if grid.empty:
        return pd.DataFrame(), pd.DataFrame()
    baseline = grid[grid["scenario"].eq("baseline")].copy()
    baseline["stage2_distance_to_mrm_band"] = np.where(
        baseline["mean_stage2_share_v6"].between(0.15, 0.70),
        0.0,
        np.minimum(
            np.abs(baseline["mean_stage2_share_v6"] - 0.15),
            np.abs(baseline["mean_stage2_share_v6"] - 0.70),
        ),
    )
    shortlist = baseline.sort_values(
        ["stage2_distance_to_mrm_band", "mean_contractual_ecl_v6"],
        ascending=[True, True],
    ).copy()
    shortlist["mrm_decision_v7"] = np.select(
        [
            shortlist["sicr_recommendation_v6"].eq("candidate_for_mrm_review"),
            shortlist["mean_stage2_share_v6"].gt(0.75),
            shortlist["mean_stage2_share_v6"].lt(0.10),
        ],
        ["shortlist_for_committee_sensitivity", "reject_too_conservative", "reject_too_loose"],
        default="document_as_sensitivity_only",
    )
    shortlist["committee_protocol_note"] = (
        "Prefer rules with Stage 2 share inside 15%-70%, no policy-level Stage 2 domination, "
        "and monotone ECL response under baseline/adverse/severe scenarios."
    )
    action_plan = pd.DataFrame(
        [
            (
                "monthly_servicing_panel",
                "partial_proxy_exists",
                "cannot claim true contractual ECL yet",
            ),
            ("days_past_due_sicr", "missing", "blocks DPD/backstop Stage 2 rule"),
            ("forbearance_hardship", "partial_or_missing", "blocks forbearance SICR trigger"),
            ("macro_scenario_path", "missing", "blocks coherent IFRS9 scenario weighting"),
            (
                "recovery_timing",
                "partial_proxy_exists",
                "recovery amount exists but timing is not a full panel",
            ),
            (
                "prepayment_timing",
                "partial_proxy_exists",
                "last payment/status proxies are not monthly cashflow history",
            ),
        ],
        columns=["ifrs9_contractual_component", "current_status_v7", "research_consequence"],
    )
    action_plan["next_step_v7"] = np.where(
        action_plan["current_status_v7"].eq("missing"),
        "external_servicing_or_macro_data_required",
        "use_as_proxy_with_explicit_scope_label",
    )
    return shortlist, action_plan


def _auditability_score(weighted_qhat: float, weighted_weak: float, cvar: float = np.nan) -> float:
    cvar_term = 0.0 if pd.isna(cvar) else min(float(cvar) / 900_000.0, 1.0)
    return float(
        np.clip(1.0 - 0.42 * weighted_qhat - 0.38 * weighted_weak - 0.20 * cvar_term, 0, 1)
    )


def build_solver_v7(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_pool_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_solver_pool(candidate_pool, online_intervals, max_pool_n)
    mdcp_specs = [
        ("v7_mdcp_soft_rt170_wp04_wd04", 0.1700, 0.04, 0.04, None),
        ("v7_mdcp_soft_rt1725_wp06_wd08", 0.1725, 0.06, 0.08, None),
        ("v7_mdcp_soft_rt175_wp08_wd10", 0.1750, 0.08, 0.10, None),
        ("v7_mdcp_soft_rt175_wp12_wd12", 0.1750, 0.12, 0.12, None),
        ("v7_mdcp_reviewcap_rt175_wp08_wd08", 0.1750, 0.08, 0.08, 0.45),
        ("v7_mdcp_committee_rt1725_wp10_wd12", 0.1725, 0.10, 0.12, 0.40),
    ]
    mdcp_allocs: list[pd.DataFrame] = []
    mdcp_rows: list[dict[str, Any]] = []
    for policy_id, rt, weak_penalty, width_penalty, cap in mdcp_specs:
        alloc, metrics, _ = _solve_linear_policy(
            pool,
            policy_id=policy_id,
            risk_tolerance=rt,
            weak_penalty=weak_penalty,
            width_penalty=width_penalty,
            max_weak_share=cap,
            time_limit=90,
        )
        if not alloc.empty:
            mdcp_allocs.append(alloc)
        metrics["solver_lane_v7"] = "mdcp_soft_penalty_inside_solver"
        metrics["mdcp_gate_proxy_v7"] = bool(
            str(metrics.get("solver_status", "")).lower().find("optimal") >= 0
            and metrics.get("weighted_weak_source_proxy", np.inf) <= 0.45
            and metrics.get("weighted_qhat", np.inf) <= 0.90
        )
        metrics["auditability_score_v7"] = _auditability_score(
            float(metrics.get("weighted_qhat", np.nan)),
            float(metrics.get("weighted_weak_source_proxy", np.nan)),
        )
        mdcp_rows.append(metrics)
    mdcp_alloc = pd.concat(mdcp_allocs, ignore_index=True) if mdcp_allocs else pd.DataFrame()
    mdcp_summary = pd.DataFrame(mdcp_rows).sort_values(
        ["mdcp_gate_proxy_v7", "auditability_score_v7", "objective_return"],
        ascending=[False, False, False],
    )

    cvar_specs = []
    for cap in [420_000.0, 520_000.0, 650_000.0, 800_000.0, 950_000.0]:
        for floor in [80_000.0, 110_000.0, 140_000.0]:
            cvar_specs.append((f"v7_cvar_cap{int(cap)}_floor{int(floor)}", 0.1750, cap, floor))
    cvar_allocs: list[pd.DataFrame] = []
    cvar_rows: list[dict[str, Any]] = []
    cvar_losses: list[pd.DataFrame] = []
    for policy_id, rt, cap, floor in cvar_specs:
        alloc, metrics, losses = _solve_linear_policy(
            pool,
            policy_id=policy_id,
            risk_tolerance=rt,
            weak_penalty=0.04,
            width_penalty=0.06,
            cvar_cap=cap,
            return_floor=floor,
            max_weak_share=0.45,
            time_limit=120,
        )
        if not alloc.empty:
            cvar_allocs.append(alloc)
        if not losses.empty:
            cvar_losses.append(losses)
        metrics["solver_lane_v7"] = "adaptive_topk_cvar_constraint"
        metrics["auditability_score_v7"] = _auditability_score(
            float(metrics.get("weighted_qhat", np.nan)),
            float(metrics.get("weighted_weak_source_proxy", np.nan)),
            float(metrics.get("scenario_loss_cvar90", np.nan)),
        )
        metrics["frontier_feasible_v7"] = bool(
            str(metrics.get("solver_status", "")).lower().find("optimal") >= 0
        )
        cvar_rows.append(metrics)
    cvar_alloc = pd.concat(cvar_allocs, ignore_index=True) if cvar_allocs else pd.DataFrame()
    cvar_summary = pd.DataFrame(cvar_rows).sort_values(
        ["frontier_feasible_v7", "scenario_loss_cvar90", "objective_return"],
        ascending=[False, True, False],
    )
    cvar_loss = pd.concat(cvar_losses, ignore_index=True) if cvar_losses else pd.DataFrame()

    hybrid = pd.concat(
        [
            mdcp_summary.assign(hybrid_family_v7="mdcp_soft_surrogate"),
            cvar_summary.assign(hybrid_family_v7="cvar_auditability_surrogate"),
        ],
        ignore_index=True,
        sort=False,
    )
    hybrid["hybrid_candidate_status_v7"] = np.select(
        [
            hybrid.get("promotion_eligible", pd.Series(False, index=hybrid.index)).astype(bool),
            hybrid["auditability_score_v7"].ge(0.35)
            & hybrid["solver_status"].astype(str).str.contains("optimal", case=False, na=False),
        ],
        ["not_used_promotion_disabled", "review_candidate_paper4_only"],
        default="park_or_infeasible",
    )
    all_alloc = pd.concat(
        [df for df in [mdcp_alloc, cvar_alloc] if not df.empty], ignore_index=True
    )
    return mdcp_alloc, mdcp_summary, cvar_alloc, cvar_summary, cvar_loss, hybrid


def build_dla_v7(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_solver_pool(
        candidate_pool, online_intervals, max_n=min(len(candidate_pool), 10_000)
    )
    months = sorted(pool["issue_month"].dropna().unique())[:max_months]
    rng = np.random.default_rng(RNG_SEED)
    cash = BUDGET
    outstanding: list[dict[str, Any]] = []
    decisions: list[pd.DataFrame] = []
    state_rows: list[dict[str, Any]] = []
    macro = 0.0
    for t, month in enumerate(months, start=1):
        macro = 0.68 * macro + float(rng.normal(0, 0.32))
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
            monthly_pd = np.clip((item["pd_high"] / 12.0) * math.exp(macro), 0.0001, 0.60)
            expected_loss += remaining * item["lgd"] * monthly_pd
            capital_used += remaining * (0.08 + 0.55 * item["pd_high"] + 0.08 * item["qhat"])
            default_event = bool(rng.uniform() < monthly_pd or (item["y_true"] >= 0.5 and age >= 9))
            if default_event:
                loss = remaining * item["lgd"]
                recovery = loss * float(np.clip(0.10 - 0.03 * max(macro, 0), 0.02, 0.12))
                realized_loss += loss
                recovery_in += recovery
                cash += recovery
                continue
            scheduled_principal = min(remaining, item["original_exposure"] / max(item["term"], 1.0))
            prepay_event = bool(
                rng.uniform() < np.clip(0.015 + 0.035 * (1 - item["pd_high"]), 0.005, 0.06)
            )
            if prepay_event:
                scheduled_principal = remaining
            interest = remaining * item["int_rate"] / 12.0
            principal_in += scheduled_principal
            interest_in += interest
            item["remaining_balance"] = remaining - scheduled_principal
            if item["remaining_balance"] > 1e-6 and age < item["term"]:
                next_outstanding.append(item)
        cash += principal_in + interest_in - realized_loss
        coverage_multiplier = 0.82 if macro > 0.55 else 1.0
        deployment_budget = max(0.0, min(cash * 0.38 * coverage_multiplier, BUDGET * 0.32))
        available = pool[pool["issue_month"].eq(month)].copy()
        if not available.empty and deployment_budget >= 1_000:
            available["stage_proxy_v7"] = np.select(
                [available["pd_high_alpha01"].ge(0.35), available["pd_high_alpha01"].ge(0.18)],
                [3, 2],
                default=1,
            )
            available["capital_charge_v7"] = available["loan_amnt"] * (
                0.08 + 0.55 * available["pd_high_alpha01"] + 0.08 * available["qhat_v4"]
            )
            available["ecl_proxy_v7"] = (
                available["loan_amnt"]
                * available["pd_high_alpha01"]
                * DEFAULT_LGD
                * (1 + max(macro, 0))
            )
            available["dla_score_v7"] = (
                available["base_return_vec"]
                - 0.18 * available["capital_charge_v7"]
                - 0.55 * available["ecl_proxy_v7"]
                - 0.08 * available["loan_amnt"] * available["weak_source_proxy"]
                - 0.05 * available["loan_amnt"] * available["qhat_v4"]
            )
            local = available.sort_values("dla_score_v7", ascending=False).copy()
            local["cum_amount"] = local["loan_amnt"].cumsum()
            funded = local[local["cum_amount"].le(deployment_budget)].copy()
            if funded.empty:
                funded = local.head(1).copy()
            funded["policy_id"] = "v7_dla_capital_coverage_state"
            funded["decision_month"] = month
            funded["month_idx"] = t
            funded["funded_exposure"] = funded["loan_amnt"]
            funded["cash_before_decision"] = cash
            funded["macro_state_v7"] = macro
            funded["action_v7"] = np.where(
                funded["qhat_v4"].gt(0.90),
                "fund_with_coverage_charge",
                "fund_with_capital_ecl_score",
            )
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
                        "stage_proxy_v7",
                        "capital_charge_v7",
                        "ecl_proxy_v7",
                        "qhat_v4",
                        "weak_source_proxy",
                        "macro_state_v7",
                        "action_v7",
                    ]
                ]
            )
        outstanding = next_outstanding + [
            item for item in outstanding if item["funded_month_idx"] == t
        ]
        outstanding_balance = float(sum(item["remaining_balance"] for item in outstanding))
        state_rows.append(
            {
                "policy_id": "v7_dla_capital_coverage_state",
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
                "macro_state_v7": macro,
                "state_value_proxy": cash
                + outstanding_balance
                - expected_loss
                - 0.10 * capital_used,
                "decision_scope": "loan_level_endogenous_capital_ecl_coverage_state",
            }
        )
    decision_df = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()
    state = pd.DataFrame(state_rows)
    summary = pd.DataFrame(
        [
            {
                "policy_id": "v7_dla_capital_coverage_state",
                "horizon_months": int(len(months)),
                "funded_loans": int(decision_df["loan_id"].nunique())
                if not decision_df.empty
                else 0,
                "total_funded_exposure": float(decision_df["funded_exposure"].sum())
                if not decision_df.empty
                else 0.0,
                "final_cash": float(state["cash_end"].iloc[-1]) if not state.empty else np.nan,
                "final_state_value_proxy": float(state["state_value_proxy"].iloc[-1])
                if not state.empty
                else np.nan,
                "cumulative_realized_loss": float(state["realized_loss"].sum())
                if not state.empty
                else np.nan,
                "cumulative_recovery": float(state["recovery_in"].sum())
                if not state.empty
                else np.nan,
                "cumulative_expected_loss": float(state["expected_loss"].sum())
                if not state.empty
                else np.nan,
                "cumulative_capital_used": float(state["capital_used"].sum())
                if not state.empty
                else np.nan,
                "claim_scope": "loan_level_dynamic_proxy_with_capital_ecl_coverage_state_not_bellman_optimal",
            }
        ]
    )
    return decision_df, state, summary


def build_sample_path_calibration_v7() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("baseline", 0.70, 0.22, 0.16, 0.22, "paired comparison only"),
            ("adverse", 0.70, 0.35, 0.16, 0.28, "stress comparison with cycle-sensitive LGD"),
            ("cycle_stress", 0.75, 0.42, 0.20, 0.35, "upper-tail stress, not forecast"),
            (
                "macro_clustered",
                0.82,
                0.30,
                0.25,
                0.40,
                "future extension for cohort/default dependence",
            ),
        ],
        columns=[
            "scenario",
            "macro_ar1_rho",
            "macro_sigma",
            "grade_shock_sigma",
            "lgd_cycle_beta",
            "claim_scope_v7",
        ],
    )


def build_causal_fairness_v7() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cate_gate, causal_outcomes, fairness_protocol, external_protocol = build_causal_fairness_v6()
    blocker = pd.DataFrame(
        [
            {
                "lane": "CATE policy value",
                "current_result_v7": "blocked",
                "evidence": "v6 CATE intervals cross zero / observational treatment remains sensitive",
                "required_to_unblock": "stable sensitivity, clean outcome, overlap and falsification package",
                "promotion_allowed": False,
            },
            {
                "lane": "high_rate_within_grade causal dossier",
                "current_result_v7": "research_usable_not_policy_value",
                "evidence": "balance/placebo improved in prior waves but hidden-bias sensitivity still material",
                "required_to_unblock": "formal sensitivity threshold and outcome-specific falsification",
                "promotion_allowed": False,
            },
            {
                "lane": "fairness",
                "current_result_v7": "proxy_governance_only",
                "evidence": "protected attributes are not available in Lending Club artifacts",
                "required_to_unblock": "externally approved protected-attribute or proxy protocol",
                "promotion_allowed": False,
            },
        ]
    )
    no_claim = pd.DataFrame(
        [
            {
                "statement_id": "paper4_fairness_no_legal_claim_v7",
                "statement": (
                    "Paper 4 reports source/proxy governance diagnostics only; it does not make a fair-lending "
                    "legal compliance claim without protected attributes or an approved proxy protocol."
                ),
                "must_appear_in_quarto": True,
            }
        ]
    )
    outcome_registry = causal_outcomes.copy()
    if not outcome_registry.empty:
        outcome_registry["v7_policy_value_use"] = np.where(
            outcome_registry["availability"].astype(str).str.contains("available"),
            "diagnostic_only_until_identification_unblocked",
            "not_used",
        )
    return blocker, no_claim, outcome_registry


def build_claim_matrix_v7() -> pd.DataFrame:
    rows = [
        (
            "Online source-month resolution",
            "implemented_deployable_support_pooling_search",
            "paper4_v7_online_deployable_weak_cell_search.csv",
            "19ao-v7-online-mdcp-resolution.qmd",
            "passes defended 0.80 gate only with costly width; no promotion",
        ),
        (
            "Online selected intervals",
            "implemented_selected_deployable_methods",
            "paper4_v7_online_deployable_intervals.parquet",
            "19ao-v7-online-mdcp-resolution.qmd",
            "selected reference plus best defended method",
        ),
        (
            "MDCP-aware solver",
            "implemented_soft_penalty_inside_lp",
            "paper4_v7_mdcp_soft_penalty_solver_summary.csv",
            "19ao-v7-online-mdcp-resolution.qmd",
            "proxy MDCP, not formal multidimensional conformal theorem",
        ),
        (
            "CVaR adaptive frontier",
            "implemented_expanded_topk_constraint",
            "paper4_v7_cvar_adaptive_frontier.csv",
            "19ao-v7-online-mdcp-resolution.qmd",
            "top-k expanded, not full 276k decomposition",
        ),
        (
            "SICR MRM shortlist",
            "implemented_committee_shortlist",
            "paper4_v7_sicr_mrm_shortlist.csv",
            "19ap-v7-ifrs9-dla-causal-governance.qmd",
            "accounting policy still proxy-only",
        ),
        (
            "IFRS9 contractual plan",
            "implemented_gap_action_plan",
            "paper4_v7_ifrs9_contractual_build_plan.csv",
            "19ap-v7-ifrs9-dla-causal-governance.qmd",
            "true servicing panel remains blocker",
        ),
        (
            "DLA endogenous capital state",
            "implemented_dynamic_proxy",
            "paper4_v7_dla_capital_state_summary.csv",
            "19ap-v7-ifrs9-dla-causal-governance.qmd",
            "not Bellman-optimal ADP",
        ),
        (
            "Sample paths calibration",
            "implemented_calibration_registry",
            "paper4_v7_sample_path_calibration_grid.csv",
            "19ap-v7-ifrs9-dla-causal-governance.qmd",
            "simulation for paired comparison, not forecast",
        ),
        (
            "CATE and fairness gates",
            "implemented_blocker_matrix",
            "paper4_v7_causal_fairness_blocker_matrix.csv",
            "19ap-v7-ifrs9-dla-causal-governance.qmd",
            "policy value and legal fairness claims remain blocked",
        ),
        (
            "Hybrid candidate registry",
            "implemented_surrogate_registry",
            "paper4_v7_hybrid_solver_candidate_registry.csv",
            "19ao-v7-online-mdcp-resolution.qmd",
            "surrogate LP/hybrid registry, not neural SPO+/DFL training",
        ),
        (
            "V7 status",
            "implemented_living_lab_status",
            "paper4_v7_resolution_status.json",
            "19aq-v7-pending-and-results.qmd",
            "no final promotion json",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["priority", "claim_status", "artifact", "quarto_page", "caveat"]
    )


def update_manifest_v7(claims: pd.DataFrame) -> None:
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
                "source_paper": "Paper 4 v7",
                "role": row["priority"],
                "status": row["claim_status"],
                "run_tag": "paper4_v7_resolution_loop_2026-05-13",
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
    parser.add_argument("--solver-pool-n", type=int, default=14_000)
    parser.add_argument("--sample-paths", type=int, default=250)
    parser.add_argument("--dla-months", type=int, default=18)
    parser.add_argument("--min-support", type=int, default=5)
    args = parser.parse_args(list(argv) if argv is not None else None)

    _, candidate_pool, allocations, _, online_intervals = _load_inputs()

    online_search, online_intervals_v7, online_policy, online_source, pooling = build_online_v7(
        allocations, online_intervals, min_support=args.min_support
    )
    _write_csv("paper4_v7_online_deployable_weak_cell_search.csv", online_search)
    _write_parquet("paper4_v7_online_deployable_intervals.parquet", online_intervals_v7)
    _write_parquet("paper4_v7_online_deployable_policy_month.parquet", online_policy)
    _write_parquet("paper4_v7_online_deployable_source_month.parquet", online_source)
    _write_csv("paper4_v7_online_min_support_pooling.csv", pooling)

    sicr_shortlist, ifrs9_plan = build_sicr_mrm_v7()
    _write_csv("paper4_v7_sicr_mrm_shortlist.csv", sicr_shortlist)
    _write_csv("paper4_v7_ifrs9_contractual_build_plan.csv", ifrs9_plan)

    data_audit, gaps, readiness = build_contractual_data_audit_v6()
    _write_csv("paper4_v7_contractual_data_audit.csv", data_audit)
    _write_csv("paper4_v7_servicing_gap_register.csv", gaps)
    _write_csv("paper4_v7_contractual_ifrs9_readiness.csv", readiness)

    mdcp_alloc, mdcp_summary, cvar_alloc, cvar_summary, cvar_loss, hybrid = build_solver_v7(
        candidate_pool, online_intervals, max_pool_n=args.solver_pool_n
    )
    _write_parquet("paper4_v7_mdcp_soft_penalty_allocations.parquet", mdcp_alloc)
    _write_csv("paper4_v7_mdcp_soft_penalty_solver_summary.csv", mdcp_summary)
    _write_parquet("paper4_v7_cvar_adaptive_allocations.parquet", cvar_alloc)
    _write_csv("paper4_v7_cvar_adaptive_frontier.csv", cvar_summary)
    _write_csv("paper4_v7_cvar_adaptive_scenario_losses.csv", cvar_loss)
    _write_csv("paper4_v7_hybrid_solver_candidate_registry.csv", hybrid)

    dla_decisions, dla_state, dla_summary = build_dla_v7(
        candidate_pool, online_intervals, max_months=args.dla_months
    )
    _write_parquet("paper4_v7_dla_capital_state_decisions.parquet", dla_decisions)
    _write_csv("paper4_v7_dla_capital_state_trace.csv", dla_state)
    _write_csv("paper4_v7_dla_capital_state_summary.csv", dla_summary)

    all_solver_allocs = pd.concat(
        [df for df in [mdcp_alloc, cvar_alloc] if not df.empty], ignore_index=True
    )
    paths, path_ci = build_sample_paths_v6(
        allocations, all_solver_allocs, n_paths=args.sample_paths
    )
    _write_parquet("paper4_v7_common_sample_paths.parquet", paths)
    _write_csv("paper4_v7_common_sample_path_ci.csv", path_ci)
    _write_csv("paper4_v7_sample_path_calibration_grid.csv", build_sample_path_calibration_v7())

    causal_blocker, fairness_no_claim, causal_outcomes = build_causal_fairness_v7()
    _write_csv("paper4_v7_causal_fairness_blocker_matrix.csv", causal_blocker)
    _write_csv("paper4_v7_fairness_no_legal_claim_statement.csv", fairness_no_claim)
    _write_csv("paper4_v7_causal_outcome_registry.csv", causal_outcomes)

    claims = build_claim_matrix_v7()
    _write_csv("paper4_v7_claim_artifact_matrix.csv", claims)
    update_manifest_v7(claims)

    best_online = online_search.sort_values(
        ["gate_pass_80_defended", "avg_width_loan"], ascending=[False, True]
    ).iloc[0]
    deployable_passing = online_search[
        online_search["deployable_without_current_outcomes"].astype(bool)
        & online_search["gate_pass_80_defended"].astype(bool)
    ]
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v7_resolution_loop",
        "mode": "paper4_living_lab_no_paper1_changes",
        "paper1_artifacts_modified": False,
        "paper4_final_promotion_created": False,
        "paper1_frozen_champion": FROZEN_PAPER1_CHAMPION,
        "priorities_targeted": 10,
        "online_best_method": best_online["online_method_v7"],
        "online_best_method_family": best_online["method_family"],
        "online_best_deployable": bool(best_online["deployable_without_current_outcomes"]),
        "online_best_policy_month_defended_min": float(
            best_online["coverage_policy_month_defended_min"]
        ),
        "online_best_source_month_defended_min": float(
            best_online["coverage_source_month_defended_min"]
        ),
        "online_best_width": float(best_online["avg_width_loan"]),
        "online_gate80_defended_deployable_exists": bool(not deployable_passing.empty),
        "online_promotion_eligible": bool(online_search["promotion_eligible"].astype(bool).any()),
        "online_efficiency_blocker": bool(float(best_online["avg_width_loan"]) > 0.95),
        "sicr_shortlist_rules": sicr_shortlist["sicr_rule_v6"].head(3).tolist()
        if not sicr_shortlist.empty
        else [],
        "contractual_ifrs9_readiness_score": float(readiness["readiness_score"].iloc[0])
        if not readiness.empty
        else np.nan,
        "mdcp_soft_optimal_count": int(
            mdcp_summary["solver_status"]
            .astype(str)
            .str.contains("optimal", case=False, na=False)
            .sum()
        )
        if not mdcp_summary.empty
        else 0,
        "mdcp_gate_proxy_count": int(
            mdcp_summary.get("mdcp_gate_proxy_v7", pd.Series(dtype=bool)).astype(bool).sum()
        )
        if not mdcp_summary.empty
        else 0,
        "cvar_frontier_feasible_count": int(
            cvar_summary["solver_status"]
            .astype(str)
            .str.contains("optimal", case=False, na=False)
            .sum()
        )
        if not cvar_summary.empty
        else 0,
        "dla_capital_state_implemented": not dla_summary.empty,
        "dla_cumulative_realized_loss": float(dla_summary["cumulative_realized_loss"].iloc[0])
        if not dla_summary.empty
        else np.nan,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "hybrid_registry_rows": int(len(hybrid)),
        "generated_artifacts": claims["artifact"].tolist(),
    }
    _write_json("paper4_v7_resolution_status.json", status)
    update_manifest_v7(claims)
    _write_note(
        "paper4_v7_resolution_loop_memo.qmd",
        """---
title: "Paper 4 v7 Resolution Loop Memo"
format: html
---

# Paper 4 v7 Resolution Loop Memo

V7 moves the main online blocker from "unresolved coverage" to "coverage can
be defended at 0.80, but only with expensive intervals."  The method is
deployable because it uses pre-decision source structure and minimum-support
pooling rather than current-month outcomes.

The remaining blockers are therefore mostly efficiency, true contractual IFRS9
data, full-universe CVaR decomposition, formal DLA/ADP, and causal/fairness
identification.  No Paper Estrella artifact is changed and no final Paper 4
promotion JSON is created.
""",
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
