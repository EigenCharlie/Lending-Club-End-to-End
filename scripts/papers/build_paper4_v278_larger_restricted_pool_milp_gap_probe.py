#!/usr/bin/env python3
"""Build Paper 4 v278 expanded restricted-pool MILP/gap probe artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers.paper4_one_swap_living_lab import (
    FAMILIES,
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    _source_summary,
    now,
    read_csv,
    read_parquet,
    write_csv,
    write_json,
)

VERSION = 278
BASE_REPAIR_VERSION = 276
ONE_SWAP_REPRICE_VERSION = 277
BOUNDED_TWO_SWAP_VERSION = 268
PREVIOUS_MILP_VERSION = 275
CANDIDATE_POOL_LIMIT = 3500
MILP_TIME_LIMIT_SECONDS = 600.0


def _source_cap_map(source_caps: pd.DataFrame, family: str) -> dict[str, float]:
    family_caps = source_caps.loc[source_caps["source_family"].astype(str).eq(family)].copy()
    family_caps["source_id"] = family_caps["source_id"].astype(str)
    return family_caps.set_index("source_id")["cap_share_v80"].astype(float).to_dict()


def _build_pool(
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    mean_returns: np.ndarray,
) -> pd.DataFrame:
    selected_ids = set(selected["loan_id"].astype(str))
    omitted = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    omitted = omitted.assign(mean_return_v278=mean_returns[omitted.index.to_numpy()])

    frontier = read_csv("paper4_v268_bounded_two_swap_primary_frontier.csv")
    frontier_add_ids = (
        set(frontier["primary_added_loan_id_v268"].dropna().astype(str))
        if not frontier.empty
        else set()
    )
    previous_pool = read_parquet("paper4_v275_restricted_pool_milp_allocations.parquet")
    previous_pool_ids = (
        set(previous_pool["loan_id"].dropna().astype(str)) if not previous_pool.empty else set()
    )
    top_ids = set(
        omitted.sort_values("mean_return_v278", ascending=False)
        .head(CANDIDATE_POOL_LIMIT)["loan_id"]
        .astype(str)
    )
    candidate_ids = top_ids | frontier_add_ids | previous_pool_ids

    selected_pool = selected.copy()
    selected_pool["pool_role_v278"] = "current_v276_selected"
    candidate_pool = omitted.loc[omitted["loan_id"].astype(str).isin(candidate_ids)].copy()
    candidate_id_series = candidate_pool["loan_id"].astype(str)
    candidate_pool["pool_role_v278"] = np.select(
        [
            candidate_id_series.isin(top_ids)
            & candidate_id_series.isin(previous_pool_ids | frontier_add_ids),
            candidate_id_series.isin(previous_pool_ids | frontier_add_ids),
            candidate_id_series.isin(top_ids),
        ],
        [
            "omitted_candidate_from_top3500_and_previous_pool_or_frontier",
            "omitted_candidate_from_previous_pool_or_v268_frontier",
            "omitted_candidate_from_top3500_mean_return",
        ],
        default="omitted_candidate_from_expanded_pool",
    )
    keep_cols = ["loan_id", "loan_amnt", *FAMILIES, "pool_role_v278"]
    pool = pd.concat([selected_pool[keep_cols], candidate_pool[keep_cols]], ignore_index=True)
    pool["loan_id"] = pool["loan_id"].astype(str)
    pool = pool.drop_duplicates("loan_id", keep="first").reset_index(drop=True)
    return pool


def _source_constraints(
    pool: pd.DataFrame,
    amounts: np.ndarray,
    source_caps: pd.DataFrame,
    var_count: int,
) -> tuple[list[np.ndarray], list[float], list[float], list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    meta: list[dict[str, Any]] = []
    for family in FAMILIES:
        caps = _source_cap_map(source_caps, family)
        for source_id in sorted(pool[family].dropna().astype(str).unique()):
            cap = float(caps.get(source_id, 1.0))
            indicator = pool[family].astype(str).eq(source_id).to_numpy(float)
            row = np.zeros(var_count)
            row[: len(pool)] = amounts * (indicator - cap)
            rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
            meta.append(
                {
                    f"constraint_type_v{VERSION}": "source_share",
                    "source_family": family,
                    "source_id": source_id,
                    f"cap_share_v{VERSION}": cap,
                }
            )
    return rows, lb, ub, meta


def _solve_restricted_pool_milp(
    pool: pd.DataFrame,
    losses_pool: np.ndarray,
    mean_returns_pool: np.ndarray,
    source_caps: pd.DataFrame,
    repair_row: pd.Series,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    n = len(pool)
    scenario_count = losses_pool.shape[0]
    var_count = n + 1 + scenario_count
    amounts = pool["loan_amnt"].to_numpy(float)
    exposure_min = float(repair_row[f"exposure_min_v{BASE_REPAIR_VERSION}"])
    exposure_max = float(repair_row[f"exposure_max_v{BASE_REPAIR_VERSION}"])
    cvar_cap = float(repair_row[f"cvar_cap_v{BASE_REPAIR_VERSION}"])
    selected_rows = int(repair_row[f"selected_rows_v{BASE_REPAIR_VERSION}"])

    c = np.zeros(var_count)
    c[:n] = -mean_returns_pool
    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    meta: list[dict[str, Any]] = []

    budget = np.zeros(var_count)
    budget[:n] = amounts
    rows.append(budget)
    lb.append(exposure_min)
    ub.append(exposure_max)
    meta.append({f"constraint_type_v{VERSION}": "budget_range"})

    cardinality = np.zeros(var_count)
    cardinality[:n] = 1.0
    rows.append(cardinality)
    lb.append(float(selected_rows))
    ub.append(float(selected_rows))
    meta.append({f"constraint_type_v{VERSION}": "selected_row_cardinality"})

    source_rows, source_lb, source_ub, source_meta = _source_constraints(
        pool=pool,
        amounts=amounts,
        source_caps=source_caps,
        var_count=var_count,
    )
    rows.extend(source_rows)
    lb.extend(source_lb)
    ub.extend(source_ub)
    meta.extend(source_meta)

    cvar_row = np.zeros(var_count)
    cvar_row[n] = 1.0
    cvar_row[n + 1 :] = 1.0 / ((1.0 - v70.ALPHA) * scenario_count)
    rows.append(cvar_row)
    lb.append(-np.inf)
    ub.append(cvar_cap)
    meta.append({f"constraint_type_v{VERSION}": "cvar_cap"})

    for scenario_idx in range(scenario_count):
        row = np.zeros(var_count)
        row[:n] = losses_pool[scenario_idx, :]
        row[n] = -1.0
        row[n + 1 + scenario_idx] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)
        meta.append({f"constraint_type_v{VERSION}": "cvar_path_excess"})

    constraint_matrix = np.vstack(rows)
    result = milp(
        c,
        integrality=np.r_[np.ones(n), np.zeros(1 + scenario_count)],
        bounds=Bounds(
            np.r_[np.zeros(n), 0.0, np.zeros(scenario_count)],
            np.r_[np.ones(n), np.inf, np.full(scenario_count, np.inf)],
        ),
        constraints=LinearConstraint(constraint_matrix, np.array(lb), np.array(ub)),
        options={"time_limit": MILP_TIME_LIMIT_SECONDS, "mip_rel_gap": 1e-6},
    )
    x = np.zeros(n)
    incumbent_available = result.x is not None
    if incumbent_available:
        x = np.rint(np.clip(result.x[:n], 0, 1)).astype(float)

    diagnostics = {
        f"milp_success_v{VERSION}": bool(result.success),
        f"milp_incumbent_available_v{VERSION}": bool(incumbent_available),
        f"milp_status_v{VERSION}": int(result.status),
        f"milp_message_v{VERSION}": str(result.message),
        f"milp_fun_v{VERSION}": float(result.fun) if result.fun is not None else np.nan,
        f"milp_dual_bound_v{VERSION}": float(getattr(result, "mip_dual_bound", np.nan)),
        f"milp_gap_v{VERSION}": float(getattr(result, "mip_gap", np.nan)),
        f"milp_node_count_v{VERSION}": int(getattr(result, "mip_node_count", -1)),
        f"time_limit_seconds_v{VERSION}": MILP_TIME_LIMIT_SECONDS,
        f"exposure_min_v{VERSION}": exposure_min,
        f"exposure_max_v{VERSION}": exposure_max,
        f"cvar_cap_v{VERSION}": cvar_cap,
        f"constraint_rows_v{VERSION}": int(len(rows)),
    }
    constraints = pd.DataFrame(meta)
    return diagnostics, x, constraints


def _update_claim_boundaries(*, improvement_found: bool, incumbent_available: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v278 restricted-pool MILP/gap probe after v277.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v278_restricted_pool_milp_summary.csv"
                ),
                "boundary": (
                    "Restricted to current v276 selected loans plus a top-3500 omitted candidate "
                    "pool, v275 pool loans, and v268 frontier loans; not full universe."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v278 finds a better restricted-pool MILP incumbent than v276.",
                "allowed": bool(improvement_found and incumbent_available),
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v278_restricted_pool_milp_summary.csv"
                ),
                "boundary": (
                    "Existence claim only inside the restricted v278 pool and solver tolerance."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v278 proves global full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v278_claim_blockers.csv"
                ),
                "boundary": "Full-universe omitted loans and branch-and-price gap remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v278 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v278_claim_blockers.csv"
                ),
                "boundary": "No final promotion, dynamic validation, or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog(*, improvement_found: bool) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v278 solves an expanded restricted-pool binary MILP over the v276 selected "
                    "loans plus top omitted candidates after v277 cleared one-swap local pricing."
                ),
                "status": (
                    "restricted_pool_milp_improvement_found"
                    if improvement_found
                    else "restricted_pool_milp_gap_or_no_improvement_recorded"
                ),
                "next_artifact": (
                    "paper4_v279_apply_restricted_pool_candidate_or_decompose_swaps.csv"
                    if improvement_found
                    else "paper4_v279_full_universe_gap_certificate_protocol.csv"
                ),
                "success_condition": (
                    "restricted-pool MILP either finds an incumbent improvement or records a "
                    "bounded negative/gap result without enabling global claims"
                ),
                "last_wave": "v278",
                "execution_result": (
                    "restricted_pool_milp_found_improvement"
                    if improvement_found
                    else "restricted_pool_milp_no_improvement_or_gap"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    key_cols = ["last_wave", "lane", "next_artifact"]
    merged_keys = set(map(tuple, additions[key_cols].astype(str).to_numpy()))
    keep = [tuple(row) not in merged_keys for row in current[key_cols].astype(str).to_numpy()]
    write_csv(path, pd.concat([current.loc[keep].copy(), additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V278_RESTRICTED_POOL_MILP_GAP_PROBE_START -->"
    end = "<!-- V278_RESTRICTED_POOL_MILP_GAP_PROBE_END -->"
    block = f"""
{start}

## Wave v278: Restricted-Pool MILP/Gap Probe After v277

Generated: {status["generated_at_utc"]}

### Objective

After v277 cleared one-swap for the v276 candidate, solve an expanded
restricted binary MILP over the v276 selected loans plus the top omitted
candidate pool. This is broader than bounded two-swap evidence but remains a
restricted-pool probe, not a full-universe certificate.

### Results

- Pool rows: `{status["pool_rows_v278"]}`.
- Current selected rows: `{status["current_selected_rows_v278"]}`.
- Omitted candidate rows: `{status["omitted_candidate_rows_v278"]}`.
- MILP incumbent available: `{status["milp_incumbent_available_v278"]}`.
- MILP success flag: `{status["milp_success_v278"]}`.
- MILP gap: `{status["milp_gap_v278"]}`.
- Objective delta vs v276:
  `{status["objective_delta_vs_v276_v278"]}`.
- CVaR90 delta vs v276:
  `{status["cvar90_delta_vs_v276_v278"]}`.
- Better restricted-pool incumbent found:
  `{status["restricted_pool_improvement_found_v278"]}`.

### Interpretation

v278 expands the evidence frontier from local swaps to a larger restricted
binary MILP. Any positive result is still restricted to the constructed pool;
any negative or gap result still does not prove full-universe optimality.

### Claim Impact

- Allowed: restricted-pool MILP/gap probe executed.
- Still prohibited: full-universe global integer optimality, Paper Estrella
  replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v278 in the living notebook. Promote only after full-universe/global,
dynamic validation and promotion gates pass.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v276_restricted_pool_milp_repair_allocations.parquet")
    repair_summary = read_csv("paper4_v276_restricted_pool_milp_repair_summary.csv")
    repair_row = repair_summary.iloc[0]
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    if universe.empty or selected.empty or source_caps.empty:
        raise RuntimeError("Missing v55 universe, v276 selected portfolio, or source caps.")

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    pool = _build_pool(universe=universe, selected=selected, mean_returns=mean_returns)
    pool_idx = idx_by_id.loc[pool["loan_id"].astype(str)].to_numpy()
    pool[f"mean_return_v{VERSION}"] = mean_returns[pool_idx]
    current_selected_ids = set(selected["loan_id"].astype(str))
    pool[f"current_selected_v{VERSION}"] = pool["loan_id"].astype(str).isin(current_selected_ids)
    pool_losses = losses[:, pool_idx]
    pool_returns = mean_returns[pool_idx]

    diagnostics, x, constraints = _solve_restricted_pool_milp(
        pool=pool,
        losses_pool=pool_losses,
        mean_returns_pool=pool_returns,
        source_caps=source_caps,
        repair_row=repair_row,
    )
    pool[f"selected_v{VERSION}"] = x.astype(int)
    pool[f"milp_action_v{VERSION}"] = np.select(
        [
            pool[f"current_selected_v{VERSION}"].to_numpy(bool) & (x < 0.5),
            ~pool[f"current_selected_v{VERSION}"].to_numpy(bool) & (x > 0.5),
            pool[f"current_selected_v{VERSION}"].to_numpy(bool) & (x > 0.5),
        ],
        ["dropped_by_restricted_pool_milp", "added_by_restricted_pool_milp", "kept_from_v276"],
        default="not_selected_candidate",
    )
    chosen = pool.loc[pool[f"selected_v{VERSION}"].eq(1), ["loan_id", "loan_amnt", *FAMILIES]]
    source_diag = _source_summary(
        universe=universe,
        portfolio=chosen,
        source_caps=source_caps,
        version=VERSION,
        ordinal="restricted-pool MILP",
    )
    source_diag[f"portfolio_label_v{VERSION}"] = "restricted_pool_milp_candidate"
    source_diag[f"claim_boundary_v{VERSION}"] = (
        "restricted-pool source diagnostic only; not full-universe proof"
    )

    incumbent_available = bool(diagnostics[f"milp_incumbent_available_v{VERSION}"])
    objective = float(pool_returns @ x) if incumbent_available else np.nan
    exposure = float(pool["loan_amnt"].to_numpy(float) @ x) if incumbent_available else np.nan
    scenario_losses = pool_losses @ x if incumbent_available else np.full(losses.shape[0], np.nan)
    cvar90 = v70._tail_cvar(scenario_losses) if incumbent_available else np.nan
    current_objective = float(repair_row[f"objective_return_v{BASE_REPAIR_VERSION}"])
    current_cvar90 = float(repair_row[f"scenario_loss_cvar90_v{BASE_REPAIR_VERSION}"])
    objective_delta = objective - current_objective if incumbent_available else np.nan
    cvar_delta = cvar90 - current_cvar90 if incumbent_available else np.nan
    selected_count = int(x.sum()) if incumbent_available else 0
    added_rows = int(
        (
            (~pool[f"current_selected_v{VERSION}"].astype(bool))
            & pool[f"selected_v{VERSION}"].eq(1)
        ).sum()
    )
    dropped_rows = int(
        (
            pool[f"current_selected_v{VERSION}"].astype(bool) & pool[f"selected_v{VERSION}"].eq(0)
        ).sum()
    )
    source_violations = (
        int(source_diag[f"source_cap_violated_v{VERSION}"].sum()) if incumbent_available else -1
    )
    budget_feasible = (
        bool(
            exposure >= float(repair_row[f"exposure_min_v{BASE_REPAIR_VERSION}"]) - 1e-7
            and exposure <= float(repair_row[f"exposure_max_v{BASE_REPAIR_VERSION}"]) + 1e-7
        )
        if incumbent_available
        else False
    )
    cvar_feasible = (
        bool(cvar90 <= float(repair_row[f"cvar_cap_v{BASE_REPAIR_VERSION}"]) + 1e-7)
        if incumbent_available
        else False
    )
    restricted_improvement = bool(
        incumbent_available
        and objective_delta > 1e-7
        and budget_feasible
        and cvar_feasible
        and source_violations == 0
        and selected_count == int(repair_row[f"selected_rows_v{BASE_REPAIR_VERSION}"])
    )

    summary = pd.DataFrame(
        [
            {
                f"probe_label_v{VERSION}": "restricted_pool_milp_after_v277_no_two_swap_improvement",
                f"base_repair_version_v{VERSION}": BASE_REPAIR_VERSION,
                f"terminal_reprice_version_v{VERSION}": ONE_SWAP_REPRICE_VERSION,
                f"bounded_two_swap_version_v{VERSION}": BOUNDED_TWO_SWAP_VERSION,
                f"previous_restricted_pool_milp_version_v{VERSION}": PREVIOUS_MILP_VERSION,
                f"candidate_pool_limit_v{VERSION}": CANDIDATE_POOL_LIMIT,
                f"pool_rows_v{VERSION}": int(len(pool)),
                f"current_selected_rows_v{VERSION}": int(
                    pool[f"current_selected_v{VERSION}"].sum()
                ),
                f"omitted_candidate_rows_v{VERSION}": int(
                    (~pool[f"current_selected_v{VERSION}"].astype(bool)).sum()
                ),
                f"selected_rows_v{VERSION}": selected_count,
                f"added_rows_v{VERSION}": added_rows,
                f"dropped_rows_v{VERSION}": dropped_rows,
                f"objective_return_v{VERSION}": objective,
                f"objective_delta_vs_v276_v{VERSION}": objective_delta,
                f"portfolio_exposure_v{VERSION}": exposure,
                f"scenario_loss_cvar90_v{VERSION}": cvar90,
                f"cvar90_delta_vs_v276_v{VERSION}": cvar_delta,
                f"source_cap_violations_v{VERSION}": source_violations,
                f"budget_feasible_v{VERSION}": budget_feasible,
                f"cvar_feasible_v{VERSION}": cvar_feasible,
                f"source_feasible_v{VERSION}": source_violations == 0,
                f"restricted_pool_improvement_found_v{VERSION}": restricted_improvement,
                f"restricted_pool_global_optimality_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"claim_boundary_v{VERSION}": (
                    "restricted-pool MILP/gap probe only; not full-universe global proof"
                ),
                **diagnostics,
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "restricted_pool_improvement_found",
                f"blocking_v{VERSION}": restricted_improvement,
                f"evidence_count_v{VERSION}": int(restricted_improvement),
                f"required_next_artifact_v{VERSION}": (
                    "paper4_v279_apply_restricted_pool_candidate_or_decompose_swaps.csv"
                    if restricted_improvement
                    else "paper4_v279_full_universe_gap_certificate_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "positive restricted-pool incumbent requires repair/decomposition"
                    if restricted_improvement
                    else "no better feasible restricted-pool incumbent was certified"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "restricted_pool_not_full_universe",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(pool)),
                f"required_next_artifact_v{VERSION}": (
                    "paper4_v279_full_universe_gap_certificate_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": "candidate pool omits most v55 loans",
            },
            {
                f"blocker_id_v{VERSION}": "global_integer_gap_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    "paper4_v279_full_universe_gap_certificate_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": "no full-universe branch-and-price gap certificate",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_final_promotion_forbidden",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
                f"claim_boundary_v{VERSION}": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v278_restricted_pool_milp_probe_executed",
                "allowed": True,
                "artifact": "paper4_v278_restricted_pool_milp_summary.csv",
                "boundary": "restricted pool only",
            },
            {
                "claim_id": "v278_restricted_pool_improvement_found",
                "allowed": restricted_improvement,
                "artifact": "paper4_v278_restricted_pool_milp_summary.csv",
                "boundary": "inside v278 restricted pool and solver result only",
            },
            {
                "claim_id": "v278_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v278_claim_blockers.csv",
                "boundary": "full-universe gap certificate missing",
            },
            {
                "claim_id": "v278_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v278_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    pool.to_parquet(
        TABLE_DIR / "paper4_v278_restricted_pool_milp_allocations.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v278_restricted_pool_milp_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v278_restricted_pool_milp_source_summary.csv", source_diag)
    write_csv(TABLE_DIR / "paper4_v278_restricted_pool_milp_constraints.csv", constraints)
    write_csv(TABLE_DIR / "paper4_v278_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v278_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries(
        improvement_found=restricted_improvement,
        incumbent_available=incumbent_available,
    )
    _update_backlog(improvement_found=restricted_improvement)

    row = summary.iloc[0]
    status = {
        "phase": "v278_restricted_pool_milp_gap_probe_after_v277",
        "schema_version": "2026-05-15.278",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_repair_version_v278": BASE_REPAIR_VERSION,
        "terminal_reprice_version_v278": ONE_SWAP_REPRICE_VERSION,
        "bounded_two_swap_version_v278": BOUNDED_TWO_SWAP_VERSION,
        "previous_restricted_pool_milp_version_v278": PREVIOUS_MILP_VERSION,
        "candidate_pool_limit_v278": CANDIDATE_POOL_LIMIT,
        "pool_rows_v278": int(row["pool_rows_v278"]),
        "current_selected_rows_v278": int(row["current_selected_rows_v278"]),
        "omitted_candidate_rows_v278": int(row["omitted_candidate_rows_v278"]),
        "selected_rows_v278": int(row["selected_rows_v278"]),
        "added_rows_v278": int(row["added_rows_v278"]),
        "dropped_rows_v278": int(row["dropped_rows_v278"]),
        "objective_return_v278": float(row["objective_return_v278"]),
        "objective_delta_vs_v276_v278": float(row["objective_delta_vs_v276_v278"]),
        "portfolio_exposure_v278": float(row["portfolio_exposure_v278"]),
        "scenario_loss_cvar90_v278": float(row["scenario_loss_cvar90_v278"]),
        "cvar90_delta_vs_v276_v278": float(row["cvar90_delta_vs_v276_v278"]),
        "source_cap_violations_v278": int(row["source_cap_violations_v278"]),
        "budget_feasible_v278": bool(row["budget_feasible_v278"]),
        "cvar_feasible_v278": bool(row["cvar_feasible_v278"]),
        "source_feasible_v278": bool(row["source_feasible_v278"]),
        "restricted_pool_improvement_found_v278": restricted_improvement,
        "milp_success_v278": bool(row["milp_success_v278"]),
        "milp_incumbent_available_v278": bool(row["milp_incumbent_available_v278"]),
        "milp_status_v278": int(row["milp_status_v278"]),
        "milp_gap_v278": float(row["milp_gap_v278"]),
        "milp_dual_bound_v278": float(row["milp_dual_bound_v278"]),
        "constraint_rows_v278": int(row["constraint_rows_v278"]),
        "claim_blocker_rows_v278": int(len(blockers)),
        "claim_matrix_rows_v278": int(len(claim_matrix)),
        "restricted_pool_global_optimality_claim_allowed_v278": False,
        "full_universe_integer_optimality_claim_allowed_v278": False,
        "paper1_promotion_allowed_v278": False,
        "paper4_working_champion_changed_v278": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v278 is a restricted-pool MILP/gap probe only; full-universe global "
            "and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v278_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v278": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
