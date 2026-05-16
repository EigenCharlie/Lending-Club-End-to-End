#!/usr/bin/env python3
"""Build Paper 4 v305 post-v304 one-swap repricing artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers.paper4_one_swap_living_lab import (
    FAMILIES,
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    _reprice_pair_columns,
    now,
    read_csv,
    read_parquet,
    write_csv,
    write_json,
)

VERSION = 305
BOUNDED_MILP_VERSION = 304
SOURCE_CAP_VERSION = 304
BASELINE_VERSION = 295
NEXT_VERSION = 306
TARGET_SELECTED_ROWS = 171
REPRICE_SCOPE = "post_v304_best_return_solution_one_drop_one_add_whole_loan_swap"


def _v304_best_solution() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, float]:
    solutions = read_csv("paper4_v304_bounded_reward_solutions.csv")
    allocations = read_parquet("paper4_v304_bounded_reward_allocations.parquet")
    source_summary = read_csv("paper4_v304_bounded_reward_source_summary.csv")
    if solutions.empty or allocations.empty or source_summary.empty:
        raise RuntimeError("Missing v304 bounded reward artifacts for v305.")

    best_row = solutions.sort_values(
        f"objective_return_v{BOUNDED_MILP_VERSION}", ascending=False
    ).iloc[0]
    reward = float(best_row[f"reward_per_imputation_penalty_v{BOUNDED_MILP_VERSION}"])
    selected = allocations.loc[
        np.isclose(
            allocations[f"reward_per_imputation_penalty_v{BOUNDED_MILP_VERSION}"].to_numpy(float),
            reward,
        )
    ].copy()
    selected["loan_id"] = selected["loan_id"].astype(str)
    for family in FAMILIES:
        selected[family] = selected[family].astype(str)

    reward_source = source_summary.loc[
        np.isclose(
            source_summary[f"reward_per_imputation_penalty_v{BOUNDED_MILP_VERSION}"].to_numpy(
                float
            ),
            reward,
        )
    ].copy()
    reward_source["source_id"] = reward_source["source_id"].astype(str)
    return selected, reward_source, best_row, reward


def _source_maps(
    selected: pd.DataFrame, source_summary: pd.DataFrame
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    current_by_family: dict[str, dict[str, float]] = {}
    cap_by_family: dict[str, dict[str, float]] = {}
    cap_col = f"cap_share_v{SOURCE_CAP_VERSION}"
    for family in FAMILIES:
        current_by_family[family] = (
            selected.groupby(family, dropna=False)["loan_amnt"].sum().astype(float).to_dict()
        )
        cap_by_family[family] = (
            source_summary.loc[source_summary["source_family"].astype(str).eq(family)]
            .set_index("source_id")[cap_col]
            .astype(float)
            .to_dict()
        )
    return current_by_family, cap_by_family


def _source_prefilter_mask(
    base_mask: np.ndarray,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    source_summary: pd.DataFrame,
    current_exposure: float,
) -> np.ndarray:
    add_amount = candidates["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    new_total = current_exposure + add_amount[:, None] - drop_amount[None, :]
    mask = base_mask.copy()
    cap_col = f"cap_share_v{SOURCE_CAP_VERSION}"
    for family in FAMILIES:
        current_by_source = selected.groupby(family, dropna=False)["loan_amnt"].sum()
        cap_by_source = (
            source_summary.loc[source_summary["source_family"].astype(str).eq(family)]
            .set_index("source_id")[cap_col]
            .astype(float)
        )
        add_source = candidates[family].astype(str).to_numpy()
        drop_source = selected[family].astype(str).to_numpy()
        current_add = np.array([current_by_source.get(x, 0.0) for x in add_source], dtype=float)
        current_drop = np.array([current_by_source.get(x, 0.0) for x in drop_source], dtype=float)
        cap_add = np.array([cap_by_source.get(x, 1.0) for x in add_source], dtype=float)
        cap_drop = np.array([cap_by_source.get(x, 1.0) for x in drop_source], dtype=float)
        same_source = add_source[:, None] == drop_source[None, :]
        add_ok = (
            current_add[:, None] + add_amount[:, None] - drop_amount[None, :] * same_source
        ) <= cap_add[:, None] * new_total + 1e-7
        drop_ok = (
            current_drop[None, :] - drop_amount[None, :] + add_amount[:, None] * same_source
        ) <= cap_drop[None, :] * new_total + 1e-7
        mask &= add_ok & drop_ok
    return mask


def _exact_source_metrics(
    add_row: pd.Series,
    drop_row: pd.Series,
    current_by_family: dict[str, dict[str, float]],
    cap_by_family: dict[str, dict[str, float]],
    new_total: float,
) -> tuple[bool, float, float, int, str, str]:
    min_slack = np.inf
    max_share = 0.0
    violations = 0
    first_family = ""
    first_source = ""
    add_amount = float(add_row["loan_amnt"])
    drop_amount = float(drop_row["loan_amnt"])
    for family in FAMILIES:
        add_source = str(add_row[family])
        drop_source = str(drop_row[family])
        sources = set(current_by_family[family]) | {add_source, drop_source}
        caps = cap_by_family[family]
        for source_id in sources:
            exposure = current_by_family[family].get(source_id, 0.0)
            if source_id == add_source:
                exposure += add_amount
            if source_id == drop_source:
                exposure -= drop_amount
            share = exposure / max(new_total, 1.0)
            cap = caps.get(source_id, 1.0)
            slack = cap - share
            min_slack = min(min_slack, slack)
            max_share = max(max_share, share)
            if share > cap + 1e-7:
                violations += 1
                if not first_family:
                    first_family = family
                    first_source = source_id
    return (
        violations == 0,
        float(min_slack),
        float(max_share),
        violations,
        first_family,
        first_source,
    )


def _candidate_pairs_for_reprice() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected, source_summary, v304_row, selected_reward = _v304_best_solution()
    v295_summary = read_csv("paper4_v295_broader_multi_swap_or_global_gap_probe.csv")
    if universe.empty or selected.empty or source_summary.empty or v295_summary.empty:
        empty = pd.DataFrame(columns=_reprice_pair_columns(VERSION))
        return empty, pd.DataFrame(), pd.DataFrame()

    universe["loan_id"] = universe["loan_id"].astype(str)
    for family in FAMILIES:
        universe[family] = universe[family].astype(str)
    selected_ids = set(selected["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].isin(selected_ids)].copy()
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    candidate_idx = idx_by_id.loc[candidates["loan_id"].astype(str)].to_numpy()
    selected = selected.reset_index(drop=True)
    candidates = candidates.reset_index(drop=True)
    selected[f"mean_return_if_dropped_v{VERSION}"] = mean_returns[selected_idx]
    candidates[f"mean_return_if_added_v{VERSION}"] = mean_returns[candidate_idx]

    policy_id = "v304_bounded_observed_proxy_best_return_reward_0"
    regime = "post_v304_best_return_solution"
    v295_row = v295_summary.iloc[0]
    current_exposure = float(v304_row[f"portfolio_exposure_v{BOUNDED_MILP_VERSION}"])
    exposure_min = float(v295_row[f"exposure_min_v{BASELINE_VERSION}"])
    exposure_max = float(v295_row[f"exposure_max_v{BASELINE_VERSION}"])
    cvar_cap = float(v304_row[f"scenario_loss_cvar90_v{BOUNDED_MILP_VERSION}"])
    current_objective_return = float(v304_row[f"objective_return_v{BOUNDED_MILP_VERSION}"])
    current_losses = losses[:, selected_idx].sum(axis=1)

    add_amount = candidates["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    add_return = candidates[f"mean_return_if_added_v{VERSION}"].to_numpy(float)
    drop_return = selected[f"mean_return_if_dropped_v{VERSION}"].to_numpy(float)
    return_mask = add_return[:, None] - drop_return[None, :] > 1e-9
    total_pairs = int(return_mask.size)
    return_pairs = int(return_mask.sum())
    exposure_after = current_exposure + add_amount[:, None] - drop_amount[None, :]
    budget_return_mask = (
        return_mask
        & (exposure_after >= exposure_min - 1e-7)
        & (exposure_after <= exposure_max + 1e-7)
    )
    budget_return_pairs = int(budget_return_mask.sum())
    source_prefilter_mask = _source_prefilter_mask(
        budget_return_mask,
        candidates,
        selected,
        source_summary,
        current_exposure,
    )
    source_prefilter_pairs = int(source_prefilter_mask.sum())

    current_by_family, cap_by_family = _source_maps(selected, source_summary)
    rows: list[dict[str, Any]] = []
    for candidate_pos, selected_pos in np.argwhere(source_prefilter_mask):
        add_row = candidates.iloc[int(candidate_pos)]
        drop_row = selected.iloc[int(selected_pos)]
        new_total = float(current_exposure + add_row["loan_amnt"] - drop_row["loan_amnt"])
        source_ok, min_slack, max_share, violations, first_family, first_source = (
            _exact_source_metrics(
                add_row,
                drop_row,
                current_by_family,
                cap_by_family,
                new_total,
            )
        )
        if not source_ok:
            continue
        swapped_losses = (
            current_losses
            + losses[:, candidate_idx[int(candidate_pos)]]
            - losses[:, selected_idx[int(selected_pos)]]
        )
        cvar_after = v70._tail_cvar(swapped_losses)
        return_delta = float(
            add_row[f"mean_return_if_added_v{VERSION}"]
            - drop_row[f"mean_return_if_dropped_v{VERSION}"]
        )
        row: dict[str, Any] = {
            "policy_id": policy_id,
            f"regime_v{VERSION}": regime,
            f"added_loan_id_v{VERSION}": str(add_row["loan_id"]),
            f"dropped_loan_id_v{VERSION}": str(drop_row["loan_id"]),
            f"added_loan_amount_v{VERSION}": float(add_row["loan_amnt"]),
            f"dropped_loan_amount_v{VERSION}": float(drop_row["loan_amnt"]),
            f"added_mean_return_v{VERSION}": float(add_row[f"mean_return_if_added_v{VERSION}"]),
            f"dropped_mean_return_v{VERSION}": float(
                drop_row[f"mean_return_if_dropped_v{VERSION}"]
            ),
            f"return_delta_v{VERSION}": return_delta,
            f"objective_return_after_swap_v{VERSION}": current_objective_return + return_delta,
            f"exposure_after_swap_v{VERSION}": new_total,
            f"budget_swap_feasible_v{VERSION}": True,
            f"source_swap_feasible_v{VERSION}": True,
            f"source_min_slack_after_swap_v{VERSION}": min_slack,
            f"max_source_share_after_swap_v{VERSION}": max_share,
            f"source_cap_violations_after_swap_v{VERSION}": violations,
            f"first_source_block_family_v{VERSION}": first_family,
            f"first_source_block_id_v{VERSION}": first_source,
            f"loss_mean_after_swap_v{VERSION}": float(swapped_losses.mean()),
            f"cvar90_after_swap_v{VERSION}": cvar_after,
            f"cvar_swap_feasible_v{VERSION}": cvar_after <= cvar_cap + 1e-7,
            f"one_swap_improves_return_v{VERSION}": (
                return_delta > 1e-9 and cvar_after <= cvar_cap + 1e-7
            ),
            f"integer_screen_scope_v{VERSION}": REPRICE_SCOPE,
            f"claim_boundary_v{VERSION}": (
                "post-v304 best-return one-swap pricing only; not multi-swap or global proof"
            ),
        }
        for family in FAMILIES:
            row[f"added_{family}_v{VERSION}"] = str(add_row[family])
            row[f"dropped_{family}_v{VERSION}"] = str(drop_row[family])
        rows.append(row)

    pairs = pd.DataFrame(rows, columns=_reprice_pair_columns(VERSION))
    cvar_feasible_pairs = (
        int(pairs[f"cvar_swap_feasible_v{VERSION}"].sum()) if not pairs.empty else 0
    )
    improving_pairs = (
        int(pairs[f"one_swap_improves_return_v{VERSION}"].sum()) if not pairs.empty else 0
    )
    best = pairs.sort_values(f"return_delta_v{VERSION}", ascending=False).head(1)
    best_feasible = (
        pairs.loc[pairs[f"one_swap_improves_return_v{VERSION}"].astype(bool)]
        .sort_values(f"return_delta_v{VERSION}", ascending=False)
        .head(1)
    )
    local_claim_boundary = (
        "post-v304 one-swap screen cleared; dynamic/global gates still missing"
        if improving_pairs == 0
        else "post-v304 one-swap screen found improvements; repair/reprice loop must continue"
    )
    summary = pd.DataFrame(
        [
            {
                "policy_id": policy_id,
                f"regime_v{VERSION}": regime,
                f"selected_reward_v{VERSION}": selected_reward,
                f"selected_rows_v{VERSION}": int(len(selected)),
                f"base_selected_rows_v{VERSION}": TARGET_SELECTED_ROWS,
                f"cardinality_restored_v{VERSION}": int(len(selected)) == TARGET_SELECTED_ROWS,
                f"candidate_add_rows_v{VERSION}": int(len(candidates)),
                f"total_pair_rows_screened_v{VERSION}": total_pairs,
                f"return_improving_pair_rows_v{VERSION}": return_pairs,
                f"budget_return_feasible_pair_rows_v{VERSION}": budget_return_pairs,
                f"source_prefilter_pair_rows_v{VERSION}": source_prefilter_pairs,
                f"source_exact_pair_rows_v{VERSION}": int(len(pairs)),
                f"cvar_feasible_pair_rows_v{VERSION}": cvar_feasible_pairs,
                f"one_swap_improving_rows_v{VERSION}": improving_pairs,
                f"best_one_swap_return_delta_v{VERSION}": float(
                    best[f"return_delta_v{VERSION}"].iloc[0]
                )
                if not best.empty
                else np.nan,
                f"best_one_swap_cvar90_after_v{VERSION}": float(
                    best[f"cvar90_after_swap_v{VERSION}"].iloc[0]
                )
                if not best.empty
                else np.nan,
                f"best_feasible_one_swap_return_delta_v{VERSION}": float(
                    best_feasible[f"return_delta_v{VERSION}"].iloc[0]
                )
                if not best_feasible.empty
                else np.nan,
                f"best_feasible_one_swap_cvar90_after_v{VERSION}": float(
                    best_feasible[f"cvar90_after_swap_v{VERSION}"].iloc[0]
                )
                if not best_feasible.empty
                else np.nan,
                f"current_exposure_v{VERSION}": current_exposure,
                f"exposure_min_v{VERSION}": exposure_min,
                f"exposure_max_v{VERSION}": exposure_max,
                f"current_loss_mean_v{VERSION}": float(current_losses.mean()),
                f"current_cvar90_v{VERSION}": cvar_cap,
                f"current_objective_return_v{VERSION}": current_objective_return,
                f"post_v304_one_swap_local_optimality_cleared_v{VERSION}": (improving_pairs == 0),
                f"dynamic_gate_ready_v{VERSION}": improving_pairs == 0,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                f"paper4_final_promotion_created_v{VERSION}": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"claim_boundary_v{VERSION}": local_claim_boundary,
            }
        ]
    )
    stage_summary = pd.DataFrame(
        [
            {f"stage_v{VERSION}": "all_pairs", f"pair_rows_v{VERSION}": total_pairs},
            {f"stage_v{VERSION}": "return_improving", f"pair_rows_v{VERSION}": return_pairs},
            {
                f"stage_v{VERSION}": "budget_return_feasible",
                f"pair_rows_v{VERSION}": budget_return_pairs,
            },
            {
                f"stage_v{VERSION}": "source_prefilter_feasible",
                f"pair_rows_v{VERSION}": source_prefilter_pairs,
            },
            {
                f"stage_v{VERSION}": "source_exact_feasible",
                f"pair_rows_v{VERSION}": int(len(pairs)),
            },
            {
                f"stage_v{VERSION}": "cvar_feasible_improving",
                f"pair_rows_v{VERSION}": cvar_feasible_pairs,
            },
        ]
    )
    stage_summary[f"claim_boundary_v{VERSION}"] = (
        "post-v304 best-return solution one-swap screen stage count only"
    )
    return pairs, summary, stage_summary


def _claim_blockers(summary: pd.DataFrame) -> pd.DataFrame:
    improving = int(summary[f"one_swap_improving_rows_v{VERSION}"].iloc[0])
    next_artifact = (
        f"paper4_v{NEXT_VERSION}_apply_next_post_v304_swap.csv"
        if improving > 0
        else f"paper4_v{NEXT_VERSION}_post_v304_dynamic_proxy_or_global_bound_gate.csv"
    )
    return pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "post_v304_one_swap_improvement_found",
                f"blocking_v{VERSION}": improving > 0,
                f"evidence_count_v{VERSION}": improving,
                f"required_next_artifact_v{VERSION}": next_artifact,
                f"claim_boundary_v{VERSION}": (
                    "feasible improving post-v304 one-swaps block local optimality"
                    if improving > 0
                    else "no feasible improving post-v304 one-swaps remain"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "bounded_solution_not_working_champion",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": next_artifact,
                f"claim_boundary_v{VERSION}": (
                    "v304/v305 remain bounded-pool plus local one-swap evidence"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "global_integer_gap_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_full_universe_branch_price_bound",
                f"claim_boundary_v{VERSION}": "one-swap screen is not a global certificate",
            },
            {
                f"blocker_id_v{VERSION}": "dynamic_online_ifrs9_gates_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_dynamic_online_ifrs9_validation",
                f"claim_boundary_v{VERSION}": (
                    "dynamic replay, online holdout and contractual IFRS9 gates are not created"
                ),
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


def _claim_matrix(*, local_optimality_cleared: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v305_post_v304_one_swap_reprice_executed",
                "allowed": True,
                "artifact": "paper4_v305_post_v304_one_swap_summary.csv",
                "boundary": "post-v304 best-return solution one-drop/one-add screen completed",
            },
            {
                "claim_id": "v305_post_v304_one_swap_local_optimality",
                "allowed": local_optimality_cleared,
                "artifact": "paper4_v305_claim_blockers.csv",
                "boundary": "allowed only within one-drop/one-add scope",
            },
            {
                "claim_id": "v305_dynamic_gate_ready",
                "allowed": local_optimality_cleared,
                "artifact": "paper4_v305_post_v304_one_swap_summary.csv",
                "boundary": "ready only to attempt proxy dynamic/global gate, not to promote",
            },
            {
                "claim_id": "v305_working_champion",
                "allowed": False,
                "artifact": "paper4_v305_claim_blockers.csv",
                "boundary": "global, dynamic, online and deployment evidence missing",
            },
            {
                "claim_id": "v305_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v305_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )


def _update_claim_boundaries(*, local_optimality_cleared: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v305 post-v304 one-swap repricing gate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v305_post_v304_one_swap_summary.csv"
                ),
                "boundary": "One-drop/one-add screen after the v304 best-return bounded solution.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v305 clears post-v304 one-swap local optimality.",
                "allowed": local_optimality_cleared,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v305_claim_blockers.csv"
                ),
                "boundary": "Scope-limited to one-drop/one-add swaps after v304.",
                "prohibited_claim_flag": not local_optimality_cleared,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v305 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v305_claim_blockers.csv"
                ),
                "boundary": "Global, dynamic, online and deployment gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v305 proves full-universe global integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v305_claim_blockers.csv"
                ),
                "boundary": "A one-swap repricing screen is not a full branch-price certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v305 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v305_claim_blockers.csv"
                ),
                "boundary": "No final promotion, dynamic validation or deployment gate is created.",
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


def _update_backlog(*, local_optimality_cleared: bool) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    next_artifact = (
        f"paper4_v{NEXT_VERSION}_post_v304_dynamic_proxy_or_global_bound_gate.csv"
        if local_optimality_cleared
        else f"paper4_v{NEXT_VERSION}_apply_next_post_v304_swap.csv"
    )
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v305 reprices the v304 best-return bounded MILP solution against "
                    "all non-selected comparable loans with an exact one-swap screen."
                ),
                "status": (
                    "post_v304_one_swap_local_optimality_cleared"
                    if local_optimality_cleared
                    else "post_v304_one_swap_improvement_found"
                ),
                "next_artifact": next_artifact,
                "success_condition": (
                    "if cleared, run a dynamic/global gate; if not, apply the next "
                    "post-v304 repair and reprice again"
                ),
                "last_wave": "v305",
                "execution_result": (
                    "post_v304_one_swap_reprice_cleared"
                    if local_optimality_cleared
                    else "post_v304_one_swap_reprice_found_improvements"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v305")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V305_POST_V304_ONE_SWAP_REPRICE_START -->"
    end = "<!-- V305_POST_V304_ONE_SWAP_REPRICE_END -->"
    best_delta = status["best_one_swap_return_delta_v305"]
    best_delta_text = (
        "not applicable; no source-exact return-improving swaps"
        if best_delta is None
        else str(best_delta)
    )
    best_feasible = status["best_feasible_one_swap_return_delta_v305"]
    best_feasible_text = (
        "not applicable; no CVaR-feasible improving swaps"
        if best_feasible is None
        else str(best_feasible)
    )
    block = f"""
{start}

## Wave v305: Post-v304 One-Swap Repricing Gate

Generated: {status["generated_at_utc"]}

### Objective

v304 found strong bounded-pool MILP solutions. v305 tests the best-return v304
solution against every non-selected comparable loan with a one-drop/one-add
screen under the same budget band, exact source caps and the v304 CVaR cap.

### Results

- Selected v304 reward: `{status["selected_reward_v305"]}`.
- Selected rows: `{status["selected_rows_v305"]}`.
- Candidate add rows: `{status["candidate_add_rows_v305"]}`.
- Pair rows screened: `{status["total_pair_rows_screened_v305"]}`.
- Return-improving pairs: `{status["return_improving_pair_rows_v305"]}`.
- Budget+return feasible pairs: `{status["budget_return_feasible_pair_rows_v305"]}`.
- Source prefilter pairs: `{status["source_prefilter_pair_rows_v305"]}`.
- Exact source-feasible pairs: `{status["source_exact_pair_rows_v305"]}`.
- CVaR-feasible improving one-swaps: `{status["one_swap_improving_rows_v305"]}`.
- Best source-exact return delta: `{best_delta_text}`.
- Best CVaR-feasible return delta: `{best_feasible_text}`.
- Post-v304 one-swap local optimality cleared:
  `{status["post_v304_one_swap_local_optimality_cleared_v305"]}`.
- Dynamic/global gate ready:
  `{status["dynamic_gate_ready_v305"]}`.

### Interpretation

v305 either clears the immediate post-solve local repricing gate for the v304
best-return allocation or records the next repair signal. Even when local
one-swap stability clears, the result is still not a full-universe integer
certificate or a live dynamic validation.

### Claim Impact

- Allowed: post-v304 one-swap repricing gate completed.
- Still prohibited: full-universe/global optimality, Paper 4 working champion,
  Paper Estrella replacement, final Paper 4 promotion, contractual IFRS9 and
  live deployability claims.

### Quarto Promotion Decision

Keep v305 in the living notebook. Promotion remains blocked.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def _float_or_none(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def main() -> None:
    started = datetime.now(UTC)
    pairs, summary, stage_summary = _candidate_pairs_for_reprice()
    top_candidates = pairs.sort_values(
        [f"one_swap_improves_return_v{VERSION}", f"return_delta_v{VERSION}"],
        ascending=[False, False],
    ).head(200)
    local_optimality_cleared = bool(
        summary[f"post_v304_one_swap_local_optimality_cleared_v{VERSION}"].iloc[0]
    )
    blockers = _claim_blockers(summary)
    claim_matrix = _claim_matrix(local_optimality_cleared=local_optimality_cleared)

    write_csv(TABLE_DIR / "paper4_v305_post_v304_one_swap_reprice.csv", pairs)
    write_csv(TABLE_DIR / "paper4_v305_post_v304_one_swap_top_candidates.csv", top_candidates)
    write_csv(TABLE_DIR / "paper4_v305_post_v304_one_swap_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v305_post_v304_one_swap_stage_summary.csv", stage_summary)
    write_csv(TABLE_DIR / "paper4_v305_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v305_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries(local_optimality_cleared=local_optimality_cleared)
    _update_backlog(local_optimality_cleared=local_optimality_cleared)

    row = summary.iloc[0]
    best_delta = row[f"best_one_swap_return_delta_v{VERSION}"]
    best_cvar = row[f"best_one_swap_cvar90_after_v{VERSION}"]
    best_feasible_delta = row[f"best_feasible_one_swap_return_delta_v{VERSION}"]
    best_feasible_cvar = row[f"best_feasible_one_swap_cvar90_after_v{VERSION}"]
    status = {
        "phase": "v305_post_v304_reprice_or_dynamic_gate",
        "schema_version": "2026-05-16.305",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "bounded_milp_version_v305": BOUNDED_MILP_VERSION,
        "baseline_version_v305": BASELINE_VERSION,
        "selected_reward_v305": float(row[f"selected_reward_v{VERSION}"]),
        "summary_rows_v305": int(len(summary)),
        "stage_summary_rows_v305": int(len(stage_summary)),
        "candidate_pair_rows_v305": int(len(pairs)),
        "top_candidate_rows_v305": int(len(top_candidates)),
        "claim_blocker_rows_v305": int(len(blockers)),
        "claim_matrix_rows_v305": int(len(claim_matrix)),
        "selected_rows_v305": int(row[f"selected_rows_v{VERSION}"]),
        "base_selected_rows_v305": int(row[f"base_selected_rows_v{VERSION}"]),
        "cardinality_restored_v305": bool(row[f"cardinality_restored_v{VERSION}"]),
        "candidate_add_rows_v305": int(row[f"candidate_add_rows_v{VERSION}"]),
        "total_pair_rows_screened_v305": int(row[f"total_pair_rows_screened_v{VERSION}"]),
        "return_improving_pair_rows_v305": int(row[f"return_improving_pair_rows_v{VERSION}"]),
        "budget_return_feasible_pair_rows_v305": int(
            row[f"budget_return_feasible_pair_rows_v{VERSION}"]
        ),
        "source_prefilter_pair_rows_v305": int(row[f"source_prefilter_pair_rows_v{VERSION}"]),
        "source_exact_pair_rows_v305": int(row[f"source_exact_pair_rows_v{VERSION}"]),
        "cvar_feasible_pair_rows_v305": int(row[f"cvar_feasible_pair_rows_v{VERSION}"]),
        "one_swap_improving_rows_v305": int(row[f"one_swap_improving_rows_v{VERSION}"]),
        "best_one_swap_return_delta_v305": _float_or_none(best_delta),
        "best_one_swap_cvar90_after_v305": _float_or_none(best_cvar),
        "best_feasible_one_swap_return_delta_v305": _float_or_none(best_feasible_delta),
        "best_feasible_one_swap_cvar90_after_v305": _float_or_none(best_feasible_cvar),
        "current_exposure_v305": float(row[f"current_exposure_v{VERSION}"]),
        "current_objective_return_v305": float(row[f"current_objective_return_v{VERSION}"]),
        "current_loss_mean_v305": float(row[f"current_loss_mean_v{VERSION}"]),
        "current_cvar90_v305": float(row[f"current_cvar90_v{VERSION}"]),
        "post_v304_one_swap_local_optimality_cleared_v305": local_optimality_cleared,
        "dynamic_gate_ready_v305": local_optimality_cleared,
        "working_champion_claim_allowed_v305": False,
        "full_universe_integer_optimality_claim_allowed_v305": False,
        "paper1_promotion_allowed_v305": False,
        "paper4_working_champion_changed_v305": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v305": (
            f"paper4_v{NEXT_VERSION}_post_v304_dynamic_proxy_or_global_bound_gate.csv"
            if local_optimality_cleared
            else f"paper4_v{NEXT_VERSION}_apply_next_post_v304_swap.csv"
        ),
        "claim_boundary": (
            "v305 is a post-v304 one-swap repricing gate; no working champion or "
            "final promotion is authorized"
        ),
    }
    write_json(STATUS_DIR / "paper4_v305_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v305": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
