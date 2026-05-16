#!/usr/bin/env python3
"""Build Paper 4 v307 post-v306 one-swap repricing artifacts."""

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

VERSION = 307
BASE_REPAIR_VERSION = 306
PREVIOUS_REPRICE_VERSION = 305
NEXT_VERSION = 308
TARGET_SELECTED_ROWS = 171
REPRICE_SCOPE = "post_v306_repair_one_drop_one_add_whole_loan_swap"


def _pair_columns() -> list[str]:
    return [
        *_reprice_pair_columns(VERSION),
        f"added_observed_v47_proxy_v{VERSION}",
        f"dropped_observed_v47_proxy_v{VERSION}",
        f"delta_missing_v47_proxy_rows_v{VERSION}",
    ]


def _source_maps(
    selected: pd.DataFrame, source_summary: pd.DataFrame
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    current_by_family: dict[str, dict[str, float]] = {}
    cap_by_family: dict[str, dict[str, float]] = {}
    cap_col = f"cap_share_v{BASE_REPAIR_VERSION}"
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
    cap_col = f"cap_share_v{BASE_REPAIR_VERSION}"
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
    selected = read_parquet("paper4_v306_apply_next_post_v304_swap_allocations.parquet")
    source_summary = read_csv("paper4_v306_apply_next_post_v304_swap_source_summary.csv")
    v306_summary = read_csv("paper4_v306_apply_next_post_v304_swap_summary.csv")
    v305_summary = read_csv("paper4_v305_post_v304_one_swap_summary.csv")
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    if any(
        df.empty
        for df in [universe, selected, source_summary, v306_summary, v305_summary, v47_panel]
    ):
        return pd.DataFrame(columns=_pair_columns()), pd.DataFrame(), pd.DataFrame()

    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    for family in FAMILIES:
        universe[family] = universe[family].astype(str)
        selected[family] = selected[family].astype(str)
    source_summary["source_id"] = source_summary["source_id"].astype(str)
    observed_ids = set(v47_panel["loan_id"].astype(str))
    selected_ids = set(selected["loan_id"].astype(str))
    current_missing_proxy_rows = int((~selected["loan_id"].isin(observed_ids)).sum())
    candidates = universe.loc[~universe["loan_id"].isin(selected_ids)].copy()

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    candidate_idx = idx_by_id.loc[candidates["loan_id"].astype(str)].to_numpy()
    selected = selected.reset_index(drop=True)
    candidates = candidates.reset_index(drop=True)
    selected[f"mean_return_if_dropped_v{VERSION}"] = mean_returns[selected_idx]
    candidates[f"mean_return_if_added_v{VERSION}"] = mean_returns[candidate_idx]

    policy_id = "v306_post_v304_best_one_swap_repair_candidate"
    regime = "post_v306_repair_candidate"
    v306_row = v306_summary.iloc[0]
    v305_row = v305_summary.iloc[0]
    current_exposure = float(v306_row[f"portfolio_exposure_v{BASE_REPAIR_VERSION}"])
    exposure_min = float(v305_row[f"exposure_min_v{PREVIOUS_REPRICE_VERSION}"])
    exposure_max = float(v305_row[f"exposure_max_v{PREVIOUS_REPRICE_VERSION}"])
    cvar_cap = float(v306_row[f"scenario_loss_cvar90_v{BASE_REPAIR_VERSION}"])
    current_objective_return = float(v306_row[f"objective_return_v{BASE_REPAIR_VERSION}"])
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
        add_observed = str(add_row["loan_id"]) in observed_ids
        drop_observed = str(drop_row["loan_id"]) in observed_ids
        delta_missing_proxy = int(not add_observed) - int(not drop_observed)
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
                "post-v306 one-swap pricing only; not multi-swap or global proof"
            ),
            f"added_observed_v47_proxy_v{VERSION}": add_observed,
            f"dropped_observed_v47_proxy_v{VERSION}": drop_observed,
            f"delta_missing_v47_proxy_rows_v{VERSION}": delta_missing_proxy,
        }
        for family in FAMILIES:
            row[f"added_{family}_v{VERSION}"] = str(add_row[family])
            row[f"dropped_{family}_v{VERSION}"] = str(drop_row[family])
        rows.append(row)

    pairs = pd.DataFrame(rows, columns=_pair_columns())
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
        "post-v306 one-swap screen cleared; dynamic/global gates still missing"
        if improving_pairs == 0
        else "post-v306 one-swap screen found improvements; repair/reprice loop must continue"
    )
    best_feasible_delta_missing = (
        int(best_feasible[f"delta_missing_v47_proxy_rows_v{VERSION}"].iloc[0])
        if not best_feasible.empty
        else 0
    )
    summary = pd.DataFrame(
        [
            {
                "policy_id": policy_id,
                f"regime_v{VERSION}": regime,
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
                f"current_missing_v47_proxy_rows_v{VERSION}": current_missing_proxy_rows,
                f"best_feasible_delta_missing_v47_proxy_rows_v{VERSION}": (
                    best_feasible_delta_missing
                ),
                f"current_exposure_v{VERSION}": current_exposure,
                f"exposure_min_v{VERSION}": exposure_min,
                f"exposure_max_v{VERSION}": exposure_max,
                f"current_loss_mean_v{VERSION}": float(current_losses.mean()),
                f"current_cvar90_v{VERSION}": cvar_cap,
                f"current_objective_return_v{VERSION}": current_objective_return,
                f"post_v306_one_swap_local_optimality_cleared_v{VERSION}": (improving_pairs == 0),
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
        "post-v306 repair one-swap screen stage count only"
    )
    return pairs, summary, stage_summary


def _claim_blockers(summary: pd.DataFrame) -> pd.DataFrame:
    improving = int(summary[f"one_swap_improving_rows_v{VERSION}"].iloc[0])
    next_artifact = (
        f"paper4_v{NEXT_VERSION}_apply_next_post_v306_swap.csv"
        if improving > 0
        else f"paper4_v{NEXT_VERSION}_dynamic_proxy_or_global_bound_after_v306.csv"
    )
    return pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "post_v306_one_swap_improvement_found",
                f"blocking_v{VERSION}": improving > 0,
                f"evidence_count_v{VERSION}": improving,
                f"required_next_artifact_v{VERSION}": next_artifact,
                f"claim_boundary_v{VERSION}": (
                    "feasible improving post-v306 one-swaps block local optimality"
                    if improving > 0
                    else "no feasible improving post-v306 one-swaps remain"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "post_v306_repair_not_working_champion",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": next_artifact,
                f"claim_boundary_v{VERSION}": "local repricing evidence only",
            },
            {
                f"blocker_id_v{VERSION}": "proxy_coverage_gap_persists",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    summary[f"current_missing_v47_proxy_rows_v{VERSION}"].iloc[0]
                ),
                f"required_next_artifact_v{VERSION}": "future_cashflow_proxy_or_ifrs9_coverage_gate",
                f"claim_boundary_v{VERSION}": "v306 portfolio still has missing v47 proxy rows",
            },
            {
                f"blocker_id_v{VERSION}": "global_dynamic_online_gates_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_global_dynamic_online_validation",
                f"claim_boundary_v{VERSION}": "no global, dynamic, online or deployment gate created",
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
                "claim_id": "v307_post_v306_one_swap_reprice_executed",
                "allowed": True,
                "artifact": "paper4_v307_post_v306_one_swap_summary.csv",
                "boundary": "post-v306 one-drop/one-add screen completed",
            },
            {
                "claim_id": "v307_post_v306_one_swap_local_optimality",
                "allowed": local_optimality_cleared,
                "artifact": "paper4_v307_claim_blockers.csv",
                "boundary": "allowed only within one-drop/one-add scope",
            },
            {
                "claim_id": "v307_dynamic_gate_ready",
                "allowed": local_optimality_cleared,
                "artifact": "paper4_v307_post_v306_one_swap_summary.csv",
                "boundary": "ready only to attempt proxy dynamic/global gate, not to promote",
            },
            {
                "claim_id": "v307_working_champion",
                "allowed": False,
                "artifact": "paper4_v307_claim_blockers.csv",
                "boundary": "global, dynamic, online and deployment evidence missing",
            },
            {
                "claim_id": "v307_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v307_claim_blockers.csv",
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
                "claim": "Paper 4 has a v307 post-v306 one-swap repricing gate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v307_post_v306_one_swap_summary.csv"
                ),
                "boundary": "One-drop/one-add screen after the v306 repaired candidate.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v307 clears post-v306 one-swap local optimality.",
                "allowed": local_optimality_cleared,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v307_claim_blockers.csv"
                ),
                "boundary": "Scope-limited to one-drop/one-add swaps after v306.",
                "prohibited_claim_flag": not local_optimality_cleared,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v307 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v307_claim_blockers.csv"
                ),
                "boundary": "Global, dynamic, online and deployment gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v307 proves full-universe global integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v307_claim_blockers.csv"
                ),
                "boundary": "A one-swap repricing screen is not a full branch-price certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v307 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v307_claim_blockers.csv"
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
        f"paper4_v{NEXT_VERSION}_dynamic_proxy_or_global_bound_after_v306.csv"
        if local_optimality_cleared
        else f"paper4_v{NEXT_VERSION}_apply_next_post_v306_swap.csv"
    )
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v307 reprices the v306 repaired candidate against all non-selected "
                    "comparable loans with exact budget/source/CVaR screens."
                ),
                "status": (
                    "post_v306_one_swap_local_optimality_cleared"
                    if local_optimality_cleared
                    else "post_v306_one_swap_improvement_found"
                ),
                "next_artifact": next_artifact,
                "success_condition": (
                    "if cleared, run a dynamic/global gate; if not, apply the next "
                    "post-v306 repair and reprice again"
                ),
                "last_wave": "v307",
                "execution_result": (
                    "post_v306_one_swap_reprice_cleared"
                    if local_optimality_cleared
                    else "post_v306_one_swap_reprice_found_improvements"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v307")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V307_POST_V306_ONE_SWAP_REPRICE_START -->"
    end = "<!-- V307_POST_V306_ONE_SWAP_REPRICE_END -->"
    best_delta = status["best_one_swap_return_delta_v307"]
    best_delta_text = (
        "not applicable; no source-exact return-improving swaps"
        if best_delta is None
        else str(best_delta)
    )
    best_feasible = status["best_feasible_one_swap_return_delta_v307"]
    best_feasible_text = (
        "not applicable; no CVaR-feasible improving swaps"
        if best_feasible is None
        else str(best_feasible)
    )
    block = f"""
{start}

## Wave v307: Post-v306 One-Swap Repricing Gate

Generated: {status["generated_at_utc"]}

### Objective

v306 applied the best feasible v305 repair. v307 tests whether that repaired
candidate has one-drop/one-add stability against the full comparable universe
under the v306 CVaR cap, exact source caps and the original budget band.

### Results

- Selected rows: `{status["selected_rows_v307"]}`.
- Candidate add rows: `{status["candidate_add_rows_v307"]}`.
- Pair rows screened: `{status["total_pair_rows_screened_v307"]}`.
- Return-improving pairs: `{status["return_improving_pair_rows_v307"]}`.
- Budget+return feasible pairs: `{status["budget_return_feasible_pair_rows_v307"]}`.
- Source prefilter pairs: `{status["source_prefilter_pair_rows_v307"]}`.
- Exact source-feasible pairs: `{status["source_exact_pair_rows_v307"]}`.
- CVaR-feasible improving one-swaps: `{status["one_swap_improving_rows_v307"]}`.
- Best source-exact return delta: `{best_delta_text}`.
- Best CVaR-feasible return delta: `{best_feasible_text}`.
- Current missing v47 proxy rows:
  `{status["current_missing_v47_proxy_rows_v307"]}`.
- Best feasible delta missing proxy rows:
  `{status["best_feasible_delta_missing_v47_proxy_rows_v307"]}`.
- Post-v306 one-swap local optimality cleared:
  `{status["post_v306_one_swap_local_optimality_cleared_v307"]}`.
- Dynamic/global gate ready:
  `{status["dynamic_gate_ready_v307"]}`.

### Interpretation

v307 either clears the post-v306 local repricing gate or records another repair
signal. The proxy coverage gap is carried explicitly so a return-improving
local move cannot quietly become an IFRS9 or live-deployment claim.

### Claim Impact

- Allowed: post-v306 one-swap repricing gate completed.
- Still prohibited: full-universe/global optimality, Paper 4 working champion,
  Paper Estrella replacement, final Paper 4 promotion, contractual IFRS9 and
  live deployability claims.

### Quarto Promotion Decision

Keep v307 in the living notebook. Promotion remains blocked.

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
        summary[f"post_v306_one_swap_local_optimality_cleared_v{VERSION}"].iloc[0]
    )
    blockers = _claim_blockers(summary)
    claim_matrix = _claim_matrix(local_optimality_cleared=local_optimality_cleared)

    write_csv(TABLE_DIR / "paper4_v307_post_v306_one_swap_reprice.csv", pairs)
    write_csv(TABLE_DIR / "paper4_v307_post_v306_one_swap_top_candidates.csv", top_candidates)
    write_csv(TABLE_DIR / "paper4_v307_post_v306_one_swap_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v307_post_v306_one_swap_stage_summary.csv", stage_summary)
    write_csv(TABLE_DIR / "paper4_v307_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v307_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries(local_optimality_cleared=local_optimality_cleared)
    _update_backlog(local_optimality_cleared=local_optimality_cleared)

    row = summary.iloc[0]
    best_delta = row[f"best_one_swap_return_delta_v{VERSION}"]
    best_cvar = row[f"best_one_swap_cvar90_after_v{VERSION}"]
    best_feasible_delta = row[f"best_feasible_one_swap_return_delta_v{VERSION}"]
    best_feasible_cvar = row[f"best_feasible_one_swap_cvar90_after_v{VERSION}"]
    status = {
        "phase": "v307_post_v306_reprice",
        "schema_version": "2026-05-16.307",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_repair_version_v307": BASE_REPAIR_VERSION,
        "previous_reprice_version_v307": PREVIOUS_REPRICE_VERSION,
        "summary_rows_v307": int(len(summary)),
        "stage_summary_rows_v307": int(len(stage_summary)),
        "candidate_pair_rows_v307": int(len(pairs)),
        "top_candidate_rows_v307": int(len(top_candidates)),
        "claim_blocker_rows_v307": int(len(blockers)),
        "claim_matrix_rows_v307": int(len(claim_matrix)),
        "selected_rows_v307": int(row[f"selected_rows_v{VERSION}"]),
        "base_selected_rows_v307": int(row[f"base_selected_rows_v{VERSION}"]),
        "cardinality_restored_v307": bool(row[f"cardinality_restored_v{VERSION}"]),
        "candidate_add_rows_v307": int(row[f"candidate_add_rows_v{VERSION}"]),
        "total_pair_rows_screened_v307": int(row[f"total_pair_rows_screened_v{VERSION}"]),
        "return_improving_pair_rows_v307": int(row[f"return_improving_pair_rows_v{VERSION}"]),
        "budget_return_feasible_pair_rows_v307": int(
            row[f"budget_return_feasible_pair_rows_v{VERSION}"]
        ),
        "source_prefilter_pair_rows_v307": int(row[f"source_prefilter_pair_rows_v{VERSION}"]),
        "source_exact_pair_rows_v307": int(row[f"source_exact_pair_rows_v{VERSION}"]),
        "cvar_feasible_pair_rows_v307": int(row[f"cvar_feasible_pair_rows_v{VERSION}"]),
        "one_swap_improving_rows_v307": int(row[f"one_swap_improving_rows_v{VERSION}"]),
        "best_one_swap_return_delta_v307": _float_or_none(best_delta),
        "best_one_swap_cvar90_after_v307": _float_or_none(best_cvar),
        "best_feasible_one_swap_return_delta_v307": _float_or_none(best_feasible_delta),
        "best_feasible_one_swap_cvar90_after_v307": _float_or_none(best_feasible_cvar),
        "current_missing_v47_proxy_rows_v307": int(
            row[f"current_missing_v47_proxy_rows_v{VERSION}"]
        ),
        "best_feasible_delta_missing_v47_proxy_rows_v307": int(
            row[f"best_feasible_delta_missing_v47_proxy_rows_v{VERSION}"]
        ),
        "current_exposure_v307": float(row[f"current_exposure_v{VERSION}"]),
        "current_objective_return_v307": float(row[f"current_objective_return_v{VERSION}"]),
        "current_loss_mean_v307": float(row[f"current_loss_mean_v{VERSION}"]),
        "current_cvar90_v307": float(row[f"current_cvar90_v{VERSION}"]),
        "post_v306_one_swap_local_optimality_cleared_v307": local_optimality_cleared,
        "dynamic_gate_ready_v307": local_optimality_cleared,
        "working_champion_claim_allowed_v307": False,
        "full_universe_integer_optimality_claim_allowed_v307": False,
        "paper1_promotion_allowed_v307": False,
        "paper4_working_champion_changed_v307": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v307": (
            f"paper4_v{NEXT_VERSION}_dynamic_proxy_or_global_bound_after_v306.csv"
            if local_optimality_cleared
            else f"paper4_v{NEXT_VERSION}_apply_next_post_v306_swap.csv"
        ),
        "claim_boundary": (
            "v307 is a post-v306 one-swap repricing gate; no working champion or "
            "final promotion is authorized"
        ),
    }
    write_json(STATUS_DIR / "paper4_v307_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v307": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
