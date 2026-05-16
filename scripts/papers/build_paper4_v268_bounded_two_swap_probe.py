#!/usr/bin/env python3
"""Build Paper 4 v268 bounded two-swap probe after v267 local clearance."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers import build_paper4_v82_one_swap_integer_pricing_probe as v82
from scripts.papers.paper4_one_swap_living_lab import (
    FAMILIES,
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    read_csv,
    read_parquet,
    write_csv,
    write_json,
)

VERSION = 268
PREVIOUS_REPAIR_VERSION = 266
TERMINAL_REPRICE_VERSION = 267
NEXT_VERSION = 269
PRIMARY_FRONTIER_LIMIT = 25


def _primary_columns() -> list[str]:
    columns = [
        "policy_id",
        f"regime_v{VERSION}",
        f"primary_rank_v{VERSION}",
        f"primary_added_loan_id_v{VERSION}",
        f"primary_dropped_loan_id_v{VERSION}",
        f"primary_added_loan_amount_v{VERSION}",
        f"primary_dropped_loan_amount_v{VERSION}",
        f"primary_return_delta_v{VERSION}",
        f"primary_exposure_after_v{VERSION}",
        f"primary_source_cap_violations_v{VERSION}",
        f"primary_first_source_block_family_v{VERSION}",
        f"primary_first_source_block_id_v{VERSION}",
        f"primary_source_min_slack_v{VERSION}",
        f"primary_max_source_share_v{VERSION}",
        f"claim_boundary_v{VERSION}",
    ]
    for family in FAMILIES:
        columns.extend(
            [f"primary_added_{family}_v{VERSION}", f"primary_dropped_{family}_v{VERSION}"]
        )
    return columns


def _candidate_columns() -> list[str]:
    columns = [
        "policy_id",
        f"regime_v{VERSION}",
        f"primary_rank_v{VERSION}",
        f"primary_added_loan_id_v{VERSION}",
        f"primary_dropped_loan_id_v{VERSION}",
        f"relief_added_loan_id_v{VERSION}",
        f"relief_dropped_loan_id_v{VERSION}",
        f"primary_return_delta_v{VERSION}",
        f"relief_return_delta_v{VERSION}",
        f"total_return_delta_v{VERSION}",
        f"objective_return_after_two_swap_v{VERSION}",
        f"exposure_after_two_swap_v{VERSION}",
        f"source_swap_feasible_v{VERSION}",
        f"source_min_slack_after_two_swap_v{VERSION}",
        f"max_source_share_after_two_swap_v{VERSION}",
        f"source_cap_violations_after_two_swap_v{VERSION}",
        f"first_source_block_family_v{VERSION}",
        f"first_source_block_id_v{VERSION}",
        f"loss_mean_after_two_swap_v{VERSION}",
        f"cvar90_after_two_swap_v{VERSION}",
        f"cvar_swap_feasible_v{VERSION}",
        f"two_swap_improves_return_v{VERSION}",
        f"bounded_multi_swap_scope_v{VERSION}",
        f"claim_boundary_v{VERSION}",
    ]
    for family in FAMILIES:
        columns.extend(
            [
                f"primary_added_{family}_v{VERSION}",
                f"primary_dropped_{family}_v{VERSION}",
                f"relief_added_{family}_v{VERSION}",
                f"relief_dropped_{family}_v{VERSION}",
            ]
        )
    return columns


def _primary_frontier(
    *,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    source_summary: pd.DataFrame,
    current_exposure: float,
    exposure_min: float,
    exposure_max: float,
) -> tuple[list[tuple[int, int, float, float]], int, int, int]:
    add_amount = candidates["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    add_return = candidates[f"mean_return_if_added_v{VERSION}"].to_numpy(float)
    drop_return = selected[f"mean_return_if_dropped_v{VERSION}"].to_numpy(float)
    return_mask = add_return[:, None] - drop_return[None, :] > 1e-9
    exposure_after = current_exposure + add_amount[:, None] - drop_amount[None, :]
    budget_mask = (
        return_mask
        & (exposure_after >= exposure_min - 1e-7)
        & (exposure_after <= exposure_max + 1e-7)
    )
    source_prefilter = v82._source_prefilter_mask(
        budget_mask,
        candidates,
        selected,
        source_summary,
        current_exposure,
    )
    primary_positions: list[tuple[int, int, float, float]] = []
    for candidate_pos, selected_pos in np.argwhere(source_prefilter):
        add_pos = int(candidate_pos)
        drop_pos = int(selected_pos)
        primary_delta = float(add_return[add_pos] - drop_return[drop_pos])
        primary_net_exposure = float(add_amount[add_pos] - drop_amount[drop_pos])
        primary_positions.append((add_pos, drop_pos, primary_delta, primary_net_exposure))
    primary_positions = sorted(primary_positions, key=lambda x: x[2], reverse=True)
    return (
        primary_positions[:PRIMARY_FRONTIER_LIMIT],
        int(return_mask.sum()),
        int(budget_mask.sum()),
        int(source_prefilter.sum()),
    )


def _build_artifacts() -> dict[str, Any]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    selected = read_parquet("paper4_v266_bounded_two_swap_repair_allocations.parquet").reset_index(
        drop=True
    )
    source_summary = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_summary = source_summary.loc[
        source_summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    repair_summary = read_csv("paper4_v266_bounded_two_swap_repair_summary.csv")
    repair_row = repair_summary.iloc[0]
    if universe.empty or selected.empty or source_summary.empty:
        raise RuntimeError("Missing universe, v266 repaired portfolio, or source caps.")

    selected_ids = set(selected["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    candidates = candidates.reset_index(drop=True)
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    candidate_idx = idx_by_id.loc[candidates["loan_id"].astype(str)].to_numpy()
    selected[f"mean_return_if_dropped_v{VERSION}"] = mean_returns[selected_idx]
    candidates[f"mean_return_if_added_v{VERSION}"] = mean_returns[candidate_idx]

    current_exposure = float(repair_row[f"portfolio_exposure_v{PREVIOUS_REPAIR_VERSION}"])
    exposure_min = float(repair_row[f"exposure_min_v{PREVIOUS_REPAIR_VERSION}"])
    exposure_max = float(repair_row[f"exposure_max_v{PREVIOUS_REPAIR_VERSION}"])
    cvar_cap = float(repair_row[f"cvar_cap_v{PREVIOUS_REPAIR_VERSION}"])
    current_objective = float(repair_row[f"objective_return_v{PREVIOUS_REPAIR_VERSION}"])
    current_losses = losses[:, selected_idx].sum(axis=1)
    current_by_family, cap_by_family = v82._source_maps(selected, source_summary)

    primary_positions, return_pairs, budget_pairs, source_prefilter_pairs = _primary_frontier(
        candidates=candidates,
        selected=selected,
        source_summary=source_summary,
        current_exposure=current_exposure,
        exposure_min=exposure_min,
        exposure_max=exposure_max,
    )

    primary_rows: list[dict[str, Any]] = []
    two_swap_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for primary_rank, (primary_add_pos, primary_drop_pos, primary_delta, primary_net) in enumerate(
        primary_positions
    ):
        primary_add = candidates.iloc[primary_add_pos]
        primary_drop = selected.iloc[primary_drop_pos]
        primary_total = current_exposure + primary_net
        (
            primary_source_ok,
            primary_min_slack,
            primary_max_share,
            primary_violations,
            primary_first_family,
            primary_first_source,
        ) = v82._exact_source_metrics(
            primary_add,
            primary_drop,
            current_by_family,
            cap_by_family,
            primary_total,
        )
        if primary_source_ok:
            continue
        primary_row: dict[str, Any] = {
            "policy_id": "v63_source_repair_return_guarded_repair",
            f"regime_v{VERSION}": "bounded_two_swap_after_v266",
            f"primary_rank_v{VERSION}": primary_rank,
            f"primary_added_loan_id_v{VERSION}": str(primary_add["loan_id"]),
            f"primary_dropped_loan_id_v{VERSION}": str(primary_drop["loan_id"]),
            f"primary_added_loan_amount_v{VERSION}": float(primary_add["loan_amnt"]),
            f"primary_dropped_loan_amount_v{VERSION}": float(primary_drop["loan_amnt"]),
            f"primary_return_delta_v{VERSION}": primary_delta,
            f"primary_exposure_after_v{VERSION}": primary_total,
            f"primary_source_cap_violations_v{VERSION}": primary_violations,
            f"primary_first_source_block_family_v{VERSION}": primary_first_family,
            f"primary_first_source_block_id_v{VERSION}": primary_first_source,
            f"primary_source_min_slack_v{VERSION}": primary_min_slack,
            f"primary_max_source_share_v{VERSION}": primary_max_share,
            f"claim_boundary_v{VERSION}": (
                "bounded primary one-swap from post-v266 source-prefilter frontier; "
                "exact source requires relief"
            ),
        }
        for family in FAMILIES:
            primary_row[f"primary_added_{family}_v{VERSION}"] = str(primary_add[family])
            primary_row[f"primary_dropped_{family}_v{VERSION}"] = str(primary_drop[family])
        primary_rows.append(primary_row)

        selected_after_primary = pd.concat(
            [
                selected.loc[~selected["loan_id"].astype(str).eq(str(primary_drop["loan_id"]))],
                primary_add.to_frame().T,
            ],
            ignore_index=True,
        ).reset_index(drop=True)
        relief_selected = selected_after_primary.loc[
            ~selected_after_primary["loan_id"].astype(str).eq(str(primary_add["loan_id"]))
        ].reset_index(drop=True)
        relief_candidates = candidates.loc[
            ~candidates["loan_id"]
            .astype(str)
            .isin({str(primary_add["loan_id"]), str(primary_drop["loan_id"])})
        ].reset_index(drop=True)
        relief_candidate_idx = idx_by_id.loc[relief_candidates["loan_id"].astype(str)].to_numpy()
        relief_selected_idx = idx_by_id.loc[relief_selected["loan_id"].astype(str)].to_numpy()
        relief_add_amount = relief_candidates["loan_amnt"].to_numpy(float)
        relief_drop_amount = relief_selected["loan_amnt"].to_numpy(float)
        relief_add_return = mean_returns[relief_candidate_idx]
        relief_drop_return = mean_returns[relief_selected_idx]
        total_delta = primary_delta + relief_add_return[:, None] - relief_drop_return[None, :]
        final_exposure = primary_total + relief_add_amount[:, None] - relief_drop_amount[None, :]
        base_mask = (
            (total_delta > 1e-9)
            & (final_exposure >= exposure_min - 1e-7)
            & (final_exposure <= exposure_max + 1e-7)
        )
        source_prefilter_mask = v82._source_prefilter_mask(
            base_mask,
            relief_candidates,
            relief_selected,
            source_summary,
            primary_total,
        )
        exact_rows = 0
        cvar_rows = 0
        current_by_family_after_primary, _cap_by_family = v82._source_maps(
            selected_after_primary,
            source_summary,
        )
        for relief_candidate_pos, relief_selected_pos in np.argwhere(source_prefilter_mask):
            relief_add = relief_candidates.iloc[int(relief_candidate_pos)]
            relief_drop = relief_selected.iloc[int(relief_selected_pos)]
            final_total = float(final_exposure[int(relief_candidate_pos), int(relief_selected_pos)])
            source_ok, min_slack, max_share, violations, first_family, first_source = (
                v82._exact_source_metrics(
                    relief_add,
                    relief_drop,
                    current_by_family_after_primary,
                    cap_by_family,
                    final_total,
                )
            )
            if not source_ok:
                continue
            exact_rows += 1
            swapped_losses = (
                current_losses
                + losses[:, idx_by_id[str(primary_add["loan_id"])]]
                + losses[:, idx_by_id[str(relief_add["loan_id"])]]
                - losses[:, idx_by_id[str(primary_drop["loan_id"])]]
                - losses[:, idx_by_id[str(relief_drop["loan_id"])]]
            )
            cvar90_after = v70._tail_cvar(swapped_losses)
            cvar_feasible = cvar90_after <= cvar_cap + 1e-7
            if cvar_feasible:
                cvar_rows += 1
            relief_delta = float(
                relief_add[f"mean_return_if_added_v{VERSION}"]
                - relief_drop[f"mean_return_if_dropped_v{VERSION}"]
            )
            total_return_delta = float(
                total_delta[int(relief_candidate_pos), int(relief_selected_pos)]
            )
            row: dict[str, Any] = {
                "policy_id": "v63_source_repair_return_guarded_repair",
                f"regime_v{VERSION}": "bounded_two_swap_after_v266",
                f"primary_rank_v{VERSION}": primary_rank,
                f"primary_added_loan_id_v{VERSION}": str(primary_add["loan_id"]),
                f"primary_dropped_loan_id_v{VERSION}": str(primary_drop["loan_id"]),
                f"relief_added_loan_id_v{VERSION}": str(relief_add["loan_id"]),
                f"relief_dropped_loan_id_v{VERSION}": str(relief_drop["loan_id"]),
                f"primary_return_delta_v{VERSION}": primary_delta,
                f"relief_return_delta_v{VERSION}": relief_delta,
                f"total_return_delta_v{VERSION}": total_return_delta,
                f"objective_return_after_two_swap_v{VERSION}": current_objective
                + total_return_delta,
                f"exposure_after_two_swap_v{VERSION}": final_total,
                f"source_swap_feasible_v{VERSION}": source_ok,
                f"source_min_slack_after_two_swap_v{VERSION}": min_slack,
                f"max_source_share_after_two_swap_v{VERSION}": max_share,
                f"source_cap_violations_after_two_swap_v{VERSION}": violations,
                f"first_source_block_family_v{VERSION}": first_family,
                f"first_source_block_id_v{VERSION}": first_source,
                f"loss_mean_after_two_swap_v{VERSION}": float(swapped_losses.mean()),
                f"cvar90_after_two_swap_v{VERSION}": cvar90_after,
                f"cvar_swap_feasible_v{VERSION}": cvar_feasible,
                f"two_swap_improves_return_v{VERSION}": total_return_delta > 1e-9,
                f"bounded_multi_swap_scope_v{VERSION}": (
                    f"top_{PRIMARY_FRONTIER_LIMIT}_post_v266_source_prefilter_two_swap"
                ),
                f"claim_boundary_v{VERSION}": (
                    "bounded two-swap source-relief probe only; not exhaustive "
                    "multi-swap/global proof"
                ),
            }
            for family in FAMILIES:
                row[f"primary_added_{family}_v{VERSION}"] = str(primary_add[family])
                row[f"primary_dropped_{family}_v{VERSION}"] = str(primary_drop[family])
                row[f"relief_added_{family}_v{VERSION}"] = str(relief_add[family])
                row[f"relief_dropped_{family}_v{VERSION}"] = str(relief_drop[family])
            two_swap_rows.append(row)
        stage_rows.append(
            {
                f"primary_rank_v{VERSION}": primary_rank,
                f"primary_added_loan_id_v{VERSION}": str(primary_add["loan_id"]),
                f"primary_dropped_loan_id_v{VERSION}": str(primary_drop["loan_id"]),
                f"base_two_swap_pair_rows_v{VERSION}": int(base_mask.sum()),
                f"source_prefilter_two_swap_pair_rows_v{VERSION}": int(source_prefilter_mask.sum()),
                f"source_exact_two_swap_pair_rows_v{VERSION}": exact_rows,
                f"cvar_feasible_two_swap_pair_rows_v{VERSION}": cvar_rows,
                f"claim_boundary_v{VERSION}": (
                    "bounded relief search from one post-v266 source-prefilter primary swap"
                ),
            }
        )

    primary_frontier = pd.DataFrame(primary_rows, columns=_primary_columns())
    stage_summary = pd.DataFrame(stage_rows)
    two_swap_candidates = pd.DataFrame(two_swap_rows, columns=_candidate_columns())
    if not two_swap_candidates.empty:
        two_swap_candidates = two_swap_candidates.sort_values(
            [f"cvar_swap_feasible_v{VERSION}", f"total_return_delta_v{VERSION}"],
            ascending=[False, False],
        ).reset_index(drop=True)
    top_candidates = two_swap_candidates.head(200).copy()
    feasible_candidates = (
        two_swap_candidates.loc[two_swap_candidates[f"cvar_swap_feasible_v{VERSION}"].astype(bool)]
        if not two_swap_candidates.empty
        else pd.DataFrame(columns=_candidate_columns())
    )
    best = feasible_candidates.iloc[0] if not feasible_candidates.empty else None
    cvar_feasible = int(len(feasible_candidates))
    base_two_swap_rows = (
        int(stage_summary[f"base_two_swap_pair_rows_v{VERSION}"].sum())
        if not stage_summary.empty
        else 0
    )
    source_prefilter_two_swap_rows = (
        int(stage_summary[f"source_prefilter_two_swap_pair_rows_v{VERSION}"].sum())
        if not stage_summary.empty
        else 0
    )
    source_exact_two_swap_rows = (
        int(stage_summary[f"source_exact_two_swap_pair_rows_v{VERSION}"].sum())
        if not stage_summary.empty
        else 0
    )
    summary = pd.DataFrame(
        [
            {
                f"probe_label_v{VERSION}": "bounded_two_swap_after_v267_local_clearance",
                f"previous_repair_version_v{VERSION}": PREVIOUS_REPAIR_VERSION,
                f"terminal_reprice_version_v{VERSION}": TERMINAL_REPRICE_VERSION,
                f"primary_frontier_limit_v{VERSION}": PRIMARY_FRONTIER_LIMIT,
                f"primary_frontier_rows_v{VERSION}": int(len(primary_frontier)),
                f"one_swap_return_improving_pair_rows_v{VERSION}": return_pairs,
                f"one_swap_budget_return_feasible_pair_rows_v{VERSION}": budget_pairs,
                f"one_swap_source_prefilter_pair_rows_v{VERSION}": source_prefilter_pairs,
                f"base_two_swap_pair_rows_v{VERSION}": base_two_swap_rows,
                f"source_prefilter_two_swap_pair_rows_v{VERSION}": source_prefilter_two_swap_rows,
                f"source_exact_two_swap_pair_rows_v{VERSION}": source_exact_two_swap_rows,
                f"cvar_feasible_two_swap_pair_rows_v{VERSION}": cvar_feasible,
                f"top_candidate_rows_v{VERSION}": int(len(top_candidates)),
                f"best_two_swap_return_delta_v{VERSION}": (
                    float(best[f"total_return_delta_v{VERSION}"]) if best is not None else np.nan
                ),
                f"best_two_swap_cvar90_after_v{VERSION}": (
                    float(best[f"cvar90_after_two_swap_v{VERSION}"]) if best is not None else np.nan
                ),
                f"best_two_swap_objective_after_v{VERSION}": (
                    float(best[f"objective_return_after_two_swap_v{VERSION}"])
                    if best is not None
                    else np.nan
                ),
                f"bounded_two_swap_improvement_found_v{VERSION}": cvar_feasible > 0,
                f"multi_swap_integer_optimality_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"claim_boundary_v{VERSION}": (
                    f"bounded two-swap source-relief probe over top {PRIMARY_FRONTIER_LIMIT} "
                    "post-v266 primary swaps only; no exhaustive multi-swap/global proof"
                ),
            }
        ]
    )
    improvement_next = (
        "paper4_v269_apply_bounded_two_swap_repair.csv"
        if cvar_feasible > 0
        else "paper4_v269_broader_three_swap_or_global_gap_probe.csv"
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "bounded_two_swap_improvement_found",
                f"blocking_v{VERSION}": cvar_feasible > 0,
                f"evidence_count_v{VERSION}": cvar_feasible,
                f"required_next_artifact_v{VERSION}": improvement_next,
                f"claim_boundary_v{VERSION}": (
                    "positive bounded two-swap candidates block multi-swap/local optimality"
                    if cvar_feasible > 0
                    else "no feasible improvement found inside this bounded two-swap frontier"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "multi_swap_search_not_exhaustive",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    "paper4_v269_broader_three_swap_or_global_gap_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "v268 searches a bounded two-swap frontier, not all multi-loan exchanges"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "global_integer_gap_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "paper4_v269_global_gap_certificate_protocol.csv",
                f"claim_boundary_v{VERSION}": "no global full-universe integer gap certificate",
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
                "claim_id": "v268_bounded_two_swap_probe_executed",
                "allowed": True,
                "artifact": "paper4_v268_bounded_two_swap_summary.csv",
                "boundary": "bounded source-relief two-swap probe after v267 local clearance",
            },
            {
                "claim_id": "v268_bounded_two_swap_improvement_found",
                "allowed": cvar_feasible > 0,
                "artifact": "paper4_v268_bounded_two_swap_top_candidates.csv",
                "boundary": "existence claim only within bounded post-v266 two-swap frontier",
            },
            {
                "claim_id": "v268_multi_swap_or_global_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v268_claim_blockers.csv",
                "boundary": "bounded two-swap probe is not exhaustive/global",
            },
            {
                "claim_id": "v268_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v268_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    return {
        "primary_frontier": primary_frontier,
        "stage_summary": stage_summary,
        "two_swap_candidates": two_swap_candidates,
        "top_candidates": top_candidates,
        "summary": summary,
        "blockers": blockers,
        "claim_matrix": claim_matrix,
    }


def _update_claim_boundaries(*, improvement_found: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v268 bounded two-swap source-relief probe after v267.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v268_bounded_two_swap_summary.csv"
                ),
                "boundary": (
                    "Searches only the top bounded post-v266 source-prefilter two-swap frontier."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v268 finds bounded two-swap improvements over the v266 candidate.",
                "allowed": improvement_found,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v268_bounded_two_swap_top_candidates.csv"
                ),
                "boundary": (
                    "Existence claim only within the bounded v268 frontier; not exhaustive."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v268 proves multi-swap or global full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v268_claim_blockers.csv"
                ),
                "boundary": "Requires broader multi-swap search or a global gap certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v268 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v268_claim_blockers.csv"
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
                    "v268 probes a bounded two-swap source-relief frontier after v267 "
                    "cleared one-swap local optimality."
                ),
                "status": (
                    "bounded_two_swap_improvement_found"
                    if improvement_found
                    else "bounded_two_swap_no_improvement_found"
                ),
                "next_artifact": (
                    "paper4_v269_apply_bounded_two_swap_repair.csv"
                    if improvement_found
                    else "paper4_v269_broader_three_swap_or_global_gap_probe.csv"
                ),
                "success_condition": (
                    "bounded two-swap frontier either finds a feasible improvement or "
                    "redirects to broader multi-swap/global evidence"
                ),
                "last_wave": "v268",
                "execution_result": (
                    "bounded_two_swap_probe_found_improvements"
                    if improvement_found
                    else "bounded_two_swap_probe_no_improvement"
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
    start = "<!-- V268_BOUNDED_TWO_SWAP_PROBE_START -->"
    end = "<!-- V268_BOUNDED_TWO_SWAP_PROBE_END -->"
    best_delta = status["best_two_swap_return_delta_v268"]
    best_delta_text = (
        "not applicable; no feasible bounded two-swap improvements"
        if best_delta is None
        else str(best_delta)
    )
    block = f"""
{start}

## Wave v268: Bounded Two-Swap Probe After v267 Local Clearance

Generated: {status["generated_at_utc"]}

### Objective

Probe whether the v266 portfolio, which v267 cleared for one-drop/one-add
swaps, still admits a bounded two-drop/two-add source-relief improvement. The
frontier is explicitly limited to the top `{status["primary_frontier_limit_v268"]}`
post-v266 source-prefilter primary swaps.

### Results

- One-swap source-prefilter primary rows:
  `{status["one_swap_source_prefilter_pair_rows_v268"]}`.
- Primary frontier rows searched:
  `{status["primary_frontier_rows_v268"]}`.
- Base two-swap rows in bounded frontier:
  `{status["base_two_swap_pair_rows_v268"]}`.
- Source-prefilter two-swap rows:
  `{status["source_prefilter_two_swap_pair_rows_v268"]}`.
- Exact source-feasible two-swap rows:
  `{status["source_exact_two_swap_pair_rows_v268"]}`.
- CVaR-feasible improving two-swap rows:
  `{status["cvar_feasible_two_swap_pair_rows_v268"]}`.
- Best bounded two-swap return delta:
  `{best_delta_text}`.

### Interpretation

v268 is bounded multi-swap evidence after the one-swap loop terminates. A
positive result would block any multi-swap optimality claim and require a v269
repair; a negative result would still not prove global optimality because the
frontier is intentionally limited.

### Claim Impact

- Allowed: bounded post-v266 two-swap probe executed.
- Still prohibited: exhaustive multi-swap optimality, full-universe global gap,
  Paper Estrella replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v268 in the living notebook. Promote only after broader integer/dynamic
and promotion gates pass.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    artifacts = _build_artifacts()
    summary = artifacts["summary"]
    row = summary.iloc[0]
    improvement_found = bool(row[f"bounded_two_swap_improvement_found_v{VERSION}"])

    write_csv(
        TABLE_DIR / "paper4_v268_bounded_two_swap_primary_frontier.csv",
        artifacts["primary_frontier"],
    )
    write_csv(
        TABLE_DIR / "paper4_v268_bounded_two_swap_stage_summary.csv", artifacts["stage_summary"]
    )
    write_csv(
        TABLE_DIR / "paper4_v268_bounded_two_swap_candidates.csv", artifacts["two_swap_candidates"]
    )
    write_csv(
        TABLE_DIR / "paper4_v268_bounded_two_swap_top_candidates.csv", artifacts["top_candidates"]
    )
    write_csv(TABLE_DIR / "paper4_v268_bounded_two_swap_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v268_claim_blockers.csv", artifacts["blockers"])
    write_csv(TABLE_DIR / "paper4_v268_claim_matrix_delta.csv", artifacts["claim_matrix"])
    _update_claim_boundaries(improvement_found=improvement_found)
    _update_backlog(improvement_found=improvement_found)

    best_delta = row[f"best_two_swap_return_delta_v{VERSION}"]
    best_cvar = row[f"best_two_swap_cvar90_after_v{VERSION}"]
    best_objective = row[f"best_two_swap_objective_after_v{VERSION}"]
    status = {
        "phase": "v268_bounded_two_swap_after_v267_local_clearance",
        "schema_version": "2026-05-15.268",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "previous_repair_version_v268": PREVIOUS_REPAIR_VERSION,
        "terminal_reprice_version_v268": TERMINAL_REPRICE_VERSION,
        "primary_frontier_limit_v268": PRIMARY_FRONTIER_LIMIT,
        "primary_frontier_rows_v268": int(row["primary_frontier_rows_v268"]),
        "one_swap_return_improving_pair_rows_v268": int(
            row["one_swap_return_improving_pair_rows_v268"]
        ),
        "one_swap_budget_return_feasible_pair_rows_v268": int(
            row["one_swap_budget_return_feasible_pair_rows_v268"]
        ),
        "one_swap_source_prefilter_pair_rows_v268": int(
            row["one_swap_source_prefilter_pair_rows_v268"]
        ),
        "base_two_swap_pair_rows_v268": int(row["base_two_swap_pair_rows_v268"]),
        "source_prefilter_two_swap_pair_rows_v268": int(
            row["source_prefilter_two_swap_pair_rows_v268"]
        ),
        "source_exact_two_swap_pair_rows_v268": int(row["source_exact_two_swap_pair_rows_v268"]),
        "cvar_feasible_two_swap_pair_rows_v268": int(row["cvar_feasible_two_swap_pair_rows_v268"]),
        "top_candidate_rows_v268": int(row["top_candidate_rows_v268"]),
        "best_two_swap_return_delta_v268": (None if pd.isna(best_delta) else float(best_delta)),
        "best_two_swap_cvar90_after_v268": None if pd.isna(best_cvar) else float(best_cvar),
        "best_two_swap_objective_after_v268": (
            None if pd.isna(best_objective) else float(best_objective)
        ),
        "bounded_two_swap_improvement_found_v268": improvement_found,
        "multi_swap_integer_optimality_claim_allowed_v268": False,
        "full_universe_integer_optimality_claim_allowed_v268": False,
        "paper1_promotion_allowed_v268": False,
        "paper4_working_champion_changed_v268": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v268": int(len(artifacts["blockers"])),
        "claim_matrix_rows_v268": int(len(artifacts["claim_matrix"])),
        "claim_boundary": (
            "v268 is bounded two-swap source-relief evidence only; exhaustive "
            "multi-swap/global and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v268_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v268": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
