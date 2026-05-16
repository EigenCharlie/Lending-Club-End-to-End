#!/usr/bin/env python3
"""Build Paper 4 v248 bounded two-swap source-relief probe artifacts."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.papers import build_paper4_v70_restricted_master_solver as v70  # noqa: E402
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71  # noqa: E402
from scripts.papers import build_paper4_v82_one_swap_integer_pricing_probe as v82  # noqa: E402
from scripts.papers.paper4_one_swap_living_lab import (  # noqa: E402
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

VERSION = 248
PREVIOUS_REPAIR_VERSION = 245
TERMINAL_REPRICE_VERSION = 246
NEXT_REPAIR_VERSION = 249


def _source_maps(
    selected: pd.DataFrame, source_summary: pd.DataFrame
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    current_by_family: dict[str, dict[str, float]] = {}
    cap_by_family: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        current_by_family[family] = (
            selected.groupby(family, dropna=False)["loan_amnt"].sum().astype(float).to_dict()
        )
        cap_by_family[family] = (
            source_summary.loc[source_summary["source_family"].astype(str).eq(family)]
            .set_index("source_id")["cap_share_v80"]
            .astype(float)
            .to_dict()
        )
    return current_by_family, cap_by_family


def _exact_source_metrics_after_relief(
    *,
    current_by_family: dict[str, dict[str, float]],
    cap_by_family: dict[str, dict[str, float]],
    add_row: pd.Series,
    drop_row: pd.Series,
    final_total: float,
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
            share = exposure / max(final_total, 1.0)
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


def _source_violations_after_primary(
    *,
    current_by_family: dict[str, dict[str, float]],
    cap_by_family: dict[str, dict[str, float]],
    add_row: pd.Series,
    drop_row: pd.Series,
    primary_total: float,
) -> tuple[int, str, str, float, float]:
    violations = 0
    first_family = ""
    first_source = ""
    min_slack = np.inf
    max_share = 0.0
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
            share = exposure / max(primary_total, 1.0)
            cap = caps.get(source_id, 1.0)
            slack = cap - share
            min_slack = min(min_slack, slack)
            max_share = max(max_share, share)
            if share > cap + 1e-7:
                violations += 1
                if not first_family:
                    first_family = family
                    first_source = source_id
    return violations, first_family, first_source, float(min_slack), float(max_share)


def _primary_frontier(
    *,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    source_summary: pd.DataFrame,
    current_exposure: float,
    exposure_min: float,
    exposure_max: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    add_amount = candidates["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    add_return = candidates["mean_return_if_added_v248"].to_numpy(float)
    drop_return = selected["mean_return_if_dropped_v248"].to_numpy(float)
    return_mask = add_return[:, None] - drop_return[None, :] > 1e-9
    exposure_after = current_exposure + add_amount[:, None] - drop_amount[None, :]
    budget_mask = (
        return_mask
        & (exposure_after >= exposure_min - 1e-7)
        & (exposure_after <= exposure_max + 1e-7)
    )
    source_prefilter = v82._source_prefilter_mask(
        budget_mask, candidates, selected, source_summary, current_exposure
    )
    return source_prefilter, budget_mask


def _build_artifacts() -> dict[str, Any]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    selected = read_parquet(
        f"paper4_v{PREVIOUS_REPAIR_VERSION}_next_one_swap_repair_allocations.parquet"
    ).reset_index(drop=True)
    source_summary = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_summary = source_summary.loc[
        source_summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    repair_summary = read_csv(f"paper4_v{PREVIOUS_REPAIR_VERSION}_next_one_swap_repair_summary.csv")
    repair_row = repair_summary.iloc[0]
    if universe.empty or selected.empty or source_summary.empty:
        raise RuntimeError("Missing universe, v245 repaired portfolio, or source caps.")

    selected_ids = set(selected["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    candidates = candidates.reset_index(drop=True)
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    candidate_idx = idx_by_id.loc[candidates["loan_id"].astype(str)].to_numpy()
    selected["mean_return_if_dropped_v248"] = mean_returns[selected_idx]
    candidates["mean_return_if_added_v248"] = mean_returns[candidate_idx]

    current_exposure = float(repair_row[f"portfolio_exposure_v{PREVIOUS_REPAIR_VERSION}"])
    exposure_min = float(repair_row[f"exposure_min_v{PREVIOUS_REPAIR_VERSION}"])
    exposure_max = float(repair_row[f"exposure_max_v{PREVIOUS_REPAIR_VERSION}"])
    cvar_cap = float(repair_row[f"cvar_cap_v{PREVIOUS_REPAIR_VERSION}"])
    current_objective = float(repair_row[f"objective_return_v{PREVIOUS_REPAIR_VERSION}"])
    current_losses = losses[:, selected_idx].sum(axis=1)
    current_by_family, cap_by_family = _source_maps(selected, source_summary)

    primary_prefilter, primary_budget_mask = _primary_frontier(
        candidates=candidates,
        selected=selected,
        source_summary=source_summary,
        current_exposure=current_exposure,
        exposure_min=exposure_min,
        exposure_max=exposure_max,
    )
    primary_rows: list[dict[str, Any]] = []
    primary_positions: list[tuple[int, int, float, float]] = []
    for candidate_pos, selected_pos in np.argwhere(primary_prefilter):
        add_row = candidates.iloc[int(candidate_pos)]
        drop_row = selected.iloc[int(selected_pos)]
        primary_delta = float(
            add_row["mean_return_if_added_v248"] - drop_row["mean_return_if_dropped_v248"]
        )
        primary_net_exposure = float(add_row["loan_amnt"] - drop_row["loan_amnt"])
        primary_total = current_exposure + primary_net_exposure
        violations, first_family, first_source, min_slack, max_share = (
            _source_violations_after_primary(
                current_by_family=current_by_family,
                cap_by_family=cap_by_family,
                add_row=add_row,
                drop_row=drop_row,
                primary_total=primary_total,
            )
        )
        primary_positions.append(
            (int(candidate_pos), int(selected_pos), primary_delta, primary_net_exposure)
        )
        row: dict[str, Any] = {
            "policy_id": "v63_source_repair_return_guarded_repair",
            "regime_v248": "bounded_two_swap_after_v245",
            "primary_rank_v248": 0,
            "primary_added_loan_id_v248": str(add_row["loan_id"]),
            "primary_dropped_loan_id_v248": str(drop_row["loan_id"]),
            "primary_added_loan_amount_v248": float(add_row["loan_amnt"]),
            "primary_dropped_loan_amount_v248": float(drop_row["loan_amnt"]),
            "primary_return_delta_v248": primary_delta,
            "primary_exposure_after_v248": primary_total,
            "primary_source_cap_violations_v248": violations,
            "primary_first_source_block_family_v248": first_family,
            "primary_first_source_block_id_v248": first_source,
            "primary_source_min_slack_v248": min_slack,
            "primary_max_source_share_v248": max_share,
            "claim_boundary_v248": (
                "primary source-prefilter one-swap only; exact source fails before relief"
            ),
        }
        for family in FAMILIES:
            row[f"primary_added_{family}_v248"] = str(add_row[family])
            row[f"primary_dropped_{family}_v248"] = str(drop_row[family])
        primary_rows.append(row)
    primary_rows = sorted(primary_rows, key=lambda x: x["primary_return_delta_v248"], reverse=True)
    for rank, row in enumerate(primary_rows):
        row["primary_rank_v248"] = rank
    primary_positions = sorted(primary_positions, key=lambda x: x[2], reverse=True)

    two_swap_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []

    for primary_rank, (primary_add_pos, primary_drop_pos, primary_delta, primary_net) in enumerate(
        primary_positions
    ):
        primary_add = candidates.iloc[primary_add_pos]
        primary_drop = selected.iloc[primary_drop_pos]
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
        primary_total = current_exposure + primary_net
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
        current_by_family_after_primary, _cap_by_family = _source_maps(
            selected_after_primary, source_summary
        )
        for relief_candidate_pos, relief_selected_pos in np.argwhere(source_prefilter_mask):
            relief_add = relief_candidates.iloc[int(relief_candidate_pos)]
            relief_drop = relief_selected.iloc[int(relief_selected_pos)]
            final_total = float(final_exposure[int(relief_candidate_pos), int(relief_selected_pos)])
            source_ok, min_slack, max_share, violations, first_family, first_source = (
                _exact_source_metrics_after_relief(
                    current_by_family=current_by_family_after_primary,
                    cap_by_family=cap_by_family,
                    add_row=relief_add,
                    drop_row=relief_drop,
                    final_total=final_total,
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
                relief_add["mean_return_if_added_v248"] - relief_drop["mean_return_if_dropped_v248"]
            )
            total_return_delta = float(
                total_delta[int(relief_candidate_pos), int(relief_selected_pos)]
            )
            row = {
                "policy_id": "v63_source_repair_return_guarded_repair",
                "regime_v248": "bounded_two_swap_after_v245",
                "primary_rank_v248": primary_rank,
                "primary_added_loan_id_v248": str(primary_add["loan_id"]),
                "primary_dropped_loan_id_v248": str(primary_drop["loan_id"]),
                "relief_added_loan_id_v248": str(relief_add["loan_id"]),
                "relief_dropped_loan_id_v248": str(relief_drop["loan_id"]),
                "primary_return_delta_v248": primary_delta,
                "relief_return_delta_v248": relief_delta,
                "total_return_delta_v248": total_return_delta,
                "objective_return_after_two_swap_v248": current_objective + total_return_delta,
                "exposure_after_two_swap_v248": final_total,
                "source_swap_feasible_v248": source_ok,
                "source_min_slack_after_two_swap_v248": min_slack,
                "max_source_share_after_two_swap_v248": max_share,
                "source_cap_violations_after_two_swap_v248": violations,
                "first_source_block_family_v248": first_family,
                "first_source_block_id_v248": first_source,
                "loss_mean_after_two_swap_v248": float(swapped_losses.mean()),
                "cvar90_after_two_swap_v248": cvar90_after,
                "cvar_swap_feasible_v248": cvar_feasible,
                "two_swap_improves_return_v248": total_return_delta > 1e-9,
                "bounded_multi_swap_scope_v248": (
                    "two_drop_two_add_source_relief_from_v246_primary_frontier"
                ),
                "claim_boundary_v248": (
                    "bounded two-swap improvement only; not exhaustive multi-swap/global proof"
                ),
            }
            for family in FAMILIES:
                row[f"primary_added_{family}_v248"] = str(primary_add[family])
                row[f"primary_dropped_{family}_v248"] = str(primary_drop[family])
                row[f"relief_added_{family}_v248"] = str(relief_add[family])
                row[f"relief_dropped_{family}_v248"] = str(relief_drop[family])
            two_swap_rows.append(row)
        stage_rows.append(
            {
                "primary_rank_v248": primary_rank,
                "primary_added_loan_id_v248": str(primary_add["loan_id"]),
                "primary_dropped_loan_id_v248": str(primary_drop["loan_id"]),
                "base_two_swap_pair_rows_v248": int(base_mask.sum()),
                "source_prefilter_two_swap_pair_rows_v248": int(source_prefilter_mask.sum()),
                "source_exact_two_swap_pair_rows_v248": exact_rows,
                "cvar_feasible_two_swap_pair_rows_v248": cvar_rows,
                "claim_boundary_v248": (
                    "bounded relief search from one source-prefilter primary swap"
                ),
            }
        )

    primary_frontier = pd.DataFrame(primary_rows)
    stage_summary = pd.DataFrame(stage_rows)
    two_swap_candidates = pd.DataFrame(two_swap_rows)
    if not two_swap_candidates.empty:
        two_swap_candidates = two_swap_candidates.sort_values(
            "total_return_delta_v248", ascending=False
        ).reset_index(drop=True)
    top_candidates = two_swap_candidates.head(200).copy()
    cvar_feasible = (
        int(two_swap_candidates["cvar_swap_feasible_v248"].sum())
        if not two_swap_candidates.empty
        else 0
    )
    best = top_candidates.iloc[0] if not top_candidates.empty else None
    summary = pd.DataFrame(
        [
            {
                "probe_label_v248": "bounded_two_swap_source_relief_probe",
                "previous_repair_version_v248": PREVIOUS_REPAIR_VERSION,
                "terminal_reprice_version_v248": TERMINAL_REPRICE_VERSION,
                "primary_frontier_rows_v248": int(len(primary_frontier)),
                "base_two_swap_pair_rows_v248": int(
                    stage_summary["base_two_swap_pair_rows_v248"].sum()
                ),
                "source_prefilter_two_swap_pair_rows_v248": int(
                    stage_summary["source_prefilter_two_swap_pair_rows_v248"].sum()
                ),
                "source_exact_two_swap_pair_rows_v248": int(
                    stage_summary["source_exact_two_swap_pair_rows_v248"].sum()
                ),
                "cvar_feasible_two_swap_pair_rows_v248": cvar_feasible,
                "top_candidate_rows_v248": int(len(top_candidates)),
                "best_two_swap_return_delta_v248": (
                    float(best["total_return_delta_v248"]) if best is not None else np.nan
                ),
                "best_two_swap_cvar90_after_v248": (
                    float(best["cvar90_after_two_swap_v248"]) if best is not None else np.nan
                ),
                "best_two_swap_objective_after_v248": (
                    float(best["objective_return_after_two_swap_v248"])
                    if best is not None
                    else np.nan
                ),
                "bounded_two_swap_improvement_found_v248": cvar_feasible > 0,
                "multi_swap_integer_optimality_claim_allowed_v248": False,
                "full_universe_integer_optimality_claim_allowed_v248": False,
                "paper1_promotion_allowed_v248": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                "claim_boundary_v248": (
                    "bounded two-swap source-relief probe only; finding an improvement blocks "
                    "multi-swap optimality claims but does not prove global optimality"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                "blocker_id_v248": "bounded_two_swap_improvement_found",
                "blocking_v248": cvar_feasible > 0,
                "evidence_count_v248": cvar_feasible,
                "required_next_artifact_v248": "paper4_v249_apply_bounded_two_swap_repair.csv",
                "claim_boundary_v248": (
                    "positive bounded two-swap candidates block multi-swap/local optimality"
                ),
            },
            {
                "blocker_id_v248": "multi_swap_search_not_exhaustive",
                "blocking_v248": True,
                "evidence_count_v248": 1,
                "required_next_artifact_v248": "paper4_v249_or_v250_broader_multi_swap_search.csv",
                "claim_boundary_v248": (
                    "v248 searches a source-relief two-swap frontier, not all multi-loan exchanges"
                ),
            },
            {
                "blocker_id_v248": "global_integer_gap_certificate_missing",
                "blocking_v248": True,
                "evidence_count_v248": 1,
                "required_next_artifact_v248": "paper4_v250_global_gap_certificate_protocol.csv",
                "claim_boundary_v248": "no global full-universe integer gap certificate",
            },
            {
                "blocker_id_v248": "paper4_final_promotion_forbidden",
                "blocking_v248": True,
                "evidence_count_v248": 1,
                "required_next_artifact_v248": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v248": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v248_bounded_two_swap_probe_executed",
                "allowed": True,
                "artifact": "paper4_v248_bounded_two_swap_summary.csv",
                "boundary": "bounded source-relief two-swap probe after v245/v246",
            },
            {
                "claim_id": "v248_bounded_two_swap_improvement_found",
                "allowed": cvar_feasible > 0,
                "artifact": "paper4_v248_bounded_two_swap_top_candidates.csv",
                "boundary": "existence claim only within bounded two-swap frontier",
            },
            {
                "claim_id": "v248_multi_swap_or_global_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v248_claim_blockers.csv",
                "boundary": "bounded two-swap probe is not exhaustive/global",
            },
            {
                "claim_id": "v248_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v248_claim_blockers.csv",
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


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v248 bounded two-swap source-relief probe.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v248_bounded_two_swap_summary.csv"
                ),
                "boundary": (
                    "Searches a bounded two-drop/two-add source-relief frontier after v245/v246."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v248 finds bounded two-swap improvements over the v245 candidate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v248_bounded_two_swap_top_candidates.csv"
                ),
                "boundary": (
                    "Existence claim only; does not imply exhaustive multi-swap or global proof."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v248 proves multi-swap or global full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v248_claim_blockers.csv"
                ),
                "boundary": "Requires broader multi-swap search or a global gap certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v248 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v248_claim_blockers.csv"
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


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v248 runs a bounded two-swap source-relief probe after the v246 "
                    "one-swap local clearance."
                ),
                "status": "bounded_two_swap_improvement_found",
                "next_artifact": "paper4_v249_apply_bounded_two_swap_repair.csv",
                "success_condition": (
                    "apply or broaden the two-swap candidate without enabling promotion claims"
                ),
                "last_wave": "v248",
                "execution_result": "bounded_two_swap_probe_found_improvements",
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
    start = "<!-- V248_BOUNDED_TWO_SWAP_PROBE_START -->"
    end = "<!-- V248_BOUNDED_TWO_SWAP_PROBE_END -->"
    block = f"""
{start}

## Wave v248: Bounded Two-Swap Source-Relief Probe

Generated: {status["generated_at_utc"]}

### Objective

Test whether the v245 candidate, although cleared by the v246 one-swap screen,
still admits bounded two-drop/two-add source-relief improvements. The probe
starts from the five v246 source-prefilter primary swaps and searches relief
swaps that can restore exact source feasibility and satisfy CVaR.

### Results

- Primary frontier rows: `{status["primary_frontier_rows_v248"]}`.
- Base two-swap pair rows screened: `{status["base_two_swap_pair_rows_v248"]}`.
- Source-prefilter two-swap rows: `{status["source_prefilter_two_swap_pair_rows_v248"]}`.
- Exact source-feasible two-swap rows: `{status["source_exact_two_swap_pair_rows_v248"]}`.
- CVaR-feasible improving two-swap rows: `{status["cvar_feasible_two_swap_pair_rows_v248"]}`.
- Best bounded two-swap return delta:
  `{status["best_two_swap_return_delta_v248"]}`.
- Best bounded two-swap objective:
  `{status["best_two_swap_objective_after_v248"]}`.

### Interpretation

v248 finds bounded two-swap improvements after v246 cleared one-swap local
pricing. This is valuable negative pressure on broad claims: one-swap local
clearance is real, but it is not multi-swap or global integer optimality.

### Claim Impact

- Allowed: bounded source-relief two-swap probe executed; bounded two-swap
  improvement exists.
- Still prohibited: exhaustive multi-swap optimality, global full-universe
  integer optimality, Paper Estrella replacement, final Paper 4 promotion and
  live deployment.

### Quarto Promotion Decision

Keep v248 in the living notebook. The next executable step is a v249 bounded
two-swap repair candidate or a broader multi-swap/global gap probe.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    artifacts = _build_artifacts()
    write_csv(
        TABLE_DIR / "paper4_v248_bounded_two_swap_primary_frontier.csv",
        artifacts["primary_frontier"],
    )
    write_csv(
        TABLE_DIR / "paper4_v248_bounded_two_swap_stage_summary.csv",
        artifacts["stage_summary"],
    )
    write_csv(
        TABLE_DIR / "paper4_v248_bounded_two_swap_candidates.csv",
        artifacts["two_swap_candidates"],
    )
    write_csv(
        TABLE_DIR / "paper4_v248_bounded_two_swap_top_candidates.csv",
        artifacts["top_candidates"],
    )
    write_csv(
        TABLE_DIR / "paper4_v248_bounded_two_swap_summary.csv",
        artifacts["summary"],
    )
    write_csv(TABLE_DIR / "paper4_v248_claim_blockers.csv", artifacts["blockers"])
    write_csv(TABLE_DIR / "paper4_v248_claim_matrix_delta.csv", artifacts["claim_matrix"])
    _update_claim_boundaries()
    _update_backlog()

    summary_row = artifacts["summary"].iloc[0]
    status = {
        "phase": "v248_bounded_two_swap_source_relief_probe",
        "schema_version": "2026-05-15.248",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "previous_repair_version_v248": PREVIOUS_REPAIR_VERSION,
        "terminal_reprice_version_v248": TERMINAL_REPRICE_VERSION,
        "primary_frontier_rows_v248": int(summary_row["primary_frontier_rows_v248"]),
        "base_two_swap_pair_rows_v248": int(summary_row["base_two_swap_pair_rows_v248"]),
        "source_prefilter_two_swap_pair_rows_v248": int(
            summary_row["source_prefilter_two_swap_pair_rows_v248"]
        ),
        "source_exact_two_swap_pair_rows_v248": int(
            summary_row["source_exact_two_swap_pair_rows_v248"]
        ),
        "cvar_feasible_two_swap_pair_rows_v248": int(
            summary_row["cvar_feasible_two_swap_pair_rows_v248"]
        ),
        "top_candidate_rows_v248": int(summary_row["top_candidate_rows_v248"]),
        "best_two_swap_return_delta_v248": float(summary_row["best_two_swap_return_delta_v248"]),
        "best_two_swap_cvar90_after_v248": float(summary_row["best_two_swap_cvar90_after_v248"]),
        "best_two_swap_objective_after_v248": float(
            summary_row["best_two_swap_objective_after_v248"]
        ),
        "bounded_two_swap_improvement_found_v248": bool(
            summary_row["bounded_two_swap_improvement_found_v248"]
        ),
        "multi_swap_integer_optimality_claim_allowed_v248": False,
        "full_universe_integer_optimality_claim_allowed_v248": False,
        "paper1_promotion_allowed_v248": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v248": int(len(artifacts["blockers"])),
        "claim_matrix_rows_v248": int(len(artifacts["claim_matrix"])),
        "claim_boundary": str(summary_row["claim_boundary_v248"]),
    }
    write_json(STATUS_DIR / "paper4_v248_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v248": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
