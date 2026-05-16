#!/usr/bin/env python3
"""Build Paper 4 v267 post-v266 one-swap repricing artifacts."""

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
    _reprice_pair_columns,
    now,
    read_csv,
    read_parquet,
    write_csv,
    write_json,
)

VERSION = 267
PREVIOUS_REPAIR_VERSION = 266
NEXT_VERSION = 268
REPRICE_SCOPE = "post_v266_bounded_two_swap_one_drop_one_add_whole_loan_swap"


def _candidate_pairs_for_reprice() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    selected = read_parquet("paper4_v266_bounded_two_swap_repair_allocations.parquet")
    source_summary = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_summary = source_summary.loc[
        source_summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    repair_summary = read_csv("paper4_v266_bounded_two_swap_repair_summary.csv")
    repair_row = repair_summary.loc[
        repair_summary["portfolio_label_v266"].eq("bounded_two_swap_repair_candidate")
    ].iloc[0]
    action = read_csv("paper4_v266_bounded_two_swap_repair_action.csv")
    if universe.empty or selected.empty or source_summary.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    selected_ids = set(selected["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    candidate_idx = idx_by_id.loc[candidates["loan_id"].astype(str)].to_numpy()
    selected = selected.reset_index(drop=True)
    candidates = candidates.reset_index(drop=True)
    selected[f"mean_return_if_dropped_v{VERSION}"] = mean_returns[selected_idx]
    candidates[f"mean_return_if_added_v{VERSION}"] = mean_returns[candidate_idx]

    policy_id = (
        str(action["policy_id"].iloc[0]) if not action.empty else "v266_bounded_two_swap_candidate"
    )
    regime = "post_v266_bounded_two_swap_repair"
    current_exposure = float(repair_row["portfolio_exposure_v266"])
    exposure_min = float(repair_row["exposure_min_v266"])
    exposure_max = float(repair_row["exposure_max_v266"])
    cvar_cap = float(repair_row["cvar_cap_v266"])
    current_objective_return = float(repair_row["objective_return_v266"])
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
    source_prefilter_mask = v82._source_prefilter_mask(
        budget_return_mask,
        candidates,
        selected,
        source_summary,
        current_exposure,
    )
    source_prefilter_pairs = int(source_prefilter_mask.sum())

    current_by_family, cap_by_family = v82._source_maps(selected, source_summary)
    rows: list[dict[str, Any]] = []
    for candidate_pos, selected_pos in np.argwhere(source_prefilter_mask):
        add_row = candidates.iloc[int(candidate_pos)]
        drop_row = selected.iloc[int(selected_pos)]
        new_total = float(current_exposure + add_row["loan_amnt"] - drop_row["loan_amnt"])
        source_ok, min_slack, max_share, violations, first_family, first_source = (
            v82._exact_source_metrics(
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
                "post-v266 bounded two-swap one-swap pricing only; not multi-swap or global proof"
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
    local_claim_boundary = (
        "post-v266 one-swap screen cleared; broader multi-swap/global proof still missing"
        if improving_pairs == 0
        else ("post-v266 one-swap screen only; repeat repair/repricing because improvements remain")
    )
    summary = pd.DataFrame(
        [
            {
                "policy_id": policy_id,
                f"regime_v{VERSION}": regime,
                f"selected_rows_v{VERSION}": int(len(selected)),
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
                f"current_exposure_v{VERSION}": current_exposure,
                f"exposure_min_v{VERSION}": exposure_min,
                f"exposure_max_v{VERSION}": exposure_max,
                f"cvar_cap_v{VERSION}": cvar_cap,
                f"current_objective_return_v{VERSION}": current_objective_return,
                f"post_bounded_two_swap_one_swap_local_optimality_cleared_v{VERSION}": (
                    improving_pairs == 0
                ),
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
        "post-v266 bounded two-swap one-swap screen stage count only"
    )
    return pairs, summary, stage_summary


def _claim_blockers(summary: pd.DataFrame) -> pd.DataFrame:
    improving = int(summary[f"one_swap_improving_rows_v{VERSION}"].iloc[0])
    improvement_next_artifact = (
        "paper4_v268_apply_next_one_swap_after_bounded_two_swap_repair.csv"
        if improving > 0
        else "paper4_v268_broader_multi_swap_or_global_gap_probe.csv"
    )
    improvement_boundary = (
        "feasible improving post-v266 one-swaps block local optimality"
        if improving > 0
        else "no feasible improving post-v266 one-swaps remain"
    )
    return pd.DataFrame(
        [
            {
                "blocker_id_v267": "post_v266_one_swap_improvement_found",
                "blocking_v267": improving > 0,
                "evidence_count_v267": improving,
                "required_next_artifact_v267": improvement_next_artifact,
                "claim_boundary_v267": improvement_boundary,
            },
            {
                "blocker_id_v267": "broader_multi_swap_search_not_exhaustive",
                "blocking_v267": True,
                "evidence_count_v267": 1,
                "required_next_artifact_v267": (
                    "paper4_v268_broader_multi_swap_or_global_gap_probe.csv"
                ),
                "claim_boundary_v267": (
                    "post-v266 one-swap screen is not an exhaustive multi-swap proof"
                ),
            },
            {
                "blocker_id_v267": "global_integer_gap_certificate_missing",
                "blocking_v267": True,
                "evidence_count_v267": 1,
                "required_next_artifact_v267": "paper4_v268_global_integer_gap_protocol.csv",
                "claim_boundary_v267": "no branch-and-price/global full-universe certificate",
            },
            {
                "blocker_id_v267": "paper4_final_promotion_forbidden",
                "blocking_v267": True,
                "evidence_count_v267": 1,
                "required_next_artifact_v267": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v267": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix(*, local_optimality_cleared: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v267_post_v266_bounded_two_swap_one_swap_reprice_executed",
                "allowed": True,
                "artifact": "paper4_v267_post_bounded_two_swap_one_swap_summary.csv",
                "boundary": "post-v266 one-drop/one-add screen completed",
            },
            {
                "claim_id": "v267_post_v266_one_swap_local_optimality",
                "allowed": local_optimality_cleared,
                "artifact": "paper4_v267_claim_blockers.csv",
                "boundary": "allowed only when no feasible improving post-v266 one-swaps remain",
            },
            {
                "claim_id": "v267_multi_swap_or_global_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v267_claim_blockers.csv",
                "boundary": "broader multi-swap/global gap evidence missing",
            },
            {
                "claim_id": "v267_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v267_claim_blockers.csv",
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
                "claim": ("Paper 4 has a v267 post-v266 bounded two-swap one-swap pricing screen."),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v267_post_bounded_two_swap_one_swap_summary.csv"
                ),
                "boundary": (
                    "One-drop/one-add screen after v266 repair; not global integer proof."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v267 proves the v266 bounded two-swap repaired portfolio is "
                    "one-swap locally optimal."
                ),
                "allowed": local_optimality_cleared,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v267_claim_blockers.csv"
                ),
                "boundary": ("Allowed only if no feasible improving post-v266 one-swaps remain."),
                "prohibited_claim_flag": not local_optimality_cleared,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v267 proves multi-swap or global full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v267_claim_blockers.csv"
                ),
                "boundary": "Requires broader multi-swap search or a global gap certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v267 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v267_claim_blockers.csv"
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


def _update_backlog(*, local_optimality_cleared: bool) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    next_artifact = (
        "paper4_v268_broader_multi_swap_or_global_gap_probe.csv"
        if local_optimality_cleared
        else "paper4_v268_apply_next_one_swap_after_bounded_two_swap_repair.csv"
    )
    status = (
        "one_swap_local_optimality_cleared_after_bounded_two_swap_repair"
        if local_optimality_cleared
        else "iterate_if_improving_swaps_remain"
    )
    execution_result = (
        "post_v266_one_swap_reprice_cleared"
        if local_optimality_cleared
        else "post_v266_one_swap_reprice_found_improvements"
    )
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v267 reruns one-swap pricing after the v266 bounded two-swap "
                    "repair over all non-selected comparable loans."
                ),
                "status": status,
                "next_artifact": next_artifact,
                "success_condition": (
                    "no feasible improving post-v266 one-swaps remain before moving to "
                    "broader multi-swap/global evidence"
                ),
                "last_wave": "v267",
                "execution_result": execution_result,
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
    start = "<!-- V267_POST_V266_BOUNDED_TWO_SWAP_ONE_SWAP_REPRICE_START -->"
    end = "<!-- V267_POST_V266_BOUNDED_TWO_SWAP_ONE_SWAP_REPRICE_END -->"
    best_delta = status["best_one_swap_return_delta_v267"]
    best_delta_text = (
        "not applicable; no feasible improving one-swaps" if best_delta is None else str(best_delta)
    )
    block = f"""
{start}

## Wave v267: Post-v266 Bounded Two-Swap One-Swap Repricing

Generated: {status["generated_at_utc"]}

### Objective

Rerun one-drop/one-add integer pricing after the v266 bounded two-swap repair,
using all non-selected loans from the comparable v55 universe as possible
additions. This tests whether the v266 repaired candidate is one-swap locally
optimal.

### Results

- Pair rows screened: `{status["total_pair_rows_screened_v267"]}`.
- Candidate add rows: `{status["candidate_add_rows_v267"]}`.
- Return-improving pairs: `{status["return_improving_pair_rows_v267"]}`.
- Exact source-feasible pairs: `{status["source_exact_pair_rows_v267"]}`.
- CVaR-feasible improving one-swaps: `{status["one_swap_improving_rows_v267"]}`.
- Best post-v266 one-swap return delta:
  `{best_delta_text}`.
- Post-v266 one-swap local optimality cleared:
  `{status["post_bounded_two_swap_one_swap_local_optimality_cleared_v267"]}`.

### Interpretation

v267 is the required re-pricing after v266 changed the portfolio with a bounded
two-swap. If feasible improving one-swaps remain, the lab should continue a
repair/reprice loop; if it clears, the remaining blocker is broader
multi-swap/global integer evidence.

### Claim Impact

- Allowed: post-v266 bounded two-swap one-swap pricing screen completed.
- Still prohibited: broader multi-swap/global integer optimality, Paper
  Estrella replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v267 in the living notebook. Promote only after broader integer/dynamic
and promotion gates pass.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    pairs, summary, stage_summary = _candidate_pairs_for_reprice()
    top_candidates = pairs.sort_values(
        [f"one_swap_improves_return_v{VERSION}", f"return_delta_v{VERSION}"],
        ascending=[False, False],
    ).head(200)
    local_optimality_cleared = bool(
        summary[f"post_bounded_two_swap_one_swap_local_optimality_cleared_v{VERSION}"].iloc[0]
    )
    blockers = _claim_blockers(summary)
    claim_matrix = _claim_matrix(local_optimality_cleared=local_optimality_cleared)

    write_csv(
        TABLE_DIR / "paper4_v267_post_bounded_two_swap_one_swap_reprice.csv",
        pairs,
    )
    write_csv(
        TABLE_DIR / "paper4_v267_post_bounded_two_swap_one_swap_top_candidates.csv",
        top_candidates,
    )
    write_csv(
        TABLE_DIR / "paper4_v267_post_bounded_two_swap_one_swap_summary.csv",
        summary,
    )
    write_csv(
        TABLE_DIR / "paper4_v267_post_bounded_two_swap_one_swap_stage_summary.csv",
        stage_summary,
    )
    write_csv(TABLE_DIR / "paper4_v267_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v267_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries(local_optimality_cleared=local_optimality_cleared)
    _update_backlog(local_optimality_cleared=local_optimality_cleared)

    row = summary.iloc[0]
    best_return_delta = row[f"best_one_swap_return_delta_v{VERSION}"]
    best_cvar90_after = row[f"best_one_swap_cvar90_after_v{VERSION}"]
    status = {
        "phase": "v267_post_v266_bounded_two_swap_one_swap_reprice",
        "schema_version": "2026-05-15.267",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "summary_rows_v267": int(len(summary)),
        "stage_summary_rows_v267": int(len(stage_summary)),
        "candidate_pair_rows_v267": int(len(pairs)),
        "top_candidate_rows_v267": int(len(top_candidates)),
        "claim_blocker_rows_v267": int(len(blockers)),
        "claim_matrix_rows_v267": int(len(claim_matrix)),
        "selected_rows_v267": int(row["selected_rows_v267"]),
        "candidate_add_rows_v267": int(row["candidate_add_rows_v267"]),
        "total_pair_rows_screened_v267": int(row["total_pair_rows_screened_v267"]),
        "return_improving_pair_rows_v267": int(row["return_improving_pair_rows_v267"]),
        "budget_return_feasible_pair_rows_v267": int(row["budget_return_feasible_pair_rows_v267"]),
        "source_prefilter_pair_rows_v267": int(row["source_prefilter_pair_rows_v267"]),
        "source_exact_pair_rows_v267": int(row["source_exact_pair_rows_v267"]),
        "cvar_feasible_pair_rows_v267": int(row["cvar_feasible_pair_rows_v267"]),
        "one_swap_improving_rows_v267": int(row["one_swap_improving_rows_v267"]),
        "best_one_swap_return_delta_v267": (
            None if pd.isna(best_return_delta) else float(best_return_delta)
        ),
        "best_one_swap_cvar90_after_v267": (
            None if pd.isna(best_cvar90_after) else float(best_cvar90_after)
        ),
        "current_exposure_v267": float(row["current_exposure_v267"]),
        "current_objective_return_v267": float(row["current_objective_return_v267"]),
        "post_bounded_two_swap_one_swap_local_optimality_cleared_v267": (local_optimality_cleared),
        "full_universe_integer_optimality_claim_allowed_v267": False,
        "paper1_promotion_allowed_v267": False,
        "paper4_working_champion_changed_v267": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v267 is post-v266 bounded two-swap one-swap pricing only; "
            "multi-swap/global and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v267_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v267": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
