#!/usr/bin/env python3
"""Build Paper 4 v294 post-v293 one-swap repricing artifacts."""

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

VERSION = 294
PREVIOUS_CHALLENGER_VERSION = 293
BASE_REPAIR_VERSION = 279
NEXT_VERSION = 295
TARGET_SELECTED_ROWS = 171
REPRICE_SCOPE = "post_v293_diverse_pool_one_drop_one_add_whole_loan_swap"


def _candidate_pairs_for_reprice() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v293_diverse_pool_allocations.parquet").reset_index(drop=True)
    source_summary = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_summary = source_summary.loc[
        source_summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    v293_summary = read_csv("paper4_v293_diverse_pool_return_gap_probe.csv")
    if universe.empty or selected.empty or source_summary.empty or v293_summary.empty:
        empty = pd.DataFrame(columns=_reprice_pair_columns(VERSION))
        summary = pd.DataFrame()
        stage = pd.DataFrame()
        return empty, summary, stage
    v293_row = v293_summary.iloc[0]

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

    policy_id = "v293_diverse_micro_relief_cardinality_milp"
    regime = "post_v293_diverse_pool_challenger"
    current_exposure = float(v293_row[f"portfolio_exposure_v{PREVIOUS_CHALLENGER_VERSION}"])
    exposure_min = float(v293_row[f"exposure_min_v{PREVIOUS_CHALLENGER_VERSION}"])
    exposure_max = float(v293_row[f"exposure_max_v{PREVIOUS_CHALLENGER_VERSION}"])
    cvar_cap = float(v293_row[f"scenario_loss_cvar90_v{PREVIOUS_CHALLENGER_VERSION}"])
    current_objective_return = float(v293_row[f"objective_return_v{PREVIOUS_CHALLENGER_VERSION}"])
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
                "post-v293 one-swap pricing only; not multi-swap or global proof"
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
        "post-v293 one-swap screen cleared; broader multi-swap/global proof still missing"
        if improving_pairs == 0
        else "post-v293 one-swap screen found improvements; repair/reprice loop must continue"
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
                f"current_exposure_v{VERSION}": current_exposure,
                f"exposure_min_v{VERSION}": exposure_min,
                f"exposure_max_v{VERSION}": exposure_max,
                f"cvar_cap_v{VERSION}": cvar_cap,
                f"current_objective_return_v{VERSION}": current_objective_return,
                f"post_v293_one_swap_local_optimality_cleared_v{VERSION}": (improving_pairs == 0),
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
        "post-v293 diverse-pool one-swap screen stage count only"
    )
    return pairs, summary, stage_summary


def _claim_blockers(summary: pd.DataFrame) -> pd.DataFrame:
    improving = int(summary[f"one_swap_improving_rows_v{VERSION}"].iloc[0])
    next_artifact = (
        f"paper4_v{NEXT_VERSION}_apply_next_post_v293_swap.csv"
        if improving > 0
        else f"paper4_v{NEXT_VERSION}_broader_multi_swap_or_global_gap_probe.csv"
    )
    return pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "post_v293_one_swap_improvement_found",
                f"blocking_v{VERSION}": improving > 0,
                f"evidence_count_v{VERSION}": improving,
                f"required_next_artifact_v{VERSION}": next_artifact,
                f"claim_boundary_v{VERSION}": (
                    "feasible improving post-v293 one-swaps block local optimality"
                    if improving > 0
                    else "no feasible improving post-v293 one-swaps remain"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "bounded_challenger_not_working_champion",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": next_artifact,
                f"claim_boundary_v{VERSION}": (
                    "v293/v294 remain bounded-pool and one-swap evidence"
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
                f"blocker_id_v{VERSION}": "dynamic_replay_and_deployment_gates_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_dynamic_replay_validation",
                f"claim_boundary_v{VERSION}": "no online or deployment validation created",
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
                "claim_id": "v294_post_v293_one_swap_reprice_executed",
                "allowed": True,
                "artifact": "paper4_v294_post_v293_one_swap_summary.csv",
                "boundary": "post-v293 one-drop/one-add screen completed",
            },
            {
                "claim_id": "v294_post_v293_one_swap_local_optimality",
                "allowed": local_optimality_cleared,
                "artifact": "paper4_v294_claim_blockers.csv",
                "boundary": "allowed only within one-drop/one-add scope",
            },
            {
                "claim_id": "v294_working_champion",
                "allowed": False,
                "artifact": "paper4_v294_claim_blockers.csv",
                "boundary": "global and dynamic evidence missing",
            },
            {
                "claim_id": "v294_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v294_claim_blockers.csv",
                "boundary": "global certificate missing",
            },
            {
                "claim_id": "v294_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v294_claim_blockers.csv",
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
                "claim": "Paper 4 has a v294 post-v293 one-swap repricing screen.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v294_post_v293_one_swap_summary.csv"
                ),
                "boundary": "One-drop/one-add screen after v293; not global integer proof.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v294 clears post-v293 one-swap local optimality.",
                "allowed": local_optimality_cleared,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v294_claim_blockers.csv"
                ),
                "boundary": "Scope-limited to one-drop/one-add swaps after v293.",
                "prohibited_claim_flag": not local_optimality_cleared,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v294 authorizes a new Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v294_claim_blockers.csv"
                ),
                "boundary": "Global evidence and dynamic gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v294 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v294_claim_blockers.csv"
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
        f"paper4_v{NEXT_VERSION}_broader_multi_swap_or_global_gap_probe.csv"
        if local_optimality_cleared
        else f"paper4_v{NEXT_VERSION}_apply_next_post_v293_swap.csv"
    )
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v294 reruns one-swap pricing after the v293 diverse-pool "
                    "bounded challenger over all non-selected comparable loans."
                ),
                "status": (
                    "post_v293_one_swap_local_optimality_cleared"
                    if local_optimality_cleared
                    else "post_v293_one_swap_improvement_found"
                ),
                "next_artifact": next_artifact,
                "success_condition": (
                    "resolve broader multi-swap/global/dynamic blockers before any "
                    "working-champion or promotion claim"
                ),
                "last_wave": "v294",
                "execution_result": (
                    "post_v293_one_swap_reprice_cleared"
                    if local_optimality_cleared
                    else "post_v293_one_swap_reprice_found_improvements"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v294")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V294_POST_V293_ONE_SWAP_REPRICE_START -->"
    end = "<!-- V294_POST_V293_ONE_SWAP_REPRICE_END -->"
    best_delta = status["best_one_swap_return_delta_v294"]
    best_delta_text = (
        "not applicable; no feasible improving one-swaps" if best_delta is None else str(best_delta)
    )
    block = f"""
{start}

## Wave v294: Post-v293 One-Swap Repricing

Generated: {status["generated_at_utc"]}

### Objective

v293 found a bounded diverse-pool challenger. v294 tests whether that challenger
has local one-drop/one-add stability against the full comparable universe under
the same budget, exact source caps and v293 CVaR cap.

### Results

- Selected rows: `{status["selected_rows_v294"]}`.
- Candidate add rows: `{status["candidate_add_rows_v294"]}`.
- Pair rows screened: `{status["total_pair_rows_screened_v294"]}`.
- Return-improving pairs: `{status["return_improving_pair_rows_v294"]}`.
- Budget+return feasible pairs: `{status["budget_return_feasible_pair_rows_v294"]}`.
- Source prefilter pairs: `{status["source_prefilter_pair_rows_v294"]}`.
- Exact source-feasible pairs: `{status["source_exact_pair_rows_v294"]}`.
- CVaR-feasible improving one-swaps: `{status["one_swap_improving_rows_v294"]}`.
- Best post-v293 one-swap return delta: `{best_delta_text}`.
- Post-v293 one-swap local optimality cleared:
  `{status["post_v293_one_swap_local_optimality_cleared_v294"]}`.

### Interpretation

v294 either clears the immediate one-swap repricing gate or identifies the next
repair action. In both cases it remains bounded/local evidence, not a global
optimality or deployment claim.

### Claim Impact

- Allowed: post-v293 one-swap repricing screen completed.
- Still prohibited: working champion replacement, full-universe optimality,
  Paper Estrella replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v294 in the living notebook. Promotion remains blocked.

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
        summary[f"post_v293_one_swap_local_optimality_cleared_v{VERSION}"].iloc[0]
    )
    blockers = _claim_blockers(summary)
    claim_matrix = _claim_matrix(local_optimality_cleared=local_optimality_cleared)

    write_csv(TABLE_DIR / "paper4_v294_post_v293_one_swap_reprice.csv", pairs)
    write_csv(TABLE_DIR / "paper4_v294_post_v293_one_swap_top_candidates.csv", top_candidates)
    write_csv(TABLE_DIR / "paper4_v294_post_v293_one_swap_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v294_post_v293_one_swap_stage_summary.csv", stage_summary)
    write_csv(TABLE_DIR / "paper4_v294_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v294_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries(local_optimality_cleared=local_optimality_cleared)
    _update_backlog(local_optimality_cleared=local_optimality_cleared)

    row = summary.iloc[0]
    best_delta = row[f"best_one_swap_return_delta_v{VERSION}"]
    best_cvar = row[f"best_one_swap_cvar90_after_v{VERSION}"]
    status = {
        "phase": "v294_post_v293_diverse_pool_one_swap_reprice",
        "schema_version": "2026-05-15.294",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "previous_challenger_version_v294": PREVIOUS_CHALLENGER_VERSION,
        "base_repair_version_v294": BASE_REPAIR_VERSION,
        "summary_rows_v294": int(len(summary)),
        "stage_summary_rows_v294": int(len(stage_summary)),
        "candidate_pair_rows_v294": int(len(pairs)),
        "top_candidate_rows_v294": int(len(top_candidates)),
        "claim_blocker_rows_v294": int(len(blockers)),
        "claim_matrix_rows_v294": int(len(claim_matrix)),
        "selected_rows_v294": int(row[f"selected_rows_v{VERSION}"]),
        "base_selected_rows_v294": int(row[f"base_selected_rows_v{VERSION}"]),
        "cardinality_restored_v294": bool(row[f"cardinality_restored_v{VERSION}"]),
        "candidate_add_rows_v294": int(row[f"candidate_add_rows_v{VERSION}"]),
        "total_pair_rows_screened_v294": int(row[f"total_pair_rows_screened_v{VERSION}"]),
        "return_improving_pair_rows_v294": int(row[f"return_improving_pair_rows_v{VERSION}"]),
        "budget_return_feasible_pair_rows_v294": int(
            row[f"budget_return_feasible_pair_rows_v{VERSION}"]
        ),
        "source_prefilter_pair_rows_v294": int(row[f"source_prefilter_pair_rows_v{VERSION}"]),
        "source_exact_pair_rows_v294": int(row[f"source_exact_pair_rows_v{VERSION}"]),
        "cvar_feasible_pair_rows_v294": int(row[f"cvar_feasible_pair_rows_v{VERSION}"]),
        "one_swap_improving_rows_v294": int(row[f"one_swap_improving_rows_v{VERSION}"]),
        "best_one_swap_return_delta_v294": (None if pd.isna(best_delta) else float(best_delta)),
        "best_one_swap_cvar90_after_v294": None if pd.isna(best_cvar) else float(best_cvar),
        "current_exposure_v294": float(row[f"current_exposure_v{VERSION}"]),
        "current_objective_return_v294": float(row[f"current_objective_return_v{VERSION}"]),
        "cvar_cap_v294": float(row[f"cvar_cap_v{VERSION}"]),
        "post_v293_one_swap_local_optimality_cleared_v294": local_optimality_cleared,
        "working_champion_claim_allowed_v294": False,
        "full_universe_integer_optimality_claim_allowed_v294": False,
        "paper1_promotion_allowed_v294": False,
        "paper4_working_champion_changed_v294": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v294": (
            f"paper4_v{NEXT_VERSION}_broader_multi_swap_or_global_gap_probe.csv"
            if local_optimality_cleared
            else f"paper4_v{NEXT_VERSION}_apply_next_post_v293_swap.csv"
        ),
        "claim_boundary": (
            "v294 is a post-v293 one-swap repricing gate; no working champion or "
            "final promotion is authorized"
        ),
    }
    write_json(STATUS_DIR / "paper4_v294_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v294": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
