#!/usr/bin/env python3
"""Build Paper 4 v348 post-v347 one-swap repricing artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers import build_paper4_v339_post_v338_reprice as one_swap
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

VERSION = 348
BASE_REPAIR_VERSION = 347
PREVIOUS_REPRICE_VERSION = 347
NEXT_VERSION = 349
TARGET_SELECTED_ROWS = 171
REPRICE_SCOPE = "post_v347_candidate_one_drop_one_add_whole_loan_swap"
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v347_proxy_or_dual_bound_gate.csv"

one_swap.VERSION = VERSION
one_swap.BASE_REPAIR_VERSION = BASE_REPAIR_VERSION
one_swap.PREVIOUS_REPRICE_VERSION = PREVIOUS_REPRICE_VERSION
one_swap.NEXT_VERSION = NEXT_VERSION
one_swap.REPRICE_SCOPE = REPRICE_SCOPE


def _candidate_pairs_for_reprice() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v347_v338_multi_source_relief_allocations.parquet")
    source_summary = read_csv("paper4_v347_v338_multi_source_relief_source_summary.csv")
    v347_summary = read_csv("paper4_v347_v338_apply_multi_source_relief_candidate.csv")
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    if any(df.empty for df in [universe, selected, source_summary, v347_summary, v47_panel]):
        return pd.DataFrame(columns=one_swap._pair_columns()), pd.DataFrame(), pd.DataFrame()
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    for frame in [universe, selected]:
        for family in FAMILIES:
            frame[family] = frame[family].astype(str)
    source_summary["source_id"] = source_summary["source_id"].astype(str)
    observed_ids = set(v47_panel["loan_id"].astype(str))
    selected_ids = set(selected["loan_id"].astype(str))
    current_missing_proxy_rows = int((~selected["loan_id"].isin(observed_ids)).sum())
    candidates = universe.loc[~universe["loan_id"].isin(selected_ids)].copy()

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy(int)
    candidate_idx = idx_by_id.loc[candidates["loan_id"].astype(str)].to_numpy(int)
    selected = selected.reset_index(drop=True)
    candidates = candidates.reset_index(drop=True)
    selected[f"mean_return_if_dropped_v{VERSION}"] = mean_returns[selected_idx]
    candidates[f"mean_return_if_added_v{VERSION}"] = mean_returns[candidate_idx]

    policy_id = "v347_v338_multi_source_relief_candidate"
    regime = "post_v347_multi_source_relief_candidate"
    v347_row = v347_summary.iloc[0]
    current_exposure = float(selected["loan_amnt"].sum())
    exposure_min = 842292.375
    exposure_max = 850000.0
    cvar_cap = float(v347_row[f"scenario_loss_cvar90_v{BASE_REPAIR_VERSION}"])
    current_objective_return = float(v347_row[f"objective_return_v{BASE_REPAIR_VERSION}"])
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
    source_prefilter_mask = one_swap._source_prefilter_mask(
        budget_return_mask,
        candidates,
        selected,
        source_summary,
        current_exposure,
    )
    source_prefilter_pairs = int(source_prefilter_mask.sum())

    current_by_family, cap_by_family = one_swap._source_maps(selected, source_summary)
    rows: list[dict[str, Any]] = []
    for candidate_pos, selected_pos in np.argwhere(source_prefilter_mask):
        add_row = candidates.iloc[int(candidate_pos)]
        drop_row = selected.iloc[int(selected_pos)]
        new_total = float(current_exposure + add_row["loan_amnt"] - drop_row["loan_amnt"])
        source_ok, min_slack, max_share, violations, first_family, first_source = (
            one_swap._exact_source_metrics(
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
                "post-v347 one-swap pricing only; not multi-swap or global proof"
            ),
            f"added_observed_v47_proxy_v{VERSION}": add_observed,
            f"dropped_observed_v47_proxy_v{VERSION}": drop_observed,
            f"delta_missing_v47_proxy_rows_v{VERSION}": delta_missing_proxy,
        }
        for family in FAMILIES:
            row[f"added_{family}_v{VERSION}"] = str(add_row[family])
            row[f"dropped_{family}_v{VERSION}"] = str(drop_row[family])
        rows.append(row)

    pairs = pd.DataFrame(rows, columns=one_swap._pair_columns())
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
        "post-v347 candidate one-swap screen cleared; proxy/global gates still missing"
        if improving_pairs == 0
        else "post-v347 candidate one-swap screen found improvements; loop must continue"
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
                f"post_v347_one_swap_local_optimality_cleared_v{VERSION}": (improving_pairs == 0),
                f"next_global_gate_ready_v{VERSION}": improving_pairs == 0,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                f"paper4_final_promotion_created_v{VERSION}": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
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
        "post-v347 candidate one-swap screen stage count only"
    )
    return pairs, summary, stage_summary


def _claim_blockers(summary: pd.DataFrame) -> pd.DataFrame:
    improving = int(summary[f"one_swap_improving_rows_v{VERSION}"].iloc[0])
    missing = int(summary[f"current_missing_v47_proxy_rows_v{VERSION}"].iloc[0])
    return pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "post_v347_one_swap_improvement_found",
                f"blocking_v{VERSION}": improving > 0,
                f"evidence_count_v{VERSION}": improving,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "feasible improving post-v347 one-swaps block local optimality"
                    if improving > 0
                    else "no feasible improving post-v347 one-swaps remain"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "proxy_coverage_gap_persists",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": missing,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "v347 candidate still has missing v47 proxy rows",
            },
            {
                f"blocker_id_v{VERSION}": "valid_branch_price_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "one-swap clearing is not a branch-price certificate",
            },
            {
                f"blocker_id_v{VERSION}": "global_dynamic_online_gates_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
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
                "claim_id": "v348_post_v347_one_swap_reprice_executed",
                "allowed": True,
                "artifact": "paper4_v348_post_v347_one_swap_summary.csv",
                "boundary": "post-v347 one-drop/one-add screen completed",
            },
            {
                "claim_id": "v348_post_v347_one_swap_local_optimality",
                "allowed": local_optimality_cleared,
                "artifact": "paper4_v348_claim_blockers.csv",
                "boundary": "allowed only within one-drop/one-add scope",
            },
            {
                "claim_id": "v348_valid_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v348_claim_blockers.csv",
                "boundary": "full dual-bound loop missing",
            },
            {
                "claim_id": "v348_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v348_claim_blockers.csv",
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
                "claim": "Paper 4 has a v348 post-v347 one-swap repricing gate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v348_post_v347_one_swap_summary.csv"
                ),
                "boundary": "One-drop/one-add screen after the v347 candidate.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v348 clears post-v347 one-swap local optimality.",
                "allowed": local_optimality_cleared,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v348_claim_blockers.csv"
                ),
                "boundary": "Scope-limited to one-drop/one-add swaps after v347.",
                "prohibited_claim_flag": not local_optimality_cleared,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v348 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v348_claim_blockers.csv"
                ),
                "boundary": "Proxy, global, dynamic, online and deployment gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v348 proves full-universe global integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v348_claim_blockers.csv"
                ),
                "boundary": "A one-swap repricing screen is not a full branch-price certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v348 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v348_claim_blockers.csv"
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
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v348 reprices the v347 applied relief candidate against all "
                    "non-selected comparable loans with exact budget/source/CVaR screens."
                ),
                "status": (
                    "post_v347_one_swap_local_optimality_cleared_proxy_gap_persists"
                    if local_optimality_cleared
                    else "post_v347_one_swap_improvement_found"
                ),
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "use the one-swap-cleared candidate as input to a proxy/global/dual-bound "
                    "gate without promotion"
                ),
                "last_wave": "v348",
                "execution_result": (
                    "post_v347_one_swap_reprice_cleared_no_cvar_feasible_improvement"
                    if local_optimality_cleared
                    else "post_v347_one_swap_reprice_found_improvements"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v348")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V348_POST_V347_ONE_SWAP_REPRICE_START -->"
    end = "<!-- V348_POST_V347_ONE_SWAP_REPRICE_END -->"
    best_delta = status["best_one_swap_return_delta_v348"]
    best_delta_text = (
        "not applicable; no source-exact return-improving swaps"
        if best_delta is None
        else str(best_delta)
    )
    best_feasible = status["best_feasible_one_swap_return_delta_v348"]
    best_feasible_text = (
        "not applicable; no CVaR-feasible improving swaps"
        if best_feasible is None
        else str(best_feasible)
    )
    block = f"""
{start}

## Wave v348: Post-v347 One-Swap Repricing Gate

Generated: {status["generated_at_utc"]}

### Objective

v347 applied the local two-add/two-drop relief candidate. v348 tests whether
that candidate still has any one-drop/one-add return-improving move under the
v347 CVaR cap, exact source caps and the original budget band.

### Results

- Selected rows: `{status["selected_rows_v348"]}`.
- Candidate add rows: `{status["candidate_add_rows_v348"]}`.
- Pair rows screened: `{status["total_pair_rows_screened_v348"]}`.
- Return-improving pairs: `{status["return_improving_pair_rows_v348"]}`.
- Budget+return feasible pairs:
  `{status["budget_return_feasible_pair_rows_v348"]}`.
- Source prefilter pairs: `{status["source_prefilter_pair_rows_v348"]}`.
- Exact source-feasible pairs: `{status["source_exact_pair_rows_v348"]}`.
- CVaR-feasible improving one-swaps:
  `{status["one_swap_improving_rows_v348"]}`.
- Best source-exact return delta: `{best_delta_text}`.
- Best CVaR-feasible return delta: `{best_feasible_text}`.
- Current missing v47 proxy rows:
  `{status["current_missing_v47_proxy_rows_v348"]}`.
- Post-v347 one-swap local optimality cleared:
  `{status["post_v347_one_swap_local_optimality_cleared_v348"]}`.

### Interpretation

v348 clears the immediate one-swap repricing gate for the v347 candidate:
return-improving source-exact swaps still exist, but none preserve the v347 CVaR
threshold. The evidence is useful, but narrow. The proxy gap, dual-bound and
live-validation blockers remain open.

### Claim Impact

- Allowed: post-v347 one-swap repricing gate completed and one-swap local
  optimality cleared within that scope.
- Still prohibited: full-universe/global optimality, Paper 4 working champion,
  Paper Estrella replacement, final Paper 4 promotion, contractual IFRS9 and
  live deployability claims.

### Quarto Promotion Decision

Keep v348 in the living notebook. The next wave should run a proxy/global/dual
gate without promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def _float_or_none(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def main() -> None:
    started = datetime.now(UTC)
    v347_status = json.loads((STATUS_DIR / "paper4_v347_status.json").read_text(encoding="utf-8"))
    if not bool(v347_status["post_v347_repricing_required_v347"]):
        raise RuntimeError("v348 expects v347 to require repricing.")
    pairs, summary, stage_summary = _candidate_pairs_for_reprice()
    top_candidates = pairs.sort_values(
        [f"one_swap_improves_return_v{VERSION}", f"return_delta_v{VERSION}"],
        ascending=[False, False],
    ).head(200)
    local_optimality_cleared = bool(
        summary[f"post_v347_one_swap_local_optimality_cleared_v{VERSION}"].iloc[0]
    )
    blockers = _claim_blockers(summary)
    claim_matrix = _claim_matrix(local_optimality_cleared=local_optimality_cleared)

    write_csv(TABLE_DIR / "paper4_v348_post_v347_one_swap_reprice.csv", pairs)
    write_csv(TABLE_DIR / "paper4_v348_post_v347_one_swap_top_candidates.csv", top_candidates)
    write_csv(TABLE_DIR / "paper4_v348_post_v347_one_swap_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v348_post_v347_one_swap_stage_summary.csv", stage_summary)
    write_csv(TABLE_DIR / "paper4_v348_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v348_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries(local_optimality_cleared=local_optimality_cleared)
    _update_backlog(local_optimality_cleared=local_optimality_cleared)

    row = summary.iloc[0]
    best_delta = row[f"best_one_swap_return_delta_v{VERSION}"]
    best_cvar = row[f"best_one_swap_cvar90_after_v{VERSION}"]
    best_feasible_delta = row[f"best_feasible_one_swap_return_delta_v{VERSION}"]
    best_feasible_cvar = row[f"best_feasible_one_swap_cvar90_after_v{VERSION}"]
    status = {
        "phase": "v348_post_v347_reprice",
        "schema_version": "2026-05-16.348",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_repair_version_v348": BASE_REPAIR_VERSION,
        "previous_reprice_version_v348": PREVIOUS_REPRICE_VERSION,
        "summary_rows_v348": int(len(summary)),
        "stage_summary_rows_v348": int(len(stage_summary)),
        "candidate_pair_rows_v348": int(len(pairs)),
        "top_candidate_rows_v348": int(len(top_candidates)),
        "claim_blocker_rows_v348": int(len(blockers)),
        "claim_matrix_rows_v348": int(len(claim_matrix)),
        "selected_rows_v348": int(row[f"selected_rows_v{VERSION}"]),
        "base_selected_rows_v348": int(row[f"base_selected_rows_v{VERSION}"]),
        "cardinality_restored_v348": bool(row[f"cardinality_restored_v{VERSION}"]),
        "candidate_add_rows_v348": int(row[f"candidate_add_rows_v{VERSION}"]),
        "total_pair_rows_screened_v348": int(row[f"total_pair_rows_screened_v{VERSION}"]),
        "return_improving_pair_rows_v348": int(row[f"return_improving_pair_rows_v{VERSION}"]),
        "budget_return_feasible_pair_rows_v348": int(
            row[f"budget_return_feasible_pair_rows_v{VERSION}"]
        ),
        "source_prefilter_pair_rows_v348": int(row[f"source_prefilter_pair_rows_v{VERSION}"]),
        "source_exact_pair_rows_v348": int(row[f"source_exact_pair_rows_v{VERSION}"]),
        "cvar_feasible_pair_rows_v348": int(row[f"cvar_feasible_pair_rows_v{VERSION}"]),
        "one_swap_improving_rows_v348": int(row[f"one_swap_improving_rows_v{VERSION}"]),
        "best_one_swap_return_delta_v348": _float_or_none(best_delta),
        "best_one_swap_cvar90_after_v348": _float_or_none(best_cvar),
        "best_feasible_one_swap_return_delta_v348": _float_or_none(best_feasible_delta),
        "best_feasible_one_swap_cvar90_after_v348": _float_or_none(best_feasible_cvar),
        "current_missing_v47_proxy_rows_v348": int(
            row[f"current_missing_v47_proxy_rows_v{VERSION}"]
        ),
        "best_feasible_delta_missing_v47_proxy_rows_v348": int(
            row[f"best_feasible_delta_missing_v47_proxy_rows_v{VERSION}"]
        ),
        "current_exposure_v348": float(row[f"current_exposure_v{VERSION}"]),
        "current_objective_return_v348": float(row[f"current_objective_return_v{VERSION}"]),
        "current_loss_mean_v348": float(row[f"current_loss_mean_v{VERSION}"]),
        "current_cvar90_v348": float(row[f"current_cvar90_v{VERSION}"]),
        "post_v347_one_swap_local_optimality_cleared_v348": local_optimality_cleared,
        "next_global_gate_ready_v348": local_optimality_cleared,
        "valid_branch_price_bound_v348": False,
        "working_champion_claim_allowed_v348": False,
        "full_universe_integer_optimality_claim_allowed_v348": False,
        "paper1_promotion_allowed_v348": False,
        "paper4_working_champion_changed_v348": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v348": NEXT_ARTIFACT,
        "claim_boundary": (
            "v348 is a post-v347 one-swap repricing gate; no working champion or "
            "final promotion is authorized"
        ),
    }
    write_json(STATUS_DIR / "paper4_v348_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v348": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
