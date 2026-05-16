#!/usr/bin/env python3
"""Build Paper 4 v288 exact-relief rank-expansion artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers.build_paper4_v286_joint_source_relief_pricing_protocol import (
    _current_source_maps,
    _solve_exact_relief_milp,
    _source_cap_maps,
    _source_metrics,
)
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

VERSION = 288
SOURCE_TIGHT_SCREEN_VERSION = 285
PREVIOUS_EXACT_RELIEF_VERSION = 286
INCUMBENT_REPAIR_VERSION = 279
NEXT_VERSION = 289
START_RANK = 201
CANDIDATE_LIMIT = 200


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v288 exact-relief rank-expansion screen.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v288_full_rank_exact_relief_resource_protocol.csv"
                ),
                "boundary": (
                    "Ranks 201-400 of the v285 source-tight candidate list only; "
                    "not a full ranked pricing or branch-price certificate."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v288 finds a return-positive exact relief column in v285 ranks 201-400.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v288_exact_relief_candidate_screen.csv"
                ),
                "boundary": (
                    "Lab-only repair signal; must be applied and repriced before any "
                    "portfolio or promotion claim."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v288 proves full-universe branch-price termination.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v288_claim_blockers.csv"
                ),
                "boundary": "No full ranked pricing loop or global dual-bound certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v288 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v288_claim_blockers.csv"
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
    obsolete_claims = {"v288 finds no return-positive exact relief column in v285 ranks 201-400."}
    replaced_claims = set(additions["claim"]) | obsolete_claims
    out = current.loc[~current["claim"].isin(replaced_claims)].copy()
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
                    "v288 expands exact source-relief pricing from the top-200 to "
                    "v285 ranks 201-400 and finds a small positive exact relief signal."
                ),
                "status": "rank201_400_exact_relief_entering_column_found",
                "next_artifact": "paper4_v289_apply_exact_relief_candidate_or_reprice.csv",
                "success_condition": (
                    "apply the v288 exact relief candidate, recompute portfolio metrics, "
                    "and rerun source-tight pricing before any claim expansion"
                ),
                "last_wave": "v288",
                "execution_result": "rank_slice_exact_relief_screen_found_small_entering_column",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v288")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V288_EXACT_RELIEF_RANK_EXPANSION_START -->"
    end = "<!-- V288_EXACT_RELIEF_RANK_EXPANSION_END -->"
    block = f"""
{start}

## Wave v288: Exact-Relief Rank Expansion

Generated: {status["generated_at_utc"]}

### Objective

Expand the exact source-relief screen beyond v286. v288 applies the same exact
source-cap MILP to v285 candidate ranks 201-400, preserving the exposure
denominator inside all source constraints and checking budget, source and CVaR.

### Results

- v285 candidates available: `{status["v285_candidate_rows_available_v288"]}`.
- Rank start: `{status["rank_start_v288"]}`.
- Candidate rows screened: `{status["candidate_rows_screened_v288"]}`.
- Unique exact relief MILPs solved:
  `{status["unique_relief_milp_signatures_v288"]}`.
- Successful relief MILP rows: `{status["relief_milp_success_rows_v288"]}`.
- Source-violating relief rows: `{status["source_violation_rows_v288"]}`.
- CVaR-feasible relief rows: `{status["cvar_feasible_rows_v288"]}`.
- Return-positive exact relief rows:
  `{status["return_positive_exact_relief_rows_v288"]}`.
- Best exact relief return delta:
  `{status["best_exact_relief_return_delta_v288"]}`.
- Exact relief entering columns found:
  `{status["exact_relief_entering_columns_found_v288"]}`.

### Interpretation

v288 changes the live-lab state: the next rank slice produces one small
return-positive exact relief column. This is useful as a repair signal, not a
promotion signal. It must be applied, audited and repriced before it can alter
any working-paper claim.

### Claim Impact

- Allowed: exact relief rank-slice screen completed; one small entering-column
  repair signal found in v285 ranks 201-400.
- Still prohibited: full ranked pricing termination, global integer optimality,
  Paper Estrella replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v288 in the living notebook. The next step is applying and repricing the
candidate, not promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v279_restricted_pool_milp_repair_allocations.parquet")
    repair_summary = read_csv("paper4_v279_restricted_pool_milp_repair_summary.csv")
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    v285_candidates = read_csv("paper4_v285_source_tight_candidate_diagnostics.csv")
    if universe.empty or selected.empty or repair_summary.empty or source_caps.empty:
        raise RuntimeError("Missing v55, v279, or source-cap inputs for v288.")
    if v285_candidates.empty:
        raise RuntimeError("Missing v285 source-tight candidate diagnostics for v288.")

    selected = selected.reset_index(drop=True)
    v285_candidates["loan_id"] = v285_candidates["loan_id"].astype(str)
    add_source_cols = ["loan_id", "state_top20"]
    start_pos = START_RANK - 1
    candidates = v285_candidates.iloc[start_pos : start_pos + CANDIDATE_LIMIT].merge(
        universe[add_source_cols].assign(loan_id=lambda df: df["loan_id"].astype(str)),
        on="loan_id",
        how="left",
    )
    repair_row = repair_summary.loc[
        repair_summary[f"portfolio_label_v{INCUMBENT_REPAIR_VERSION}"].eq(
            "restricted_pool_milp_repair_candidate"
        )
    ].iloc[0]
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    selected[f"mean_return_if_dropped_v{VERSION}"] = mean_returns[selected_idx]
    current_losses = losses[:, selected_idx].sum(axis=1)

    current_exposure = float(repair_row[f"portfolio_exposure_v{INCUMBENT_REPAIR_VERSION}"])
    exposure_min = float(repair_row[f"exposure_min_v{INCUMBENT_REPAIR_VERSION}"])
    exposure_max = float(repair_row[f"exposure_max_v{INCUMBENT_REPAIR_VERSION}"])
    cvar_cap = float(repair_row[f"cvar_cap_v{INCUMBENT_REPAIR_VERSION}"])
    current_objective_return = float(repair_row[f"objective_return_v{INCUMBENT_REPAIR_VERSION}"])
    selected_amounts = selected["loan_amnt"].to_numpy(float)
    selected_returns = selected[f"mean_return_if_dropped_v{VERSION}"].to_numpy(float)
    current_by_family = _current_source_maps(selected)
    cap_by_family = _source_cap_maps(source_caps)

    solution_cache: dict[tuple[object, ...], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    drop_rows: list[dict[str, Any]] = []
    for offset, add_row in enumerate(candidates.itertuples(index=False), start=0):
        add = pd.Series(add_row._asdict())
        candidate_rank = START_RANK + offset
        cache_key = tuple(add[col] for col in ["loan_amnt", *FAMILIES])
        if cache_key not in solution_cache:
            solution_cache[cache_key] = _solve_exact_relief_milp(
                add_row=add,
                selected=selected,
                selected_amounts=selected_amounts,
                selected_returns=selected_returns,
                current_exposure=current_exposure,
                exposure_min=exposure_min,
                exposure_max=exposure_max,
                current_by_family=current_by_family,
                cap_by_family=cap_by_family,
            )
        solution = solution_cache[cache_key]
        drop_mask = solution["drop_mask"]
        dropped = selected.loc[drop_mask].copy()
        drop_amount = float(dropped["loan_amnt"].sum())
        drop_return = float(dropped[f"mean_return_if_dropped_v{VERSION}"].sum())
        new_exposure = current_exposure + float(add["loan_amnt"]) - drop_amount
        added_idx = int(idx_by_id.loc[str(add["loan_id"])])
        new_losses = current_losses + losses[:, added_idx]
        if drop_mask.any():
            new_losses = new_losses - losses[:, selected_idx[drop_mask]].sum(axis=1)
        cvar_after = v70._tail_cvar(new_losses)
        min_slack, max_share, violations, first_family, first_source = _source_metrics(
            add_row=add,
            selected=selected,
            drop_mask=drop_mask,
            current_by_family=current_by_family,
            cap_by_family=cap_by_family,
            new_exposure=new_exposure,
        )
        return_delta = float(add[f"mean_return_v{SOURCE_TIGHT_SCREEN_VERSION}"]) - drop_return
        exact_relief_feasible = (
            bool(solution["success"])
            and new_exposure >= exposure_min - 1e-7
            and new_exposure <= exposure_max + 1e-7
            and violations == 0
            and cvar_after <= cvar_cap + 1e-7
        )
        entering_column = exact_relief_feasible and return_delta > 1e-9
        rows.append(
            {
                f"candidate_rank_v{VERSION}": candidate_rank,
                f"added_loan_id_v{VERSION}": str(add["loan_id"]),
                f"added_loan_amount_v{VERSION}": float(add["loan_amnt"]),
                f"added_mean_return_v{VERSION}": float(
                    add[f"mean_return_v{SOURCE_TIGHT_SCREEN_VERSION}"]
                ),
                f"pricing_block_ids_v{VERSION}": str(
                    add[f"pricing_block_ids_v{SOURCE_TIGHT_SCREEN_VERSION}"]
                ),
                f"relief_milp_success_v{VERSION}": bool(solution["success"]),
                f"relief_milp_status_v{VERSION}": int(solution["status"]),
                f"relief_milp_gap_v{VERSION}": float(solution["mip_gap"]),
                f"drop_count_v{VERSION}": int(len(dropped)),
                f"drop_exposure_v{VERSION}": drop_amount,
                f"drop_mean_return_v{VERSION}": drop_return,
                f"return_delta_after_exact_relief_v{VERSION}": return_delta,
                f"objective_return_after_exact_relief_v{VERSION}": (
                    current_objective_return + return_delta
                ),
                f"exposure_after_exact_relief_v{VERSION}": new_exposure,
                f"source_min_slack_after_exact_relief_v{VERSION}": min_slack,
                f"max_source_share_after_exact_relief_v{VERSION}": max_share,
                f"source_cap_violations_after_exact_relief_v{VERSION}": violations,
                f"first_source_block_family_v{VERSION}": first_family,
                f"first_source_block_id_v{VERSION}": first_source,
                f"cvar90_after_exact_relief_v{VERSION}": cvar_after,
                f"budget_source_cvar_feasible_exact_relief_v{VERSION}": exact_relief_feasible,
                f"return_positive_exact_relief_v{VERSION}": return_delta > 1e-9,
                f"exact_relief_entering_column_v{VERSION}": entering_column,
                f"cardinality_preserved_v{VERSION}": int(len(dropped)) == 1,
                f"cardinality_after_exact_relief_v{VERSION}": int(len(selected) - len(dropped) + 1),
                f"claim_boundary_v{VERSION}": (
                    "rank 201-400 exact joint source-relief diagnostic only; not global pricing"
                ),
            }
        )
        for drop_rank, (_, drop) in enumerate(
            dropped.sort_values(f"mean_return_if_dropped_v{VERSION}").iterrows(),
            start=1,
        ):
            drop_rows.append(
                {
                    f"candidate_rank_v{VERSION}": candidate_rank,
                    f"added_loan_id_v{VERSION}": str(add["loan_id"]),
                    f"drop_rank_v{VERSION}": drop_rank,
                    f"dropped_loan_id_v{VERSION}": str(drop["loan_id"]),
                    f"dropped_loan_amount_v{VERSION}": float(drop["loan_amnt"]),
                    f"dropped_mean_return_v{VERSION}": float(
                        drop[f"mean_return_if_dropped_v{VERSION}"]
                    ),
                    f"claim_boundary_v{VERSION}": (
                        "selected by exact source-relief MILP for diagnostic pricing only"
                    ),
                }
            )

    candidate_screen = pd.DataFrame(rows).sort_values(
        f"return_delta_after_exact_relief_v{VERSION}", ascending=False
    )
    drop_bundle = pd.DataFrame(drop_rows)
    success_rows = int(candidate_screen[f"relief_milp_success_v{VERSION}"].sum())
    source_violation_rows = int(
        candidate_screen[f"source_cap_violations_after_exact_relief_v{VERSION}"].gt(0).sum()
    )
    cvar_feasible_rows = int(
        candidate_screen[f"cvar90_after_exact_relief_v{VERSION}"].le(cvar_cap + 1e-7).sum()
    )
    return_positive_rows = int(candidate_screen[f"return_positive_exact_relief_v{VERSION}"].sum())
    entering_rows = int(candidate_screen[f"exact_relief_entering_column_v{VERSION}"].sum())
    best = candidate_screen.head(1)
    summary = pd.DataFrame(
        [
            {
                f"protocol_id_v{VERSION}": "rank201_400_exact_source_relief_expansion",
                f"source_tight_screen_version_v{VERSION}": SOURCE_TIGHT_SCREEN_VERSION,
                f"previous_exact_relief_version_v{VERSION}": PREVIOUS_EXACT_RELIEF_VERSION,
                f"incumbent_repair_version_v{VERSION}": INCUMBENT_REPAIR_VERSION,
                f"v285_candidate_rows_available_v{VERSION}": int(len(v285_candidates)),
                f"rank_start_v{VERSION}": START_RANK,
                f"candidate_screen_limit_v{VERSION}": CANDIDATE_LIMIT,
                f"candidate_rows_screened_v{VERSION}": int(len(candidate_screen)),
                f"unique_relief_milp_signatures_v{VERSION}": int(len(solution_cache)),
                f"relief_milp_success_rows_v{VERSION}": success_rows,
                f"source_violation_rows_v{VERSION}": source_violation_rows,
                f"cvar_feasible_rows_v{VERSION}": cvar_feasible_rows,
                f"return_positive_exact_relief_rows_v{VERSION}": return_positive_rows,
                f"exact_relief_entering_column_rows_v{VERSION}": entering_rows,
                f"best_exact_relief_added_loan_id_v{VERSION}": str(
                    best[f"added_loan_id_v{VERSION}"].iloc[0]
                )
                if not best.empty
                else "",
                f"best_exact_relief_return_delta_v{VERSION}": float(
                    best[f"return_delta_after_exact_relief_v{VERSION}"].iloc[0]
                )
                if not best.empty
                else np.nan,
                f"best_exact_relief_drop_count_v{VERSION}": int(
                    best[f"drop_count_v{VERSION}"].iloc[0]
                )
                if not best.empty
                else 0,
                f"exact_relief_entering_columns_found_v{VERSION}": entering_rows > 0,
                f"valid_branch_price_bound_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_apply_exact_relief_candidate_or_reprice.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "rank 201-400 exact source-relief screen found a lab-only repair "
                    "signal; application and repricing remain required"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": (
                    "rank201_400_exact_relief_entering_column_requires_repair"
                ),
                f"blocking_v{VERSION}": entering_rows > 0,
                f"evidence_count_v{VERSION}": entering_rows,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_apply_exact_relief_candidate_or_reprice.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "positive rank-slice relief column must be applied and repriced"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "remaining_ranked_exact_relief_screen_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    len(v285_candidates) - (START_RANK - 1) - len(candidate_screen)
                ),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_apply_exact_relief_candidate_or_reprice.csv"
                ),
                f"claim_boundary_v{VERSION}": "v288 screens ranks 201-400 only",
            },
            {
                f"blocker_id_v{VERSION}": "branch_price_dual_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_branch_price_dual_bound_loop",
                f"claim_boundary_v{VERSION}": "no dual-bound loop or termination certificate",
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
                "claim_id": "v288_exact_relief_rank_expansion_executed",
                "allowed": True,
                "artifact": "paper4_v288_full_rank_exact_relief_resource_protocol.csv",
                "boundary": "v285 ranks 201-400 only",
            },
            {
                "claim_id": "v288_rank201_400_return_positive_exact_relief_column_found",
                "allowed": entering_rows > 0,
                "artifact": "paper4_v288_exact_relief_candidate_screen.csv",
                "boundary": "lab-only repair signal; apply and reprice next",
            },
            {
                "claim_id": "v288_valid_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v288_claim_blockers.csv",
                "boundary": "dual-bound loop missing",
            },
            {
                "claim_id": "v288_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v288_claim_blockers.csv",
                "boundary": "global certificate missing",
            },
            {
                "claim_id": "v288_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v288_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v288_full_rank_exact_relief_resource_protocol.csv", summary)
    write_csv(TABLE_DIR / "paper4_v288_exact_relief_candidate_screen.csv", candidate_screen)
    write_csv(TABLE_DIR / "paper4_v288_exact_relief_drop_bundles.csv", drop_bundle)
    write_csv(TABLE_DIR / "paper4_v288_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v288_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = summary.iloc[0]
    status = {
        "phase": "v288_exact_relief_rank_expansion",
        "schema_version": "2026-05-15.288",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "source_tight_screen_version_v288": SOURCE_TIGHT_SCREEN_VERSION,
        "previous_exact_relief_version_v288": PREVIOUS_EXACT_RELIEF_VERSION,
        "incumbent_repair_version_v288": INCUMBENT_REPAIR_VERSION,
        "v285_candidate_rows_available_v288": int(row[f"v285_candidate_rows_available_v{VERSION}"]),
        "rank_start_v288": START_RANK,
        "candidate_screen_limit_v288": CANDIDATE_LIMIT,
        "candidate_rows_screened_v288": int(row[f"candidate_rows_screened_v{VERSION}"]),
        "unique_relief_milp_signatures_v288": int(row[f"unique_relief_milp_signatures_v{VERSION}"]),
        "relief_milp_success_rows_v288": success_rows,
        "source_violation_rows_v288": source_violation_rows,
        "cvar_feasible_rows_v288": cvar_feasible_rows,
        "return_positive_exact_relief_rows_v288": return_positive_rows,
        "exact_relief_entering_column_rows_v288": entering_rows,
        "best_exact_relief_added_loan_id_v288": str(
            row[f"best_exact_relief_added_loan_id_v{VERSION}"]
        ),
        "best_exact_relief_return_delta_v288": float(
            row[f"best_exact_relief_return_delta_v{VERSION}"]
        ),
        "best_exact_relief_drop_count_v288": int(row[f"best_exact_relief_drop_count_v{VERSION}"]),
        "exact_relief_entering_columns_found_v288": bool(
            row[f"exact_relief_entering_columns_found_v{VERSION}"]
        ),
        "valid_branch_price_bound_v288": False,
        "full_universe_integer_optimality_claim_allowed_v288": False,
        "paper1_promotion_allowed_v288": False,
        "paper4_working_champion_changed_v288": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "drop_bundle_rows_v288": int(len(drop_bundle)),
        "claim_blocker_rows_v288": int(len(blockers)),
        "claim_matrix_rows_v288": int(len(claim_matrix)),
        "next_artifact_v288": (
            f"paper4_v{NEXT_VERSION}_apply_exact_relief_candidate_or_reprice.csv"
        ),
        "claim_boundary": (
            "v288 found a small exact-relief repair signal in v285 ranks 201-400; "
            "application, repricing, global pricing and promotion remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v288_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v288": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
