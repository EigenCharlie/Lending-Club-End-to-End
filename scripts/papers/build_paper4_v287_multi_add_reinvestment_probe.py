#!/usr/bin/env python3
"""Build Paper 4 v287 multi-add reinvestment probe artifacts."""

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
    now,
    read_csv,
    read_parquet,
    write_csv,
    write_json,
)

VERSION = 287
EXACT_RELIEF_VERSION = 286
INCUMBENT_REPAIR_VERSION = 279
NEXT_VERSION = 288
TOP_RELIEF_STATE_LIMIT = 25


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v287 multi-add reinvestment probe.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v287_multi_add_reinvestment_probe.csv"
                ),
                "boundary": (
                    "Top-25 v286 relief states only; probes one additional reinvestment "
                    "loan after exact relief, not full branch-price termination."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v287 finds no source-feasible second-add reinvestment column in "
                    "the top-25 exact relief states."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v287_reinvestment_state_screen.csv"
                ),
                "boundary": (
                    "Second-add screen only; broader exact relief ranks and multi-add "
                    "sets remain open."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v287 proves full-universe branch-price termination.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v287_claim_blockers.csv"
                ),
                "boundary": "No full ranked pricing loop or global dual-bound certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v287 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v287_claim_blockers.csv"
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


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v287 tests whether a second small add can reinvest after the "
                    "best v286 exact relief bundles; all return-recovering second adds "
                    "are blocked by source caps before CVaR."
                ),
                "status": "top25_reinvestment_second_add_blocked_by_source_caps",
                "next_artifact": "paper4_v288_full_rank_exact_relief_resource_protocol.csv",
                "success_condition": (
                    "either expand exact relief beyond top-200 with a resource protocol "
                    "or formulate a richer multi-add source-relief master"
                ),
                "last_wave": "v287",
                "execution_result": "no_source_feasible_second_add_reinvestment_column",
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
    start = "<!-- V287_MULTI_ADD_REINVESTMENT_PROBE_START -->"
    end = "<!-- V287_MULTI_ADD_REINVESTMENT_PROBE_END -->"
    block = f"""
{start}

## Wave v287: Multi-Add Reinvestment Probe

Generated: {status["generated_at_utc"]}

### Objective

Follow the v286 near-miss screen by asking whether a second small add can
recover the return lost by exact source relief. v287 takes the best 25 v286
relief states, screens all omitted loans that fit remaining budget and would
make net return positive, then applies exact source caps before any CVaR claim.

### Results

- v286 states available: `{status["v286_relief_states_available_v287"]}`.
- Reinvestment states screened: `{status["reinvestment_states_screened_v287"]}`.
- Budget-eligible second-add rows:
  `{status["budget_eligible_second_add_rows_v287"]}`.
- Return-recovering second-add rows:
  `{status["return_recovering_second_add_rows_v287"]}`.
- Source-feasible second-add rows:
  `{status["source_feasible_second_add_rows_v287"]}`.
- CVaR-feasible second-add rows:
  `{status["cvar_feasible_second_add_rows_v287"]}`.
- Reinvestment entering columns found:
  `{status["reinvestment_entering_columns_found_v287"]}`.
- Valid branch-price bound produced:
  `{status["valid_branch_price_bound_v287"]}`.

### Interpretation

v287 closes a tempting loophole from v286. There are many small loans that fit
the remaining budget and would recover the exact-relief return loss, but none
survive the exact source-cap screen. The blocker is therefore source capacity,
not CVaR, for these top relief states.

### Claim Impact

- Allowed: top-25 second-add reinvestment probe completed; no source-feasible
  reinvestment column found in that screened set.
- Still prohibited: full ranked pricing termination, global integer optimality,
  Paper Estrella replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v287 in the living notebook. The next live-lab step is a resource-aware
full-rank exact relief protocol or a richer multi-add source-relief master.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def _source_cap_maps(source_caps: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        family: {
            str(k): float(v)
            for k, v in source_caps.loc[source_caps["source_family"].astype(str).eq(family)]
            .set_index("source_id")["cap_share_v80"]
            .to_dict()
            .items()
        }
        for family in FAMILIES
    }


def _source_check_after_add(
    *,
    portfolio: pd.DataFrame,
    candidate: pd.Series,
    cap_by_family: dict[str, dict[str, float]],
    new_exposure: float,
) -> tuple[bool, float, int, str]:
    current_by_family = {
        family: {
            str(k): float(v)
            for k, v in portfolio.groupby(family, dropna=False)["loan_amnt"].sum().to_dict().items()
        }
        for family in FAMILIES
    }
    min_slack = np.inf
    violations = 0
    first_block = ""
    for family in FAMILIES:
        source_id = str(candidate[family])
        sources = set(current_by_family[family]) | {source_id}
        for source in sources:
            exposure = current_by_family[family].get(source, 0.0)
            if source == source_id:
                exposure += float(candidate["loan_amnt"])
            cap = cap_by_family[family].get(source, 1.0)
            share = exposure / max(new_exposure, 1.0)
            slack = cap - share
            min_slack = min(min_slack, slack)
            if share > cap + 1e-7:
                violations += 1
                if not first_block:
                    first_block = f"{family}={source}"
    return violations == 0, float(min_slack), int(violations), first_block


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v279_restricted_pool_milp_repair_allocations.parquet")
    repair_summary = read_csv("paper4_v279_restricted_pool_milp_repair_summary.csv")
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    v286_screen = read_csv("paper4_v286_joint_source_relief_candidate_screen.csv")
    v286_drops = read_csv("paper4_v286_joint_source_relief_drop_bundles.csv")
    if universe.empty or selected.empty or repair_summary.empty or source_caps.empty:
        raise RuntimeError("Missing v55, v279, or source-cap inputs for v287.")
    if v286_screen.empty or v286_drops.empty:
        raise RuntimeError("Missing v286 exact relief inputs for v287.")

    keep_cols = ["loan_id", "loan_amnt", *FAMILIES]
    selected = selected.reset_index(drop=True)
    repair_row = repair_summary.loc[
        repair_summary[f"portfolio_label_v{INCUMBENT_REPAIR_VERSION}"].eq(
            "restricted_pool_milp_repair_candidate"
        )
    ].iloc[0]
    exposure_max = float(repair_row[f"exposure_max_v{INCUMBENT_REPAIR_VERSION}"])
    cvar_cap = float(repair_row[f"cvar_cap_v{INCUMBENT_REPAIR_VERSION}"])
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_ids = set(selected["loan_id"].astype(str))
    cap_by_family = _source_cap_maps(source_caps)

    states = v286_screen.head(TOP_RELIEF_STATE_LIMIT).copy()
    rows: list[dict[str, Any]] = []
    source_candidate_rows: list[dict[str, Any]] = []
    for state_rank, state in states.iterrows():
        candidate_rank = int(state[f"candidate_rank_v{EXACT_RELIEF_VERSION}"])
        first_add_id = str(state[f"added_loan_id_v{EXACT_RELIEF_VERSION}"])
        drop_ids = set(
            v286_drops.loc[
                v286_drops[f"candidate_rank_v{EXACT_RELIEF_VERSION}"].eq(candidate_rank),
                f"dropped_loan_id_v{EXACT_RELIEF_VERSION}",
            ].astype(str)
        )
        portfolio = selected.loc[~selected["loan_id"].astype(str).isin(drop_ids), keep_cols].copy()
        first_add = universe.loc[universe["loan_id"].astype(str).eq(first_add_id), keep_cols].copy()
        portfolio = pd.concat([portfolio, first_add], ignore_index=True)
        portfolio_exposure = float(portfolio["loan_amnt"].sum())
        headroom = max(0.0, exposure_max - portfolio_exposure)
        base_return_delta = float(state[f"return_delta_after_exact_relief_v{EXACT_RELIEF_VERSION}"])
        used_ids = selected_ids | drop_ids | {first_add_id}
        omitted = universe.loc[~universe["loan_id"].astype(str).isin(used_ids)].copy()
        omitted[f"mean_return_v{VERSION}"] = mean_returns[omitted.index.to_numpy()]
        budget_eligible = omitted.loc[omitted["loan_amnt"].le(headroom + 1e-7)].copy()
        return_recovering = budget_eligible.loc[
            budget_eligible[f"mean_return_v{VERSION}"] + base_return_delta > 1e-9
        ].copy()
        portfolio_idx = idx_by_id.loc[portfolio["loan_id"].astype(str)].to_numpy()
        portfolio_losses = losses[:, portfolio_idx].sum(axis=1)

        source_feasible_rows = 0
        cvar_feasible_rows = 0
        best_net_delta = np.nan
        best_second_add_id = ""
        best_second_add_return = np.nan
        first_source_block = ""
        for _, candidate in return_recovering.sort_values(
            f"mean_return_v{VERSION}", ascending=False
        ).iterrows():
            ok, min_slack, violations, first_block = _source_check_after_add(
                portfolio=portfolio,
                candidate=candidate,
                cap_by_family=cap_by_family,
                new_exposure=portfolio_exposure + float(candidate["loan_amnt"]),
            )
            if not ok:
                if not first_source_block:
                    first_source_block = first_block
                continue
            source_feasible_rows += 1
            second_idx = int(idx_by_id.loc[str(candidate["loan_id"])])
            cvar_after = v70._tail_cvar(portfolio_losses + losses[:, second_idx])
            net_delta = base_return_delta + float(candidate[f"mean_return_v{VERSION}"])
            source_candidate_rows.append(
                {
                    f"state_rank_v{VERSION}": int(state_rank) + 1,
                    f"v286_candidate_rank_v{VERSION}": candidate_rank,
                    f"first_added_loan_id_v{VERSION}": first_add_id,
                    f"second_added_loan_id_v{VERSION}": str(candidate["loan_id"]),
                    f"second_added_amount_v{VERSION}": float(candidate["loan_amnt"]),
                    f"second_added_mean_return_v{VERSION}": float(
                        candidate[f"mean_return_v{VERSION}"]
                    ),
                    f"net_return_delta_after_second_add_v{VERSION}": net_delta,
                    f"source_min_slack_after_second_add_v{VERSION}": min_slack,
                    f"source_cap_violations_after_second_add_v{VERSION}": violations,
                    f"cvar90_after_second_add_v{VERSION}": cvar_after,
                    f"cvar_feasible_second_add_v{VERSION}": cvar_after <= cvar_cap + 1e-7,
                    f"claim_boundary_v{VERSION}": (
                        "source-feasible reinvestment candidate only; no global pricing"
                    ),
                }
            )
            if cvar_after <= cvar_cap + 1e-7:
                cvar_feasible_rows += 1
                if np.isnan(best_net_delta) or net_delta > best_net_delta:
                    best_net_delta = net_delta
                    best_second_add_id = str(candidate["loan_id"])
                    best_second_add_return = float(candidate[f"mean_return_v{VERSION}"])

        rows.append(
            {
                f"state_rank_v{VERSION}": int(state_rank) + 1,
                f"v286_candidate_rank_v{VERSION}": candidate_rank,
                f"first_added_loan_id_v{VERSION}": first_add_id,
                f"base_return_delta_v{VERSION}": base_return_delta,
                f"portfolio_exposure_before_second_add_v{VERSION}": portfolio_exposure,
                f"budget_headroom_v{VERSION}": headroom,
                f"budget_eligible_second_add_rows_v{VERSION}": int(len(budget_eligible)),
                f"return_recovering_second_add_rows_v{VERSION}": int(len(return_recovering)),
                f"source_feasible_second_add_rows_v{VERSION}": source_feasible_rows,
                f"cvar_feasible_second_add_rows_v{VERSION}": cvar_feasible_rows,
                f"best_second_add_loan_id_v{VERSION}": best_second_add_id,
                f"best_second_add_mean_return_v{VERSION}": best_second_add_return,
                f"best_net_return_delta_after_second_add_v{VERSION}": best_net_delta,
                f"first_source_blocker_v{VERSION}": first_source_block,
                f"reinvestment_entering_column_found_v{VERSION}": cvar_feasible_rows > 0,
                f"claim_boundary_v{VERSION}": (
                    "top-v286-state second-add reinvestment diagnostic only"
                ),
            }
        )

    state_screen = pd.DataFrame(rows)
    source_candidate_columns = [
        f"state_rank_v{VERSION}",
        f"v286_candidate_rank_v{VERSION}",
        f"first_added_loan_id_v{VERSION}",
        f"second_added_loan_id_v{VERSION}",
        f"second_added_amount_v{VERSION}",
        f"second_added_mean_return_v{VERSION}",
        f"net_return_delta_after_second_add_v{VERSION}",
        f"source_min_slack_after_second_add_v{VERSION}",
        f"source_cap_violations_after_second_add_v{VERSION}",
        f"cvar90_after_second_add_v{VERSION}",
        f"cvar_feasible_second_add_v{VERSION}",
        f"claim_boundary_v{VERSION}",
    ]
    source_candidates = pd.DataFrame(source_candidate_rows, columns=source_candidate_columns)
    budget_rows = int(state_screen[f"budget_eligible_second_add_rows_v{VERSION}"].sum())
    return_rows = int(state_screen[f"return_recovering_second_add_rows_v{VERSION}"].sum())
    source_rows = int(state_screen[f"source_feasible_second_add_rows_v{VERSION}"].sum())
    cvar_rows = int(state_screen[f"cvar_feasible_second_add_rows_v{VERSION}"].sum())
    summary = pd.DataFrame(
        [
            {
                f"probe_id_v{VERSION}": "top25_exact_relief_second_add_reinvestment_probe",
                f"exact_relief_version_v{VERSION}": EXACT_RELIEF_VERSION,
                f"incumbent_repair_version_v{VERSION}": INCUMBENT_REPAIR_VERSION,
                f"v286_relief_states_available_v{VERSION}": int(len(v286_screen)),
                f"reinvestment_state_limit_v{VERSION}": TOP_RELIEF_STATE_LIMIT,
                f"reinvestment_states_screened_v{VERSION}": int(len(state_screen)),
                f"budget_eligible_second_add_rows_v{VERSION}": budget_rows,
                f"return_recovering_second_add_rows_v{VERSION}": return_rows,
                f"source_feasible_second_add_rows_v{VERSION}": source_rows,
                f"cvar_feasible_second_add_rows_v{VERSION}": cvar_rows,
                f"reinvestment_entering_column_rows_v{VERSION}": cvar_rows,
                f"reinvestment_entering_columns_found_v{VERSION}": cvar_rows > 0,
                f"valid_branch_price_bound_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_full_rank_exact_relief_resource_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "top-25 second-add reinvestment screen only; broader exact relief "
                    "and branch-price bounds remain missing"
                ),
            }
        ]
    )
    stage_summary = pd.DataFrame(
        [
            {
                f"stage_v{VERSION}": "budget_eligible_second_add",
                f"row_count_v{VERSION}": budget_rows,
                f"claim_boundary_v{VERSION}": "fits remaining exposure headroom",
            },
            {
                f"stage_v{VERSION}": "return_recovering_second_add",
                f"row_count_v{VERSION}": return_rows,
                f"claim_boundary_v{VERSION}": "would make exact-relief net return positive",
            },
            {
                f"stage_v{VERSION}": "source_feasible_second_add",
                f"row_count_v{VERSION}": source_rows,
                f"claim_boundary_v{VERSION}": "exact source caps after second add",
            },
            {
                f"stage_v{VERSION}": "cvar_feasible_second_add",
                f"row_count_v{VERSION}": cvar_rows,
                f"claim_boundary_v{VERSION}": "source-feasible and within CVaR cap",
            },
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "top25_second_add_blocked_by_source_caps",
                f"blocking_v{VERSION}": source_rows == 0,
                f"evidence_count_v{VERSION}": return_rows,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_full_rank_exact_relief_resource_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "return-recovering second adds exist but none pass exact source caps"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "full_rank_exact_relief_screen_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(v286_screen) - len(state_screen)),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_full_rank_exact_relief_resource_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": "v287 screens top-25 v286 states, not all states",
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
                "claim_id": "v287_multi_add_reinvestment_probe_executed",
                "allowed": True,
                "artifact": "paper4_v287_multi_add_reinvestment_probe.csv",
                "boundary": "top-25 v286 relief states only",
            },
            {
                "claim_id": "v287_no_source_feasible_second_add_reinvestment",
                "allowed": source_rows == 0,
                "artifact": "paper4_v287_reinvestment_state_screen.csv",
                "boundary": "one second-add screen only; not all multi-add sets",
            },
            {
                "claim_id": "v287_valid_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v287_claim_blockers.csv",
                "boundary": "dual-bound loop missing",
            },
            {
                "claim_id": "v287_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v287_claim_blockers.csv",
                "boundary": "global certificate missing",
            },
            {
                "claim_id": "v287_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v287_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v287_multi_add_reinvestment_probe.csv", summary)
    write_csv(TABLE_DIR / "paper4_v287_reinvestment_state_screen.csv", state_screen)
    write_csv(TABLE_DIR / "paper4_v287_reinvestment_stage_summary.csv", stage_summary)
    write_csv(
        TABLE_DIR / "paper4_v287_source_feasible_second_add_candidates.csv", source_candidates
    )
    write_csv(TABLE_DIR / "paper4_v287_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v287_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = summary.iloc[0]
    status = {
        "phase": "v287_multi_add_reinvestment_probe",
        "schema_version": "2026-05-15.287",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "exact_relief_version_v287": EXACT_RELIEF_VERSION,
        "incumbent_repair_version_v287": INCUMBENT_REPAIR_VERSION,
        "v286_relief_states_available_v287": int(row[f"v286_relief_states_available_v{VERSION}"]),
        "reinvestment_state_limit_v287": TOP_RELIEF_STATE_LIMIT,
        "reinvestment_states_screened_v287": int(row[f"reinvestment_states_screened_v{VERSION}"]),
        "budget_eligible_second_add_rows_v287": budget_rows,
        "return_recovering_second_add_rows_v287": return_rows,
        "source_feasible_second_add_rows_v287": source_rows,
        "cvar_feasible_second_add_rows_v287": cvar_rows,
        "reinvestment_entering_column_rows_v287": cvar_rows,
        "reinvestment_entering_columns_found_v287": bool(
            row[f"reinvestment_entering_columns_found_v{VERSION}"]
        ),
        "valid_branch_price_bound_v287": False,
        "full_universe_integer_optimality_claim_allowed_v287": False,
        "paper1_promotion_allowed_v287": False,
        "paper4_working_champion_changed_v287": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "source_feasible_candidate_rows_v287": int(len(source_candidates)),
        "stage_summary_rows_v287": int(len(stage_summary)),
        "claim_blocker_rows_v287": int(len(blockers)),
        "claim_matrix_rows_v287": int(len(claim_matrix)),
        "next_artifact_v287": (
            f"paper4_v{NEXT_VERSION}_full_rank_exact_relief_resource_protocol.csv"
        ),
        "claim_boundary": (
            "v287 screens second-add reinvestment over top-25 v286 states only; "
            "global pricing, branch-price bounds and promotion remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v287_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v287": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
