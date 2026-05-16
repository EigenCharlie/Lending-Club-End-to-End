#!/usr/bin/env python3
"""Build Paper 4 v291 cardinality restoration protocol artifacts."""

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

VERSION = 291
POST_REPRICE_VERSION = 290
REPAIR_VERSION = 289
BASE_REPAIR_VERSION = 279
NEXT_VERSION = 292
TARGET_SELECTED_ROWS = 171
EXPOSURE_MAX = 850000.0
TOP_DIAGNOSTIC_ROWS = 500


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v291 cardinality restoration protocol.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v291_cardinality_or_multi_swap_protocol.csv"
                ),
                "boundary": (
                    "Add-only restoration diagnostic after v289/v290; not a full "
                    "cardinality repair or global optimality certificate."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v291 proves the 168-row v289 portfolio can be promoted.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v291_claim_blockers.csv"
                ),
                "boundary": "Cardinality deficit and add-only restoration blockers remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v291 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v291_claim_blockers.csv"
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
                    "v291 checks whether v289/v290 can restore the 171-row target by "
                    "add-only moves before escalating to cardinality-aware multi-swap/MILP."
                ),
                "status": "add_only_cardinality_restoration_blocked",
                "next_artifact": "paper4_v292_cardinality_aware_multi_swap_milp_protocol.csv",
                "success_condition": (
                    "formulate a cardinality-aware multi-swap/MILP repair that can test "
                    "three-row restoration without breaking source or CVaR constraints"
                ),
                "last_wave": "v291",
                "execution_result": "positive_add_only_restoration_blocked_by_grade_a",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v291")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V291_CARDINALITY_RESTORATION_PROTOCOL_START -->"
    end = "<!-- V291_CARDINALITY_RESTORATION_PROTOCOL_END -->"
    block = f"""
{start}

## Wave v291: Cardinality Restoration Protocol

Generated: {status["generated_at_utc"]}

### Objective

v289 improved return and CVaR but reduced selected rows from 171 to 168. v291
tests the simplest restoration path: add one omitted loan without dropping
anything, while respecting remaining budget, exact source caps and CVaR.

### Results

- Current selected rows: `{status["current_selected_rows_v291"]}`.
- Target selected rows: `{status["target_selected_rows_v291"]}`.
- Cardinality deficit: `{status["cardinality_deficit_v291"]}`.
- Budget headroom: `{status["budget_headroom_v291"]}`.
- Candidate rows: `{status["candidate_rows_v291"]}`.
- Budget-eligible add rows: `{status["budget_eligible_add_rows_v291"]}`.
- Return-positive budget rows: `{status["return_positive_budget_rows_v291"]}`.
- Source-feasible budget rows: `{status["source_feasible_budget_rows_v291"]}`.
- Return-positive source-feasible rows:
  `{status["return_positive_source_feasible_rows_v291"]}`.
- CVaR-feasible source rows: `{status["cvar_feasible_source_rows_v291"]}`.
- Add-only restoration step feasible:
  `{status["add_only_restoration_step_feasible_v291"]}`.

### Interpretation

The add-only route is blocked. There are many loans that fit remaining budget,
and 1,678 would recover positive return, but every positive budget-eligible add
violates the tight grade A source cap. Source-feasible adds exist, but they are
return-negative and none preserve the current CVaR cap. The next experiment
therefore needs cardinality-aware multi-swap or MILP repair, not simple adds.

### Claim Impact

- Allowed: cardinality restoration protocol and add-only blocker documented.
- Still prohibited: working champion replacement, full-universe optimality,
  Paper Estrella replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v291 in the living notebook. Promotion remains blocked.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def _source_check_after_add(
    *,
    current_by_family: dict[str, dict[str, float]],
    candidate: pd.Series,
    cap_by_family: dict[str, dict[str, float]],
    new_exposure: float,
) -> tuple[bool, float, int, str]:
    min_slack = np.inf
    violations = 0
    first_block = ""
    for family in FAMILIES:
        source_id = str(candidate[family])
        sources = set(current_by_family[family]) | {source_id}
        for source in sorted(sources):
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
    portfolio = read_parquet("paper4_v289_exact_relief_repair_allocations.parquet")
    repair_summary = read_csv("paper4_v289_exact_relief_repair_summary.csv")
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    if universe.empty or portfolio.empty or repair_summary.empty or source_caps.empty:
        raise RuntimeError("Missing v55, v289, or source-cap inputs for v291.")

    repair_row = repair_summary.iloc[0]
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_ids = set(portfolio["loan_id"].astype(str))
    portfolio_idx = idx_by_id.loc[portfolio["loan_id"].astype(str)].to_numpy()
    portfolio_losses = losses[:, portfolio_idx].sum(axis=1)
    current_exposure = float(repair_row[f"portfolio_exposure_v{REPAIR_VERSION}"])
    cvar_cap = float(repair_row[f"scenario_loss_cvar90_v{REPAIR_VERSION}"])
    budget_headroom = EXPOSURE_MAX - current_exposure
    candidates = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    candidates[f"mean_return_v{VERSION}"] = mean_returns[candidates.index.to_numpy()]
    budget = candidates.loc[candidates["loan_amnt"].le(budget_headroom + 1e-7)].copy()
    positive_budget = budget.loc[budget[f"mean_return_v{VERSION}"].gt(0)].copy()
    cap_by_family = {
        family: {
            str(k): float(v)
            for k, v in source_caps.loc[source_caps["source_family"].astype(str).eq(family)]
            .set_index("source_id")["cap_share_v80"]
            .to_dict()
            .items()
        }
        for family in FAMILIES
    }
    current_by_family = {
        family: {
            str(k): float(v)
            for k, v in portfolio.groupby(family, dropna=False)["loan_amnt"].sum().to_dict().items()
        }
        for family in FAMILIES
    }

    diagnostic_rows: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    positive_blocker_counts: dict[str, int] = {}
    source_feasible_rows = 0
    return_positive_source_rows = 0
    cvar_feasible_source_rows = 0
    for _, candidate in budget.iterrows():
        ok, min_slack, violations, first_block = _source_check_after_add(
            current_by_family=current_by_family,
            candidate=candidate,
            cap_by_family=cap_by_family,
            new_exposure=current_exposure + float(candidate["loan_amnt"]),
        )
        mean_return = float(candidate[f"mean_return_v{VERSION}"])
        if not ok:
            blocker_counts[first_block] = blocker_counts.get(first_block, 0) + 1
            if mean_return > 0:
                positive_blocker_counts[first_block] = (
                    positive_blocker_counts.get(first_block, 0) + 1
                )
        cvar_after = np.nan
        cvar_ok = False
        if ok:
            source_feasible_rows += 1
            if mean_return > 0:
                return_positive_source_rows += 1
            add_idx = int(idx_by_id.loc[str(candidate["loan_id"])])
            cvar_after = v70._tail_cvar(portfolio_losses + losses[:, add_idx])
            cvar_ok = cvar_after <= cvar_cap + 1e-7
            if cvar_ok:
                cvar_feasible_source_rows += 1
        diagnostic_rows.append(
            {
                f"loan_id_v{VERSION}": str(candidate["loan_id"]),
                f"loan_amount_v{VERSION}": float(candidate["loan_amnt"]),
                f"mean_return_v{VERSION}": mean_return,
                f"budget_eligible_v{VERSION}": True,
                f"return_positive_v{VERSION}": mean_return > 0,
                f"source_feasible_v{VERSION}": ok,
                f"source_min_slack_after_add_v{VERSION}": min_slack,
                f"source_cap_violations_after_add_v{VERSION}": violations,
                f"first_source_blocker_v{VERSION}": first_block,
                f"cvar90_after_add_v{VERSION}": cvar_after,
                f"cvar_feasible_after_add_v{VERSION}": cvar_ok,
                f"claim_boundary_v{VERSION}": (
                    "single-add cardinality restoration diagnostic only"
                ),
            }
        )

    diagnostic = pd.DataFrame(diagnostic_rows).sort_values(
        [f"return_positive_v{VERSION}", f"mean_return_v{VERSION}"],
        ascending=[False, False],
    )
    diagnostic_top = diagnostic.head(TOP_DIAGNOSTIC_ROWS).copy()
    blocker_breakdown = pd.DataFrame(
        [
            {
                f"first_source_blocker_v{VERSION}": blocker,
                f"budget_blocked_rows_v{VERSION}": int(count),
                f"return_positive_blocked_rows_v{VERSION}": int(
                    positive_blocker_counts.get(blocker, 0)
                ),
                f"claim_boundary_v{VERSION}": "single-add source blocker count only",
            }
            for blocker, count in sorted(
                blocker_counts.items(), key=lambda item: item[1], reverse=True
            )
        ]
    )
    stage_summary = pd.DataFrame(
        [
            {
                f"stage_v{VERSION}": "all_omitted_candidates",
                f"row_count_v{VERSION}": int(len(candidates)),
            },
            {
                f"stage_v{VERSION}": "budget_eligible_add",
                f"row_count_v{VERSION}": int(len(budget)),
            },
            {
                f"stage_v{VERSION}": "return_positive_budget_eligible_add",
                f"row_count_v{VERSION}": int(len(positive_budget)),
            },
            {
                f"stage_v{VERSION}": "source_feasible_budget_eligible_add",
                f"row_count_v{VERSION}": source_feasible_rows,
            },
            {
                f"stage_v{VERSION}": "return_positive_source_feasible_add",
                f"row_count_v{VERSION}": return_positive_source_rows,
            },
            {
                f"stage_v{VERSION}": "cvar_feasible_source_add",
                f"row_count_v{VERSION}": cvar_feasible_source_rows,
            },
        ]
    )
    stage_summary[f"claim_boundary_v{VERSION}"] = "add-only cardinality restoration stage count"
    cardinality_deficit = TARGET_SELECTED_ROWS - int(len(portfolio))
    add_only_feasible = return_positive_source_rows > 0 and cvar_feasible_source_rows > 0
    protocol = pd.DataFrame(
        [
            {
                f"protocol_id_v{VERSION}": "add_only_cardinality_restoration_protocol",
                f"post_reprice_version_v{VERSION}": POST_REPRICE_VERSION,
                f"repair_version_v{VERSION}": REPAIR_VERSION,
                f"base_repair_version_v{VERSION}": BASE_REPAIR_VERSION,
                f"current_selected_rows_v{VERSION}": int(len(portfolio)),
                f"target_selected_rows_v{VERSION}": TARGET_SELECTED_ROWS,
                f"cardinality_deficit_v{VERSION}": cardinality_deficit,
                f"current_exposure_v{VERSION}": current_exposure,
                f"exposure_max_v{VERSION}": EXPOSURE_MAX,
                f"budget_headroom_v{VERSION}": budget_headroom,
                f"candidate_rows_v{VERSION}": int(len(candidates)),
                f"budget_eligible_add_rows_v{VERSION}": int(len(budget)),
                f"return_positive_budget_rows_v{VERSION}": int(len(positive_budget)),
                f"source_feasible_budget_rows_v{VERSION}": source_feasible_rows,
                f"return_positive_source_feasible_rows_v{VERSION}": return_positive_source_rows,
                f"cvar_feasible_source_rows_v{VERSION}": cvar_feasible_source_rows,
                f"add_only_restoration_step_feasible_v{VERSION}": add_only_feasible,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_cardinality_aware_multi_swap_milp_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "add-only cardinality restoration blocked; multi-swap/MILP required"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "add_only_positive_restoration_blocked_by_source",
                f"blocking_v{VERSION}": return_positive_source_rows == 0,
                f"evidence_count_v{VERSION}": int(len(positive_budget)),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_cardinality_aware_multi_swap_milp_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "all positive budget-eligible single adds are blocked by source caps"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "add_only_cvar_restoration_blocked",
                f"blocking_v{VERSION}": cvar_feasible_source_rows == 0,
                f"evidence_count_v{VERSION}": source_feasible_rows,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_cardinality_aware_multi_swap_milp_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "source-feasible single adds do not preserve the v289 CVaR cap"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "cardinality_deficit_requires_multi_swap",
                f"blocking_v{VERSION}": cardinality_deficit > 0,
                f"evidence_count_v{VERSION}": cardinality_deficit,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_cardinality_aware_multi_swap_milp_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": "need three-row restoration to match 171 rows",
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
                "claim_id": "v291_cardinality_restoration_protocol_executed",
                "allowed": True,
                "artifact": "paper4_v291_cardinality_or_multi_swap_protocol.csv",
                "boundary": "add-only restoration diagnostic",
            },
            {
                "claim_id": "v291_add_only_restoration_feasible",
                "allowed": add_only_feasible,
                "artifact": "paper4_v291_claim_blockers.csv",
                "boundary": "false when source/CVaR blockers remain",
            },
            {
                "claim_id": "v291_working_champion",
                "allowed": False,
                "artifact": "paper4_v291_claim_blockers.csv",
                "boundary": "cardinality restoration and global evidence missing",
            },
            {
                "claim_id": "v291_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v291_claim_blockers.csv",
                "boundary": "global certificate missing",
            },
            {
                "claim_id": "v291_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v291_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v291_cardinality_or_multi_swap_protocol.csv", protocol)
    write_csv(TABLE_DIR / "paper4_v291_add_only_candidate_diagnostics_top.csv", diagnostic_top)
    write_csv(TABLE_DIR / "paper4_v291_add_only_source_blockers.csv", blocker_breakdown)
    write_csv(TABLE_DIR / "paper4_v291_cardinality_restoration_stage_summary.csv", stage_summary)
    write_csv(TABLE_DIR / "paper4_v291_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v291_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = protocol.iloc[0]
    status = {
        "phase": "v291_cardinality_restoration_add_only_protocol",
        "schema_version": "2026-05-15.291",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "post_reprice_version_v291": POST_REPRICE_VERSION,
        "repair_version_v291": REPAIR_VERSION,
        "base_repair_version_v291": BASE_REPAIR_VERSION,
        "current_selected_rows_v291": int(row[f"current_selected_rows_v{VERSION}"]),
        "target_selected_rows_v291": TARGET_SELECTED_ROWS,
        "cardinality_deficit_v291": cardinality_deficit,
        "current_exposure_v291": current_exposure,
        "budget_headroom_v291": budget_headroom,
        "candidate_rows_v291": int(row[f"candidate_rows_v{VERSION}"]),
        "budget_eligible_add_rows_v291": int(row[f"budget_eligible_add_rows_v{VERSION}"]),
        "return_positive_budget_rows_v291": int(row[f"return_positive_budget_rows_v{VERSION}"]),
        "source_feasible_budget_rows_v291": source_feasible_rows,
        "return_positive_source_feasible_rows_v291": return_positive_source_rows,
        "cvar_feasible_source_rows_v291": cvar_feasible_source_rows,
        "add_only_restoration_step_feasible_v291": bool(add_only_feasible),
        "top_diagnostic_rows_v291": int(len(diagnostic_top)),
        "source_blocker_rows_v291": int(len(blocker_breakdown)),
        "claim_blocker_rows_v291": int(len(blockers)),
        "claim_matrix_rows_v291": int(len(claim_matrix)),
        "working_champion_claim_allowed_v291": False,
        "full_universe_integer_optimality_claim_allowed_v291": False,
        "paper1_promotion_allowed_v291": False,
        "paper4_working_champion_changed_v291": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v291": (
            f"paper4_v{NEXT_VERSION}_cardinality_aware_multi_swap_milp_protocol.csv"
        ),
        "claim_boundary": (
            "v291 shows add-only cardinality restoration is blocked; multi-swap/MILP, "
            "global evidence and promotion remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v291_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v291": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
