#!/usr/bin/env python3
"""Build Paper 4 v263 bounded two-swap repair candidate artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime

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
    _source_summary,
    now,
    read_csv,
    read_parquet,
    write_csv,
    write_json,
)

VERSION = 263
PREVIOUS_REPAIR_VERSION = 260
PROBE_VERSION = 262
NEXT_REPRICE_VERSION = 264


def _best_two_swap() -> pd.Series:
    top = read_csv("paper4_v262_bounded_two_swap_top_candidates.csv")
    if top.empty:
        raise RuntimeError("Missing v262 bounded two-swap top candidates.")
    return top.sort_values("total_return_delta_v262", ascending=False).iloc[0]


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v263 bounded two-swap repair candidate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v263_bounded_two_swap_repair_summary.csv"
                ),
                "boundary": (
                    "Applies the best v262 bounded two-swap only; requires post-repair repricing."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v263 repaired portfolio is post-repair locally optimal.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v263_claim_blockers.csv"
                ),
                "boundary": "Post-repair one-swap/two-swap pricing has not been rerun after v263.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v263 proves multi-swap or global full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v263_claim_blockers.csv"
                ),
                "boundary": "No exhaustive multi-swap search or global gap certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v263 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v263_claim_blockers.csv"
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
                    "v263 applies the best v262 bounded two-swap repair and recomputes "
                    "portfolio metrics."
                ),
                "status": "post_repair_pricing_required",
                "next_artifact": "paper4_v264_post_bounded_two_swap_reprice.csv",
                "success_condition": (
                    "post-v263 one-swap/two-swap pricing finds no feasible improving exchanges"
                ),
                "last_wave": "v263",
                "execution_result": "bounded_two_swap_repair_candidate_created",
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


def _update_notebook(status: dict[str, object]) -> None:
    start = "<!-- V263_BOUNDED_TWO_SWAP_REPAIR_START -->"
    end = "<!-- V263_BOUNDED_TWO_SWAP_REPAIR_END -->"
    block = f"""
{start}

## Wave v263: Bounded Two-Swap Repair Candidate

Generated: {status["generated_at_utc"]}

### Objective

Apply the best v262 bounded two-drop/two-add source-relief candidate and
recompute return, exposure, source and CVaR metrics. This tests whether the
bounded multi-swap signal can become a feasible repaired portfolio candidate.

### Results

- Added loans: `{status["primary_added_loan_id_v263"]}`,
  `{status["relief_added_loan_id_v263"]}`.
- Dropped loans: `{status["primary_dropped_loan_id_v263"]}`,
  `{status["relief_dropped_loan_id_v263"]}`.
- Selected rows after repair: `{status["selected_rows_v263"]}`.
- Return delta vs v260:
  `{status["delta_return_vs_v260_v263"]}`.
- CVaR90 delta vs v260:
  `{status["delta_cvar90_vs_v260_v263"]}`.
- Budget feasible: `{status["budget_feasible_v263"]}`.
- Source feasible: `{status["source_feasible_v263"]}`.
- CVaR feasible: `{status["cvar_feasible_v263"]}`.

### Interpretation

v263 converts the v262 bounded two-swap finding into a feasible repair
candidate. The next required step is post-repair repricing; no local/global
optimality or promotion claim is enabled.

### Claim Impact

- Allowed: bounded two-swap repair candidate created.
- Still prohibited: post-repair local optimality, exhaustive multi-swap/global
  integer optimality, Paper Estrella replacement, final Paper 4 promotion and
  live deployment.

### Quarto Promotion Decision

Keep v263 in the living notebook. Promote only after post-repair pricing,
broader multi-swap/global and dynamic validation gates pass.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    previous = read_parquet("paper4_v260_next_one_swap_repair_allocations.parquet")
    previous_summary = read_csv("paper4_v260_next_one_swap_repair_summary.csv")
    previous_row = previous_summary.iloc[0]
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    best = _best_two_swap()
    if universe.empty or previous.empty:
        raise RuntimeError("Missing v55 universe or v260 repaired portfolio.")

    add_ids = [
        str(best["primary_added_loan_id_v262"]),
        str(best["relief_added_loan_id_v262"]),
    ]
    drop_ids = [
        str(best["primary_dropped_loan_id_v262"]),
        str(best["relief_dropped_loan_id_v262"]),
    ]
    add_rows = universe.loc[universe["loan_id"].astype(str).isin(add_ids)].copy()
    if len(add_rows) != 2:
        raise RuntimeError(f"Could not find both added loans for v263: {add_ids}")

    repaired = previous.loc[~previous["loan_id"].astype(str).isin(drop_ids)].copy()
    if len(repaired) != len(previous) - 2:
        raise RuntimeError(f"Could not drop both selected loans for v263: {drop_ids}")
    keep_cols = ["loan_id", "loan_amnt", *FAMILIES]
    repaired = pd.concat([repaired[keep_cols], add_rows[keep_cols]], ignore_index=True)
    repaired["loan_id"] = repaired["loan_id"].astype(str)

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    repaired_idx = idx_by_id.loc[repaired["loan_id"].astype(str)].to_numpy()
    scenario_losses = losses[:, repaired_idx].sum(axis=1)
    repaired[f"mean_return_v{VERSION}"] = mean_returns[repaired_idx]
    repaired[f"selected_v{VERSION}"] = 1
    repaired[f"portfolio_label_v{VERSION}"] = "bounded_two_swap_repair_candidate"
    repaired[f"repair_action_v{VERSION}"] = np.where(
        repaired["loan_id"].isin(add_ids),
        "added_from_v262_best_bounded_two_swap",
        "kept_from_v260",
    )
    repaired[f"claim_boundary_v{VERSION}"] = (
        "bounded two-swap repair candidate only; requires post-repair repricing"
    )

    source_summary = _source_summary(
        universe=universe,
        portfolio=repaired,
        source_caps=source_caps,
        version=VERSION,
        ordinal="post-v261 bounded two-swap",
    )
    source_summary[f"portfolio_label_v{VERSION}"] = "bounded_two_swap_repair_candidate"
    source_summary[f"claim_boundary_v{VERSION}"] = "bounded two-swap source diagnostic only"
    objective_return = float(repaired[f"mean_return_v{VERSION}"].sum())
    exposure = float(repaired["loan_amnt"].sum())
    cvar90 = v70._tail_cvar(scenario_losses)
    budget_feasible = exposure >= float(previous_row["exposure_min_v260"]) - 1e-7
    budget_feasible = (
        budget_feasible and exposure <= float(previous_row["exposure_max_v260"]) + 1e-7
    )
    cvar_feasible = cvar90 <= float(previous_row["cvar_cap_v260"]) + 1e-7
    source_feasible = not source_summary[f"source_cap_violated_v{VERSION}"].astype(bool).any()
    feasible = bool(budget_feasible and cvar_feasible and source_feasible)

    action = pd.DataFrame(
        [
            {
                "policy_id": str(best["policy_id"]),
                f"regime_v{VERSION}": "bounded_two_swap_after_v260",
                f"primary_added_loan_id_v{VERSION}": add_ids[0],
                f"primary_dropped_loan_id_v{VERSION}": drop_ids[0],
                f"relief_added_loan_id_v{VERSION}": add_ids[1],
                f"relief_dropped_loan_id_v{VERSION}": drop_ids[1],
                f"primary_return_delta_v{VERSION}": float(best["primary_return_delta_v262"]),
                f"relief_return_delta_v{VERSION}": float(best["relief_return_delta_v262"]),
                f"total_return_delta_v{VERSION}": float(best["total_return_delta_v262"]),
                f"cvar90_after_repair_v{VERSION}": cvar90,
                f"exposure_after_repair_v{VERSION}": exposure,
                f"source_cap_violations_after_repair_v{VERSION}": int(
                    source_summary[f"source_cap_violated_v{VERSION}"].sum()
                ),
                f"claim_boundary_v{VERSION}": (
                    "best v262 bounded two-swap applied; not post-repair optimality"
                ),
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                f"portfolio_label_v{VERSION}": "bounded_two_swap_repair_candidate",
                f"selected_rows_v{VERSION}": int(len(repaired)),
                f"portfolio_exposure_v{VERSION}": exposure,
                f"objective_return_v{VERSION}": objective_return,
                f"scenario_loss_mean_v{VERSION}": float(scenario_losses.mean()),
                f"scenario_loss_cvar90_v{VERSION}": cvar90,
                f"source_cap_violations_v{VERSION}": int(
                    source_summary[f"source_cap_violated_v{VERSION}"].sum()
                ),
                f"max_source_share_v{VERSION}": float(
                    source_summary[f"source_share_v{VERSION}"].max()
                ),
                f"min_source_slack_v{VERSION}": float(
                    source_summary[f"source_slack_v{VERSION}"].min()
                ),
                f"delta_return_vs_v{PREVIOUS_REPAIR_VERSION}_v{VERSION}": objective_return
                - float(previous_row["objective_return_v260"]),
                f"delta_cvar90_vs_v{PREVIOUS_REPAIR_VERSION}_v{VERSION}": cvar90
                - float(previous_row["scenario_loss_cvar90_v260"]),
                f"delta_exposure_vs_v{PREVIOUS_REPAIR_VERSION}_v{VERSION}": exposure
                - float(previous_row["portfolio_exposure_v260"]),
                f"exposure_min_v{VERSION}": float(previous_row["exposure_min_v260"]),
                f"exposure_max_v{VERSION}": float(previous_row["exposure_max_v260"]),
                f"cvar_cap_v{VERSION}": float(previous_row["cvar_cap_v260"]),
                f"budget_feasible_v{VERSION}": budget_feasible,
                f"cvar_feasible_v{VERSION}": cvar_feasible,
                f"source_feasible_v{VERSION}": source_feasible,
                f"repair_candidate_feasible_v{VERSION}": feasible,
                f"post_repair_pricing_required_v{VERSION}": True,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": (
                    "bounded two-swap repair candidate; must rerun post-repair pricing"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "bounded_two_swap_repair_candidate_created",
                f"blocking_v{VERSION}": not feasible,
                f"evidence_count_v{VERSION}": int(feasible),
                f"required_next_artifact_v{VERSION}": (
                    "paper4_v263_bounded_two_swap_repair_summary.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "candidate exists only if budget/source/CVaR remain feasible"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "post_repair_pricing_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    "paper4_v264_post_bounded_two_swap_reprice.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "repair must be re-priced after changing the portfolio again"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "global_integer_gap_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "paper4_v264_global_integer_gap_protocol.csv",
                f"claim_boundary_v{VERSION}": (
                    "no branch-and-price/global full-universe integer certificate"
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
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v263_bounded_two_swap_repair_executed",
                "allowed": True,
                "artifact": "paper4_v263_bounded_two_swap_repair_summary.csv",
                "boundary": "best v262 bounded two-swap applied",
            },
            {
                "claim_id": "v263_repair_candidate_feasible",
                "allowed": feasible,
                "artifact": "paper4_v263_bounded_two_swap_repair_summary.csv",
                "boundary": "budget/source/CVaR feasibility only",
            },
            {
                "claim_id": "v263_post_repair_local_optimality",
                "allowed": False,
                "artifact": "paper4_v263_claim_blockers.csv",
                "boundary": "post-repair pricing not rerun after v263",
            },
            {
                "claim_id": "v263_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v263_claim_blockers.csv",
                "boundary": "global gap certificate missing",
            },
            {
                "claim_id": "v263_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v263_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    repaired.to_parquet(
        TABLE_DIR / "paper4_v263_bounded_two_swap_repair_allocations.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v263_bounded_two_swap_repair_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v263_bounded_two_swap_repair_action.csv", action)
    write_csv(
        TABLE_DIR / "paper4_v263_bounded_two_swap_repair_source_summary.csv",
        source_summary,
    )
    write_csv(TABLE_DIR / "paper4_v263_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v263_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    summary_row = summary.iloc[0]
    status = {
        "phase": "v263_bounded_two_swap_repair",
        "schema_version": "2026-05-15.263",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "allocation_rows_v263": int(len(repaired)),
        "summary_rows_v263": int(len(summary)),
        "action_rows_v263": int(len(action)),
        "source_summary_rows_v263": int(len(source_summary)),
        "claim_blocker_rows_v263": int(len(blockers)),
        "claim_matrix_rows_v263": int(len(claim_matrix)),
        "primary_added_loan_id_v263": add_ids[0],
        "primary_dropped_loan_id_v263": drop_ids[0],
        "relief_added_loan_id_v263": add_ids[1],
        "relief_dropped_loan_id_v263": drop_ids[1],
        "selected_rows_v263": int(summary_row["selected_rows_v263"]),
        "portfolio_exposure_v263": float(summary_row["portfolio_exposure_v263"]),
        "objective_return_v263": float(summary_row["objective_return_v263"]),
        "scenario_loss_cvar90_v263": float(summary_row["scenario_loss_cvar90_v263"]),
        "source_cap_violations_v263": int(summary_row["source_cap_violations_v263"]),
        "delta_return_vs_v260_v263": float(summary_row["delta_return_vs_v260_v263"]),
        "delta_cvar90_vs_v260_v263": float(summary_row["delta_cvar90_vs_v260_v263"]),
        "delta_exposure_vs_v260_v263": float(summary_row["delta_exposure_vs_v260_v263"]),
        "budget_feasible_v263": bool(summary_row["budget_feasible_v263"]),
        "source_feasible_v263": bool(summary_row["source_feasible_v263"]),
        "cvar_feasible_v263": bool(summary_row["cvar_feasible_v263"]),
        "repair_candidate_feasible_v263": bool(summary_row["repair_candidate_feasible_v263"]),
        "post_repair_local_optimality_claim_allowed_v263": False,
        "full_universe_integer_optimality_claim_allowed_v263": False,
        "paper1_promotion_allowed_v263": False,
        "paper4_working_champion_changed_v263": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v263 creates a bounded two-swap repaired candidate only; post-repair "
            "pricing and global integer claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v263_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v263": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
