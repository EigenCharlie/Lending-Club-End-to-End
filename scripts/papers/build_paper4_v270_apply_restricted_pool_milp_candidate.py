#!/usr/bin/env python3
"""Build Paper 4 v270 restricted-pool MILP repair candidate artifacts."""

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

VERSION = 270
BASE_REPAIR_VERSION = 266
MILP_PROBE_VERSION = 269
NEXT_REPRICE_VERSION = 271


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v270 restricted-pool MILP repair candidate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v270_restricted_pool_milp_repair_summary.csv"
                ),
                "boundary": (
                    "Applies the v269 restricted-pool MILP incumbent only; requires "
                    "post-repair repricing."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v270 repaired portfolio is post-repair locally optimal.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v270_claim_blockers.csv"
                ),
                "boundary": "Post-repair pricing has not been rerun after v270.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v270 proves full-universe global integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v270_claim_blockers.csv"
                ),
                "boundary": "v269/v270 are restricted-pool artifacts; no global gap certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v270 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v270_claim_blockers.csv"
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
                    "v270 applies the v269 restricted-pool MILP incumbent as a repair "
                    "candidate and records the resulting portfolio metrics."
                ),
                "status": "post_restricted_pool_milp_repair_pricing_required",
                "next_artifact": "paper4_v271_post_restricted_pool_milp_reprice.csv",
                "success_condition": (
                    "post-v270 one-swap/two-swap pricing finds no feasible improving exchanges"
                ),
                "last_wave": "v270",
                "execution_result": "restricted_pool_milp_repair_candidate_created",
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
    start = "<!-- V270_RESTRICTED_POOL_MILP_REPAIR_START -->"
    end = "<!-- V270_RESTRICTED_POOL_MILP_REPAIR_END -->"
    block = f"""
{start}

## Wave v270: Restricted-Pool MILP Repair Candidate

Generated: {status["generated_at_utc"]}

### Objective

Apply the v269 restricted-pool MILP incumbent as a candidate portfolio and
recompute exposure, return, source and CVaR metrics. This converts a restricted
search signal into a concrete repair candidate for the next repricing loop.

### Results

- Added loans: `{status["added_rows_v270"]}`.
- Dropped loans: `{status["dropped_rows_v270"]}`.
- Selected rows after repair: `{status["selected_rows_v270"]}`.
- Return delta vs v266:
  `{status["delta_return_vs_v266_v270"]}`.
- CVaR90 delta vs v266:
  `{status["delta_cvar90_vs_v266_v270"]}`.
- Budget feasible: `{status["budget_feasible_v270"]}`.
- Source feasible: `{status["source_feasible_v270"]}`.
- CVaR feasible: `{status["cvar_feasible_v270"]}`.

### Interpretation

v270 creates a feasible restricted-pool MILP repair candidate. It does not make
the portfolio locally or globally optimal; the next required step is post-v270
repricing and then broader/global validation.

### Claim Impact

- Allowed: restricted-pool MILP repair candidate created.
- Still prohibited: post-repair local optimality, full-universe global integer
  optimality, Paper Estrella replacement, final Paper 4 promotion and live
  deployment.

### Quarto Promotion Decision

Keep v270 in the living notebook. Promote only after post-repair pricing,
full-universe/global, dynamic validation and promotion gates pass.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    milp_pool = read_parquet("paper4_v269_restricted_pool_milp_allocations.parquet")
    base_summary = read_csv("paper4_v266_bounded_two_swap_repair_summary.csv")
    base_row = base_summary.iloc[0]
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    if universe.empty or milp_pool.empty or source_caps.empty:
        raise RuntimeError("Missing v55 universe, v269 MILP pool, or source caps.")

    selected = milp_pool.loc[milp_pool["selected_v269"].eq(1)].copy()
    if selected.empty:
        raise RuntimeError("v269 did not produce a selected restricted-pool incumbent.")
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    scenario_losses = losses[:, selected_idx].sum(axis=1)
    selected[f"mean_return_v{VERSION}"] = mean_returns[selected_idx]
    selected[f"selected_v{VERSION}"] = 1
    selected[f"portfolio_label_v{VERSION}"] = "restricted_pool_milp_repair_candidate"
    selected[f"repair_action_v{VERSION}"] = selected["milp_action_v269"]
    selected[f"claim_boundary_v{VERSION}"] = (
        "restricted-pool MILP repair candidate only; requires post-repair repricing"
    )

    selected_out = selected[
        [
            "loan_id",
            "loan_amnt",
            *FAMILIES,
            f"mean_return_v{VERSION}",
            f"selected_v{VERSION}",
            f"portfolio_label_v{VERSION}",
            f"repair_action_v{VERSION}",
            f"claim_boundary_v{VERSION}",
        ]
    ].copy()
    source_summary = _source_summary(
        universe=universe,
        portfolio=selected_out,
        source_caps=source_caps,
        version=VERSION,
        ordinal="restricted-pool MILP repair",
    )
    source_summary[f"portfolio_label_v{VERSION}"] = "restricted_pool_milp_repair_candidate"
    source_summary[f"claim_boundary_v{VERSION}"] = (
        "restricted-pool MILP repair source diagnostic only"
    )

    objective_return = float(selected_out[f"mean_return_v{VERSION}"].sum())
    exposure = float(selected_out["loan_amnt"].sum())
    cvar90 = v70._tail_cvar(scenario_losses)
    source_violations = int(source_summary[f"source_cap_violated_v{VERSION}"].sum())
    budget_feasible = (
        exposure >= float(base_row[f"exposure_min_v{BASE_REPAIR_VERSION}"]) - 1e-7
        and exposure <= float(base_row[f"exposure_max_v{BASE_REPAIR_VERSION}"]) + 1e-7
    )
    cvar_feasible = cvar90 <= float(base_row[f"cvar_cap_v{BASE_REPAIR_VERSION}"]) + 1e-7
    source_feasible = source_violations == 0
    feasible = bool(budget_feasible and cvar_feasible and source_feasible)
    added = selected.loc[selected["milp_action_v269"].eq("added_by_restricted_pool_milp")].copy()
    dropped = milp_pool.loc[
        milp_pool["milp_action_v269"].eq("dropped_by_restricted_pool_milp")
    ].copy()

    action = pd.concat(
        [
            added.assign(action_v270="added_by_restricted_pool_milp"),
            dropped.assign(action_v270="dropped_by_restricted_pool_milp"),
        ],
        ignore_index=True,
    )
    action = action[["action_v270", "loan_id", "loan_amnt", "mean_return_v269", *FAMILIES]].rename(
        columns={"mean_return_v269": "mean_return_v270"}
    )
    action[f"claim_boundary_v{VERSION}"] = (
        "restricted-pool MILP repair action list; not full-universe proof"
    )
    summary = pd.DataFrame(
        [
            {
                f"portfolio_label_v{VERSION}": "restricted_pool_milp_repair_candidate",
                f"source_probe_version_v{VERSION}": MILP_PROBE_VERSION,
                f"selected_rows_v{VERSION}": int(len(selected_out)),
                f"added_rows_v{VERSION}": int(len(added)),
                f"dropped_rows_v{VERSION}": int(len(dropped)),
                f"portfolio_exposure_v{VERSION}": exposure,
                f"objective_return_v{VERSION}": objective_return,
                f"scenario_loss_mean_v{VERSION}": float(scenario_losses.mean()),
                f"scenario_loss_cvar90_v{VERSION}": cvar90,
                f"source_cap_violations_v{VERSION}": source_violations,
                f"max_source_share_v{VERSION}": float(
                    source_summary[f"source_share_v{VERSION}"].max()
                ),
                f"min_source_slack_v{VERSION}": float(
                    source_summary[f"source_slack_v{VERSION}"].min()
                ),
                f"delta_return_vs_v{BASE_REPAIR_VERSION}_v{VERSION}": objective_return
                - float(base_row[f"objective_return_v{BASE_REPAIR_VERSION}"]),
                f"delta_cvar90_vs_v{BASE_REPAIR_VERSION}_v{VERSION}": cvar90
                - float(base_row[f"scenario_loss_cvar90_v{BASE_REPAIR_VERSION}"]),
                f"delta_exposure_vs_v{BASE_REPAIR_VERSION}_v{VERSION}": exposure
                - float(base_row[f"portfolio_exposure_v{BASE_REPAIR_VERSION}"]),
                f"exposure_min_v{VERSION}": float(base_row[f"exposure_min_v{BASE_REPAIR_VERSION}"]),
                f"exposure_max_v{VERSION}": float(base_row[f"exposure_max_v{BASE_REPAIR_VERSION}"]),
                f"cvar_cap_v{VERSION}": float(base_row[f"cvar_cap_v{BASE_REPAIR_VERSION}"]),
                f"budget_feasible_v{VERSION}": budget_feasible,
                f"cvar_feasible_v{VERSION}": cvar_feasible,
                f"source_feasible_v{VERSION}": source_feasible,
                f"repair_candidate_feasible_v{VERSION}": feasible,
                f"post_repair_pricing_required_v{VERSION}": True,
                f"restricted_pool_global_optimality_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"claim_boundary_v{VERSION}": (
                    "restricted-pool MILP repair candidate; must rerun post-repair pricing"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "restricted_pool_milp_repair_candidate_created",
                f"blocking_v{VERSION}": not feasible,
                f"evidence_count_v{VERSION}": int(feasible),
                f"required_next_artifact_v{VERSION}": (
                    "paper4_v270_restricted_pool_milp_repair_summary.csv"
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
                    "paper4_v271_post_restricted_pool_milp_reprice.csv"
                ),
                f"claim_boundary_v{VERSION}": "repair must be re-priced after v270",
            },
            {
                f"blocker_id_v{VERSION}": "global_integer_gap_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    "paper4_v271_full_universe_gap_certificate_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "restricted-pool repair is not a full-universe gap certificate"
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
                "claim_id": "v270_restricted_pool_milp_repair_executed",
                "allowed": True,
                "artifact": "paper4_v270_restricted_pool_milp_repair_summary.csv",
                "boundary": "v269 restricted-pool incumbent applied",
            },
            {
                "claim_id": "v270_repair_candidate_feasible",
                "allowed": feasible,
                "artifact": "paper4_v270_restricted_pool_milp_repair_summary.csv",
                "boundary": "budget/source/CVaR feasibility only",
            },
            {
                "claim_id": "v270_post_repair_local_optimality",
                "allowed": False,
                "artifact": "paper4_v270_claim_blockers.csv",
                "boundary": "post-repair pricing missing",
            },
            {
                "claim_id": "v270_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v270_claim_blockers.csv",
                "boundary": "full-universe gap certificate missing",
            },
            {
                "claim_id": "v270_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v270_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    selected_out.to_parquet(
        TABLE_DIR / "paper4_v270_restricted_pool_milp_repair_allocations.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v270_restricted_pool_milp_repair_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v270_restricted_pool_milp_repair_action.csv", action)
    write_csv(
        TABLE_DIR / "paper4_v270_restricted_pool_milp_repair_source_summary.csv",
        source_summary,
    )
    write_csv(TABLE_DIR / "paper4_v270_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v270_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    summary_row = summary.iloc[0]
    status = {
        "phase": "v270_restricted_pool_milp_repair",
        "schema_version": "2026-05-15.270",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "source_probe_version_v270": MILP_PROBE_VERSION,
        "allocation_rows_v270": int(len(selected_out)),
        "summary_rows_v270": int(len(summary)),
        "action_rows_v270": int(len(action)),
        "source_summary_rows_v270": int(len(source_summary)),
        "claim_blocker_rows_v270": int(len(blockers)),
        "claim_matrix_rows_v270": int(len(claim_matrix)),
        "selected_rows_v270": int(summary_row["selected_rows_v270"]),
        "added_rows_v270": int(summary_row["added_rows_v270"]),
        "dropped_rows_v270": int(summary_row["dropped_rows_v270"]),
        "portfolio_exposure_v270": float(summary_row["portfolio_exposure_v270"]),
        "objective_return_v270": float(summary_row["objective_return_v270"]),
        "scenario_loss_cvar90_v270": float(summary_row["scenario_loss_cvar90_v270"]),
        "source_cap_violations_v270": int(summary_row["source_cap_violations_v270"]),
        "delta_return_vs_v266_v270": float(summary_row["delta_return_vs_v266_v270"]),
        "delta_cvar90_vs_v266_v270": float(summary_row["delta_cvar90_vs_v266_v270"]),
        "delta_exposure_vs_v266_v270": float(summary_row["delta_exposure_vs_v266_v270"]),
        "budget_feasible_v270": bool(summary_row["budget_feasible_v270"]),
        "source_feasible_v270": bool(summary_row["source_feasible_v270"]),
        "cvar_feasible_v270": bool(summary_row["cvar_feasible_v270"]),
        "repair_candidate_feasible_v270": bool(summary_row["repair_candidate_feasible_v270"]),
        "post_repair_local_optimality_claim_allowed_v270": False,
        "restricted_pool_global_optimality_claim_allowed_v270": False,
        "full_universe_integer_optimality_claim_allowed_v270": False,
        "paper1_promotion_allowed_v270": False,
        "paper4_working_champion_changed_v270": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v270 creates a restricted-pool MILP repair candidate only; post-repair "
            "pricing and global integer claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v270_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v270": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
