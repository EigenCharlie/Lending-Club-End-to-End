#!/usr/bin/env python3
"""Build Paper 4 v260 one-swap repair after the v258 bounded two-swap."""

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

VERSION = 260
PREVIOUS_REPAIR_VERSION = 258
PRICING_VERSION = 259
NEXT_REPRICE_VERSION = 261


def _best_one_swap() -> pd.Series:
    top = read_csv("paper4_v259_post_bounded_two_swap_one_swap_top_candidates.csv")
    if top.empty:
        raise RuntimeError("Missing v259 post-bounded-two-swap top candidates.")
    return top.sort_values("return_delta_v259", ascending=False).iloc[0]


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v260 one-swap repair after the v258 two-swap.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v260_next_one_swap_repair_summary.csv"
                ),
                "boundary": (
                    "Applies the best v259 post-v258 one-swap; requires post-repair repricing."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v260 repaired portfolio is post-repair locally optimal.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v260_claim_blockers.csv"
                ),
                "boundary": "Post-repair pricing has not been rerun after v260.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v260 proves multi-swap or global full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v260_claim_blockers.csv"
                ),
                "boundary": "Requires broader multi-swap search or a global gap certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v260 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v260_claim_blockers.csv"
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
                    "v260 applies the best v259 post-v258 one-swap repair and "
                    "recomputes portfolio metrics."
                ),
                "status": "post_repair_pricing_required",
                "next_artifact": "paper4_v261_post_repair_one_swap_reprice.csv",
                "success_condition": (
                    "post-v260 one-swap pricing finds no feasible improving exchanges"
                ),
                "last_wave": "v260",
                "execution_result": "post_bounded_two_swap_one_swap_repair_candidate_created",
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
    start = "<!-- V260_NEXT_ONE_SWAP_AFTER_BOUNDED_TWO_SWAP_REPAIR_START -->"
    end = "<!-- V260_NEXT_ONE_SWAP_AFTER_BOUNDED_TWO_SWAP_REPAIR_END -->"
    block = f"""
{start}

## Wave v260: One-Swap Repair After the v258 Bounded Two-Swap

Generated: {status["generated_at_utc"]}

### Objective

Apply the best v259 post-v258 one-drop/one-add swap and recompute portfolio
return, exposure, source and CVaR metrics. This continues the local repair loop
after v259 showed that the v258 bounded two-swap candidate was not one-swap
locally optimal.

### Results

- Added loan: `{status["added_loan_id_v260"]}`.
- Dropped loan: `{status["dropped_loan_id_v260"]}`.
- Selected rows after repair: `{status["selected_rows_v260"]}`.
- Return delta vs v258:
  `{status["delta_return_vs_v258_v260"]}`.
- CVaR90 delta vs v258:
  `{status["delta_cvar90_vs_v258_v260"]}`.
- Budget feasible: `{status["budget_feasible_v260"]}`.
- Source feasible: `{status["source_feasible_v260"]}`.
- CVaR feasible: `{status["cvar_feasible_v260"]}`.

### Interpretation

v260 converts the best v259 signal into a feasible repaired portfolio
candidate. The next required experiment is v261 post-repair repricing; no
local/global optimality or promotion claim is enabled.

### Claim Impact

- Allowed: one-swap repair after the v258 bounded two-swap created.
- Still prohibited: post-repair local optimality, broader multi-swap/global
  integer optimality, Paper Estrella replacement, final Paper 4 promotion and
  live deployment.

### Quarto Promotion Decision

Keep v260 in the living notebook. Promote only after post-repair pricing and
broader integer/dynamic/promotion gates pass.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    previous = read_parquet("paper4_v258_bounded_two_swap_repair_allocations.parquet")
    previous_summary = read_csv("paper4_v258_bounded_two_swap_repair_summary.csv")
    previous_row = previous_summary.iloc[0]
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    best = _best_one_swap()
    if universe.empty or previous.empty:
        raise RuntimeError("Missing v55 universe or v258 repaired portfolio.")

    add_id = str(best["added_loan_id_v259"])
    drop_id = str(best["dropped_loan_id_v259"])
    add_row = universe.loc[universe["loan_id"].astype(str).eq(add_id)].head(1).copy()
    if add_row.empty:
        raise RuntimeError(f"Could not find added loan {add_id} in v55 universe.")

    repaired = previous.loc[~previous["loan_id"].astype(str).eq(drop_id)].copy()
    if len(repaired) != len(previous) - 1:
        raise RuntimeError(f"Could not drop selected loan {drop_id} from v258 portfolio.")
    keep_cols = ["loan_id", "loan_amnt", *FAMILIES]
    repaired = pd.concat([repaired[keep_cols], add_row[keep_cols]], ignore_index=True)
    repaired["loan_id"] = repaired["loan_id"].astype(str)

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    repaired_idx = idx_by_id.loc[repaired["loan_id"].astype(str)].to_numpy()
    scenario_losses = losses[:, repaired_idx].sum(axis=1)
    repaired[f"mean_return_v{VERSION}"] = mean_returns[repaired_idx]
    repaired[f"selected_v{VERSION}"] = 1
    repaired[f"portfolio_label_v{VERSION}"] = "next_one_swap_repair_candidate"
    repaired[f"repair_action_v{VERSION}"] = np.where(
        repaired["loan_id"].eq(add_id),
        "added_from_v259_best_post_bounded_two_swap_one_swap",
        "kept_from_v258",
    )
    repaired[f"claim_boundary_v{VERSION}"] = (
        "post-bounded-two-swap one-swap repair candidate only; requires post-repair repricing"
    )

    source_summary = _source_summary(
        universe=universe,
        portfolio=repaired,
        source_caps=source_caps,
        version=VERSION,
        ordinal="post-bounded-two-swap",
    )
    source_summary[f"portfolio_label_v{VERSION}"] = "next_one_swap_repair_candidate"
    source_summary[f"claim_boundary_v{VERSION}"] = (
        "post-bounded-two-swap one-swap source diagnostic only"
    )
    objective_return = float(repaired[f"mean_return_v{VERSION}"].sum())
    exposure = float(repaired["loan_amnt"].sum())
    cvar90 = v70._tail_cvar(scenario_losses)
    budget_feasible = exposure >= float(previous_row["exposure_min_v258"]) - 1e-7
    budget_feasible = (
        budget_feasible and exposure <= float(previous_row["exposure_max_v258"]) + 1e-7
    )
    cvar_feasible = cvar90 <= float(previous_row["cvar_cap_v258"]) + 1e-7
    source_feasible = not source_summary[f"source_cap_violated_v{VERSION}"].astype(bool).any()
    feasible = bool(budget_feasible and cvar_feasible and source_feasible)

    action = pd.DataFrame(
        [
            {
                "policy_id": str(best["policy_id"]),
                f"regime_v{VERSION}": "post_v258_bounded_two_swap_one_swap_repair",
                f"added_loan_id_v{VERSION}": add_id,
                f"dropped_loan_id_v{VERSION}": drop_id,
                f"added_loan_amount_v{VERSION}": float(best["added_loan_amount_v259"]),
                f"dropped_loan_amount_v{VERSION}": float(best["dropped_loan_amount_v259"]),
                f"added_mean_return_v{VERSION}": float(best["added_mean_return_v259"]),
                f"dropped_mean_return_v{VERSION}": float(best["dropped_mean_return_v259"]),
                f"return_delta_v{VERSION}": float(best["return_delta_v259"]),
                f"cvar90_after_repair_v{VERSION}": cvar90,
                f"exposure_after_repair_v{VERSION}": exposure,
                f"source_cap_violations_after_repair_v{VERSION}": int(
                    source_summary[f"source_cap_violated_v{VERSION}"].sum()
                ),
                f"claim_boundary_v{VERSION}": (
                    "best v259 one-swap applied; not post-repair optimality"
                ),
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                f"portfolio_label_v{VERSION}": "next_one_swap_repair_candidate",
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
                - float(previous_row["objective_return_v258"]),
                f"delta_cvar90_vs_v{PREVIOUS_REPAIR_VERSION}_v{VERSION}": cvar90
                - float(previous_row["scenario_loss_cvar90_v258"]),
                f"delta_exposure_vs_v{PREVIOUS_REPAIR_VERSION}_v{VERSION}": exposure
                - float(previous_row["portfolio_exposure_v258"]),
                f"exposure_min_v{VERSION}": float(previous_row["exposure_min_v258"]),
                f"exposure_max_v{VERSION}": float(previous_row["exposure_max_v258"]),
                f"cvar_cap_v{VERSION}": float(previous_row["cvar_cap_v258"]),
                f"budget_feasible_v{VERSION}": budget_feasible,
                f"cvar_feasible_v{VERSION}": cvar_feasible,
                f"source_feasible_v{VERSION}": source_feasible,
                f"repair_candidate_feasible_v{VERSION}": feasible,
                f"post_repair_pricing_required_v{VERSION}": True,
                f"post_repair_one_swap_optimality_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": (
                    "post-bounded-two-swap one-swap repair candidate; "
                    "must rerun post-repair pricing"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "post_bounded_two_swap_one_swap_repair_created",
                f"blocking_v{VERSION}": not feasible,
                f"evidence_count_v{VERSION}": int(feasible),
                f"required_next_artifact_v{VERSION}": (
                    "paper4_v260_next_one_swap_repair_summary.csv"
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
                    "paper4_v261_post_repair_one_swap_reprice.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "repair must be re-priced after changing the portfolio again"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "broader_multi_swap_search_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    "paper4_v261_or_v262_broader_multi_swap_search.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "one-swap after bounded two-swap does not exhaust multi-loan exchanges"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "global_integer_gap_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    "paper4_v261_global_integer_gap_protocol.csv"
                ),
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
                "claim_id": "v260_post_bounded_two_swap_one_swap_repair_executed",
                "allowed": True,
                "artifact": "paper4_v260_next_one_swap_repair_summary.csv",
                "boundary": "best v259 post-v258 one-swap applied",
            },
            {
                "claim_id": "v260_repair_candidate_feasible",
                "allowed": feasible,
                "artifact": "paper4_v260_next_one_swap_repair_summary.csv",
                "boundary": "budget/source/CVaR feasibility only",
            },
            {
                "claim_id": "v260_post_repair_local_optimality",
                "allowed": False,
                "artifact": "paper4_v260_claim_blockers.csv",
                "boundary": "post-repair pricing not rerun after v260",
            },
            {
                "claim_id": "v260_multi_swap_or_global_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v260_claim_blockers.csv",
                "boundary": "broader multi-swap/global gap evidence missing",
            },
            {
                "claim_id": "v260_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v260_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    repaired.to_parquet(
        TABLE_DIR / "paper4_v260_next_one_swap_repair_allocations.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v260_next_one_swap_repair_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v260_next_one_swap_repair_action.csv", action)
    write_csv(
        TABLE_DIR / "paper4_v260_next_one_swap_repair_source_summary.csv",
        source_summary,
    )
    write_csv(TABLE_DIR / "paper4_v260_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v260_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = summary.iloc[0]
    action_row = action.iloc[0]
    status = {
        "phase": "v260_next_one_swap_after_bounded_two_swap_repair",
        "schema_version": "2026-05-15.260",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "allocation_rows_v260": int(len(repaired)),
        "summary_rows_v260": int(len(summary)),
        "action_rows_v260": int(len(action)),
        "source_summary_rows_v260": int(len(source_summary)),
        "claim_blocker_rows_v260": int(len(blockers)),
        "claim_matrix_rows_v260": int(len(claim_matrix)),
        "added_loan_id_v260": str(action_row["added_loan_id_v260"]),
        "dropped_loan_id_v260": str(action_row["dropped_loan_id_v260"]),
        "selected_rows_v260": int(row["selected_rows_v260"]),
        "portfolio_exposure_v260": float(row["portfolio_exposure_v260"]),
        "objective_return_v260": float(row["objective_return_v260"]),
        "scenario_loss_cvar90_v260": float(row["scenario_loss_cvar90_v260"]),
        "source_cap_violations_v260": int(row["source_cap_violations_v260"]),
        "delta_return_vs_v258_v260": float(row["delta_return_vs_v258_v260"]),
        "delta_cvar90_vs_v258_v260": float(row["delta_cvar90_vs_v258_v260"]),
        "delta_exposure_vs_v258_v260": float(row["delta_exposure_vs_v258_v260"]),
        "budget_feasible_v260": bool(row["budget_feasible_v260"]),
        "source_feasible_v260": bool(row["source_feasible_v260"]),
        "cvar_feasible_v260": bool(row["cvar_feasible_v260"]),
        "repair_candidate_feasible_v260": bool(row["repair_candidate_feasible_v260"]),
        "post_repair_one_swap_optimality_claim_allowed_v260": False,
        "full_universe_integer_optimality_claim_allowed_v260": False,
        "paper1_promotion_allowed_v260": False,
        "paper4_working_champion_changed_v260": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v260 creates a post-bounded-two-swap one-swap repaired candidate only; "
            "post-repair pricing and global integer claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v260_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v260": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
