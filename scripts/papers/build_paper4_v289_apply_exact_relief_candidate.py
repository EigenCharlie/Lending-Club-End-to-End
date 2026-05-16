#!/usr/bin/env python3
"""Build Paper 4 v289 exact-relief repair application artifacts."""

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
    _source_summary,
    now,
    read_csv,
    read_parquet,
    write_csv,
    write_json,
)

VERSION = 289
SIGNAL_VERSION = 288
BASE_REPAIR_VERSION = 279
NEXT_VERSION = 290


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v289 applied exact-relief repair candidate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v289_exact_relief_repair_summary.csv"
                ),
                "boundary": (
                    "Applied lab repair candidate only; cardinality changes and "
                    "post-application repricing are still required."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v289 is a new Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v289_claim_blockers.csv"
                ),
                "boundary": (
                    "Repair improves objective slightly but changes cardinality and has "
                    "not survived post-repair pricing."
                ),
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v289 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v289_claim_blockers.csv"
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
                    "v289 applies the v288 exact-relief repair signal and audits "
                    "return, exposure, CVaR, source caps and cardinality."
                ),
                "status": "exact_relief_repair_candidate_applied_requires_repricing",
                "next_artifact": "paper4_v290_post_exact_relief_reprice.csv",
                "success_condition": (
                    "post-v289 repricing finds no feasible improving local or source-tight "
                    "columns, or records the next repair signal"
                ),
                "last_wave": "v289",
                "execution_result": "small_return_gain_with_cardinality_change",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v289")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V289_APPLY_EXACT_RELIEF_REPAIR_START -->"
    end = "<!-- V289_APPLY_EXACT_RELIEF_REPAIR_END -->"
    block = f"""
{start}

## Wave v289: Apply Exact-Relief Repair Candidate

Generated: {status["generated_at_utc"]}

### Objective

Apply the v288 exact-relief signal to the v279 repaired portfolio and audit the
resulting candidate. This is an application step, not a promotion step: the
repair drops four selected loans, adds one source-tight candidate, and therefore
changes cardinality.

### Results

- Added loan: `{status["added_loan_id_v289"]}`.
- Dropped rows: `{status["dropped_rows_v289"]}`.
- Selected rows after repair: `{status["selected_rows_v289"]}`.
- Objective return: `{status["objective_return_v289"]}`.
- Delta return vs v279: `{status["delta_return_vs_v279_v289"]}`.
- CVaR90 after repair: `{status["scenario_loss_cvar90_v289"]}`.
- Delta CVaR90 vs v279: `{status["delta_cvar90_vs_v279_v289"]}`.
- Source cap violations: `{status["source_cap_violations_v289"]}`.
- Cardinality changed: `{status["cardinality_changed_v289"]}`.
- Post-repair pricing required: `{status["post_repair_pricing_required_v289"]}`.

### Interpretation

v289 confirms that the v288 signal is mechanically feasible and slightly
improves return while lowering CVaR. The tradeoff is structural: the repair
uses a 4-drop/1-add move, reducing selected rows from 171 to 168. That makes it
a living-lab repair candidate requiring repricing and policy review, not a new
champion.

### Claim Impact

- Allowed: applied exact-relief repair candidate with audited metrics.
- Still prohibited: working champion replacement, full-universe optimality,
  Paper Estrella replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v289 in the living notebook. The next step is post-repair repricing.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v279_restricted_pool_milp_repair_allocations.parquet")
    base_summary = read_csv("paper4_v279_restricted_pool_milp_repair_summary.csv")
    v288_screen = read_csv("paper4_v288_exact_relief_candidate_screen.csv")
    v288_drops = read_csv("paper4_v288_exact_relief_drop_bundles.csv")
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    if universe.empty or selected.empty or base_summary.empty or v288_screen.empty:
        raise RuntimeError("Missing v55, v279, or v288 inputs for v289.")

    signal = v288_screen.loc[
        v288_screen[f"exact_relief_entering_column_v{SIGNAL_VERSION}"].astype(bool)
    ].sort_values(f"return_delta_after_exact_relief_v{SIGNAL_VERSION}", ascending=False)
    if signal.empty:
        raise RuntimeError("v289 requires a positive v288 exact-relief signal.")
    signal_row = signal.iloc[0]
    candidate_rank = int(signal_row[f"candidate_rank_v{SIGNAL_VERSION}"])
    add_id = str(signal_row[f"added_loan_id_v{SIGNAL_VERSION}"])
    drop_ids = set(
        v288_drops.loc[
            v288_drops[f"candidate_rank_v{SIGNAL_VERSION}"].eq(candidate_rank),
            f"dropped_loan_id_v{SIGNAL_VERSION}",
        ].astype(str)
    )
    if not drop_ids:
        raise RuntimeError("Could not find v288 drop bundle for v289 signal.")

    keep_cols = ["loan_id", "loan_amnt", *FAMILIES]
    add_row = universe.loc[universe["loan_id"].astype(str).eq(add_id), keep_cols].copy()
    if add_row.empty:
        raise RuntimeError(f"Could not find added loan {add_id} in v55 universe.")
    repaired = selected.loc[~selected["loan_id"].astype(str).isin(drop_ids), keep_cols].copy()
    repaired = pd.concat([repaired, add_row], ignore_index=True)
    repaired["loan_id"] = repaired["loan_id"].astype(str)

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    repaired_idx = idx_by_id.loc[repaired["loan_id"].astype(str)].to_numpy()
    scenario_losses = losses[:, repaired_idx].sum(axis=1)
    repaired[f"mean_return_v{VERSION}"] = mean_returns[repaired_idx]
    repaired[f"selected_v{VERSION}"] = 1
    repaired[f"portfolio_label_v{VERSION}"] = "exact_relief_repair_candidate"
    repaired[f"repair_action_v{VERSION}"] = np.where(
        repaired["loan_id"].eq(add_id),
        f"added_from_v{SIGNAL_VERSION}_exact_relief_signal",
        f"kept_from_v{BASE_REPAIR_VERSION}",
    )
    repaired[f"claim_boundary_v{VERSION}"] = (
        "applied exact-relief repair candidate only; requires post-repair repricing"
    )
    source_summary = _source_summary(
        universe=universe,
        portfolio=repaired,
        source_caps=source_caps,
        version=VERSION,
        ordinal="exact-relief",
    )

    base_row = base_summary.loc[
        base_summary[f"portfolio_label_v{BASE_REPAIR_VERSION}"].eq(
            "restricted_pool_milp_repair_candidate"
        )
    ].iloc[0]
    objective_return = float(repaired[f"mean_return_v{VERSION}"].sum())
    exposure = float(repaired["loan_amnt"].sum())
    cvar90 = v70._tail_cvar(scenario_losses)
    source_violations = int(source_summary[f"source_cap_violated_v{VERSION}"].sum())
    action = pd.DataFrame(
        [
            {
                f"candidate_rank_v{VERSION}": candidate_rank,
                f"added_loan_id_v{VERSION}": add_id,
                f"added_loan_amount_v{VERSION}": float(
                    signal_row[f"added_loan_amount_v{SIGNAL_VERSION}"]
                ),
                f"added_mean_return_v{VERSION}": float(
                    signal_row[f"added_mean_return_v{SIGNAL_VERSION}"]
                ),
                f"dropped_loan_ids_v{VERSION}": "|".join(sorted(drop_ids)),
                f"dropped_rows_v{VERSION}": int(len(drop_ids)),
                f"drop_exposure_v{VERSION}": float(signal_row[f"drop_exposure_v{SIGNAL_VERSION}"]),
                f"drop_mean_return_v{VERSION}": float(
                    signal_row[f"drop_mean_return_v{SIGNAL_VERSION}"]
                ),
                f"return_delta_v{VERSION}": float(
                    signal_row[f"return_delta_after_exact_relief_v{SIGNAL_VERSION}"]
                ),
                f"claim_boundary_v{VERSION}": (
                    "v288 exact relief signal applied; post-repair pricing still required"
                ),
            }
        ]
    )
    budget_feasible = exposure >= float(base_row[f"exposure_min_v{BASE_REPAIR_VERSION}"]) - 1e-7
    budget_feasible = (
        budget_feasible
        and exposure <= float(base_row[f"exposure_max_v{BASE_REPAIR_VERSION}"]) + 1e-7
    )
    cvar_feasible = cvar90 <= float(base_row[f"cvar_cap_v{BASE_REPAIR_VERSION}"]) + 1e-7
    source_feasible = source_violations == 0
    summary = pd.DataFrame(
        [
            {
                f"portfolio_label_v{VERSION}": "exact_relief_repair_candidate",
                f"signal_version_v{VERSION}": SIGNAL_VERSION,
                f"base_repair_version_v{VERSION}": BASE_REPAIR_VERSION,
                f"candidate_rank_v{VERSION}": candidate_rank,
                f"selected_rows_v{VERSION}": int(len(repaired)),
                f"base_selected_rows_v{VERSION}": int(
                    base_row[f"selected_rows_v{BASE_REPAIR_VERSION}"]
                ),
                f"added_rows_v{VERSION}": 1,
                f"dropped_rows_v{VERSION}": int(len(drop_ids)),
                f"cardinality_delta_vs_v{BASE_REPAIR_VERSION}_v{VERSION}": int(
                    len(repaired) - int(base_row[f"selected_rows_v{BASE_REPAIR_VERSION}"])
                ),
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
                f"budget_feasible_v{VERSION}": budget_feasible,
                f"cvar_feasible_v{VERSION}": cvar_feasible,
                f"source_feasible_v{VERSION}": source_feasible,
                f"repair_candidate_feasible_v{VERSION}": (
                    budget_feasible and cvar_feasible and source_feasible
                ),
                f"cardinality_changed_v{VERSION}": int(len(drop_ids)) != 1,
                f"post_repair_pricing_required_v{VERSION}": True,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": f"paper4_v{NEXT_VERSION}_post_exact_relief_reprice.csv",
                f"claim_boundary_v{VERSION}": (
                    "feasible applied repair candidate with cardinality change; repricing required"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "post_repair_pricing_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_post_exact_relief_reprice.csv"
                ),
                f"claim_boundary_v{VERSION}": "must rerun pricing after applying v289",
            },
            {
                f"blocker_id_v{VERSION}": "cardinality_policy_changed",
                f"blocking_v{VERSION}": int(len(drop_ids)) != 1,
                f"evidence_count_v{VERSION}": int(len(drop_ids)) - 1,
                f"required_next_artifact_v{VERSION}": "cardinality_policy_review_or_reprice",
                f"claim_boundary_v{VERSION}": "4-drop/1-add move changes selected-row count",
            },
            {
                f"blocker_id_v{VERSION}": "global_integer_gap_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_global_gap_or_branch_price_certificate",
                f"claim_boundary_v{VERSION}": "applied repair is not a global certificate",
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
                "claim_id": "v289_exact_relief_repair_applied",
                "allowed": True,
                "artifact": "paper4_v289_exact_relief_repair_summary.csv",
                "boundary": "applied repair candidate only",
            },
            {
                "claim_id": "v289_small_return_gain_confirmed",
                "allowed": objective_return
                > float(base_row[f"objective_return_v{BASE_REPAIR_VERSION}"]),
                "artifact": "paper4_v289_exact_relief_repair_summary.csv",
                "boundary": "lab metric, not champion claim",
            },
            {
                "claim_id": "v289_working_champion",
                "allowed": False,
                "artifact": "paper4_v289_claim_blockers.csv",
                "boundary": "repricing and cardinality review missing",
            },
            {
                "claim_id": "v289_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v289_claim_blockers.csv",
                "boundary": "global certificate missing",
            },
            {
                "claim_id": "v289_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v289_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    repaired.to_parquet(
        TABLE_DIR / "paper4_v289_exact_relief_repair_allocations.parquet", index=False
    )
    write_csv(TABLE_DIR / "paper4_v289_exact_relief_repair_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v289_exact_relief_repair_action.csv", action)
    write_csv(TABLE_DIR / "paper4_v289_exact_relief_repair_source_summary.csv", source_summary)
    write_csv(TABLE_DIR / "paper4_v289_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v289_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = summary.iloc[0]
    status = {
        "phase": "v289_apply_exact_relief_repair_candidate",
        "schema_version": "2026-05-15.289",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "signal_version_v289": SIGNAL_VERSION,
        "base_repair_version_v289": BASE_REPAIR_VERSION,
        "candidate_rank_v289": candidate_rank,
        "added_loan_id_v289": add_id,
        "dropped_rows_v289": int(len(drop_ids)),
        "selected_rows_v289": int(row[f"selected_rows_v{VERSION}"]),
        "base_selected_rows_v289": int(row[f"base_selected_rows_v{VERSION}"]),
        "cardinality_delta_vs_v279_v289": int(
            row[f"cardinality_delta_vs_v{BASE_REPAIR_VERSION}_v{VERSION}"]
        ),
        "portfolio_exposure_v289": float(row[f"portfolio_exposure_v{VERSION}"]),
        "objective_return_v289": float(row[f"objective_return_v{VERSION}"]),
        "scenario_loss_cvar90_v289": float(row[f"scenario_loss_cvar90_v{VERSION}"]),
        "source_cap_violations_v289": source_violations,
        "delta_return_vs_v279_v289": float(
            row[f"delta_return_vs_v{BASE_REPAIR_VERSION}_v{VERSION}"]
        ),
        "delta_cvar90_vs_v279_v289": float(
            row[f"delta_cvar90_vs_v{BASE_REPAIR_VERSION}_v{VERSION}"]
        ),
        "budget_feasible_v289": bool(row[f"budget_feasible_v{VERSION}"]),
        "cvar_feasible_v289": bool(row[f"cvar_feasible_v{VERSION}"]),
        "source_feasible_v289": bool(row[f"source_feasible_v{VERSION}"]),
        "repair_candidate_feasible_v289": bool(row[f"repair_candidate_feasible_v{VERSION}"]),
        "cardinality_changed_v289": bool(row[f"cardinality_changed_v{VERSION}"]),
        "post_repair_pricing_required_v289": True,
        "working_champion_claim_allowed_v289": False,
        "full_universe_integer_optimality_claim_allowed_v289": False,
        "paper1_promotion_allowed_v289": False,
        "paper4_working_champion_changed_v289": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v289": int(len(blockers)),
        "claim_matrix_rows_v289": int(len(claim_matrix)),
        "next_artifact_v289": f"paper4_v{NEXT_VERSION}_post_exact_relief_reprice.csv",
        "claim_boundary": (
            "v289 is an applied lab repair candidate only; cardinality review, repricing, "
            "global pricing and promotion remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v289_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v289": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
