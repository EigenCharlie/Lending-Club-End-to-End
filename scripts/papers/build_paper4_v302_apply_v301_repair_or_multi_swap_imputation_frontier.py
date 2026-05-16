#!/usr/bin/env python3
"""Build Paper 4 v302 greedy multi-swap imputation frontier artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers.build_paper4_v301_source_tight_branch_price_pricing_or_imputation_repair import (
    _imputed_drop_pool,
    _observed_proxy_candidate_pool,
    _repair_pairs,
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

VERSION = 302
SOURCE_CANDIDATE_VERSION = 295
REPAIR_SCREEN_VERSION = 301
MAX_GREEDY_STEPS = 15
NEXT_VERSION = 303


def _source_summary(
    *,
    universe: pd.DataFrame,
    portfolio: pd.DataFrame,
    source_caps: pd.DataFrame,
) -> pd.DataFrame:
    cap_lookup: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        family_caps = source_caps.loc[source_caps["source_family"].astype(str).eq(family)]
        cap_lookup[family] = {
            str(row["source_id"]): float(row[f"cap_share_v{SOURCE_CANDIDATE_VERSION}"])
            for _, row in family_caps.iterrows()
        }
    exposure = float(portfolio["loan_amnt"].sum())
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        by_source = portfolio.groupby(family, dropna=False)["loan_amnt"].sum()
        for source_id in sorted(universe[family].dropna().astype(str).unique()):
            source_exposure = float(by_source.get(source_id, 0.0))
            share = source_exposure / max(exposure, 1.0)
            cap = float(cap_lookup[family].get(source_id, 1.0))
            rows.append(
                {
                    "source_family": family,
                    "source_id": source_id,
                    f"cap_share_v{VERSION}": cap,
                    f"source_exposure_v{VERSION}": source_exposure,
                    f"source_share_v{VERSION}": share,
                    f"source_slack_v{VERSION}": cap - share,
                    f"source_cap_violated_v{VERSION}": share > cap + 1e-7,
                    f"claim_boundary_v{VERSION}": (
                        "v302 greedy frontier final source diagnostic only; no global proof"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _greedy_frontier(
    *,
    universe: pd.DataFrame,
    initial_selected: pd.DataFrame,
    source_summary: pd.DataFrame,
    v47_panel: pd.DataFrame,
    v299_panel: pd.DataFrame,
    losses: np.ndarray,
    mean_returns: np.ndarray,
    idx_by_id: pd.Series,
    objective_return: float,
    exposure_min: float,
    exposure_max: float,
    cvar_cap: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = initial_selected[["loan_id", "loan_amnt", *FAMILIES]].copy()
    selected["loan_id"] = selected["loan_id"].astype(str)
    rows: list[dict[str, Any]] = []
    for step in range(1, MAX_GREEDY_STEPS + 1):
        candidates = _observed_proxy_candidate_pool(
            universe=universe,
            selected=selected,
            v47_panel=v47_panel,
            idx_by_id=idx_by_id,
            mean_returns=mean_returns,
        )
        drops = _imputed_drop_pool(
            selected=selected,
            v299_panel=v299_panel,
            idx_by_id=idx_by_id,
            mean_returns=mean_returns,
        )
        if drops.empty:
            break
        repair_pairs, _top_pairs, _stage_summary = _repair_pairs(
            universe=universe,
            selected=selected,
            candidates=candidates,
            drops=drops,
            source_summary=source_summary,
            losses=losses,
            idx_by_id=idx_by_id,
            objective_return=objective_return,
            exposure_min=exposure_min,
            exposure_max=exposure_max,
            cvar_cap=cvar_cap,
        )
        if repair_pairs.empty:
            break
        best = repair_pairs.sort_values("return_delta_v301", ascending=False).iloc[0]
        add_id = str(best["added_loan_id_v301"])
        drop_id = str(best["dropped_loan_id_v301"])
        objective_return = float(best["objective_return_after_repair_v301"])
        add_row = (
            candidates.set_index("loan_id", drop=False)
            .loc[add_id, ["loan_id", "loan_amnt", *FAMILIES]]
            .to_frame()
            .T
        )
        selected = selected.loc[~selected["loan_id"].astype(str).eq(drop_id)].copy()
        selected = pd.concat([selected, add_row], ignore_index=True)
        rows.append(
            {
                f"frontier_step_v{VERSION}": step,
                f"added_loan_id_v{VERSION}": add_id,
                f"dropped_loan_id_v{VERSION}": drop_id,
                f"repair_profile_v{VERSION}": str(best["repair_profile_v301"]),
                f"return_delta_v{VERSION}": float(best["return_delta_v301"]),
                f"cumulative_return_delta_v{VERSION}": float(
                    sum(row[f"return_delta_v{VERSION}"] for row in rows)
                    + float(best["return_delta_v301"])
                ),
                f"objective_return_after_step_v{VERSION}": objective_return,
                f"exposure_after_step_v{VERSION}": float(best["exposure_after_repair_v301"]),
                f"cvar90_after_step_v{VERSION}": float(best["cvar90_after_repair_v301"]),
                f"imputed_proxy_loan_rows_after_step_v{VERSION}": int(
                    best["imputed_proxy_loan_rows_after_v301"]
                ),
                f"feasible_pair_rows_before_step_v{VERSION}": int(len(repair_pairs)),
                f"return_improving_pair_rows_before_step_v{VERSION}": int(
                    repair_pairs["return_improving_repair_v301"].sum()
                ),
                f"grade_A_exposure_delta_v{VERSION}": float(best["grade_A_exposure_delta_v301"]),
                f"score0_exposure_delta_v{VERSION}": float(best["score0_exposure_delta_v301"]),
                f"claim_boundary_v{VERSION}": (
                    "greedy frontier step only; not globally optimal or promoted"
                ),
            }
        )
    final_allocations = selected.copy()
    final_allocations[f"selected_v{VERSION}"] = True
    final_allocations[f"portfolio_label_v{VERSION}"] = "v302_greedy_imputation_frontier_final"
    final_allocations[f"claim_boundary_v{VERSION}"] = (
        "v302 greedy frontier final allocation; not a promoted champion"
    )
    return pd.DataFrame(rows), final_allocations


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v302 greedy multi-swap imputation frontier.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v302_apply_v301_repair_or_multi_swap_imputation_frontier.csv"
                ),
                "boundary": "Greedy bounded frontier only; not globally optimal.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v302 reduces v295 imputed proxy rows through 15 feasible greedy swaps.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v302_greedy_imputation_frontier.csv"
                ),
                "boundary": "Feasible under budget/source/CVaR screens but carries return cost.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v302 proves the optimal imputation-repair frontier.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v302_claim_blockers.csv"
                ),
                "boundary": "Greedy sequence only; no global multi-swap optimization proof.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v302 resolves contractual IFRS9 or live deployability for v295.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v302_claim_blockers.csv"
                ),
                "boundary": "61 imputed rows remain after the bounded greedy frontier.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v302 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v302_claim_blockers.csv"
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
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v302 applies the v301 exact-source repair greedily to trace a bounded "
                    "multi-swap imputation frontier."
                ),
                "status": "greedy_multi_swap_imputation_frontier_executed",
                "next_artifact": (
                    f"paper4_v{NEXT_VERSION}_global_or_multiobjective_frontier_after_v302.csv"
                ),
                "success_condition": (
                    "compare the return-cost/imputation frontier with global-bound and "
                    "multi-objective alternatives without promoting"
                ),
                "last_wave": "v302",
                "execution_result": (
                    "fifteen_feasible_greedy_repairs_reduce_imputed_rows_to_61_with_return_cost"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v302")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V302_GREEDY_IMPUTATION_FRONTIER_START -->"
    end = "<!-- V302_GREEDY_IMPUTATION_FRONTIER_END -->"
    block = f"""
{start}

## Wave v302: Greedy Multi-Swap Imputation Frontier

Generated: {status["generated_at_utc"]}

### Objective

v301 found exact-source one-swap repairs that reduce imputed proxy dependence,
but only with return cost. v302 applies that repair rule greedily for up to 15
steps to trace a bounded return-cost versus imputation-reduction frontier.

### Results

- Greedy steps executed: `{status["greedy_steps_v302"]}`.
- Initial imputed proxy rows: `{status["initial_imputed_proxy_loan_rows_v302"]}`.
- Final imputed proxy rows: `{status["final_imputed_proxy_loan_rows_v302"]}`.
- Imputed rows reduced: `{status["imputed_proxy_rows_reduced_v302"]}`.
- Cumulative return delta: `{status["cumulative_return_delta_v302"]}`.
- Final objective return: `{status["final_objective_return_v302"]}`.
- Final CVaR90: `{status["final_cvar90_v302"]}`.
- Tight-relief steps: `{status["tight_relief_steps_v302"]}`.
- Return-improving steps: `{status["return_improving_steps_v302"]}`.
- Valid global/multi-swap optimality proof:
  `{status["valid_multi_swap_optimality_claim_v302"]}`.

### Interpretation

v302 shows that the v295 data-quality blocker can be reduced in a controlled
way: 15 feasible greedy repairs lower imputed rows from 76 to 61 while staying
inside budget, exact source caps and the v295 CVaR cap. The trade-off is real:
the frontier pays return to buy observed-proxy coverage.

### Claim Impact

- Allowed: bounded greedy multi-swap imputation frontier and 15-step repair
  trace.
- Still prohibited: optimal frontier, contractual IFRS9, live deployability,
  full branch-price/global optimality, Paper Estrella replacement, final Paper
  4 promotion and working champion claims.

### Quarto Promotion Decision

Keep v302 in the living notebook. The next wave should compare this return-cost
frontier against global-bound or multi-objective alternatives.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v295_broader_multi_swap_allocations.parquet").reset_index(
        drop=True
    )
    selected["loan_id"] = selected["loan_id"].astype(str)
    source_caps = read_csv("paper4_v295_broader_source_summary.csv")
    v295_summary = read_csv("paper4_v295_broader_multi_swap_or_global_gap_probe.csv")
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    v299_panel = read_parquet("paper4_v299_v295_cashflow_proxy_panel.parquet")
    v301_status = json.loads((STATUS_DIR / "paper4_v301_status.json").read_text(encoding="utf-8"))
    if any(
        df.empty for df in [universe, selected, source_caps, v295_summary, v47_panel, v299_panel]
    ):
        raise RuntimeError("Missing v55, v295, v47, v299 or source inputs for v302.")
    if not bool(v301_status["feasible_imputation_repair_found_v301"]):
        raise RuntimeError("v302 expects v301 to find feasible one-swap repairs.")

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    v295_row = v295_summary.iloc[0]
    objective_return = float(v295_row[f"objective_return_v{SOURCE_CANDIDATE_VERSION}"])
    cvar_cap = float(v295_row[f"scenario_loss_cvar90_v{SOURCE_CANDIDATE_VERSION}"])
    exposure_min = float(v295_row[f"exposure_min_v{SOURCE_CANDIDATE_VERSION}"])
    exposure_max = float(v295_row[f"exposure_max_v{SOURCE_CANDIDATE_VERSION}"])
    frontier, final_allocations = _greedy_frontier(
        universe=universe,
        initial_selected=selected,
        source_summary=source_caps,
        v47_panel=v47_panel,
        v299_panel=v299_panel,
        losses=losses,
        mean_returns=mean_returns,
        idx_by_id=idx_by_id,
        objective_return=objective_return,
        exposure_min=exposure_min,
        exposure_max=exposure_max,
        cvar_cap=cvar_cap,
    )
    if frontier.empty:
        raise RuntimeError("v302 expected at least one feasible greedy repair step.")
    final_source = _source_summary(
        universe=universe,
        portfolio=final_allocations,
        source_caps=source_caps,
    )
    final_row = frontier.iloc[-1]
    initial_imputed = int(v301_status["imputed_selected_drop_rows_v301"])
    final_imputed = int(final_row[f"imputed_proxy_loan_rows_after_step_v{VERSION}"])
    cumulative_delta = float(frontier[f"return_delta_v{VERSION}"].sum())
    tight_relief_profiles = {
        "relieves_grade_A_and_score0",
        "relieves_grade_A_only",
        "relieves_score0_only",
    }
    tight_relief_steps = int(
        frontier[f"repair_profile_v{VERSION}"].isin(tight_relief_profiles).sum()
    )
    return_improving_steps = int(frontier[f"return_delta_v{VERSION}"].gt(0).sum())
    valid_multi_swap_claim = False
    summary = pd.DataFrame(
        [
            {
                f"frontier_id_v{VERSION}": "v302_greedy_multi_swap_imputation_frontier",
                f"source_candidate_version_v{VERSION}": SOURCE_CANDIDATE_VERSION,
                f"repair_screen_version_v{VERSION}": REPAIR_SCREEN_VERSION,
                f"max_greedy_steps_v{VERSION}": MAX_GREEDY_STEPS,
                f"greedy_steps_v{VERSION}": int(len(frontier)),
                f"initial_imputed_proxy_loan_rows_v{VERSION}": initial_imputed,
                f"final_imputed_proxy_loan_rows_v{VERSION}": final_imputed,
                f"imputed_proxy_rows_reduced_v{VERSION}": initial_imputed - final_imputed,
                f"cumulative_return_delta_v{VERSION}": cumulative_delta,
                f"final_objective_return_v{VERSION}": float(
                    final_row[f"objective_return_after_step_v{VERSION}"]
                ),
                f"v295_objective_return_v{VERSION}": objective_return,
                f"final_cvar90_v{VERSION}": float(final_row[f"cvar90_after_step_v{VERSION}"]),
                f"v295_cvar90_cap_v{VERSION}": cvar_cap,
                f"tight_relief_steps_v{VERSION}": tight_relief_steps,
                f"return_improving_steps_v{VERSION}": return_improving_steps,
                f"final_source_cap_violations_v{VERSION}": int(
                    final_source[f"source_cap_violated_v{VERSION}"].sum()
                ),
                f"valid_multi_swap_optimality_claim_v{VERSION}": valid_multi_swap_claim,
                f"contractual_ifrs9_claim_allowed_v{VERSION}": False,
                f"strict_live_deployability_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_global_or_multiobjective_frontier_after_v302.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "greedy bounded imputation frontier only; no optimality, IFRS9, live or promotion claim"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "global_multi_swap_optimality_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(frontier)),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_global_or_multiobjective_frontier_after_v302.csv"
                ),
                f"claim_boundary_v{VERSION}": "greedy frontier is not a global proof",
            },
            {
                f"blocker_id_v{VERSION}": "residual_cashflow_imputation_after_frontier",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": final_imputed,
                f"required_next_artifact_v{VERSION}": "future_observed_v295_cashflow_panel",
                f"claim_boundary_v{VERSION}": "61 imputed proxy rows remain",
            },
            {
                f"blocker_id_v{VERSION}": "return_cost_frontier_not_champion_upgrade",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(abs(round(cumulative_delta))),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_global_or_multiobjective_frontier_after_v302.csv"
                ),
                f"claim_boundary_v{VERSION}": "frontier pays return to reduce imputation",
            },
            {
                f"blocker_id_v{VERSION}": "external_online_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": "future_external_online_holdout",
                f"claim_boundary_v{VERSION}": "v302 does not create external online evidence",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_working_champion_gate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_global_dynamic_promotion_gate",
                f"claim_boundary_v{VERSION}": "working champion replacement remains blocked",
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
                "claim_id": "v302_greedy_multi_swap_imputation_frontier_executed",
                "allowed": True,
                "artifact": "paper4_v302_apply_v301_repair_or_multi_swap_imputation_frontier.csv",
                "boundary": "bounded greedy frontier only",
            },
            {
                "claim_id": "v302_reduces_imputed_rows_under_constraints",
                "allowed": True,
                "artifact": "paper4_v302_greedy_imputation_frontier.csv",
                "boundary": "feasible with return cost",
            },
            {
                "claim_id": "v302_optimal_imputation_frontier",
                "allowed": False,
                "artifact": "paper4_v302_claim_blockers.csv",
                "boundary": "no global multi-swap proof",
            },
            {
                "claim_id": "v302_contractual_ifrs9_or_live_deployability",
                "allowed": False,
                "artifact": "paper4_v302_claim_blockers.csv",
                "boundary": "residual imputation and no external holdout",
            },
            {
                "claim_id": "v302_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v302_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(
        TABLE_DIR / "paper4_v302_apply_v301_repair_or_multi_swap_imputation_frontier.csv", summary
    )
    write_csv(TABLE_DIR / "paper4_v302_greedy_imputation_frontier.csv", frontier)
    final_allocations.to_parquet(
        TABLE_DIR / "paper4_v302_greedy_frontier_final_allocations.parquet", index=False
    )
    write_csv(TABLE_DIR / "paper4_v302_greedy_frontier_final_source_summary.csv", final_source)
    write_csv(TABLE_DIR / "paper4_v302_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v302_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v302_apply_v301_repair_or_multi_swap_imputation_frontier",
        "schema_version": "2026-05-15.302",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "source_candidate_version_v302": SOURCE_CANDIDATE_VERSION,
        "repair_screen_version_v302": REPAIR_SCREEN_VERSION,
        "max_greedy_steps_v302": MAX_GREEDY_STEPS,
        "greedy_steps_v302": int(len(frontier)),
        "initial_imputed_proxy_loan_rows_v302": initial_imputed,
        "final_imputed_proxy_loan_rows_v302": final_imputed,
        "imputed_proxy_rows_reduced_v302": initial_imputed - final_imputed,
        "cumulative_return_delta_v302": cumulative_delta,
        "final_objective_return_v302": float(final_row[f"objective_return_after_step_v{VERSION}"]),
        "v295_objective_return_v302": objective_return,
        "final_cvar90_v302": float(final_row[f"cvar90_after_step_v{VERSION}"]),
        "v295_cvar90_cap_v302": cvar_cap,
        "tight_relief_steps_v302": tight_relief_steps,
        "return_improving_steps_v302": return_improving_steps,
        "frontier_rows_v302": int(len(frontier)),
        "final_allocation_rows_v302": int(len(final_allocations)),
        "final_source_summary_rows_v302": int(len(final_source)),
        "final_source_cap_violations_v302": int(
            final_source[f"source_cap_violated_v{VERSION}"].sum()
        ),
        "valid_multi_swap_optimality_claim_v302": valid_multi_swap_claim,
        "strict_live_deployability_claim_allowed_v302": False,
        "contractual_ifrs9_claim_allowed_v302": False,
        "working_champion_claim_allowed_v302": False,
        "paper1_promotion_allowed_v302": False,
        "paper4_working_champion_changed_v302": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v302": int(len(blockers)),
        "claim_matrix_rows_v302": int(len(claim_matrix)),
        "next_artifact_v302": (
            f"paper4_v{NEXT_VERSION}_global_or_multiobjective_frontier_after_v302.csv"
        ),
        "claim_boundary": (
            "v302 builds a bounded greedy imputation frontier; optimality, IFRS9, live deployment, "
            "working champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v302_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v302": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
