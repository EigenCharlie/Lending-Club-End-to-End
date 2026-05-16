#!/usr/bin/env python3
"""Build Paper 4 v326 by applying the best v325 post-v324 one-swap signal."""

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

VERSION = 326
BASE_VERSION = 324
REPRICE_VERSION = 325
NEXT_VERSION = 327
TARGET_SELECTED_ROWS = 171
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_post_v326_reprice.csv"


def _period_distribution(portfolio: pd.DataFrame) -> dict[str, int]:
    return {
        str(period): int(count)
        for period, count in portfolio["period"].astype(str).value_counts().sort_index().items()
    }


def _portfolio_metrics(
    *,
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    losses: np.ndarray,
    mean_returns: np.ndarray,
) -> dict[str, Any]:
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    scenario_losses = losses[:, idx].sum(axis=1)
    return {
        "selected_rows": int(len(selected)),
        "portfolio_exposure": float(selected["loan_amnt"].sum()),
        "objective_return": float(mean_returns[idx].sum()),
        "scenario_loss_mean": float(scenario_losses.mean()),
        "scenario_loss_cvar90": v70._tail_cvar(scenario_losses),
        "period_distribution": _period_distribution(selected),
    }


def _cap_lookup(source_caps: pd.DataFrame, family: str) -> dict[str, float]:
    family_caps = source_caps.loc[source_caps["source_family"].astype(str).eq(family)]
    return {
        str(row["source_id"]): float(row[f"cap_share_v{BASE_VERSION}"])
        for _, row in family_caps.iterrows()
    }


def _source_summary(
    *,
    selected: pd.DataFrame,
    universe: pd.DataFrame,
    source_caps: pd.DataFrame,
) -> pd.DataFrame:
    cap_lookup = {family: _cap_lookup(source_caps, family) for family in FAMILIES}
    exposure = float(selected["loan_amnt"].sum())
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        by_source = selected.groupby(family, dropna=False)["loan_amnt"].sum()
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
                        "v326 post-v324 applied-swap source diagnostic only"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v326 applies the best v325 post-v324 feasible one-swap.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v326_apply_next_post_v324_swap.csv"
                ),
                "boundary": "Applied one-swap candidate only; post-v326 repricing required.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v326 improves return and CVaR versus the v324 relaxed repair.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v326_apply_next_post_v324_swap.csv"
                ),
                "boundary": "Static common-scenario proxy only; not live or global evidence.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v326 preserves v324 relaxed proxy coverage exactly.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v326_claim_blockers.csv"
                ),
                "boundary": "The applied swap increases missing proxy rows by one.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v326 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v326_claim_blockers.csv"
                ),
                "boundary": "Post-v326 repricing, global and live gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v326 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v326_claim_blockers.csv"
                ),
                "boundary": "No final promotion or deployment gate is created.",
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
                    "v326 applies the best v325 feasible one-swap to the v324 relaxed repair."
                ),
                "status": "post_v324_best_feasible_swap_applied_requires_reprice",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "rerun post-v326 repricing before any local-optimal or champion claim"
                ),
                "last_wave": "v326",
                "execution_result": "return_and_cvar_improve_proxy_missing_worsens_by_one",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v326")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V326_APPLY_NEXT_POST_V324_SWAP_START -->"
    end = "<!-- V326_APPLY_NEXT_POST_V324_SWAP_END -->"
    block = f"""
{start}

## Wave v326: Apply Next Post-v324 Swap

Generated: {status["generated_at_utc"]}

### Objective

v325 found four CVaR-feasible return-improving one-swaps for the v324 relaxed
repair. v326 applies the best feasible swap and recalculates the candidate book.

### Results

- Applied added loan: `{status["applied_added_loan_id_v326"]}`.
- Applied dropped loan: `{status["applied_dropped_loan_id_v326"]}`.
- Return delta vs v324:
  `{status["delta_return_vs_v324_v326"]}`.
- CVaR90 delta vs v324:
  `{status["delta_cvar90_vs_v324_v326"]}`.
- Observed proxy rows: `{status["observed_proxy_rows_v326"]}`.
- Missing proxy rows: `{status["missing_proxy_rows_v326"]}`.
- Missing proxy delta vs v324:
  `{status["missing_proxy_delta_vs_v324_v326"]}`.
- Post-v326 repricing required:
  `{status["post_v326_repricing_required_v326"]}`.

### Interpretation

v326 recovers some of the return given back in v324 and lowers CVaR, but it does
so by sacrificing one observed proxy loan. The resulting book is a new candidate
that must be repriced; it is not locally optimal, live deployable, or a working
champion.

### Claim Impact

- Allowed: applied one-swap candidate, static return/CVaR improvement versus
  v324 relaxed repair.
- Still prohibited: local optimality after v326, full-universe optimality,
  contractual IFRS9, live deployment, Paper Estrella replacement, final Paper 4
  promotion and working champion claims.

### Quarto Promotion Decision

Keep v326 in the living notebook. The next wave should reprice the v326
candidate without promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    base = read_parquet("paper4_v324_relaxed_proxy_gap_allocations.parquet").reset_index(drop=True)
    v325_top = read_csv("paper4_v325_post_v324_one_swap_top_candidates.csv")
    v324_summary = read_csv("paper4_v324_v320_proxy_gap_repair_or_branch_price_protocol.csv")
    v320_summary = read_csv("paper4_v320_matched_period_milp_summary.csv")
    v316_summary = read_csv("paper4_v316_apply_next_post_v314_swap_summary.csv")
    source_caps = read_csv("paper4_v324_relaxed_proxy_gap_source_summary.csv")
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    if any(
        df.empty
        for df in [
            universe,
            base,
            v325_top,
            v324_summary,
            v320_summary,
            v316_summary,
            source_caps,
            v47_panel,
        ]
    ):
        raise RuntimeError("Missing inputs for v326 applied swap.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    base["loan_id"] = base["loan_id"].astype(str)
    for family in FAMILIES:
        universe[family] = universe[family].astype(str)
        base[family] = base[family].astype(str)
    observed_ids = set(v47_panel["loan_id"].astype(str))
    feasible = v325_top.loc[v325_top["one_swap_improves_return_v325"].astype(bool)].copy()
    if feasible.empty:
        raise RuntimeError("v326 requires a feasible improving v325 one-swap.")
    best = feasible.sort_values("return_delta_v325", ascending=False).iloc[0]
    add_id = str(best["added_loan_id_v325"])
    drop_id = str(best["dropped_loan_id_v325"])

    added = universe.loc[universe["loan_id"].eq(add_id)].copy()
    kept = base.loc[~base["loan_id"].eq(drop_id)].copy()
    if added.empty or len(kept) != TARGET_SELECTED_ROWS - 1:
        raise RuntimeError("v326 could not apply the selected add/drop pair.")
    selected = pd.concat([kept, added], ignore_index=True, sort=False)
    selected["loan_id"] = selected["loan_id"].astype(str)

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    metrics = _portfolio_metrics(
        universe=universe,
        selected=selected,
        losses=losses,
        mean_returns=mean_returns,
    )
    v324_row = v324_summary.iloc[0]
    v320_row = v320_summary.iloc[0]
    v316_row = v316_summary.iloc[0]
    base_return = float(v324_row["relaxed_objective_return_v324"])
    base_cvar = float(v324_row["relaxed_cvar90_v324"])
    observed_rows = int(selected["loan_id"].isin(observed_ids).sum())
    missing_rows = int(len(selected) - observed_rows)
    source_summary = _source_summary(selected=selected, universe=universe, source_caps=source_caps)
    source_cap_violations = int(source_summary[f"source_cap_violated_v{VERSION}"].sum())
    min_source_slack = float(source_summary[f"source_slack_v{VERSION}"].min())
    period_distribution = metrics["period_distribution"]

    allocations = selected[
        [
            "loan_id",
            "loan_amnt",
            *FAMILIES,
        ]
    ].copy()
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    allocations[f"mean_return_v{VERSION}"] = mean_returns[
        idx_by_id.loc[allocations["loan_id"].astype(str)].to_numpy()
    ]
    allocations[f"selected_v{VERSION}"] = True
    allocations[f"observed_v47_proxy_v{VERSION}"] = allocations["loan_id"].isin(observed_ids)
    allocations[f"portfolio_label_v{VERSION}"] = "post_v324_best_feasible_swap_candidate"
    allocations[f"claim_boundary_v{VERSION}"] = (
        "applied post-v324 one-swap candidate; post-v326 repricing required"
    )
    action = pd.DataFrame(
        [
            {
                f"action_v{VERSION}": "add_v325_best_feasible",
                "loan_id": add_id,
                "loan_amnt": float(best["added_loan_amount_v325"]),
                f"mean_return_v{VERSION}": float(best["added_mean_return_v325"]),
                f"observed_v47_proxy_v{VERSION}": bool(best["added_observed_v47_proxy_v325"]),
                f"claim_boundary_v{VERSION}": "applied v325 best feasible add leg",
            },
            {
                f"action_v{VERSION}": "drop_v325_best_feasible",
                "loan_id": drop_id,
                "loan_amnt": float(best["dropped_loan_amount_v325"]),
                f"mean_return_v{VERSION}": float(best["dropped_mean_return_v325"]),
                f"observed_v47_proxy_v{VERSION}": bool(best["dropped_observed_v47_proxy_v325"]),
                f"claim_boundary_v{VERSION}": "applied v325 best feasible drop leg",
            },
        ]
    )
    summary = pd.DataFrame(
        [
            {
                f"gate_id_v{VERSION}": "v326_apply_next_post_v324_swap",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"reprice_version_v{VERSION}": REPRICE_VERSION,
                f"applied_added_loan_id_v{VERSION}": add_id,
                f"applied_dropped_loan_id_v{VERSION}": drop_id,
                f"selected_rows_v{VERSION}": int(metrics["selected_rows"]),
                f"cardinality_restored_v{VERSION}": int(metrics["selected_rows"])
                == TARGET_SELECTED_ROWS,
                f"portfolio_exposure_v{VERSION}": float(metrics["portfolio_exposure"]),
                f"objective_return_v{VERSION}": float(metrics["objective_return"]),
                f"delta_return_vs_v324_v{VERSION}": float(
                    metrics["objective_return"] - base_return
                ),
                f"delta_return_vs_v320_v{VERSION}": float(
                    metrics["objective_return"] - float(v320_row["objective_return_v320"])
                ),
                f"delta_return_vs_v316_v{VERSION}": float(
                    metrics["objective_return"] - float(v316_row["objective_return_v316"])
                ),
                f"scenario_loss_mean_v{VERSION}": float(metrics["scenario_loss_mean"]),
                f"scenario_loss_cvar90_v{VERSION}": float(metrics["scenario_loss_cvar90"]),
                f"delta_cvar90_vs_v324_v{VERSION}": float(
                    metrics["scenario_loss_cvar90"] - base_cvar
                ),
                f"delta_cvar90_vs_v320_v{VERSION}": float(
                    metrics["scenario_loss_cvar90"] - float(v320_row["scenario_loss_cvar90_v320"])
                ),
                f"delta_cvar90_vs_v316_v{VERSION}": float(
                    metrics["scenario_loss_cvar90"] - float(v316_row["scenario_loss_cvar90_v316"])
                ),
                f"observed_proxy_rows_v{VERSION}": observed_rows,
                f"missing_proxy_rows_v{VERSION}": missing_rows,
                f"observed_proxy_delta_vs_v324_v{VERSION}": int(
                    observed_rows - int(v324_row["relaxed_observed_proxy_rows_v324"])
                ),
                f"missing_proxy_delta_vs_v324_v{VERSION}": int(
                    missing_rows - int(v324_row["relaxed_missing_proxy_rows_v324"])
                ),
                f"period_distribution_v{VERSION}": json.dumps(period_distribution, sort_keys=True),
                f"source_cap_violations_v{VERSION}": source_cap_violations,
                f"min_source_slack_v{VERSION}": min_source_slack,
                f"post_v326_repricing_required_v{VERSION}": True,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "applied one-swap candidate only; post-v326 repricing/global gates required"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "post_v326_reprice_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "applied swap must be repriced before local claim",
            },
            {
                f"blocker_id_v{VERSION}": "proxy_coverage_worsens_vs_v324",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    summary.iloc[0][f"missing_proxy_delta_vs_v324_v{VERSION}"]
                ),
                f"required_next_artifact_v{VERSION}": "future_proxy_coverage_repair_gate",
                f"claim_boundary_v{VERSION}": "best return/CVaR swap adds one missing proxy row",
            },
            {
                f"blocker_id_v{VERSION}": "global_dynamic_online_gates_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_global_dynamic_online_validation",
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
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v326_best_v325_swap_applied",
                "allowed": True,
                "artifact": "paper4_v326_apply_next_post_v324_swap.csv",
                "boundary": "applied one-swap candidate only",
            },
            {
                "claim_id": "v326_improves_return_and_cvar_vs_v324",
                "allowed": True,
                "artifact": "paper4_v326_apply_next_post_v324_swap.csv",
                "boundary": "static common-scenario proxy only",
            },
            {
                "claim_id": "v326_preserves_v324_proxy_coverage",
                "allowed": False,
                "artifact": "paper4_v326_claim_blockers.csv",
                "boundary": "missing proxy rows increase by one",
            },
            {
                "claim_id": "v326_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v326_claim_blockers.csv",
                "boundary": "repricing and global gates missing",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v326_apply_next_post_v324_swap.csv", summary)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v326_post_v324_swap_allocations.parquet", index=False
    )
    write_csv(TABLE_DIR / "paper4_v326_post_v324_swap_actions.csv", action)
    write_csv(TABLE_DIR / "paper4_v326_post_v324_swap_source_summary.csv", source_summary)
    write_csv(TABLE_DIR / "paper4_v326_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v326_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = summary.iloc[0]
    status = {
        "phase": "v326_apply_next_post_v324_swap",
        "schema_version": "2026-05-16.326",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v326": BASE_VERSION,
        "reprice_version_v326": REPRICE_VERSION,
        "applied_added_loan_id_v326": add_id,
        "applied_dropped_loan_id_v326": drop_id,
        "selected_rows_v326": int(row[f"selected_rows_v{VERSION}"]),
        "cardinality_restored_v326": bool(row[f"cardinality_restored_v{VERSION}"]),
        "portfolio_exposure_v326": float(row[f"portfolio_exposure_v{VERSION}"]),
        "objective_return_v326": float(row[f"objective_return_v{VERSION}"]),
        "delta_return_vs_v324_v326": float(row[f"delta_return_vs_v324_v{VERSION}"]),
        "delta_return_vs_v320_v326": float(row[f"delta_return_vs_v320_v{VERSION}"]),
        "delta_return_vs_v316_v326": float(row[f"delta_return_vs_v316_v{VERSION}"]),
        "scenario_loss_mean_v326": float(row[f"scenario_loss_mean_v{VERSION}"]),
        "scenario_loss_cvar90_v326": float(row[f"scenario_loss_cvar90_v{VERSION}"]),
        "delta_cvar90_vs_v324_v326": float(row[f"delta_cvar90_vs_v324_v{VERSION}"]),
        "delta_cvar90_vs_v320_v326": float(row[f"delta_cvar90_vs_v320_v{VERSION}"]),
        "delta_cvar90_vs_v316_v326": float(row[f"delta_cvar90_vs_v316_v{VERSION}"]),
        "observed_proxy_rows_v326": int(row[f"observed_proxy_rows_v{VERSION}"]),
        "missing_proxy_rows_v326": int(row[f"missing_proxy_rows_v{VERSION}"]),
        "observed_proxy_delta_vs_v324_v326": int(row[f"observed_proxy_delta_vs_v324_v{VERSION}"]),
        "missing_proxy_delta_vs_v324_v326": int(row[f"missing_proxy_delta_vs_v324_v{VERSION}"]),
        "source_cap_violations_v326": source_cap_violations,
        "min_source_slack_v326": min_source_slack,
        "post_v326_repricing_required_v326": True,
        "working_champion_claim_allowed_v326": False,
        "full_universe_integer_optimality_claim_allowed_v326": False,
        "paper1_promotion_allowed_v326": False,
        "paper4_working_champion_changed_v326": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v326": int(len(blockers)),
        "claim_matrix_rows_v326": int(len(claim_matrix)),
        "next_artifact_v326": row[f"next_artifact_v{VERSION}"],
        "claim_boundary": (
            "v326 applies one v325 feasible swap; repricing, global proof, live deployment, "
            "working champion, and final promotion remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v326_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v326": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
