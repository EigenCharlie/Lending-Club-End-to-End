#!/usr/bin/env python3
"""Build Paper 4 v347 by applying the v346 multi-source relief candidate."""

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

VERSION = 347
BASE_VERSION = 338
RELIEF_VERSION = 346
NEXT_VERSION = 348
TARGET_SELECTED_ROWS = 171
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_post_v347_reprice.csv"


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
    idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy(int)
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
                        "v347 applied multi-source relief candidate source diagnostic only"
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
                "claim": "v347 applies the v346 local two-add/two-drop relief candidate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v347_v338_apply_multi_source_relief_candidate.csv"
                ),
                "boundary": "Applied local candidate only; post-v347 repricing required.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v347 improves static return and CVaR versus v338.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v347_v338_apply_multi_source_relief_candidate.csv"
                ),
                "boundary": "Static common-scenario proxy only; not live or global evidence.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v347 preserves v338 proxy coverage exactly.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v347_claim_blockers.csv"
                ),
                "boundary": "The applied two-swap increases missing proxy rows by one.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v347 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v347_claim_blockers.csv"
                ),
                "boundary": "Post-v347 repricing, global and live gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v347 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v347_claim_blockers.csv"
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
                    "v347 applies the v346 local two-add/two-drop multi-source relief "
                    "candidate to v338 and recalculates static portfolio diagnostics."
                ),
                "status": "multi_source_relief_candidate_applied_requires_reprice",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "rerun post-v347 repricing before any local-optimal or champion claim"
                ),
                "last_wave": "v347",
                "execution_result": "return_and_cvar_improve_proxy_missing_worsens_by_one",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v347")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V347_V338_APPLY_MULTI_SOURCE_RELIEF_CANDIDATE_START -->"
    end = "<!-- V347_V338_APPLY_MULTI_SOURCE_RELIEF_CANDIDATE_END -->"
    block = f"""
{start}

## Wave v347: Apply v346 Multi-Source Relief Candidate

Generated: {status["generated_at_utc"]}

### Objective

v346 found one local two-add/two-drop source/CVaR feasible entering candidate.
v347 applies that action to the v338 book and recalculates the candidate
portfolio diagnostics before any repricing or champion language.

### Results

- Applied added loans: `{status["applied_added_loan_ids_v347"]}`.
- Applied dropped loans: `{status["applied_dropped_loan_ids_v347"]}`.
- Return delta vs v338:
  `{status["delta_return_vs_v338_v347"]}`.
- CVaR90 delta vs v338:
  `{status["delta_cvar90_vs_v338_v347"]}`.
- Observed proxy rows: `{status["observed_proxy_rows_v347"]}`.
- Missing proxy rows: `{status["missing_proxy_rows_v347"]}`.
- Missing proxy delta vs v338:
  `{status["missing_proxy_delta_vs_v338_v347"]}`.
- Source cap violations:
  `{status["source_cap_violations_v347"]}`.
- Post-v347 repricing required:
  `{status["post_v347_repricing_required_v347"]}`.

### Interpretation

v347 turns the v346 local pricing signal into a concrete candidate portfolio.
It improves static return and CVaR relative to v338 while preserving exposure,
cardinality and source caps. The tradeoff is proxy coverage: missing proxy rows
increase from 74 to 75. The candidate therefore remains a living-lab object, not
a working champion.

### Claim Impact

- Allowed: applied v346 two-swap candidate; static return/CVaR improvement
  versus v338.
- Still prohibited: post-v347 local optimality, full-universe optimality,
  contractual IFRS9, live deployment, Paper Estrella replacement, final Paper 4
  promotion and working champion claims.

### Quarto Promotion Decision

Keep v347 in the living notebook. The next wave should reprice the v347
candidate without promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    base = read_parquet("paper4_v338_post_v336_swap_allocations.parquet").reset_index(drop=True)
    v338_summary = read_csv("paper4_v338_apply_next_post_v336_swap.csv")
    source_caps = read_csv("paper4_v338_post_v336_swap_source_summary.csv")
    entering = read_csv("paper4_v346_second_order_entering_candidates.csv")
    v346_status = json.loads((STATUS_DIR / "paper4_v346_status.json").read_text(encoding="utf-8"))
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    if any(df.empty for df in [universe, base, v338_summary, source_caps, entering, v47_panel]):
        raise RuntimeError("Missing inputs for v347 applied multi-source relief candidate.")
    if int(v346_status["cvar_feasible_entering_rows_v346"]) <= 0:
        raise RuntimeError("v347 expects v346 to identify a local entering candidate.")
    if bool(v346_status["working_champion_claim_allowed_v346"]):
        raise RuntimeError("v347 expects v346 to remain below working-champion status.")
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    base["loan_id"] = base["loan_id"].astype(str)
    for frame in [universe, base]:
        for family in FAMILIES:
            frame[family] = frame[family].astype(str)
    v47_panel["loan_id"] = v47_panel["loan_id"].astype(str)

    best = entering.sort_values(f"return_delta_v{RELIEF_VERSION}", ascending=False).iloc[0]
    add_ids = [
        str(best[f"seed_added_loan_id_v{RELIEF_VERSION}"]),
        str(best[f"second_added_loan_id_v{RELIEF_VERSION}"]),
    ]
    drop_ids = [
        str(best[f"seed_dropped_loan_id_v{RELIEF_VERSION}"]),
        str(best[f"second_dropped_loan_id_v{RELIEF_VERSION}"]),
    ]
    add_rows = universe.loc[universe["loan_id"].isin(add_ids), ["loan_id", "loan_amnt", *FAMILIES]]
    if len(add_rows) != len(add_ids):
        raise RuntimeError("Could not find all v347 added loans in v55 universe.")

    allocations = base.loc[~base["loan_id"].isin(drop_ids), ["loan_id", "loan_amnt", *FAMILIES]]
    if len(allocations) != len(base) - len(drop_ids):
        raise RuntimeError("Could not drop all v347 loans from the v338 portfolio.")
    allocations = pd.concat([allocations, add_rows], ignore_index=True)
    allocations["loan_id"] = allocations["loan_id"].astype(str)
    for family in FAMILIES:
        allocations[family] = allocations[family].astype(str)

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    allocation_idx = idx_by_id.loc[allocations["loan_id"].astype(str)].to_numpy(int)
    allocations[f"mean_return_v{VERSION}"] = mean_returns[allocation_idx]
    allocations[f"selected_v{VERSION}"] = 1
    allocations[f"portfolio_label_v{VERSION}"] = "v338_plus_v346_multi_source_relief_candidate"
    allocations[f"relief_action_v{VERSION}"] = np.where(
        allocations["loan_id"].isin(add_ids),
        "added_from_v346_local_two_swap",
        "kept_from_v338",
    )
    allocations[f"claim_boundary_v{VERSION}"] = (
        "applied multi-source relief candidate only; post-v347 repricing required"
    )

    metrics = _portfolio_metrics(
        universe=universe,
        selected=allocations,
        losses=losses,
        mean_returns=mean_returns,
    )
    source_summary = _source_summary(
        selected=allocations,
        universe=universe,
        source_caps=source_caps,
    )
    v338_row = v338_summary.iloc[0]
    observed_ids = set(v47_panel["loan_id"].astype(str))
    observed_proxy_rows = int(allocations["loan_id"].isin(observed_ids).sum())
    missing_proxy_rows = int((~allocations["loan_id"].isin(observed_ids)).sum())
    base_observed_proxy_rows = int(base["loan_id"].isin(observed_ids).sum())
    base_missing_proxy_rows = int((~base["loan_id"].isin(observed_ids)).sum())

    universe_returns = pd.Series(mean_returns, index=universe["loan_id"].astype(str))
    action_rows: list[dict[str, Any]] = []
    for action_rank, (add_id, drop_id, action_label) in enumerate(
        [
            (add_ids[0], drop_ids[0], "seed_action_from_v346"),
            (add_ids[1], drop_ids[1], "second_relief_action_from_v346"),
        ],
        start=1,
    ):
        add_row = universe.loc[universe["loan_id"].eq(add_id)].iloc[0]
        drop_row = base.loc[base["loan_id"].eq(drop_id)].iloc[0]
        add_observed = add_id in observed_ids
        drop_observed = drop_id in observed_ids
        action_rows.append(
            {
                f"action_rank_v{VERSION}": action_rank,
                f"action_label_v{VERSION}": action_label,
                f"added_loan_id_v{VERSION}": add_id,
                f"dropped_loan_id_v{VERSION}": drop_id,
                f"added_loan_amount_v{VERSION}": float(add_row["loan_amnt"]),
                f"dropped_loan_amount_v{VERSION}": float(drop_row["loan_amnt"]),
                f"added_mean_return_v{VERSION}": float(universe_returns.loc[add_id]),
                f"dropped_mean_return_v{VERSION}": float(universe_returns.loc[drop_id]),
                f"return_delta_v{VERSION}": float(
                    universe_returns.loc[add_id] - universe_returns.loc[drop_id]
                ),
                f"added_observed_v47_proxy_v{VERSION}": add_observed,
                f"dropped_observed_v47_proxy_v{VERSION}": drop_observed,
                f"delta_missing_v47_proxy_rows_v{VERSION}": int(not add_observed)
                - int(not drop_observed),
                f"claim_boundary_v{VERSION}": (
                    "component action inside the applied v346 local two-swap candidate"
                ),
            }
        )

    source_cap_violations = int(source_summary[f"source_cap_violated_v{VERSION}"].sum())
    min_source_slack = float(source_summary[f"source_slack_v{VERSION}"].min())
    max_source_share = float(source_summary[f"source_share_v{VERSION}"].max())
    delta_return_vs_v338 = float(metrics["objective_return"] - v338_row["objective_return_v338"])
    delta_loss_mean_vs_v338 = float(
        metrics["scenario_loss_mean"] - v338_row["scenario_loss_mean_v338"]
    )
    delta_cvar90_vs_v338 = float(
        metrics["scenario_loss_cvar90"] - v338_row["scenario_loss_cvar90_v338"]
    )
    missing_proxy_delta = missing_proxy_rows - base_missing_proxy_rows
    observed_proxy_delta = observed_proxy_rows - base_observed_proxy_rows
    cardinality_preserved = int(metrics["selected_rows"]) == TARGET_SELECTED_ROWS
    exposure_preserved = (
        abs(float(metrics["portfolio_exposure"]) - float(v338_row["portfolio_exposure_v338"]))
        <= 1e-7
    )

    summary = pd.DataFrame(
        [
            {
                f"gate_id_v{VERSION}": "v347_v338_apply_multi_source_relief_candidate",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"relief_protocol_version_v{VERSION}": RELIEF_VERSION,
                f"applied_added_loan_ids_v{VERSION}": "|".join(sorted(add_ids)),
                f"applied_dropped_loan_ids_v{VERSION}": "|".join(sorted(drop_ids)),
                f"selected_rows_v{VERSION}": int(metrics["selected_rows"]),
                f"cardinality_preserved_v{VERSION}": cardinality_preserved,
                f"portfolio_exposure_v{VERSION}": float(metrics["portfolio_exposure"]),
                f"exposure_preserved_vs_v338_v{VERSION}": exposure_preserved,
                f"objective_return_v{VERSION}": float(metrics["objective_return"]),
                f"delta_return_vs_v338_v{VERSION}": delta_return_vs_v338,
                f"scenario_loss_mean_v{VERSION}": float(metrics["scenario_loss_mean"]),
                f"delta_loss_mean_vs_v338_v{VERSION}": delta_loss_mean_vs_v338,
                f"scenario_loss_cvar90_v{VERSION}": float(metrics["scenario_loss_cvar90"]),
                f"delta_cvar90_vs_v338_v{VERSION}": delta_cvar90_vs_v338,
                f"observed_proxy_rows_v{VERSION}": observed_proxy_rows,
                f"missing_proxy_rows_v{VERSION}": missing_proxy_rows,
                f"observed_proxy_delta_vs_v338_v{VERSION}": observed_proxy_delta,
                f"missing_proxy_delta_vs_v338_v{VERSION}": missing_proxy_delta,
                f"period_distribution_v{VERSION}": json.dumps(
                    metrics["period_distribution"], sort_keys=True
                ),
                f"source_cap_violations_v{VERSION}": source_cap_violations,
                f"min_source_slack_v{VERSION}": min_source_slack,
                f"max_source_share_v{VERSION}": max_source_share,
                f"return_improves_vs_v338_v{VERSION}": delta_return_vs_v338 > 1e-9,
                f"cvar_improves_vs_v338_v{VERSION}": delta_cvar90_vs_v338 < -1e-9,
                f"post_v347_repricing_required_v{VERSION}": True,
                f"valid_branch_price_bound_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "applied v346 local relief candidate only; repricing/global/live gates "
                    "remain required"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "post_v347_repricing_required",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "applied candidate has not been repriced yet",
            },
            {
                f"blocker_id_v{VERSION}": "proxy_coverage_worsens_vs_v338",
                f"blocking_v{VERSION}": missing_proxy_delta > 0,
                f"evidence_count_v{VERSION}": missing_proxy_delta,
                f"required_next_artifact_v{VERSION}": "future_proxy_repair_or_ifrs9_gate",
                f"claim_boundary_v{VERSION}": "missing proxy rows increase versus v338",
            },
            {
                f"blocker_id_v{VERSION}": "valid_branch_price_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_branch_price_dual_bound_loop",
                f"claim_boundary_v{VERSION}": "no dual-bound loop or termination certificate",
            },
            {
                f"blocker_id_v{VERSION}": "contractual_ifrs9_and_live_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": missing_proxy_rows,
                f"required_next_artifact_v{VERSION}": "future_contractual_or_live_holdout_gate",
                f"claim_boundary_v{VERSION}": "proxy rows remain missing and live evidence is absent",
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
                "claim_id": "v347_multi_source_relief_candidate_applied",
                "allowed": True,
                "artifact": "paper4_v347_v338_apply_multi_source_relief_candidate.csv",
                "boundary": "local applied candidate only; post-v347 repricing required",
            },
            {
                "claim_id": "v347_static_return_and_cvar_improve_vs_v338",
                "allowed": delta_return_vs_v338 > 1e-9 and delta_cvar90_vs_v338 < -1e-9,
                "artifact": "paper4_v347_v338_apply_multi_source_relief_candidate.csv",
                "boundary": "common-scenario static proxy only",
            },
            {
                "claim_id": "v347_proxy_coverage_preserved_vs_v338",
                "allowed": missing_proxy_delta == 0,
                "artifact": "paper4_v347_claim_blockers.csv",
                "boundary": "missing proxy rows increase by one",
            },
            {
                "claim_id": "v347_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v347_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    allocations.to_parquet(TABLE_DIR / "paper4_v347_v338_multi_source_relief_allocations.parquet")
    write_csv(TABLE_DIR / "paper4_v347_v338_apply_multi_source_relief_candidate.csv", summary)
    write_csv(TABLE_DIR / "paper4_v347_v338_multi_source_relief_actions.csv", action_rows)
    write_csv(
        TABLE_DIR / "paper4_v347_v338_multi_source_relief_source_summary.csv",
        source_summary,
    )
    write_csv(TABLE_DIR / "paper4_v347_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v347_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = summary.iloc[0]
    status = {
        "phase": "v347_v338_apply_multi_source_relief_candidate",
        "schema_version": "2026-05-16.347",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v347": BASE_VERSION,
        "relief_protocol_version_v347": RELIEF_VERSION,
        "applied_added_loan_ids_v347": str(row[f"applied_added_loan_ids_v{VERSION}"]),
        "applied_dropped_loan_ids_v347": str(row[f"applied_dropped_loan_ids_v{VERSION}"]),
        "selected_rows_v347": int(row[f"selected_rows_v{VERSION}"]),
        "cardinality_preserved_v347": bool(row[f"cardinality_preserved_v{VERSION}"]),
        "portfolio_exposure_v347": float(row[f"portfolio_exposure_v{VERSION}"]),
        "exposure_preserved_vs_v338_v347": bool(row[f"exposure_preserved_vs_v338_v{VERSION}"]),
        "objective_return_v347": float(row[f"objective_return_v{VERSION}"]),
        "delta_return_vs_v338_v347": delta_return_vs_v338,
        "scenario_loss_mean_v347": float(row[f"scenario_loss_mean_v{VERSION}"]),
        "delta_loss_mean_vs_v338_v347": delta_loss_mean_vs_v338,
        "scenario_loss_cvar90_v347": float(row[f"scenario_loss_cvar90_v{VERSION}"]),
        "delta_cvar90_vs_v338_v347": delta_cvar90_vs_v338,
        "observed_proxy_rows_v347": observed_proxy_rows,
        "missing_proxy_rows_v347": missing_proxy_rows,
        "observed_proxy_delta_vs_v338_v347": observed_proxy_delta,
        "missing_proxy_delta_vs_v338_v347": missing_proxy_delta,
        "period_distribution_v347": metrics["period_distribution"],
        "source_cap_violations_v347": source_cap_violations,
        "min_source_slack_v347": min_source_slack,
        "max_source_share_v347": max_source_share,
        "return_improves_vs_v338_v347": delta_return_vs_v338 > 1e-9,
        "cvar_improves_vs_v338_v347": delta_cvar90_vs_v338 < -1e-9,
        "post_v347_repricing_required_v347": True,
        "valid_branch_price_bound_v347": False,
        "full_universe_integer_optimality_claim_allowed_v347": False,
        "working_champion_claim_allowed_v347": False,
        "paper1_promotion_allowed_v347": False,
        "paper4_working_champion_changed_v347": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v347": int(len(blockers)),
        "claim_matrix_rows_v347": int(len(claim_matrix)),
        "next_artifact_v347": NEXT_ARTIFACT,
        "claim_boundary": (
            "v347 applies the v346 local two-swap candidate, but repricing, "
            "full branch-price, IFRS9, live, champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v347_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v347": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
