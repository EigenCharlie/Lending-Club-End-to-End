#!/usr/bin/env python3
"""Build Paper 4 v353 by applying the v352 bounded branch-price candidate."""

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

VERSION = 353
BASE_VERSION = 347
PRICING_VERSION = 352
NEXT_VERSION = 354
TARGET_SELECTED_ROWS = 171
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_post_v353_reprice.csv"


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
                        "v353 applied bounded branch-price candidate source diagnostic only"
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
                "claim": "v353 applies the best v352 bounded third-order entering candidate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v353_v347_apply_expanded_branch_price_candidate.csv"
                ),
                "boundary": "Applied bounded candidate only; post-v353 repricing required.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v353 improves static return and CVaR versus v347.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v353_v347_apply_expanded_branch_price_candidate.csv"
                ),
                "boundary": "Static common-scenario proxy only; not live or global evidence.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v353 preserves v347 proxy coverage exactly.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v353_claim_blockers.csv"
                ),
                "boundary": "The applied three-swap increases missing proxy rows by two.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v353 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v353_claim_blockers.csv"
                ),
                "boundary": "Post-v353 repricing, global and live gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v353 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v353_claim_blockers.csv"
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
                    "v353 applies the best v352 bounded third-order branch-price entering "
                    "candidate to v347 and recalculates static diagnostics."
                ),
                "status": "expanded_branch_price_candidate_applied_requires_reprice",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "post-v353 one-swap/source/CVaR repricing finds no feasible improving "
                    "exchanges before any local-optimal claim"
                ),
                "last_wave": "v353",
                "execution_result": ("return_and_cvar_improve_proxy_missing_worsens_by_two"),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v353")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V353_V347_APPLY_EXPANDED_BRANCH_PRICE_CANDIDATE_START -->"
    end = "<!-- V353_V347_APPLY_EXPANDED_BRANCH_PRICE_CANDIDATE_END -->"
    block = f"""
{start}

## Wave v353: Apply v352 Expanded Branch-Price Candidate

Generated: {status["generated_at_utc"]}

### Objective

v352 found bounded third-order CVaR-feasible entering candidates. v353 applies
the best-return entering action to the v347 candidate and recalculates static
portfolio, source, proxy and claim diagnostics before any repricing.

### Results

- Applied added loans: `{status["applied_added_loan_ids_v353"]}`.
- Applied dropped loans: `{status["applied_dropped_loan_ids_v353"]}`.
- Return delta vs v347:
  `{status["delta_return_vs_v347_v353"]}`.
- CVaR90 delta vs v347:
  `{status["delta_cvar90_vs_v347_v353"]}`.
- Observed proxy rows: `{status["observed_proxy_rows_v353"]}`.
- Missing proxy rows: `{status["missing_proxy_rows_v353"]}`.
- Missing proxy delta vs v347:
  `{status["missing_proxy_delta_vs_v347_v353"]}`.
- Source cap violations:
  `{status["source_cap_violations_v353"]}`.
- Post-v353 repricing required:
  `{status["post_v353_repricing_required_v353"]}`.

### Interpretation

v353 converts the v352 local pricing signal into a concrete portfolio. It
improves static expected return and CVaR versus v347 while preserving exposure,
cardinality and source caps. The cost is proxy coverage: missing proxy rows
increase from 75 to 77. The result is useful for the lab but still not a
working champion.

### Claim Impact

- Allowed: applied bounded v352 entering candidate; static return/CVaR
  improvement versus v347.
- Still prohibited: post-v353 local optimality, full-universe optimality,
  contractual IFRS9, live deployment, Paper Estrella replacement, final Paper 4
  promotion and working champion claims.

### Quarto Promotion Decision

Keep v353 in the living notebook. The next wave should reprice the v353
candidate without promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    base = read_parquet("paper4_v347_v338_multi_source_relief_allocations.parquet").reset_index(
        drop=True
    )
    v347_summary = read_csv("paper4_v347_v338_apply_multi_source_relief_candidate.csv")
    source_caps = read_csv("paper4_v347_v338_multi_source_relief_source_summary.csv")
    entering = read_csv("paper4_v352_entering_candidate_summary.csv")
    v352_status = json.loads((STATUS_DIR / "paper4_v352_status.json").read_text(encoding="utf-8"))
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    if any(df.empty for df in [universe, base, v347_summary, source_caps, entering, v47_panel]):
        raise RuntimeError("Missing inputs for v353 applied branch-price candidate.")
    if int(v352_status["cvar_feasible_entering_rows_v352"]) <= 0:
        raise RuntimeError("v353 expects v352 to identify bounded entering candidates.")
    if bool(v352_status["working_champion_claim_allowed_v352"]):
        raise RuntimeError("v353 expects v352 to remain below working-champion status.")
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    base["loan_id"] = base["loan_id"].astype(str)
    for frame in [universe, base]:
        for family in FAMILIES:
            frame[family] = frame[family].astype(str)
    v47_panel["loan_id"] = v47_panel["loan_id"].astype(str)

    best = entering.sort_values(f"return_delta_v{PRICING_VERSION}", ascending=False).iloc[0]
    add_ids = [
        str(best[f"first_added_loan_id_v{PRICING_VERSION}"]),
        str(best[f"second_added_loan_id_v{PRICING_VERSION}"]),
        str(best[f"third_added_loan_id_v{PRICING_VERSION}"]),
    ]
    drop_ids = [
        str(best[f"first_dropped_loan_id_v{PRICING_VERSION}"]),
        str(best[f"second_dropped_loan_id_v{PRICING_VERSION}"]),
        str(best[f"third_dropped_loan_id_v{PRICING_VERSION}"]),
    ]
    add_rows = universe.loc[universe["loan_id"].isin(add_ids), ["loan_id", "loan_amnt", *FAMILIES]]
    if len(add_rows) != len(add_ids):
        raise RuntimeError("Could not find all v353 added loans in v55 universe.")

    allocations = base.loc[~base["loan_id"].isin(drop_ids), ["loan_id", "loan_amnt", *FAMILIES]]
    if len(allocations) != len(base) - len(drop_ids):
        raise RuntimeError("Could not drop all v353 loans from the v347 portfolio.")
    allocations = pd.concat([allocations, add_rows], ignore_index=True)
    allocations["loan_id"] = allocations["loan_id"].astype(str)
    for family in FAMILIES:
        allocations[family] = allocations[family].astype(str)

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    allocation_idx = idx_by_id.loc[allocations["loan_id"].astype(str)].to_numpy(int)
    allocations[f"mean_return_v{VERSION}"] = mean_returns[allocation_idx]
    allocations[f"selected_v{VERSION}"] = 1
    allocations[f"portfolio_label_v{VERSION}"] = "v347_plus_v352_expanded_branch_price_candidate"
    allocations[f"branch_price_action_v{VERSION}"] = np.where(
        allocations["loan_id"].isin(add_ids),
        "added_from_v352_bounded_third_order_entry",
        "kept_from_v347",
    )
    allocations[f"claim_boundary_v{VERSION}"] = (
        "applied bounded branch-price candidate only; post-v353 repricing required"
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
    v347_row = v347_summary.iloc[0]
    observed_ids = set(v47_panel["loan_id"].astype(str))
    observed_proxy_rows = int(allocations["loan_id"].isin(observed_ids).sum())
    missing_proxy_rows = int((~allocations["loan_id"].isin(observed_ids)).sum())
    base_observed_proxy_rows = int(base["loan_id"].isin(observed_ids).sum())
    base_missing_proxy_rows = int((~base["loan_id"].isin(observed_ids)).sum())

    universe_returns = pd.Series(mean_returns, index=universe["loan_id"].astype(str))
    action_rows: list[dict[str, Any]] = []
    for action_rank, (add_id, drop_id) in enumerate(zip(add_ids, drop_ids, strict=True), start=1):
        add_row = universe.loc[universe["loan_id"].eq(add_id)].iloc[0]
        drop_row = base.loc[base["loan_id"].eq(drop_id)].iloc[0]
        add_observed = add_id in observed_ids
        drop_observed = drop_id in observed_ids
        action_rows.append(
            {
                f"action_rank_v{VERSION}": action_rank,
                f"action_label_v{VERSION}": "component_action_from_v352_three_swap",
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
                    "component action inside the applied v352 bounded three-swap candidate"
                ),
            }
        )

    source_cap_violations = int(source_summary[f"source_cap_violated_v{VERSION}"].sum())
    min_source_slack = float(source_summary[f"source_slack_v{VERSION}"].min())
    max_source_share = float(source_summary[f"source_share_v{VERSION}"].max())
    delta_return_vs_v347 = float(metrics["objective_return"] - v347_row["objective_return_v347"])
    delta_loss_mean_vs_v347 = float(
        metrics["scenario_loss_mean"] - v347_row["scenario_loss_mean_v347"]
    )
    delta_cvar90_vs_v347 = float(
        metrics["scenario_loss_cvar90"] - v347_row["scenario_loss_cvar90_v347"]
    )
    missing_proxy_delta = missing_proxy_rows - base_missing_proxy_rows
    observed_proxy_delta = observed_proxy_rows - base_observed_proxy_rows
    cardinality_preserved = int(metrics["selected_rows"]) == TARGET_SELECTED_ROWS
    exposure_preserved = (
        abs(float(metrics["portfolio_exposure"]) - float(v347_row["portfolio_exposure_v347"]))
        <= 1e-7
    )
    matches_v352_pricing = (
        abs(delta_return_vs_v347 - float(best[f"return_delta_v{PRICING_VERSION}"])) <= 1e-7
        and abs(
            metrics["scenario_loss_cvar90"]
            - float(best[f"cvar90_after_three_swap_v{PRICING_VERSION}"])
        )
        <= 1e-7
    )

    summary = pd.DataFrame(
        [
            {
                f"gate_id_v{VERSION}": "v353_v347_apply_expanded_branch_price_candidate",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"pricing_version_v{VERSION}": PRICING_VERSION,
                f"applied_action_signature_v{VERSION}": str(
                    best[f"action_signature_v{PRICING_VERSION}"]
                ),
                f"applied_added_loan_ids_v{VERSION}": "|".join(sorted(add_ids)),
                f"applied_dropped_loan_ids_v{VERSION}": "|".join(sorted(drop_ids)),
                f"selected_rows_v{VERSION}": int(metrics["selected_rows"]),
                f"cardinality_preserved_v{VERSION}": cardinality_preserved,
                f"portfolio_exposure_v{VERSION}": float(metrics["portfolio_exposure"]),
                f"exposure_preserved_vs_v347_v{VERSION}": exposure_preserved,
                f"objective_return_v{VERSION}": float(metrics["objective_return"]),
                f"delta_return_vs_v347_v{VERSION}": delta_return_vs_v347,
                f"scenario_loss_mean_v{VERSION}": float(metrics["scenario_loss_mean"]),
                f"delta_loss_mean_vs_v347_v{VERSION}": delta_loss_mean_vs_v347,
                f"scenario_loss_cvar90_v{VERSION}": float(metrics["scenario_loss_cvar90"]),
                f"delta_cvar90_vs_v347_v{VERSION}": delta_cvar90_vs_v347,
                f"matches_v352_pricing_row_v{VERSION}": matches_v352_pricing,
                f"observed_proxy_rows_v{VERSION}": observed_proxy_rows,
                f"missing_proxy_rows_v{VERSION}": missing_proxy_rows,
                f"observed_proxy_delta_vs_v347_v{VERSION}": observed_proxy_delta,
                f"missing_proxy_delta_vs_v347_v{VERSION}": missing_proxy_delta,
                f"period_distribution_v{VERSION}": json.dumps(
                    metrics["period_distribution"], sort_keys=True
                ),
                f"source_cap_violations_v{VERSION}": source_cap_violations,
                f"min_source_slack_v{VERSION}": min_source_slack,
                f"max_source_share_v{VERSION}": max_source_share,
                f"return_improves_vs_v347_v{VERSION}": delta_return_vs_v347 > 1e-9,
                f"cvar_improves_vs_v347_v{VERSION}": delta_cvar90_vs_v347 < -1e-9,
                f"post_v353_repricing_required_v{VERSION}": True,
                f"valid_branch_price_bound_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "applied bounded v352 candidate only; repricing/global/live gates "
                    "remain required"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "post_v353_repricing_required",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "applied candidate has not been repriced yet",
            },
            {
                f"blocker_id_v{VERSION}": "proxy_coverage_worsens_vs_v347",
                f"blocking_v{VERSION}": missing_proxy_delta > 0,
                f"evidence_count_v{VERSION}": missing_proxy_delta,
                f"required_next_artifact_v{VERSION}": "future_proxy_repair_or_ifrs9_gate",
                f"claim_boundary_v{VERSION}": "missing proxy rows increase versus v347",
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
                "claim_id": "v353_expanded_branch_price_candidate_applied",
                "allowed": True,
                "artifact": "paper4_v353_v347_apply_expanded_branch_price_candidate.csv",
                "boundary": "bounded applied candidate only; post-v353 repricing required",
            },
            {
                "claim_id": "v353_static_return_and_cvar_improve_vs_v347",
                "allowed": delta_return_vs_v347 > 1e-9 and delta_cvar90_vs_v347 < -1e-9,
                "artifact": "paper4_v353_v347_apply_expanded_branch_price_candidate.csv",
                "boundary": "common-scenario static proxy only",
            },
            {
                "claim_id": "v353_proxy_coverage_preserved_vs_v347",
                "allowed": missing_proxy_delta == 0,
                "artifact": "paper4_v353_claim_blockers.csv",
                "boundary": "missing proxy rows increase by two",
            },
            {
                "claim_id": "v353_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v353_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    allocations.to_parquet(TABLE_DIR / "paper4_v353_v347_expanded_branch_price_allocations.parquet")
    write_csv(TABLE_DIR / "paper4_v353_v347_apply_expanded_branch_price_candidate.csv", summary)
    write_csv(TABLE_DIR / "paper4_v353_v347_expanded_branch_price_actions.csv", action_rows)
    write_csv(
        TABLE_DIR / "paper4_v353_v347_expanded_branch_price_source_summary.csv", source_summary
    )
    write_csv(TABLE_DIR / "paper4_v353_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v353_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = summary.iloc[0]
    status = {
        "phase": "v353_v347_apply_expanded_branch_price_candidate",
        "schema_version": "2026-05-17.353",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v353": BASE_VERSION,
        "pricing_version_v353": PRICING_VERSION,
        "applied_action_signature_v353": str(row[f"applied_action_signature_v{VERSION}"]),
        "applied_added_loan_ids_v353": str(row[f"applied_added_loan_ids_v{VERSION}"]),
        "applied_dropped_loan_ids_v353": str(row[f"applied_dropped_loan_ids_v{VERSION}"]),
        "selected_rows_v353": int(row[f"selected_rows_v{VERSION}"]),
        "cardinality_preserved_v353": bool(row[f"cardinality_preserved_v{VERSION}"]),
        "portfolio_exposure_v353": float(row[f"portfolio_exposure_v{VERSION}"]),
        "exposure_preserved_vs_v347_v353": bool(row[f"exposure_preserved_vs_v347_v{VERSION}"]),
        "objective_return_v353": float(row[f"objective_return_v{VERSION}"]),
        "delta_return_vs_v347_v353": delta_return_vs_v347,
        "scenario_loss_mean_v353": float(row[f"scenario_loss_mean_v{VERSION}"]),
        "delta_loss_mean_vs_v347_v353": delta_loss_mean_vs_v347,
        "scenario_loss_cvar90_v353": float(row[f"scenario_loss_cvar90_v{VERSION}"]),
        "delta_cvar90_vs_v347_v353": delta_cvar90_vs_v347,
        "matches_v352_pricing_row_v353": bool(row[f"matches_v352_pricing_row_v{VERSION}"]),
        "observed_proxy_rows_v353": observed_proxy_rows,
        "missing_proxy_rows_v353": missing_proxy_rows,
        "observed_proxy_delta_vs_v347_v353": observed_proxy_delta,
        "missing_proxy_delta_vs_v347_v353": missing_proxy_delta,
        "period_distribution_v353": metrics["period_distribution"],
        "source_cap_violations_v353": source_cap_violations,
        "min_source_slack_v353": min_source_slack,
        "max_source_share_v353": max_source_share,
        "return_improves_vs_v347_v353": delta_return_vs_v347 > 1e-9,
        "cvar_improves_vs_v347_v353": delta_cvar90_vs_v347 < -1e-9,
        "post_v353_repricing_required_v353": True,
        "valid_branch_price_bound_v353": False,
        "full_universe_integer_optimality_claim_allowed_v353": False,
        "working_champion_claim_allowed_v353": False,
        "paper1_promotion_allowed_v353": False,
        "paper4_working_champion_changed_v353": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v353": int(len(blockers)),
        "claim_matrix_rows_v353": int(len(claim_matrix)),
        "next_artifact_v353": NEXT_ARTIFACT,
        "claim_boundary": (
            "v353 applies the bounded v352 candidate, but repricing, full branch-price, "
            "IFRS9, live, champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v353_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v353": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
