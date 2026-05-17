#!/usr/bin/env python3
"""Build Paper 4 v350 v347 dual-bound readiness artifacts after proxy gate."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

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

VERSION = 350
BASE_VERSION = 347
REPRICE_VERSION = 348
PROXY_GATE_VERSION = 349
NEXT_VERSION = 351
TIGHT_SOURCE_SLACK_THRESHOLD = 2e-4
DIRECT_MIP_BINARY_GUARD = 50_000
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v347_branch_price_or_dual_bound_loop.csv"


def _source_slack_hotspots(source_summary: pd.DataFrame) -> pd.DataFrame:
    out = source_summary.copy()
    out[f"cap_share_v{VERSION}"] = pd.to_numeric(out[f"cap_share_v{BASE_VERSION}"], errors="coerce")
    out[f"source_exposure_v{VERSION}"] = pd.to_numeric(
        out[f"source_exposure_v{BASE_VERSION}"], errors="coerce"
    )
    out[f"source_share_v{VERSION}"] = pd.to_numeric(
        out[f"source_share_v{BASE_VERSION}"], errors="coerce"
    )
    out[f"source_slack_v{VERSION}"] = pd.to_numeric(
        out[f"source_slack_v{BASE_VERSION}"], errors="coerce"
    )
    out = out.sort_values(
        [f"source_slack_v{VERSION}", "source_family", "source_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    out[f"source_slack_rank_v{VERSION}"] = np.arange(1, len(out) + 1)
    out[f"source_tight_flag_v{VERSION}"] = (
        out[f"source_slack_v{VERSION}"] <= TIGHT_SOURCE_SLACK_THRESHOLD
    )
    out[f"branch_price_priority_v{VERSION}"] = out[f"source_slack_rank_v{VERSION}"].where(
        out[f"source_tight_flag_v{VERSION}"], other=np.nan
    )
    out[f"required_next_artifact_v{VERSION}"] = np.where(
        out[f"source_tight_flag_v{VERSION}"],
        NEXT_ARTIFACT,
        "none_for_non_tight_source_scope",
    )
    out[f"claim_boundary_v{VERSION}"] = (
        "v347 source-slack hotspot map only; no branch-price termination certificate"
    )
    return out[
        [
            "source_family",
            "source_id",
            f"cap_share_v{VERSION}",
            f"source_exposure_v{VERSION}",
            f"source_share_v{VERSION}",
            f"source_slack_v{VERSION}",
            f"source_slack_rank_v{VERSION}",
            f"source_tight_flag_v{VERSION}",
            f"branch_price_priority_v{VERSION}",
            f"required_next_artifact_v{VERSION}",
            f"claim_boundary_v{VERSION}",
        ]
    ]


def _candidate_partition(
    *,
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    observed_ids: set[str],
    v349_pool_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_ids = set(selected["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    candidates["loan_id"] = candidates["loan_id"].astype(str)
    candidates[f"observed_v47_proxy_v{VERSION}"] = candidates["loan_id"].isin(observed_ids)
    observed_omitted = int(candidates[f"observed_v47_proxy_v{VERSION}"].sum())
    unobserved_omitted = int((~candidates[f"observed_v47_proxy_v{VERSION}"]).sum())
    v349_row = v349_pool_summary.iloc[0]
    full_omitted = int(len(candidates))
    partition = pd.DataFrame(
        [
            {
                f"partition_id_v{VERSION}": "v347_full_v55_omitted_partition_after_v349",
                f"full_universe_rows_v{VERSION}": int(len(universe)),
                f"selected_rows_v{VERSION}": int(len(selected_ids)),
                f"full_omitted_candidate_rows_v{VERSION}": full_omitted,
                f"observed_omitted_candidate_rows_v{VERSION}": observed_omitted,
                f"unobserved_omitted_candidate_rows_v{VERSION}": unobserved_omitted,
                f"v349_pool_rows_v{VERSION}": int(v349_row["pool_rows_v349"]),
                f"v349_observed_candidate_rows_v{VERSION}": int(
                    v349_row["observed_omitted_candidate_rows_v349"]
                ),
                f"v349_pool_includes_all_observed_omitted_v{VERSION}": bool(
                    v349_row["expanded_pool_includes_all_observed_omitted_v349"]
                ),
                f"v349_pool_share_of_full_omitted_v{VERSION}": (
                    int(v349_row["observed_omitted_candidate_rows_v349"]) / max(full_omitted, 1)
                ),
                f"unobserved_share_of_full_omitted_v{VERSION}": (
                    unobserved_omitted / max(full_omitted, 1)
                ),
                f"claim_boundary_v{VERSION}": (
                    "v349 covers all observed omitted candidates, not all full-v55 omitted loans"
                ),
            }
        ]
    )
    return candidates, partition


def _source_tight_candidate_map(
    *,
    candidates: pd.DataFrame,
    hotspots: pd.DataFrame,
    mean_returns: np.ndarray,
    idx_by_id: pd.Series,
) -> pd.DataFrame:
    candidates = candidates.copy()
    candidate_idx = idx_by_id.loc[candidates["loan_id"].astype(str)].to_numpy(int)
    candidates[f"mean_return_v{VERSION}"] = mean_returns[candidate_idx]
    tight = hotspots.loc[hotspots[f"source_tight_flag_v{VERSION}"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for _, source_row in tight.iterrows():
        family = str(source_row["source_family"])
        source_id = str(source_row["source_id"])
        if family not in FAMILIES:
            continue
        block = candidates.loc[candidates[family].astype(str).eq(source_id)].copy()
        observed = block.loc[block[f"observed_v47_proxy_v{VERSION}"]].copy()
        positive = block.loc[block[f"mean_return_v{VERSION}"].gt(0)].copy()
        top = block.sort_values(f"mean_return_v{VERSION}", ascending=False).head(1)
        rows.append(
            {
                f"pricing_block_id_v{VERSION}": f"{family}={source_id}",
                "source_family": family,
                "source_id": source_id,
                f"source_slack_v{VERSION}": float(source_row[f"source_slack_v{VERSION}"]),
                f"source_slack_rank_v{VERSION}": int(source_row[f"source_slack_rank_v{VERSION}"]),
                f"candidate_rows_v{VERSION}": int(len(block)),
                f"observed_candidate_rows_v{VERSION}": int(len(observed)),
                f"unobserved_candidate_rows_v{VERSION}": int(len(block) - len(observed)),
                f"positive_return_candidate_rows_v{VERSION}": int(len(positive)),
                f"candidate_exposure_v{VERSION}": float(block["loan_amnt"].sum()),
                f"top_candidate_loan_id_v{VERSION}": str(top["loan_id"].iloc[0])
                if not top.empty
                else "",
                f"top_candidate_mean_return_v{VERSION}": float(
                    top[f"mean_return_v{VERSION}"].iloc[0]
                )
                if not top.empty
                else None,
                f"top_candidate_observed_v47_proxy_v{VERSION}": bool(
                    top[f"observed_v47_proxy_v{VERSION}"].iloc[0]
                )
                if not top.empty
                else False,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "source-tight candidate map only; no entering-column or dual-bound claim"
                ),
            }
        )
    return pd.DataFrame(rows)


def _requirement_register(
    *,
    statuses: dict[int, dict[str, Any]],
    full_binary_variables: int,
    observed_proxy_rows: int,
    missing_proxy_rows: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                f"requirement_id_v{VERSION}": "post_v347_one_swap_local_optimality_cleared",
                f"met_v{VERSION}": bool(
                    statuses[REPRICE_VERSION][
                        f"post_v347_one_swap_local_optimality_cleared_v{REPRICE_VERSION}"
                    ]
                ),
                f"evidence_artifact_v{VERSION}": "paper4_v348_post_v347_one_swap_summary.csv",
                f"required_next_artifact_v{VERSION}": "none_for_one_swap_scope",
                f"claim_boundary_v{VERSION}": "local one-swap evidence only",
            },
            {
                f"requirement_id_v{VERSION}": "economic_proxy_repair_missing",
                f"met_v{VERSION}": not bool(
                    statuses[PROXY_GATE_VERSION][
                        f"strict_v347_repair_feasible_v{PROXY_GATE_VERSION}"
                    ]
                )
                and not bool(
                    statuses[PROXY_GATE_VERSION][
                        f"relaxed_v338_return_repair_feasible_v{PROXY_GATE_VERSION}"
                    ]
                ),
                f"evidence_artifact_v{VERSION}": "paper4_v349_proxy_repair_tier_summary.csv",
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "proxy repair is infeasible under economic floors",
            },
            {
                f"requirement_id_v{VERSION}": "proxy_gap_persists",
                f"met_v{VERSION}": missing_proxy_rows > 0,
                f"evidence_artifact_v{VERSION}": "paper4_v347_v338_apply_multi_source_relief_candidate.csv",
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": f"{observed_proxy_rows} observed and {missing_proxy_rows} missing proxy rows",
            },
            {
                f"requirement_id_v{VERSION}": "direct_full_mip_guard_met",
                f"met_v{VERSION}": full_binary_variables <= DIRECT_MIP_BINARY_GUARD,
                f"evidence_artifact_v{VERSION}": "paper4_v350_full_universe_candidate_partition.csv",
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "full-v55 binary variable count exceeds direct-MIP guard",
            },
            {
                f"requirement_id_v{VERSION}": "branch_price_dual_bound_loop_executed",
                f"met_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "missing",
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "no branch-price dual-bound loop has been run",
            },
            {
                f"requirement_id_v{VERSION}": "paper4_final_promotion_absent",
                f"met_v{VERSION}": not FORBIDDEN_FINAL_PROMOTION.exists(),
                f"evidence_artifact_v{VERSION}": "status/paper4_final_promotion.json absent",
                f"required_next_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
                f"claim_boundary_v{VERSION}": "final promotion remains forbidden",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v350 creates a post-v349 v347 dual-bound readiness register.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v350_v347_dual_bound_after_proxy_gate.csv"
                ),
                "boundary": "Readiness register and source map only; no branch-price certificate.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v350 identifies source-tight v347 branch-price blockers.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v350_v347_source_slack_hotspots.csv"
                ),
                "boundary": "Hotspot map only; not proof of global optimality.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v350 proves a valid full-universe dual-bound certificate.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v350_claim_blockers.csv"
                ),
                "boundary": "No branch-price or full-v55 dual-bound loop is executed.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v350 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v350_claim_blockers.csv"
                ),
                "boundary": "Proxy, global, dynamic, online and deployment gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v350 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v350_claim_blockers.csv"
                ),
                "boundary": "No final promotion, working champion or deployment gate is created.",
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
                    "v350 converts the v348/v349 evidence into a v347 dual-bound readiness "
                    "register, source-tight map and full-v55 candidate partition."
                ),
                "status": "dual_bound_readiness_register_created_no_certificate",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "run an actual branch-price or dual-bound loop without promotion"
                ),
                "last_wave": "v350",
                "execution_result": "source_tight_and_direct_mip_guard_block_global_claim",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v350")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V350_V347_DUAL_BOUND_AFTER_PROXY_GATE_START -->"
    end = "<!-- V350_V347_DUAL_BOUND_AFTER_PROXY_GATE_END -->"
    block = f"""
{start}

## Wave v350: v347 Dual-Bound Readiness After Proxy Gate

Generated: {status["generated_at_utc"]}

### Objective

v348 cleared one-swap local optimality for the v347 candidate and v349 showed
that economic proxy repair is infeasible in the all-observed-omitted pool. v350
turns that evidence into a dual-bound readiness register and branch-price
hotspot map.

### Results

- Full-v55 binary variables: `{status["full_binary_variables_v350"]}`.
- Direct full-MIP guard met:
  `{status["direct_full_mip_guard_met_v350"]}`.
- Observed omitted candidates:
  `{status["observed_omitted_candidate_rows_v350"]}`.
- Unobserved omitted candidates:
  `{status["unobserved_omitted_candidate_rows_v350"]}`.
- Tight source rows: `{status["tight_source_rows_v350"]}`.
- Unique source-tight candidate rows:
  `{status["unique_source_tight_candidate_rows_v350"]}`.
- Positive source-tight candidate rows:
  `{status["positive_source_tight_candidate_rows_v350"]}`.
- Branch-price dual-bound loop executed:
  `{status["branch_price_dual_bound_loop_executed_v350"]}`.
- Valid branch-price bound:
  `{status["valid_branch_price_bound_v350"]}`.

### Interpretation

v350 does not prove a global certificate; it makes the missing certificate
explicit. The direct full-v55 MIP remains beyond the guard, the proxy gap
persists, and the tight source frontier is still grade A / score decile 0. The
next executable wave should run an actual branch-price or dual-bound loop.

### Claim Impact

- Allowed: v347 dual-bound readiness register and source-tight blocker map.
- Still prohibited: full-universe branch-price termination, valid global
  integer optimality, contractual IFRS9, live deployability, Paper Estrella
  replacement, final Paper 4 promotion and working champion claims.

### Quarto Promotion Decision

Keep v350 in the living notebook. The next wave should execute a branch-price
or dual-bound loop without promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v347_v338_multi_source_relief_allocations.parquet").reset_index(
        drop=True
    )
    source_summary = read_csv("paper4_v347_v338_multi_source_relief_source_summary.csv")
    v347_summary = read_csv("paper4_v347_v338_apply_multi_source_relief_candidate.csv")
    v349_pool = read_csv("paper4_v349_proxy_repair_pool_summary.csv")
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    statuses = {
        REPRICE_VERSION: json.loads(
            (STATUS_DIR / f"paper4_v{REPRICE_VERSION}_status.json").read_text(encoding="utf-8")
        ),
        PROXY_GATE_VERSION: json.loads(
            (STATUS_DIR / f"paper4_v{PROXY_GATE_VERSION}_status.json").read_text(encoding="utf-8")
        ),
    }
    if any(
        df.empty for df in [universe, selected, source_summary, v347_summary, v349_pool, v47_panel]
    ):
        raise RuntimeError("Missing v350 dual-bound readiness inputs.")
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    for frame in [universe, selected]:
        for family in FAMILIES:
            frame[family] = frame[family].astype(str)
    observed_ids = set(v47_panel["loan_id"].astype(str))
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    del losses
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    hotspots = _source_slack_hotspots(source_summary)
    candidates, partition = _candidate_partition(
        universe=universe,
        selected=selected,
        observed_ids=observed_ids,
        v349_pool_summary=v349_pool,
    )
    source_tight_map = _source_tight_candidate_map(
        candidates=candidates,
        hotspots=hotspots,
        mean_returns=mean_returns,
        idx_by_id=idx_by_id,
    )
    v347_row = v347_summary.iloc[0]
    observed_proxy_rows = int(v347_row["observed_proxy_rows_v347"])
    missing_proxy_rows = int(v347_row["missing_proxy_rows_v347"])
    full_binary_variables = int(len(universe))
    register = _requirement_register(
        statuses=statuses,
        full_binary_variables=full_binary_variables,
        observed_proxy_rows=observed_proxy_rows,
        missing_proxy_rows=missing_proxy_rows,
    )
    tight_source_rows = int(hotspots[f"source_tight_flag_v{VERSION}"].sum())
    tight_masks = [
        candidates[str(row["source_family"])].astype(str).eq(str(row["source_id"])).to_numpy()
        for _, row in hotspots.loc[
            hotspots[f"source_tight_flag_v{VERSION}"].astype(bool)
        ].iterrows()
        if str(row["source_family"]) in FAMILIES
    ]
    tight_mask = (
        np.logical_or.reduce(tight_masks) if tight_masks else np.zeros(len(candidates), bool)
    )
    candidates_with_returns = candidates.copy()
    candidate_idx = idx_by_id.loc[candidates_with_returns["loan_id"].astype(str)].to_numpy(int)
    candidates_with_returns[f"mean_return_v{VERSION}"] = mean_returns[candidate_idx]
    unique_source_tight_candidates = int(
        candidates_with_returns.loc[tight_mask, "loan_id"].nunique()
    )
    positive_source_tight_candidates = int(
        candidates_with_returns.loc[
            tight_mask & candidates_with_returns[f"mean_return_v{VERSION}"].gt(0), "loan_id"
        ].nunique()
    )
    direct_full_mip_guard_met = full_binary_variables <= DIRECT_MIP_BINARY_GUARD
    branch_price_dual_bound_loop_executed = False
    valid_branch_price_bound = False
    main = pd.DataFrame(
        [
            {
                f"gate_id_v{VERSION}": "v350_v347_dual_bound_after_proxy_gate",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"reprice_version_v{VERSION}": REPRICE_VERSION,
                f"proxy_gate_version_v{VERSION}": PROXY_GATE_VERSION,
                f"full_binary_variables_v{VERSION}": full_binary_variables,
                f"direct_full_mip_binary_guard_v{VERSION}": DIRECT_MIP_BINARY_GUARD,
                f"direct_full_mip_guard_met_v{VERSION}": direct_full_mip_guard_met,
                f"selected_rows_v{VERSION}": int(len(selected)),
                f"observed_proxy_rows_v{VERSION}": observed_proxy_rows,
                f"missing_proxy_rows_v{VERSION}": missing_proxy_rows,
                f"full_omitted_candidate_rows_v{VERSION}": int(
                    partition[f"full_omitted_candidate_rows_v{VERSION}"].iloc[0]
                ),
                f"observed_omitted_candidate_rows_v{VERSION}": int(
                    partition[f"observed_omitted_candidate_rows_v{VERSION}"].iloc[0]
                ),
                f"unobserved_omitted_candidate_rows_v{VERSION}": int(
                    partition[f"unobserved_omitted_candidate_rows_v{VERSION}"].iloc[0]
                ),
                f"tight_source_rows_v{VERSION}": tight_source_rows,
                f"unique_source_tight_candidate_rows_v{VERSION}": unique_source_tight_candidates,
                f"positive_source_tight_candidate_rows_v{VERSION}": positive_source_tight_candidates,
                f"requirement_rows_v{VERSION}": int(len(register)),
                f"requirements_met_v{VERSION}": int(register[f"met_v{VERSION}"].sum()),
                f"branch_price_dual_bound_loop_executed_v{VERSION}": (
                    branch_price_dual_bound_loop_executed
                ),
                f"valid_branch_price_bound_v{VERSION}": valid_branch_price_bound,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "dual-bound readiness register only; no branch-price certificate or promotion"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "direct_full_mip_guard_exceeded",
                f"blocking_v{VERSION}": not direct_full_mip_guard_met,
                f"evidence_count_v{VERSION}": full_binary_variables,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "full-v55 binary variable count exceeds direct guard",
            },
            {
                f"blocker_id_v{VERSION}": "branch_price_dual_bound_loop_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "no dual-bound loop has been executed",
            },
            {
                f"blocker_id_v{VERSION}": "source_tight_branch_price_frontier",
                f"blocking_v{VERSION}": tight_source_rows > 0,
                f"evidence_count_v{VERSION}": tight_source_rows,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "tight source rows require branch-price handling",
            },
            {
                f"blocker_id_v{VERSION}": "proxy_gap_persists",
                f"blocking_v{VERSION}": missing_proxy_rows > 0,
                f"evidence_count_v{VERSION}": missing_proxy_rows,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "v347 still contains missing proxy rows",
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
                "claim_id": "v350_dual_bound_readiness_register_executed",
                "allowed": True,
                "artifact": "paper4_v350_v347_dual_bound_after_proxy_gate.csv",
                "boundary": "readiness register and source map only",
            },
            {
                "claim_id": "v350_source_tight_frontier_identified",
                "allowed": tight_source_rows > 0,
                "artifact": "paper4_v350_v347_source_slack_hotspots.csv",
                "boundary": "hotspot map only; no bound",
            },
            {
                "claim_id": "v350_valid_full_universe_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v350_claim_blockers.csv",
                "boundary": "dual-bound loop missing",
            },
            {
                "claim_id": "v350_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v350_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v350_v347_dual_bound_after_proxy_gate.csv", main)
    write_csv(TABLE_DIR / "paper4_v350_v347_source_slack_hotspots.csv", hotspots)
    write_csv(TABLE_DIR / "paper4_v350_full_universe_candidate_partition.csv", partition)
    write_csv(TABLE_DIR / "paper4_v350_v347_source_tight_candidate_map.csv", source_tight_map)
    write_csv(TABLE_DIR / "paper4_v350_dual_bound_requirement_register.csv", register)
    write_csv(TABLE_DIR / "paper4_v350_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v350_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = main.iloc[0]
    status = {
        "phase": "v350_v347_dual_bound_after_proxy_gate",
        "schema_version": "2026-05-17.350",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v350": BASE_VERSION,
        "reprice_version_v350": REPRICE_VERSION,
        "proxy_gate_version_v350": PROXY_GATE_VERSION,
        "full_binary_variables_v350": full_binary_variables,
        "direct_full_mip_binary_guard_v350": DIRECT_MIP_BINARY_GUARD,
        "direct_full_mip_guard_met_v350": direct_full_mip_guard_met,
        "selected_rows_v350": int(row[f"selected_rows_v{VERSION}"]),
        "observed_proxy_rows_v350": observed_proxy_rows,
        "missing_proxy_rows_v350": missing_proxy_rows,
        "full_omitted_candidate_rows_v350": int(row[f"full_omitted_candidate_rows_v{VERSION}"]),
        "observed_omitted_candidate_rows_v350": int(
            row[f"observed_omitted_candidate_rows_v{VERSION}"]
        ),
        "unobserved_omitted_candidate_rows_v350": int(
            row[f"unobserved_omitted_candidate_rows_v{VERSION}"]
        ),
        "tight_source_rows_v350": tight_source_rows,
        "unique_source_tight_candidate_rows_v350": unique_source_tight_candidates,
        "positive_source_tight_candidate_rows_v350": positive_source_tight_candidates,
        "requirement_rows_v350": int(len(register)),
        "requirements_met_v350": int(row[f"requirements_met_v{VERSION}"]),
        "branch_price_dual_bound_loop_executed_v350": branch_price_dual_bound_loop_executed,
        "valid_branch_price_bound_v350": valid_branch_price_bound,
        "full_universe_integer_optimality_claim_allowed_v350": False,
        "working_champion_claim_allowed_v350": False,
        "paper1_promotion_allowed_v350": False,
        "paper4_working_champion_changed_v350": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v350": int(len(blockers)),
        "claim_matrix_rows_v350": int(len(claim_matrix)),
        "next_artifact_v350": NEXT_ARTIFACT,
        "claim_boundary": (
            "v350 creates a readiness register only; branch-price dual-bound, proxy, "
            "live, champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v350_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v350": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
