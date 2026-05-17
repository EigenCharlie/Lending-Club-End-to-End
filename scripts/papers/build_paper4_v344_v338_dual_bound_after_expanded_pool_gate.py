#!/usr/bin/env python3
"""Build Paper 4 v344 v338 dual-bound readiness artifacts after v343."""

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

VERSION = 344
BASE_VERSION = 338
REFERENCE_VERSION = 316
LOCAL_REPRICE_VERSION = 339
DYNAMIC_GATE_VERSION = 340
CASHFLOW_GATE_VERSION = 341
BOUNDED_REPAIR_VERSION = 342
EXPANDED_POOL_GATE_VERSION = 343
DIRECT_MIP_GUARD_VERSION = 283
NEXT_VERSION = 345
TIGHT_SOURCE_SLACK_THRESHOLD = 2e-4
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v338_source_tight_branch_price_screen.csv"


def _source_slack_hotspots(source_summary: pd.DataFrame) -> pd.DataFrame:
    cap_col = f"cap_share_v{BASE_VERSION}"
    exposure_col = f"source_exposure_v{BASE_VERSION}"
    share_col = f"source_share_v{BASE_VERSION}"
    slack_col = f"source_slack_v{BASE_VERSION}"
    out = source_summary.copy()
    out[f"cap_share_v{VERSION}"] = pd.to_numeric(out[cap_col], errors="coerce")
    out[f"source_exposure_v{VERSION}"] = pd.to_numeric(out[exposure_col], errors="coerce")
    out[f"source_share_v{VERSION}"] = pd.to_numeric(out[share_col], errors="coerce")
    out[f"source_slack_v{VERSION}"] = pd.to_numeric(out[slack_col], errors="coerce")
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
        "v338 source-slack hotspot map only; no branch-price termination certificate"
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
    v343_pool_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_ids = set(selected["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    candidates["loan_id"] = candidates["loan_id"].astype(str)
    candidates[f"observed_v47_proxy_v{VERSION}"] = candidates["loan_id"].isin(observed_ids)
    observed_omitted = int(candidates[f"observed_v47_proxy_v{VERSION}"].sum())
    unobserved_omitted = int((~candidates[f"observed_v47_proxy_v{VERSION}"]).sum())
    v343_pool_row = v343_pool_summary.iloc[0]
    expanded_observed_candidates = int(v343_pool_row["observed_omitted_candidate_rows_v343"])
    full_omitted = int(len(candidates))
    partition = pd.DataFrame(
        [
            {
                f"partition_id_v{VERSION}": "v338_full_v55_omitted_partition_after_v343",
                f"full_universe_rows_v{VERSION}": int(len(universe)),
                f"selected_rows_v{VERSION}": int(len(selected_ids)),
                f"full_omitted_candidate_rows_v{VERSION}": full_omitted,
                f"observed_omitted_candidate_rows_v{VERSION}": observed_omitted,
                f"unobserved_omitted_candidate_rows_v{VERSION}": unobserved_omitted,
                f"expanded_pool_rows_v{VERSION}": int(v343_pool_row["pool_rows_v343"]),
                f"expanded_pool_observed_candidate_rows_v{VERSION}": expanded_observed_candidates,
                f"expanded_pool_includes_all_observed_omitted_v{VERSION}": bool(
                    v343_pool_row["expanded_pool_includes_all_observed_omitted_v343"]
                ),
                f"expanded_pool_share_of_full_omitted_v{VERSION}": (
                    expanded_observed_candidates / max(full_omitted, 1)
                ),
                f"unobserved_share_of_full_omitted_v{VERSION}": (
                    unobserved_omitted / max(full_omitted, 1)
                ),
                f"claim_boundary_v{VERSION}": (
                    "v343 covers all observed omitted candidates, not all full-v55 omitted loans"
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
    direct_mip_guard: int,
    observed_proxy_rows: int,
    imputed_proxy_rows: int,
) -> pd.DataFrame:
    direct_guard_met = full_binary_variables <= direct_mip_guard
    rows = [
        {
            f"requirement_id_v{VERSION}": "post_v338_one_swap_local_optimality_cleared",
            f"met_v{VERSION}": bool(
                statuses[LOCAL_REPRICE_VERSION][
                    f"post_v338_one_swap_local_optimality_cleared_v{LOCAL_REPRICE_VERSION}"
                ]
            ),
            f"evidence_artifact_v{VERSION}": "paper4_v339_post_v338_one_swap_summary.csv",
            f"required_next_artifact_v{VERSION}": "none_for_one_swap_scope",
            f"claim_boundary_v{VERSION}": "local one-swap evidence only",
        },
        {
            f"requirement_id_v{VERSION}": "dynamic_proxy_gate_executed",
            f"met_v{VERSION}": bool(
                statuses[DYNAMIC_GATE_VERSION][
                    f"dynamic_proxy_replay_executed_v{DYNAMIC_GATE_VERSION}"
                ]
            ),
            f"evidence_artifact_v{VERSION}": (
                "paper4_v340_dynamic_proxy_or_global_bound_after_v338.csv"
            ),
            f"required_next_artifact_v{VERSION}": "none_for_static_proxy_scope",
            f"claim_boundary_v{VERSION}": "periodized proxy replay only",
        },
        {
            f"requirement_id_v{VERSION}": "cashflow_online_gate_executed",
            f"met_v{VERSION}": imputed_proxy_rows >= 0 and observed_proxy_rows > 0,
            f"evidence_artifact_v{VERSION}": "paper4_v341_v338_cashflow_online_ifrs9_gate.csv",
            f"required_next_artifact_v{VERSION}": "future_contractual_or_live_holdout_gate",
            f"claim_boundary_v{VERSION}": "cashflow proxy gate only; imputation remains",
        },
        {
            f"requirement_id_v{VERSION}": "bounded_proxy_gap_repair_tested",
            f"met_v{VERSION}": not bool(
                statuses[BOUNDED_REPAIR_VERSION][
                    f"relaxed_repair_feasible_v{BOUNDED_REPAIR_VERSION}"
                ]
            ),
            f"evidence_artifact_v{VERSION}": (
                "paper4_v342_v338_proxy_gap_repair_or_branch_price_protocol.csv"
            ),
            f"required_next_artifact_v{VERSION}": "none_for_bounded_pool_scope",
            f"claim_boundary_v{VERSION}": "bounded-pool infeasibility, not global proof",
        },
        {
            f"requirement_id_v{VERSION}": "expanded_pool_proxy_gap_repair_tested",
            f"met_v{VERSION}": bool(
                statuses[EXPANDED_POOL_GATE_VERSION][
                    f"expanded_pool_includes_all_observed_omitted_v{EXPANDED_POOL_GATE_VERSION}"
                ]
            )
            and not bool(
                statuses[EXPANDED_POOL_GATE_VERSION][
                    f"relaxed_repair_feasible_v{EXPANDED_POOL_GATE_VERSION}"
                ]
            ),
            f"evidence_artifact_v{VERSION}": (
                "paper4_v343_v338_expanded_pool_or_dual_bound_gate.csv"
            ),
            f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
            f"claim_boundary_v{VERSION}": (
                "expanded observed-pool infeasibility, still not full-v55 certificate"
            ),
        },
        {
            f"requirement_id_v{VERSION}": "direct_full_v55_mip_within_guard",
            f"met_v{VERSION}": direct_guard_met,
            f"evidence_artifact_v{VERSION}": "paper4_v344_full_universe_candidate_partition.csv",
            f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
            f"claim_boundary_v{VERSION}": "full-v55 binary count exceeds direct-MIP guard",
        },
        {
            f"requirement_id_v{VERSION}": "branch_price_dual_bound_loop_executed",
            f"met_v{VERSION}": False,
            f"evidence_artifact_v{VERSION}": "paper4_v344_dual_bound_requirement_register.csv",
            f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
            f"claim_boundary_v{VERSION}": "dual-bound loop not yet executed",
        },
        {
            f"requirement_id_v{VERSION}": "valid_full_universe_gap_certificate",
            f"met_v{VERSION}": False,
            f"evidence_artifact_v{VERSION}": "paper4_v344_claim_blockers.csv",
            f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
            f"claim_boundary_v{VERSION}": "no valid branch-price or full-v55 gap certificate",
        },
        {
            f"requirement_id_v{VERSION}": "observed_or_contractual_ifrs9_complete",
            f"met_v{VERSION}": imputed_proxy_rows == 0,
            f"evidence_artifact_v{VERSION}": "paper4_v341_v338_cashflow_online_ifrs9_gate.csv",
            f"required_next_artifact_v{VERSION}": "future_observed_or_contractual_ifrs9_gate",
            f"claim_boundary_v{VERSION}": "v338 still has imputed proxy loan rows",
        },
        {
            f"requirement_id_v{VERSION}": "paper4_final_promotion_absent",
            f"met_v{VERSION}": not FORBIDDEN_FINAL_PROMOTION.exists(),
            f"evidence_artifact_v{VERSION}": "reports/paper_material/paper4/status",
            f"required_next_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
            f"claim_boundary_v{VERSION}": "final promotion remains forbidden",
        },
    ]
    return pd.DataFrame(rows)


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": (
                    "Paper 4 has a v344 v338 dual-bound readiness gate after expanded-pool repair."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v344_v338_dual_bound_after_expanded_pool_gate.csv"
                ),
                "boundary": (
                    "Readiness and blocker gate only; no branch-price termination certificate."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v344 documents that expanded-pool repair failure is not a full-universe "
                    "optimality certificate."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v344_full_universe_candidate_partition.csv"
                ),
                "boundary": (
                    "v343 covers observed omitted candidates only; full-v55 omitted loans remain."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v344 proves a valid full-universe branch-price or dual-bound certificate.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v344_claim_blockers.csv"
                ),
                "boundary": "No dual-bound loop, branch-price termination, or full-v55 solve exists.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v344 resolves contractual IFRS9 or live deployability for v338.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v344_claim_blockers.csv"
                ),
                "boundary": "v341 still relies on imputed proxy rows and internal online replay.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v344 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v344_claim_blockers.csv"
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
                    "v344 converts v343 expanded-pool repair infeasibility into a "
                    "full-v55 dual-bound readiness gate and source-tight next-step map."
                ),
                "status": "dual_bound_readiness_gate_created_no_certificate",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "execute a v338 source-tight branch-price screen or record a sharper "
                    "no-entering-column blocker without promoting"
                ),
                "last_wave": "v344",
                "execution_result": ("full_v55_dual_bound_missing_source_tight_screen_required"),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v344")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V344_V338_DUAL_BOUND_AFTER_EXPANDED_POOL_START -->"
    end = "<!-- V344_V338_DUAL_BOUND_AFTER_EXPANDED_POOL_END -->"
    block = f"""
{start}

## Wave v344: v338 Dual-Bound Readiness After Expanded Pool

Generated: {status["generated_at_utc"]}

### Objective

v343 showed that all observed omitted candidates still cannot repair v338's
proxy gap under the static v338 CVaR cap. v344 asks the next harder question:
does that failure amount to a full-universe/global certificate? It does not.
This wave partitions the full v55 omitted universe, checks the direct-MIP guard,
and maps source-tight branch-price work for the next executable screen.

### Results

- Full-v55 binary variables: `{status["full_universe_rows_v344"]}`.
- Full omitted candidate rows: `{status["full_omitted_candidate_rows_v344"]}`.
- Observed omitted candidate rows: `{status["observed_omitted_candidate_rows_v344"]}`.
- Unobserved omitted candidate rows: `{status["unobserved_omitted_candidate_rows_v344"]}`.
- Expanded pool share of full omitted:
  `{status["expanded_pool_share_of_full_omitted_v344"]}`.
- Direct full-v55 MIP guard exceeded:
  `{status["direct_full_mip_guard_exceeded_v344"]}`.
- Source-tight rows:
  `{status["source_tight_rows_v344"]}`.
- Source-tight candidate rows:
  `{status["source_tight_candidate_rows_v344"]}`.
- Unique source-tight candidate rows:
  `{status["unique_source_tight_candidate_rows_v344"]}`.
- Valid full-universe gap certificate:
  `{status["valid_full_universe_gap_certificate_v344"]}`.
- Unmet dual-bound requirement rows:
  `{status["unmet_requirement_rows_v344"]}`.

### Interpretation

v344 closes one misleading exit: expanded-pool infeasibility is not a global
proof because the full-v55 omitted universe still contains
`{status["unobserved_omitted_candidate_rows_v344"]}` unobserved loans outside
that repair pool. The next useful route is a source-tight branch-price screen
focused on the tight v338 governance blocks.

### Claim Impact

- Allowed: dual-bound readiness gate, full-v55 omitted partition, and
  source-tight next-step map.
- Still prohibited: valid full-universe branch-price/dual-bound certificate,
  contractual IFRS9, live deployability, Paper Estrella replacement, final
  Paper 4 promotion and working champion claims.

### Quarto Promotion Decision

Keep v344 in the living notebook. The next wave should execute the v338
source-tight branch-price screen without promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v338_post_v336_swap_allocations.parquet").reset_index(drop=True)
    source_summary = read_csv("paper4_v338_post_v336_swap_source_summary.csv")
    v343_pool_summary = read_csv("paper4_v343_proxy_gap_pool_summary.csv")
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    if any(df.empty for df in [universe, selected, source_summary, v343_pool_summary, v47_panel]):
        raise RuntimeError("Missing v344 full-universe dual-bound inputs.")

    statuses = {
        version: json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())
        for version in [
            DIRECT_MIP_GUARD_VERSION,
            LOCAL_REPRICE_VERSION,
            DYNAMIC_GATE_VERSION,
            CASHFLOW_GATE_VERSION,
            BOUNDED_REPAIR_VERSION,
            EXPANDED_POOL_GATE_VERSION,
        ]
    }
    if bool(statuses[EXPANDED_POOL_GATE_VERSION]["relaxed_repair_feasible_v343"]):
        raise RuntimeError("v344 expects v343 expanded-pool relaxed repair to be infeasible.")
    if not bool(
        statuses[EXPANDED_POOL_GATE_VERSION]["expanded_pool_includes_all_observed_omitted_v343"]
    ):
        raise RuntimeError("v344 expects v343 to include all observed omitted candidates.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    for family in FAMILIES:
        universe[family] = universe[family].astype(str)
        selected[family] = selected[family].astype(str)
    observed_ids = set(v47_panel["loan_id"].astype(str))
    _losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))

    candidates, partition = _candidate_partition(
        universe=universe,
        selected=selected,
        observed_ids=observed_ids,
        v343_pool_summary=v343_pool_summary,
    )
    hotspots = _source_slack_hotspots(source_summary)
    source_tight_candidates = _source_tight_candidate_map(
        candidates=candidates,
        hotspots=hotspots,
        mean_returns=mean_returns,
        idx_by_id=idx_by_id,
    )

    partition_row = partition.iloc[0]
    full_binary_variables = int(partition_row[f"full_universe_rows_v{VERSION}"])
    direct_mip_guard = int(
        statuses[DIRECT_MIP_GUARD_VERSION]["max_binary_vars_for_direct_mip_v283"]
    )
    observed_proxy_rows = int(statuses[CASHFLOW_GATE_VERSION]["observed_proxy_loan_rows_v341"])
    imputed_proxy_rows = int(statuses[CASHFLOW_GATE_VERSION]["imputed_proxy_loan_rows_v341"])
    requirements = _requirement_register(
        statuses=statuses,
        full_binary_variables=full_binary_variables,
        direct_mip_guard=direct_mip_guard,
        observed_proxy_rows=observed_proxy_rows,
        imputed_proxy_rows=imputed_proxy_rows,
    )

    tight = hotspots.loc[hotspots[f"source_tight_flag_v{VERSION}"].astype(bool)]
    source_tight_candidate_rows = int(source_tight_candidates[f"candidate_rows_v{VERSION}"].sum())
    source_tight_observed_rows = int(
        source_tight_candidates[f"observed_candidate_rows_v{VERSION}"].sum()
    )
    positive_tight_rows = int(
        source_tight_candidates[f"positive_return_candidate_rows_v{VERSION}"].sum()
    )
    tight_mask = pd.Series(False, index=candidates.index)
    for _, source_row in tight.iterrows():
        family = str(source_row["source_family"])
        source_id = str(source_row["source_id"])
        if family in FAMILIES:
            tight_mask = tight_mask | candidates[family].astype(str).eq(source_id)
    unique_tight = candidates.loc[tight_mask].copy()
    unique_tight_idx = idx_by_id.loc[unique_tight["loan_id"].astype(str)].to_numpy(int)
    unique_tight[f"mean_return_v{VERSION}"] = mean_returns[unique_tight_idx]
    unique_source_tight_candidate_rows = int(len(unique_tight))
    unique_source_tight_observed_rows = int(unique_tight[f"observed_v47_proxy_v{VERSION}"].sum())
    unique_positive_tight_rows = int(unique_tight[f"mean_return_v{VERSION}"].gt(0).sum())
    direct_guard_exceeded = full_binary_variables > direct_mip_guard
    valid_gap_certificate = False
    branch_price_loop_executed = False

    summary = pd.DataFrame(
        [
            {
                f"gate_id_v{VERSION}": "v344_v338_dual_bound_after_expanded_pool_gate",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"reference_version_v{VERSION}": REFERENCE_VERSION,
                f"local_reprice_version_v{VERSION}": LOCAL_REPRICE_VERSION,
                f"dynamic_gate_version_v{VERSION}": DYNAMIC_GATE_VERSION,
                f"cashflow_gate_version_v{VERSION}": CASHFLOW_GATE_VERSION,
                f"bounded_repair_version_v{VERSION}": BOUNDED_REPAIR_VERSION,
                f"expanded_pool_gate_version_v{VERSION}": EXPANDED_POOL_GATE_VERSION,
                f"direct_mip_guard_version_v{VERSION}": DIRECT_MIP_GUARD_VERSION,
                f"full_binary_variables_v{VERSION}": full_binary_variables,
                f"direct_mip_binary_guard_v{VERSION}": direct_mip_guard,
                f"direct_full_mip_attempted_v{VERSION}": False,
                f"direct_full_mip_guard_exceeded_v{VERSION}": direct_guard_exceeded,
                f"branch_price_dual_bound_loop_executed_v{VERSION}": branch_price_loop_executed,
                f"valid_full_universe_gap_certificate_v{VERSION}": valid_gap_certificate,
                f"selected_rows_v{VERSION}": int(partition_row[f"selected_rows_v{VERSION}"]),
                f"full_omitted_candidate_rows_v{VERSION}": int(
                    partition_row[f"full_omitted_candidate_rows_v{VERSION}"]
                ),
                f"observed_omitted_candidate_rows_v{VERSION}": int(
                    partition_row[f"observed_omitted_candidate_rows_v{VERSION}"]
                ),
                f"unobserved_omitted_candidate_rows_v{VERSION}": int(
                    partition_row[f"unobserved_omitted_candidate_rows_v{VERSION}"]
                ),
                f"expanded_pool_share_of_full_omitted_v{VERSION}": float(
                    partition_row[f"expanded_pool_share_of_full_omitted_v{VERSION}"]
                ),
                f"source_summary_rows_v{VERSION}": int(len(hotspots)),
                f"source_tight_threshold_v{VERSION}": TIGHT_SOURCE_SLACK_THRESHOLD,
                f"source_tight_rows_v{VERSION}": int(len(tight)),
                f"source_tight_candidate_rows_v{VERSION}": source_tight_candidate_rows,
                f"source_tight_observed_candidate_rows_v{VERSION}": source_tight_observed_rows,
                f"source_tight_positive_return_candidate_rows_v{VERSION}": positive_tight_rows,
                f"unique_source_tight_candidate_rows_v{VERSION}": (
                    unique_source_tight_candidate_rows
                ),
                f"unique_source_tight_observed_candidate_rows_v{VERSION}": (
                    unique_source_tight_observed_rows
                ),
                f"unique_source_tight_positive_return_candidate_rows_v{VERSION}": (
                    unique_positive_tight_rows
                ),
                f"observed_proxy_rows_v{VERSION}": observed_proxy_rows,
                f"imputed_proxy_rows_v{VERSION}": imputed_proxy_rows,
                f"unmet_requirement_rows_v{VERSION}": int(
                    (~requirements[f"met_v{VERSION}"].astype(bool)).sum()
                ),
                f"contractual_ifrs9_claim_allowed_v{VERSION}": False,
                f"strict_live_deployability_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "dual-bound readiness gate only; no branch-price termination or promotion"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "valid_full_universe_gap_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": full_binary_variables,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "no valid full-v55 branch-price or dual-bound certificate",
            },
            {
                f"blocker_id_v{VERSION}": "direct_full_mip_resource_guard_exceeded",
                f"blocking_v{VERSION}": direct_guard_exceeded,
                f"evidence_count_v{VERSION}": int(full_binary_variables - direct_mip_guard),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "direct full-v55 MIP remains resource guarded",
            },
            {
                f"blocker_id_v{VERSION}": "branch_price_dual_loop_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "branch-price dual loop not executed",
            },
            {
                f"blocker_id_v{VERSION}": "source_tight_pricing_screen_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(tight)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "source-tight candidates require explicit pricing screen",
            },
            {
                f"blocker_id_v{VERSION}": "proxy_gap_repair_infeasible_after_expanded_pool",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 2,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "v343 strict and relaxed expanded-pool tiers infeasible",
            },
            {
                f"blocker_id_v{VERSION}": "contractual_ifrs9_and_live_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": imputed_proxy_rows,
                f"required_next_artifact_v{VERSION}": "future_contractual_or_live_holdout_gate",
                f"claim_boundary_v{VERSION}": "imputed proxy rows and internal online replay remain",
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
                "claim_id": "v344_dual_bound_readiness_gate_executed",
                "allowed": True,
                "artifact": "paper4_v344_v338_dual_bound_after_expanded_pool_gate.csv",
                "boundary": "readiness and blocker gate only",
            },
            {
                "claim_id": "v344_expanded_pool_failure_not_full_universe_certificate",
                "allowed": True,
                "artifact": "paper4_v344_full_universe_candidate_partition.csv",
                "boundary": "observed omitted pool is not full-v55 omitted universe",
            },
            {
                "claim_id": "v344_valid_full_universe_gap_certificate",
                "allowed": False,
                "artifact": "paper4_v344_claim_blockers.csv",
                "boundary": "dual-bound loop missing",
            },
            {
                "claim_id": "v344_source_tight_branch_price_screen_completed",
                "allowed": False,
                "artifact": "paper4_v344_source_tight_candidate_map.csv",
                "boundary": "next screen required",
            },
            {
                "claim_id": "v344_contractual_ifrs9_or_live_deployability",
                "allowed": False,
                "artifact": "paper4_v344_claim_blockers.csv",
                "boundary": "imputation and external holdout blockers remain",
            },
            {
                "claim_id": "v344_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v344_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v344_v338_dual_bound_after_expanded_pool_gate.csv", summary)
    write_csv(TABLE_DIR / "paper4_v344_full_universe_candidate_partition.csv", partition)
    write_csv(TABLE_DIR / "paper4_v344_v338_source_slack_hotspots.csv", hotspots)
    write_csv(TABLE_DIR / "paper4_v344_source_tight_candidate_map.csv", source_tight_candidates)
    write_csv(TABLE_DIR / "paper4_v344_dual_bound_requirement_register.csv", requirements)
    write_csv(TABLE_DIR / "paper4_v344_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v344_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = summary.iloc[0]
    status = {
        "phase": "v344_v338_dual_bound_after_expanded_pool_gate",
        "schema_version": "2026-05-16.344",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v344": BASE_VERSION,
        "reference_version_v344": REFERENCE_VERSION,
        "local_reprice_version_v344": LOCAL_REPRICE_VERSION,
        "dynamic_gate_version_v344": DYNAMIC_GATE_VERSION,
        "cashflow_gate_version_v344": CASHFLOW_GATE_VERSION,
        "bounded_repair_version_v344": BOUNDED_REPAIR_VERSION,
        "expanded_pool_gate_version_v344": EXPANDED_POOL_GATE_VERSION,
        "direct_mip_guard_version_v344": DIRECT_MIP_GUARD_VERSION,
        "full_universe_rows_v344": full_binary_variables,
        "full_binary_variables_v344": full_binary_variables,
        "direct_mip_binary_guard_v344": direct_mip_guard,
        "direct_full_mip_attempted_v344": False,
        "direct_full_mip_guard_exceeded_v344": direct_guard_exceeded,
        "branch_price_dual_bound_loop_executed_v344": branch_price_loop_executed,
        "valid_full_universe_gap_certificate_v344": valid_gap_certificate,
        "selected_rows_v344": int(row[f"selected_rows_v{VERSION}"]),
        "full_omitted_candidate_rows_v344": int(row[f"full_omitted_candidate_rows_v{VERSION}"]),
        "observed_omitted_candidate_rows_v344": int(
            row[f"observed_omitted_candidate_rows_v{VERSION}"]
        ),
        "unobserved_omitted_candidate_rows_v344": int(
            row[f"unobserved_omitted_candidate_rows_v{VERSION}"]
        ),
        "expanded_pool_share_of_full_omitted_v344": float(
            row[f"expanded_pool_share_of_full_omitted_v{VERSION}"]
        ),
        "source_summary_rows_v344": int(row[f"source_summary_rows_v{VERSION}"]),
        "source_tight_threshold_v344": TIGHT_SOURCE_SLACK_THRESHOLD,
        "source_tight_rows_v344": int(row[f"source_tight_rows_v{VERSION}"]),
        "source_tight_candidate_rows_v344": source_tight_candidate_rows,
        "source_tight_observed_candidate_rows_v344": source_tight_observed_rows,
        "source_tight_positive_return_candidate_rows_v344": positive_tight_rows,
        "unique_source_tight_candidate_rows_v344": unique_source_tight_candidate_rows,
        "unique_source_tight_observed_candidate_rows_v344": unique_source_tight_observed_rows,
        "unique_source_tight_positive_return_candidate_rows_v344": unique_positive_tight_rows,
        "observed_proxy_rows_v344": observed_proxy_rows,
        "imputed_proxy_rows_v344": imputed_proxy_rows,
        "requirement_rows_v344": int(len(requirements)),
        "unmet_requirement_rows_v344": int(row[f"unmet_requirement_rows_v{VERSION}"]),
        "contractual_ifrs9_claim_allowed_v344": False,
        "strict_live_deployability_claim_allowed_v344": False,
        "working_champion_claim_allowed_v344": False,
        "full_universe_integer_optimality_claim_allowed_v344": False,
        "paper1_promotion_allowed_v344": False,
        "paper4_working_champion_changed_v344": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v344": int(len(blockers)),
        "claim_matrix_rows_v344": int(len(claim_matrix)),
        "next_artifact_v344": NEXT_ARTIFACT,
        "claim_boundary": (
            "v344 converts expanded-pool infeasibility into a dual-bound readiness gate; "
            "full-universe, IFRS9, live, champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v344_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v344": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
