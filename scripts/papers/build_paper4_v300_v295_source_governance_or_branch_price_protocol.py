#!/usr/bin/env python3
"""Build Paper 4 v300 source-governance/branch-price protocol artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from scripts.papers.paper4_one_swap_living_lab import (
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

VERSION = 300
SOURCE_CANDIDATE_VERSION = 295
GLOBAL_GATE_VERSION = 297
CASHFLOW_PROXY_VERSION = 299
NEXT_VERSION = 301
TIGHT_SLACK_THRESHOLD = 1e-4
WATCH_SLACK_THRESHOLD = 0.15


def _source_slack_hotspots(source_summary: pd.DataFrame) -> pd.DataFrame:
    hotspots = source_summary.copy()
    hotspots["source_id"] = hotspots["source_id"].astype(str)
    hotspots = hotspots.sort_values(
        [f"source_slack_v{SOURCE_CANDIDATE_VERSION}", "source_family", "source_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    hotspots[f"source_slack_rank_v{VERSION}"] = range(1, len(hotspots) + 1)
    hotspots[f"source_tight_flag_v{VERSION}"] = hotspots[
        f"source_slack_v{SOURCE_CANDIDATE_VERSION}"
    ].le(TIGHT_SLACK_THRESHOLD)
    hotspots[f"source_watch_flag_v{VERSION}"] = hotspots[
        f"source_slack_v{SOURCE_CANDIDATE_VERSION}"
    ].le(WATCH_SLACK_THRESHOLD)
    hotspots[f"source_governance_severity_v{VERSION}"] = hotspots.apply(
        lambda row: (
            "critical_tight"
            if bool(row[f"source_tight_flag_v{VERSION}"])
            else "watch"
            if bool(row[f"source_watch_flag_v{VERSION}"])
            else "monitor"
        ),
        axis=1,
    )
    hotspots[f"branch_price_priority_v{VERSION}"] = hotspots[f"source_slack_rank_v{VERSION}"].where(
        hotspots[f"source_tight_flag_v{VERSION}"], 0
    )
    hotspots[f"required_next_artifact_v{VERSION}"] = hotspots.apply(
        lambda row: (
            f"paper4_v{NEXT_VERSION}_source_tight_branch_price_pricing_or_imputation_repair.csv"
            if bool(row[f"source_tight_flag_v{VERSION}"])
            else "future_source_monitoring_only"
        ),
        axis=1,
    )
    hotspots[f"claim_boundary_v{VERSION}"] = (
        "v300 source-governance hotspot map only; no branch-price bound or promotion claim"
    )
    ordered_cols = [
        f"source_slack_rank_v{VERSION}",
        "source_family",
        "source_id",
        f"cap_share_v{SOURCE_CANDIDATE_VERSION}",
        f"source_exposure_v{SOURCE_CANDIDATE_VERSION}",
        f"source_share_v{SOURCE_CANDIDATE_VERSION}",
        f"source_slack_v{SOURCE_CANDIDATE_VERSION}",
        f"source_cap_violated_v{SOURCE_CANDIDATE_VERSION}",
        f"source_tight_flag_v{VERSION}",
        f"source_watch_flag_v{VERSION}",
        f"source_governance_severity_v{VERSION}",
        f"branch_price_priority_v{VERSION}",
        f"required_next_artifact_v{VERSION}",
        f"claim_boundary_v{VERSION}",
    ]
    return hotspots[ordered_cols]


def _proxy_imputation_by_source(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work["loan_id"] = work["loan_id"].astype(str)
    work["proxy_source_v299"] = work["proxy_source_v299"].astype(str)
    work[f"proxy_imputed_flag_v{VERSION}"] = work["proxy_source_v299"].str.startswith("imputed")
    rows: list[pd.DataFrame] = []
    for family in ("grade", "score_decile", "period"):
        family_work = work.copy()
        family_work[f"source_family_v{VERSION}"] = family
        family_work[f"source_id_v{VERSION}"] = family_work[family].astype(str)
        rows.append(family_work)
    stacked = pd.concat(rows, ignore_index=True)
    grouped = (
        stacked.groupby(
            [f"source_family_v{VERSION}", f"source_id_v{VERSION}", "proxy_source_v299"],
            dropna=False,
        )
        .agg(
            loan_rows_v300=("loan_id", "nunique"),
            panel_rows_v300=("loan_id", "size"),
            imputed_panel_rows_v300=(f"proxy_imputed_flag_v{VERSION}", "sum"),
            ecl_proxy_total_v300=("ecl_proxy_v29", "sum"),
            net_cash_proxy_total_v300=("net_cash_proxy_v47", "sum"),
        )
        .reset_index()
    )
    grouped[f"imputed_loan_rows_v{VERSION}"] = grouped.apply(
        lambda row: (
            int(row["loan_rows_v300"]) if str(row["proxy_source_v299"]).startswith("imputed") else 0
        ),
        axis=1,
    )
    grouped[f"observed_loan_rows_v{VERSION}"] = grouped.apply(
        lambda row: (
            int(row["loan_rows_v300"])
            if str(row["proxy_source_v299"]) == "observed_v47_proxy"
            else 0
        ),
        axis=1,
    )
    grouped[f"contractual_ifrs9_claim_allowed_v{VERSION}"] = False
    grouped[f"claim_boundary_v{VERSION}"] = (
        "v300 source-level proxy-imputation map; imputed rows are not contractual IFRS9"
    )
    return grouped.sort_values(
        [f"source_family_v{VERSION}", f"source_id_v{VERSION}", "proxy_source_v299"]
    ).reset_index(drop=True)


def _requirement_register(
    *,
    hotspots: pd.DataFrame,
    v297_status: dict[str, Any],
    v299_status: dict[str, Any],
) -> pd.DataFrame:
    tight = hotspots.loc[hotspots[f"source_tight_flag_v{VERSION}"].astype(bool)].copy()
    rows: list[dict[str, Any]] = [
        {
            f"requirement_id_v{VERSION}": "valid_full_universe_gap_certificate",
            f"met_v{VERSION}": False,
            f"evidence_count_v{VERSION}": int(v297_status["full_binary_variables_v297"]),
            f"evidence_artifact_v{VERSION}": "paper4_v297_global_dynamic_gate_summary.csv",
            f"required_next_artifact_v{VERSION}": (
                f"paper4_v{NEXT_VERSION}_source_tight_branch_price_pricing_or_imputation_repair.csv"
            ),
            f"claim_boundary_v{VERSION}": (
                "full-v55 direct model exceeds guard; branch-price or valid dual bound required"
            ),
        },
        {
            f"requirement_id_v{VERSION}": "direct_full_mip_resource_guard",
            f"met_v{VERSION}": False,
            f"evidence_count_v{VERSION}": int(v297_status["direct_mip_binary_guard_v297"]),
            f"evidence_artifact_v{VERSION}": "paper4_v297_status.json",
            f"required_next_artifact_v{VERSION}": (
                f"paper4_v{NEXT_VERSION}_source_tight_branch_price_pricing_or_imputation_repair.csv"
            ),
            f"claim_boundary_v{VERSION}": "276869 binary variables exceed the direct-MIP guard",
        },
        {
            f"requirement_id_v{VERSION}": "observed_cashflow_for_imputed_v295_loans",
            f"met_v{VERSION}": False,
            f"evidence_count_v{VERSION}": int(v299_status["imputed_proxy_loan_rows_v299"]),
            f"evidence_artifact_v{VERSION}": "paper4_v299_v295_cashflow_proxy_source_summary.csv",
            f"required_next_artifact_v{VERSION}": "future_observed_v295_cashflow_panel",
            f"claim_boundary_v{VERSION}": "76 v295 loans still use IFRS9-inspired imputation",
        },
        {
            f"requirement_id_v{VERSION}": "external_online_holdout_for_v295",
            f"met_v{VERSION}": False,
            f"evidence_count_v{VERSION}": 0,
            f"evidence_artifact_v{VERSION}": "paper4_v299_v295_online_temporal_summary.csv",
            f"required_next_artifact_v{VERSION}": "future_external_online_holdout",
            f"claim_boundary_v{VERSION}": "v299 online replay is internal selected-book evidence",
        },
        {
            f"requirement_id_v{VERSION}": "final_promotion_absence_guardrail",
            f"met_v{VERSION}": not FORBIDDEN_FINAL_PROMOTION.exists(),
            f"evidence_count_v{VERSION}": int(FORBIDDEN_FINAL_PROMOTION.exists()),
            f"evidence_artifact_v{VERSION}": "reports/paper_material/paper4/status",
            f"required_next_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
            f"claim_boundary_v{VERSION}": "Paper Estrella replacement and final Paper 4 remain forbidden",
        },
    ]
    for row in tight.itertuples(index=False):
        rows.append(
            {
                f"requirement_id_v{VERSION}": (
                    f"source_tight_branch_price_{row.source_family}_{row.source_id}"
                ),
                f"met_v{VERSION}": False,
                f"evidence_count_v{VERSION}": 1,
                f"evidence_artifact_v{VERSION}": "paper4_v300_v295_source_slack_hotspots.csv",
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_source_tight_branch_price_pricing_or_imputation_repair.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    f"{row.source_family}={row.source_id} is source-tight; needs dual-priced "
                    "relief or explicit branch-price blocker"
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
                "claim": "Paper 4 has a v300 v295 source-governance branch-price protocol.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v300_v295_source_governance_or_branch_price_protocol.csv"
                ),
                "boundary": "Protocol and requirement register only; no global bound.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v300 identifies v295 source-tight branch-price hotspots.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v300_v295_source_slack_hotspots.csv"
                ),
                "boundary": "Hotspot map only; not an entering-column or termination certificate.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v300 proves a valid full-universe branch-price bound.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v300_claim_blockers.csv"
                ),
                "boundary": "No dual-bound loop or branch-price termination certificate exists.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v300 resolves contractual IFRS9 or live deployability for v295.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v300_claim_blockers.csv"
                ),
                "boundary": "76 imputed proxy loans and no external online holdout remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v300 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v300_claim_blockers.csv"
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
                    "v300 converts v295-v299 blockers into a source-governance and "
                    "branch-price requirement protocol."
                ),
                "status": "source_governance_branch_price_protocol_created",
                "next_artifact": (
                    f"paper4_v{NEXT_VERSION}_source_tight_branch_price_pricing_or_imputation_repair.csv"
                ),
                "success_condition": (
                    "execute a source-tight branch-price/pricing repair or record a stronger "
                    "dual-bound blocker without promoting"
                ),
                "last_wave": "v300",
                "execution_result": "grade_A_and_score0_tight_cashflow_imputation_global_bound_blocked",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v300")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V300_SOURCE_GOVERNANCE_BRANCH_PRICE_PROTOCOL_START -->"
    end = "<!-- V300_SOURCE_GOVERNANCE_BRANCH_PRICE_PROTOCOL_END -->"
    block = f"""
{start}

## Wave v300: Source Governance / Branch-Price Protocol

Generated: {status["generated_at_utc"]}

### Objective

v299 made v295 cashflow/online evidence candidate-specific, but left two hard
blockers: source-tight global evidence and imputed IFRS9 proxy rows. v300 turns
those blockers into an executable protocol for branch-price/source-governance
work without promoting the candidate.

### Results

- Source summary rows: `{status["source_summary_rows_v300"]}`.
- Tight source rows: `{status["source_tight_rows_v300"]}`.
- Tightest source: `{status["tightest_source_family_v300"]}={status["tightest_source_id_v300"]}`.
- Tightest source slack: `{status["tightest_source_slack_v300"]}`.
- Second tight source: `{status["second_tight_source_family_v300"]}={status["second_tight_source_id_v300"]}`.
- Second tight source slack: `{status["second_tight_source_slack_v300"]}`.
- v295 imputed proxy loan rows: `{status["proxy_imputed_loan_rows_v300"]}`.
- v295 observed proxy loan rows: `{status["proxy_observed_loan_rows_v300"]}`.
- Full-v55 binary variables: `{status["full_binary_variables_v300"]}`.
- Direct MIP guard: `{status["direct_mip_binary_guard_v300"]}`.
- Valid branch-price bound: `{status["valid_branch_price_bound_v300"]}`.

### Interpretation

The source picture is extremely localized: grade A and score decile 0 are the
binding governance hotspots. The cashflow picture is also concentrated: v295 is
fully covered only after explicit proxy imputation. v300 is therefore useful as
a handoff map, not as a claim upgrade.

### Claim Impact

- Allowed: v300 source-governance/branch-price protocol and hotspot register.
- Still prohibited: full-universe global optimality, contractual IFRS9, live
  deployability, Paper Estrella replacement, final Paper 4 promotion and
  working champion claims.

### Quarto Promotion Decision

Keep v300 in the living notebook. The next wave should execute a source-tight
branch-price/pricing repair or record a stronger blocker.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    source_summary = read_csv("paper4_v295_broader_source_summary.csv")
    cashflow_panel = read_parquet("paper4_v299_v295_cashflow_proxy_panel.parquet")
    v295_status = json.loads((STATUS_DIR / "paper4_v295_status.json").read_text(encoding="utf-8"))
    v297_status = json.loads((STATUS_DIR / "paper4_v297_status.json").read_text(encoding="utf-8"))
    v299_status = json.loads((STATUS_DIR / "paper4_v299_status.json").read_text(encoding="utf-8"))
    if source_summary.empty or cashflow_panel.empty:
        raise RuntimeError("Missing v295 source summary or v299 cashflow panel for v300.")
    if bool(v297_status["valid_full_universe_gap_certificate_v297"]):
        raise RuntimeError("v300 expects the v297 full-universe gap certificate to remain missing.")
    if int(v299_status["imputed_proxy_loan_rows_v299"]) <= 0:
        raise RuntimeError("v300 expects v299 proxy imputation blockers to remain open.")

    hotspots = _source_slack_hotspots(source_summary)
    proxy_map = _proxy_imputation_by_source(cashflow_panel)
    requirements = _requirement_register(
        hotspots=hotspots,
        v297_status=v297_status,
        v299_status=v299_status,
    )
    tight = hotspots.loc[hotspots[f"source_tight_flag_v{VERSION}"].astype(bool)].copy()
    tightest = hotspots.iloc[0]
    second = hotspots.iloc[1]
    selected_rows = int(v295_status["selected_rows_v295"])
    imputed_rows = int(v299_status["imputed_proxy_loan_rows_v299"])
    observed_rows = int(v299_status["observed_proxy_loan_rows_v299"])
    valid_branch_price_bound = False
    strict_live_allowed = False
    contractual_ifrs9_allowed = False

    protocol = pd.DataFrame(
        [
            {
                f"protocol_id_v{VERSION}": "v295_source_governance_branch_price_protocol",
                f"source_candidate_version_v{VERSION}": SOURCE_CANDIDATE_VERSION,
                f"global_gate_version_v{VERSION}": GLOBAL_GATE_VERSION,
                f"cashflow_proxy_version_v{VERSION}": CASHFLOW_PROXY_VERSION,
                f"selected_rows_v{VERSION}": selected_rows,
                f"source_summary_rows_v{VERSION}": int(len(hotspots)),
                f"source_tight_rows_v{VERSION}": int(len(tight)),
                f"source_cap_violations_v{VERSION}": int(
                    hotspots[f"source_cap_violated_v{SOURCE_CANDIDATE_VERSION}"].sum()
                ),
                f"tightest_source_family_v{VERSION}": str(tightest["source_family"]),
                f"tightest_source_id_v{VERSION}": str(tightest["source_id"]),
                f"tightest_source_slack_v{VERSION}": float(
                    tightest[f"source_slack_v{SOURCE_CANDIDATE_VERSION}"]
                ),
                f"second_tight_source_family_v{VERSION}": str(second["source_family"]),
                f"second_tight_source_id_v{VERSION}": str(second["source_id"]),
                f"second_tight_source_slack_v{VERSION}": float(
                    second[f"source_slack_v{SOURCE_CANDIDATE_VERSION}"]
                ),
                f"proxy_observed_loan_rows_v{VERSION}": observed_rows,
                f"proxy_imputed_loan_rows_v{VERSION}": imputed_rows,
                f"proxy_imputed_loan_share_v{VERSION}": imputed_rows / max(selected_rows, 1),
                f"full_binary_variables_v{VERSION}": int(v297_status["full_binary_variables_v297"]),
                f"direct_mip_binary_guard_v{VERSION}": int(
                    v297_status["direct_mip_binary_guard_v297"]
                ),
                f"direct_full_mip_guard_exceeded_v{VERSION}": (
                    int(v297_status["full_binary_variables_v297"])
                    > int(v297_status["direct_mip_binary_guard_v297"])
                ),
                f"valid_branch_price_bound_v{VERSION}": valid_branch_price_bound,
                f"source_governance_protocol_executed_v{VERSION}": True,
                f"contractual_ifrs9_claim_allowed_v{VERSION}": contractual_ifrs9_allowed,
                f"strict_live_deployability_claim_allowed_v{VERSION}": strict_live_allowed,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_source_tight_branch_price_pricing_or_imputation_repair.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "protocol only; source governance and branch-price requirements remain open"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "source_tight_branch_prices_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(tight)),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_source_tight_branch_price_pricing_or_imputation_repair.csv"
                ),
                f"claim_boundary_v{VERSION}": "grade A and score decile 0 are branch-price hotspots",
            },
            {
                f"blocker_id_v{VERSION}": "full_universe_gap_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v297_status["full_binary_variables_v297"]),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_source_tight_branch_price_pricing_or_imputation_repair.csv"
                ),
                f"claim_boundary_v{VERSION}": "no valid global dual-bound certificate exists",
            },
            {
                f"blocker_id_v{VERSION}": "cashflow_proxy_imputation_required",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": imputed_rows,
                f"required_next_artifact_v{VERSION}": "future_observed_v295_cashflow_panel",
                f"claim_boundary_v{VERSION}": "imputed rows block contractual IFRS9",
            },
            {
                f"blocker_id_v{VERSION}": "external_online_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": "future_external_online_holdout",
                f"claim_boundary_v{VERSION}": "v299 online replay is internal selected-book only",
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
                "claim_id": "v300_source_governance_branch_price_protocol_executed",
                "allowed": True,
                "artifact": "paper4_v300_v295_source_governance_or_branch_price_protocol.csv",
                "boundary": "protocol and requirement register only",
            },
            {
                "claim_id": "v300_source_tight_hotspots_identified",
                "allowed": True,
                "artifact": "paper4_v300_v295_source_slack_hotspots.csv",
                "boundary": "hotspot map, not branch-price termination",
            },
            {
                "claim_id": "v300_valid_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v300_claim_blockers.csv",
                "boundary": "dual-bound loop missing",
            },
            {
                "claim_id": "v300_contractual_ifrs9_or_live_deployability",
                "allowed": False,
                "artifact": "paper4_v300_claim_blockers.csv",
                "boundary": "imputation and external holdout blockers remain",
            },
            {
                "claim_id": "v300_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v300_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(
        TABLE_DIR / "paper4_v300_v295_source_governance_or_branch_price_protocol.csv",
        protocol,
    )
    write_csv(TABLE_DIR / "paper4_v300_v295_source_slack_hotspots.csv", hotspots)
    write_csv(TABLE_DIR / "paper4_v300_v295_proxy_imputation_by_source.csv", proxy_map)
    write_csv(TABLE_DIR / "paper4_v300_branch_price_requirement_register.csv", requirements)
    write_csv(TABLE_DIR / "paper4_v300_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v300_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    protocol_row = protocol.iloc[0]
    status = {
        "phase": "v300_v295_source_governance_or_branch_price_protocol",
        "schema_version": "2026-05-15.300",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "source_candidate_version_v300": SOURCE_CANDIDATE_VERSION,
        "global_gate_version_v300": GLOBAL_GATE_VERSION,
        "cashflow_proxy_version_v300": CASHFLOW_PROXY_VERSION,
        "selected_rows_v300": selected_rows,
        "source_summary_rows_v300": int(protocol_row[f"source_summary_rows_v{VERSION}"]),
        "source_tight_rows_v300": int(protocol_row[f"source_tight_rows_v{VERSION}"]),
        "source_cap_violations_v300": int(protocol_row[f"source_cap_violations_v{VERSION}"]),
        "tightest_source_family_v300": str(protocol_row[f"tightest_source_family_v{VERSION}"]),
        "tightest_source_id_v300": str(protocol_row[f"tightest_source_id_v{VERSION}"]),
        "tightest_source_slack_v300": float(protocol_row[f"tightest_source_slack_v{VERSION}"]),
        "second_tight_source_family_v300": str(
            protocol_row[f"second_tight_source_family_v{VERSION}"]
        ),
        "second_tight_source_id_v300": str(protocol_row[f"second_tight_source_id_v{VERSION}"]),
        "second_tight_source_slack_v300": float(
            protocol_row[f"second_tight_source_slack_v{VERSION}"]
        ),
        "proxy_imputation_source_rows_v300": int(len(proxy_map)),
        "proxy_imputed_loan_rows_v300": imputed_rows,
        "proxy_observed_loan_rows_v300": observed_rows,
        "proxy_imputed_loan_share_v300": float(
            protocol_row[f"proxy_imputed_loan_share_v{VERSION}"]
        ),
        "full_binary_variables_v300": int(protocol_row[f"full_binary_variables_v{VERSION}"]),
        "direct_mip_binary_guard_v300": int(protocol_row[f"direct_mip_binary_guard_v{VERSION}"]),
        "direct_full_mip_guard_exceeded_v300": bool(
            protocol_row[f"direct_full_mip_guard_exceeded_v{VERSION}"]
        ),
        "requirement_rows_v300": int(len(requirements)),
        "unmet_requirement_rows_v300": int((~requirements[f"met_v{VERSION}"].astype(bool)).sum()),
        "valid_branch_price_bound_v300": valid_branch_price_bound,
        "source_governance_protocol_executed_v300": True,
        "strict_live_deployability_claim_allowed_v300": strict_live_allowed,
        "contractual_ifrs9_claim_allowed_v300": contractual_ifrs9_allowed,
        "working_champion_claim_allowed_v300": False,
        "paper1_promotion_allowed_v300": False,
        "paper4_working_champion_changed_v300": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v300": int(len(blockers)),
        "claim_matrix_rows_v300": int(len(claim_matrix)),
        "next_artifact_v300": (
            f"paper4_v{NEXT_VERSION}_source_tight_branch_price_pricing_or_imputation_repair.csv"
        ),
        "claim_boundary": (
            "v300 maps source-governance, imputation and global-bound blockers; "
            "branch-price, IFRS9, live deployability, working champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v300_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v300": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
