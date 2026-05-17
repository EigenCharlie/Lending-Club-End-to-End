#!/usr/bin/env python3
"""Build Paper 4 v472 IFRS9 proxy boundary probe artifacts."""

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
    write_csv,
    write_json,
)

VERSION = 472
PRIOR_SPO_DLA_VERSION = 471
NEXT_ARTIFACT = "paper4_v473_domain_execution_synthesis.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v472_ifrs9_proxy_boundary_probe.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _first_row(name: str) -> pd.Series:
    data = pd.read_csv(TABLE_DIR / name)
    if data.empty:
        raise RuntimeError(f"Expected non-empty artifact: {name}")
    return data.iloc[0]


def _proxy_summary() -> pd.DataFrame:
    v341 = _first_row("paper4_v341_v338_cashflow_proxy_summary.csv")
    v323 = _first_row("paper4_v323_v320_cashflow_online_ifrs9_gate.csv")
    readiness = pd.read_csv(TABLE_DIR / "paper4_v36_ifrs9_readiness_matrix.csv")
    audit = pd.read_csv(TABLE_DIR / "paper4_v36_ifrs9_contractual_data_audit.csv")
    return pd.DataFrame(
        [
            {
                "summary_id_v472": "ifrs9_proxy_boundary_probe",
                "latest_proxy_candidate_v472": str(v341["candidate_version_v341"]),
                "v341_selected_rows_v472": int(v341["selected_rows_v341"]),
                "v341_observed_proxy_loan_rows_v472": int(v341["observed_proxy_loan_rows_v341"]),
                "v341_imputed_proxy_loan_rows_v472": int(v341["imputed_proxy_loan_rows_v341"]),
                "v341_cashflow_proxy_panel_rows_v472": int(
                    v341["cashflow_proxy_panel_rows_v341"]
                ),
                "v341_post_imputation_coverage_share_v472": float(
                    v341["post_imputation_coverage_share_v341"]
                ),
                "v341_observed_coverage_share_v472": float(
                    v341["observed_coverage_share_v341"]
                ),
                "v341_contractual_ifrs9_claim_allowed_v472": bool(
                    v341["contractual_ifrs9_claim_allowed_v341"]
                ),
                "v323_observed_proxy_loan_rows_v472": int(
                    v323["observed_proxy_loan_rows_v323"]
                ),
                "v323_imputed_proxy_loan_rows_v472": int(
                    v323["imputed_proxy_loan_rows_v323"]
                ),
                "readiness_available_or_proxy_requirements_v472": int(
                    readiness.loc[
                        readiness["availability_v36"].eq("available_or_proxy"),
                        "requirements",
                    ].iloc[0]
                ),
                "readiness_missing_requirements_v472": int(
                    readiness.loc[
                        readiness["availability_v36"].eq("missing"),
                        "requirements",
                    ].iloc[0]
                ),
                "contractual_audit_requirement_rows_v472": len(audit),
                "contractual_audit_claim_allowed_rows_v472": int(
                    audit["contractual_claim_allowed"].astype(bool).sum()
                ),
                "claim_boundary_v472": (
                    "IFRS9-inspired proxy boundary only; no contractual accounting claim"
                ),
            }
        ]
    )


def _requirement_audit() -> pd.DataFrame:
    audit = pd.read_csv(TABLE_DIR / "paper4_v36_ifrs9_contractual_data_audit.csv").copy()
    return audit.rename(
        columns={
            "requirement": "requirement_v472",
            "availability_v36": "availability_v472",
            "role_for_contractual_ifrs9": "role_for_contractual_ifrs9_v472",
            "contractual_claim_allowed": "contractual_claim_allowed_v472",
            "claim_boundary_v36": "claim_boundary_v472",
        }
    )[
        [
            "requirement_v472",
            "availability_v472",
            "role_for_contractual_ifrs9_v472",
            "contractual_claim_allowed_v472",
            "claim_boundary_v472",
        ]
    ]


def _frontier_ifrs9_gap() -> pd.DataFrame:
    v467 = _read_status(467)
    v341 = _first_row("paper4_v341_v338_cashflow_proxy_summary.csv")
    return pd.DataFrame(
        [
            {
                "gap_id_v472": "current_frontier_v353_ifrs9_proxy_gate_missing",
                "current_local_frontier_v472": str(v467["local_frontier_candidate_v467"]),
                "latest_ifrs9_proxy_candidate_v472": str(v341["candidate_version_v341"]),
                "current_frontier_proxy_panel_available_v472": False,
                "current_frontier_contractual_ifrs9_claim_allowed_v472": False,
                "latest_proxy_observed_coverage_share_v472": float(
                    v341["observed_coverage_share_v341"]
                ),
                "latest_proxy_imputed_rows_v472": int(v341["imputed_proxy_loan_rows_v341"]),
                "claim_boundary_v472": (
                    "v353 cannot inherit v338 cashflow/IFRS9 proxy claims"
                ),
            }
        ]
    )


def _blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v472": "contractual_ifrs9_requirements_missing",
                "blocking_v472": True,
                "evidence_count_v472": 5,
                "required_next_artifact_v472": "future_contractual_servicing_panel",
                "claim_boundary_v472": "five contractual requirements remain missing",
            },
            {
                "blocker_id_v472": "v338_proxy_uses_imputation",
                "blocking_v472": True,
                "evidence_count_v472": 74,
                "required_next_artifact_v472": "future_observed_proxy_coverage_repair",
                "claim_boundary_v472": "v338 proxy panel includes explicit imputed rows",
            },
            {
                "blocker_id_v472": "v353_ifrs9_proxy_gate_missing",
                "blocking_v472": True,
                "evidence_count_v472": 1,
                "required_next_artifact_v472": "future_v353_cashflow_proxy_gate",
                "claim_boundary_v472": "current local frontier lacks cashflow proxy gate",
            },
            {
                "blocker_id_v472": "production_accounting_validation_missing",
                "blocking_v472": True,
                "evidence_count_v472": 1,
                "required_next_artifact_v472": "future_accounting_review_and_validation",
                "claim_boundary_v472": "no accounting review or contractual validation",
            },
            {
                "blocker_id_v472": "paper4_final_promotion_forbidden",
                "blocking_v472": True,
                "evidence_count_v472": 1,
                "required_next_artifact_v472": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v472": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v472_ifrs9_proxy_boundary_created",
                "allowed": True,
                "artifact": "paper4_v472_ifrs9_proxy_boundary_summary.csv",
                "boundary": "IFRS9-inspired proxy boundary only",
            },
            {
                "claim_id": "v472_contractual_requirement_gap_documented",
                "allowed": True,
                "artifact": "paper4_v472_ifrs9_requirement_audit.csv",
                "boundary": "requirement audit only",
            },
            {
                "claim_id": "v472_v353_has_ifrs9_proxy_or_contractual_validation",
                "allowed": False,
                "artifact": "paper4_v472_current_frontier_ifrs9_gap.csv",
                "boundary": "v353 cashflow proxy gate missing",
            },
            {
                "claim_id": "v472_contractual_ifrs9_or_accounting_compliance",
                "allowed": False,
                "artifact": "paper4_v472_ifrs9_blocker_register.csv",
                "boundary": "contractual and accounting validation missing",
            },
            {
                "claim_id": "v472_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "no champion or final promotion claim",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v472 summarizes IFRS9-inspired proxy evidence.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v472_ifrs9_proxy_boundary_summary.csv"
                ),
                "boundary": "Proxy evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v472 documents contractual IFRS9 requirement gaps.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v472_ifrs9_requirement_audit.csv"
                ),
                "boundary": "Requirement audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v472 validates v353 with IFRS9 proxy or contractual cashflows.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v472_current_frontier_ifrs9_gap.csv"
                ),
                "boundary": "v353 lacks a cashflow proxy gate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v472 proves contractual IFRS9 or accounting compliance.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v472_ifrs9_blocker_register.csv"
                ),
                "boundary": "Contractual servicing, macro and accounting review are missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v472 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v472_ifrs9_blocker_register.csv"
                ),
                "boundary": (
                    "No final promotion artifact, champion replacement or deployment gate "
                    "is created."
                ),
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "IFRS9 Proxy",
                "executable_item": "v472 refreshes IFRS9 proxy boundaries.",
                "status": "ifrs9_proxy_boundary_probe_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v473 synthesizes v466-v472 domain execution results",
                "last_wave": "v472",
                "execution_result": "proxy_evidence_allowed_contractual_ifrs9_blocked",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v472")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 IFRS9 Proxy Boundary Probe v472

Generated: {status["generated_at_utc"]}

## Result

v472 summarizes IFRS9-inspired proxy evidence and keeps contractual/accounting
claims blocked. The latest cashflow proxy anchor is v338, with explicit imputed
rows; the current v353 local frontier has no candidate-specific cashflow proxy
gate.

## Counts

- Latest proxy candidate: `{status["latest_proxy_candidate_v472"]}`.
- v341 observed proxy loan rows: `{status["v341_observed_proxy_loan_rows_v472"]}`.
- v341 imputed proxy loan rows: `{status["v341_imputed_proxy_loan_rows_v472"]}`.
- Missing contractual requirements: `{status["readiness_missing_requirements_v472"]}`.
- Current frontier proxy panel available: `{status["current_frontier_proxy_panel_available_v472"]}`.
- Contractual IFRS9 claim allowed: `{status["contractual_ifrs9_claim_allowed_v472"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v472 is an IFRS9-inspired proxy boundary artifact. It does not provide
contractual cashflows, accounting validation, production IFRS9 staging, v353
cashflow validation, Paper Estrella replacement, or final Paper 4 promotion.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V472_IFRS9_PROXY_BOUNDARY_PROBE_START -->"
    end = "<!-- V472_IFRS9_PROXY_BOUNDARY_PROBE_END -->"
    block = f"""
{start}

## Wave v472: IFRS9 Proxy Boundary Probe

Generated: {status["generated_at_utc"]}

### Objective

v472 refreshes IFRS9 proxy boundaries after the SPO-DLA formal boundary probe.

### Results

- Latest proxy candidate:
  `{status["latest_proxy_candidate_v472"]}`.
- v341 observed proxy loan rows:
  `{status["v341_observed_proxy_loan_rows_v472"]}`.
- v341 imputed proxy loan rows:
  `{status["v341_imputed_proxy_loan_rows_v472"]}`.
- Missing contractual requirements:
  `{status["readiness_missing_requirements_v472"]}`.
- Contractual audit claim-allowed rows:
  `{status["contractual_audit_claim_allowed_rows_v472"]}`.
- Current local frontier:
  `{status["current_local_frontier_v472"]}`.
- Current frontier proxy panel available:
  `{status["current_frontier_proxy_panel_available_v472"]}`.
- Contractual IFRS9 claim allowed:
  `{status["contractual_ifrs9_claim_allowed_v472"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v472"]}`.

### Interpretation

The IFRS9 lane supports transparent proxy language and requirement-gap
language. It does not support contractual accounting claims, and v353 cannot
inherit the v338 cashflow proxy gate.

### Claim Impact

- Allowed: IFRS9-inspired proxy evidence and contractual requirement-gap audit.
- Still prohibited: v353 IFRS9 validation, contractual/accounting compliance,
  working-champion language, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v472 in the living notebook. v473 should synthesize the six domain lanes
executed from v466 through v472.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v471 = _read_status(PRIOR_SPO_DLA_VERSION)
    if v471["next_artifact_v471"] != "paper4_v472_ifrs9_proxy_boundary_probe.md":
        raise RuntimeError("v472 expects v471 to route to IFRS9 proxy boundary probe.")

    summary = _proxy_summary()
    requirement = _requirement_audit()
    gap = _frontier_ifrs9_gap()
    blockers = _blocker_register()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v472_ifrs9_proxy_boundary_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v472_ifrs9_requirement_audit.csv", requirement)
    write_csv(TABLE_DIR / "paper4_v472_current_frontier_ifrs9_gap.csv", gap)
    write_csv(TABLE_DIR / "paper4_v472_ifrs9_blocker_register.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v472_claim_matrix_delta.csv", claim_matrix)

    row = summary.iloc[0]
    gap_row = gap.iloc[0]
    status = {
        "phase": "v472_ifrs9_proxy_boundary_probe",
        "schema_version": "2026-05-17.472",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_spo_dla_version_v472": PRIOR_SPO_DLA_VERSION,
        "ifrs9_proxy_boundary_probe_created_v472": True,
        "latest_proxy_candidate_v472": str(row["latest_proxy_candidate_v472"]),
        "v341_selected_rows_v472": int(row["v341_selected_rows_v472"]),
        "v341_observed_proxy_loan_rows_v472": int(row["v341_observed_proxy_loan_rows_v472"]),
        "v341_imputed_proxy_loan_rows_v472": int(row["v341_imputed_proxy_loan_rows_v472"]),
        "v341_post_imputation_coverage_share_v472": float(
            row["v341_post_imputation_coverage_share_v472"]
        ),
        "v341_observed_coverage_share_v472": float(row["v341_observed_coverage_share_v472"]),
        "readiness_available_or_proxy_requirements_v472": int(
            row["readiness_available_or_proxy_requirements_v472"]
        ),
        "readiness_missing_requirements_v472": int(
            row["readiness_missing_requirements_v472"]
        ),
        "contractual_audit_requirement_rows_v472": int(
            row["contractual_audit_requirement_rows_v472"]
        ),
        "contractual_audit_claim_allowed_rows_v472": int(
            row["contractual_audit_claim_allowed_rows_v472"]
        ),
        "current_local_frontier_v472": str(gap_row["current_local_frontier_v472"]),
        "current_frontier_proxy_panel_available_v472": bool(
            gap_row["current_frontier_proxy_panel_available_v472"]
        ),
        "current_frontier_contractual_ifrs9_claim_allowed_v472": bool(
            gap_row["current_frontier_contractual_ifrs9_claim_allowed_v472"]
        ),
        "contractual_ifrs9_claim_allowed_v472": False,
        "accounting_compliance_claim_allowed_v472": False,
        "working_champion_claim_allowed_v472": False,
        "paper1_promotion_allowed_v472": False,
        "paper4_working_champion_changed_v472": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v472": NEXT_ARTIFACT,
        "claim_boundary": (
            "v472 supports IFRS9-inspired proxy language only; contractual IFRS9, "
            "accounting compliance, champion and final promotion claims remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v472 must not create final Paper 4 promotion.")

    PROBE_MD.write_text(_probe_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v472": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
