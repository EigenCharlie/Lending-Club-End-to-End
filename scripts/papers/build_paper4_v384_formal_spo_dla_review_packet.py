#!/usr/bin/env python3
"""Build Paper 4 v384 formal SPO/DLA review-packet artifacts."""

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
    write_csv,
    write_json,
)

VERSION = 384
PRIOR_AUDIT_VERSION = 383
PRIOR_CONTRACT_VERSION = 375
PRIOR_SOURCE_VERSION = 381
DEPENDENCY_VERSION = 32
NEXT_VERSION = 385
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_validation_gap_triage.md"
PACKET_MD = NOTEBOOK.parent / "paper4_v384_formal_spo_dla_review_packet.md"


def _dependency_review(v32_dependencies: pd.DataFrame) -> pd.DataFrame:
    if v32_dependencies.empty:
        raise RuntimeError("Missing v32 SPO dependency blockers.")
    out = v32_dependencies.rename(
        columns={
            "package": "package_v384",
            "available_v32": "available_v384",
            "version_v32": "version_v384",
            "import_error_v32": "import_error_v384",
            "formal_differentiable_spo_claim_allowed": (
                "formal_differentiable_spo_claim_allowed_v384"
            ),
            "decision_v32": "decision_v384",
            "future_install_path": "future_install_path_v384",
        }
    ).copy()
    out["review_use_v384"] = out["decision_v384"].map(
        {
            "dependency_blocked_for_differentiable_spo": "blocks formal differentiable SPO+ route",
            "usable_for_oracle_or_surrogate_route": "usable for historical oracle/surrogate audit",
            "environment_context_only": "environment context only",
        }
    )
    out["claim_boundary_v384"] = "dependency review only"
    return out[
        [
            "package_v384",
            "available_v384",
            "version_v384",
            "import_error_v384",
            "formal_differentiable_spo_claim_allowed_v384",
            "decision_v384",
            "review_use_v384",
            "future_install_path_v384",
            "claim_boundary_v384",
        ]
    ]


def _formal_review_packet() -> pd.DataFrame:
    rows = [
        {
            "packet_item_id_v384": "spo_anchor_source_verified",
            "review_domain_v384": "SPO/SPO+ literature",
            "input_artifact_v384": "paper4_v381_verified_literature_source_log.csv",
            "current_evidence_count_v384": 1,
            "readiness_v384": "context_ready",
            "allowed_language_v384": "SPO provides related-work framing for decision-aware learning",
            "blocked_language_v384": "Paper 4 implements or proves formal SPO+ training",
            "claim_boundary_v384": "verified citation only",
        },
        {
            "packet_item_id_v384": "oracle_surrogate_route_documented",
            "review_domain_v384": "Historical decision audit",
            "input_artifact_v384": "paper4_v16_spo_method_search_registry.csv",
            "current_evidence_count_v384": 1,
            "readiness_v384": "historical_audit_ready",
            "allowed_language_v384": "decision regret/oracle validation can be discussed as proxy evidence",
            "blocked_language_v384": "formal differentiable SPO+ theorem claim",
            "claim_boundary_v384": "oracle/surrogate only",
        },
        {
            "packet_item_id_v384": "differentiable_dependency_path_blocked",
            "review_domain_v384": "Differentiable optimization dependencies",
            "input_artifact_v384": "paper4_v32_spo_dependency_blockers.csv",
            "current_evidence_count_v384": 3,
            "readiness_v384": "blocked",
            "allowed_language_v384": "dependencies are an explicit blocker",
            "blocked_language_v384": "differentiable SPO+ implementation is complete",
            "claim_boundary_v384": "dependency-blocker evidence only",
        },
        {
            "packet_item_id_v384": "dla_state_action_review_missing",
            "review_domain_v384": "DLA formal model",
            "input_artifact_v384": "paper4_v375_live_gate_data_contract.csv",
            "current_evidence_count_v384": 2,
            "readiness_v384": "historical_audit_only",
            "allowed_language_v384": "DLA can be framed as historical audit workflow",
            "blocked_language_v384": "approved DLA formal theorem or optimality proof",
            "claim_boundary_v384": "formal review not approved",
        },
        {
            "packet_item_id_v384": "cvar_solver_scope_bounded",
            "review_domain_v384": "CVaR solver formalism",
            "input_artifact_v384": "paper4_v382_global_solver_scope_decision.csv",
            "current_evidence_count_v384": 5,
            "readiness_v384": "bounded_scope_ready",
            "allowed_language_v384": "bounded/gap CVaR solver evidence can be reported",
            "blocked_language_v384": "full-v55 global or integer optimality",
            "claim_boundary_v384": "bounded/gap only",
        },
        {
            "packet_item_id_v384": "live_legal_ifrs_gates_unmet",
            "review_domain_v384": "External claim gates",
            "input_artifact_v384": "paper4_v375_claim_permission_register.csv",
            "current_evidence_count_v384": 0,
            "readiness_v384": "blocked",
            "allowed_language_v384": "proxy/offline limitations can be stated",
            "blocked_language_v384": "live, legal or contractual IFRS9 approval",
            "claim_boundary_v384": "external gates unmet",
        },
        {
            "packet_item_id_v384": "final_promotion_absent",
            "review_domain_v384": "Promotion governance",
            "input_artifact_v384": "paper4_final_promotion_gate_not_created",
            "current_evidence_count_v384": 0,
            "readiness_v384": "blocked",
            "allowed_language_v384": "Paper Estrella remains protected",
            "blocked_language_v384": "Paper 4 final promotion or champion replacement",
            "claim_boundary_v384": "final promotion forbidden",
        },
    ]
    return pd.DataFrame(rows)


def _claim_readiness_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "formal_claim_id_v384": "bounded_historical_spo_dla_audit_language",
                "allowed_v384": True,
                "supporting_packet_item_v384": "oracle_surrogate_route_documented",
                "required_next_artifact_v384": "none_for_bounded_audit_language",
                "claim_boundary_v384": "historical audit only",
            },
            {
                "formal_claim_id_v384": "bounded_cvar_solver_formal_context",
                "allowed_v384": True,
                "supporting_packet_item_v384": "cvar_solver_scope_bounded",
                "required_next_artifact_v384": "none_for_bounded_gap_language",
                "claim_boundary_v384": "bounded/gap only",
            },
            {
                "formal_claim_id_v384": "formal_differentiable_spo_plus_claim",
                "allowed_v384": False,
                "supporting_packet_item_v384": "differentiable_dependency_path_blocked",
                "required_next_artifact_v384": "approved_dependency_and_theorem_route",
                "claim_boundary_v384": "dependencies and proof missing",
            },
            {
                "formal_claim_id_v384": "formal_dla_optimality_or_policy_theorem",
                "allowed_v384": False,
                "supporting_packet_item_v384": "dla_state_action_review_missing",
                "required_next_artifact_v384": "approved_formal_dla_review",
                "claim_boundary_v384": "review not approved",
            },
            {
                "formal_claim_id_v384": "formal_crc_or_decision_risk_guarantee",
                "allowed_v384": False,
                "supporting_packet_item_v384": "spo_anchor_source_verified",
                "required_next_artifact_v384": "implemented_and_reviewed_formal_risk_control",
                "claim_boundary_v384": "related-work source only",
            },
            {
                "formal_claim_id_v384": "live_legal_global_or_final_claim",
                "allowed_v384": False,
                "supporting_packet_item_v384": "live_legal_ifrs_gates_unmet",
                "required_next_artifact_v384": "external_gate_approval_and_final_promotion_gate",
                "claim_boundary_v384": "external gates blocked",
            },
        ]
    )


def _claim_blockers(dependency_review: pd.DataFrame) -> pd.DataFrame:
    blocked_dependencies = int(
        dependency_review["decision_v384"].eq("dependency_blocked_for_differentiable_spo").sum()
    )
    return pd.DataFrame(
        [
            {
                "blocker_id_v384": "formal_review_not_approved",
                "blocking_v384": True,
                "evidence_count_v384": 2,
                "required_next_artifact_v384": "approved_formal_spo_dla_review",
                "claim_boundary_v384": "v375 records historical audit rows, not approval",
            },
            {
                "blocker_id_v384": "differentiable_spo_dependencies_blocked",
                "blocking_v384": True,
                "evidence_count_v384": blocked_dependencies,
                "required_next_artifact_v384": "approved_dependency_and_theorem_route",
                "claim_boundary_v384": "cvxpy/cvxpylayers/torch route is blocked",
            },
            {
                "blocker_id_v384": "formal_theorem_or_proof_missing",
                "blocking_v384": True,
                "evidence_count_v384": 1,
                "required_next_artifact_v384": "formal_theorem_or_review_packet_approval",
                "claim_boundary_v384": "review packet is not a theorem",
            },
            {
                "blocker_id_v384": "external_live_legal_global_gates_blocked",
                "blocking_v384": True,
                "evidence_count_v384": 4,
                "required_next_artifact_v384": "external_gate_approval_and_full_v55_certificate",
                "claim_boundary_v384": "live/legal/IFRS9/global gates remain unmet",
            },
            {
                "blocker_id_v384": "paper4_final_promotion_forbidden",
                "blocking_v384": True,
                "evidence_count_v384": 1,
                "required_next_artifact_v384": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v384": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v384_formal_review_packet_created",
                "allowed": True,
                "artifact": "paper4_v384_formal_spo_dla_review_packet.md",
                "boundary": "packet only",
            },
            {
                "claim_id": "v384_historical_spo_dla_audit_language_allowed",
                "allowed": True,
                "artifact": "paper4_v384_formal_claim_readiness_matrix.csv",
                "boundary": "historical audit only",
            },
            {
                "claim_id": "v384_formal_spo_plus_or_dla_theorem_claim",
                "allowed": False,
                "artifact": "paper4_v384_claim_blockers.csv",
                "boundary": "dependencies, approval and theorem missing",
            },
            {
                "claim_id": "v384_formal_crc_decision_risk_guarantee",
                "allowed": False,
                "artifact": "paper4_v384_formal_claim_readiness_matrix.csv",
                "boundary": "related-work context only",
            },
            {
                "claim_id": "v384_live_legal_global_claim",
                "allowed": False,
                "artifact": "paper4_v384_claim_blockers.csv",
                "boundary": "external gates blocked",
            },
            {
                "claim_id": "v384_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v384 assembles a formal SPO/DLA review packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v384_formal_spo_dla_review_packet.md"
                ),
                "boundary": "Review packet only; not approval or theorem.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v384 permits bounded historical SPO/DLA audit language.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v384_formal_claim_readiness_matrix.csv"
                ),
                "boundary": "Historical audit/oracle-surrogate wording only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v384 approves formal SPO+, DLA or CRC theorem claims.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v384_claim_blockers.csv"
                ),
                "boundary": "Dependencies, formal proof and review approval remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v384 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v384_claim_blockers.csv"
                ),
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
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
                "lane": "Validation",
                "executable_item": (
                    "v384 assembles a formal SPO/DLA review packet and keeps formal claims blocked."
                ),
                "status": "formal_spo_dla_review_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v385 separates the known old Quarto regression failure from current guardrails"
                ),
                "last_wave": "v384",
                "execution_result": "formal_review_packet_created_formal_claims_blocked",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v384")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _packet_markdown(status: dict[str, Any], packet: pd.DataFrame) -> str:
    item_lines = "\n".join(
        (
            f"- `{row['packet_item_id_v384']}` ({row['readiness_v384']}): "
            f"{row['claim_boundary_v384']}."
        )
        for _, row in packet.iterrows()
    )
    return f"""# Paper 4 Formal SPO/DLA Review Packet v384

Generated: {status["generated_at_utc"]}

v384 packages the current formal-method evidence. It supports bounded historical
SPO/DLA audit language, not formal SPO+, DLA optimality or CRC/decision-risk
guarantee claims.

## Packet Items

{item_lines}

## Dependency Summary

- Dependency review rows: `{status["dependency_review_rows_v384"]}`.
- Blocked differentiable dependencies: `{status["differentiable_dependency_blocked_rows_v384"]}`.
- Oracle/surrogate usable rows: `{status["oracle_surrogate_usable_rows_v384"]}`.

## Required Caveat

This packet is not a formal approval. It must not be used to claim formal SPO+
training, DLA optimality, CRC/decision-risk guarantees, live deployment,
legal/IFRS9 compliance, full-v55 global optimality, Paper Estrella replacement
or final Paper 4 promotion.

## Next Executable Wave

Build `{status["next_artifact_v384"]}` to separate known old full-regression
Quarto failure from the current Paper 4 living-lab guardrails.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V384_FORMAL_SPO_DLA_REVIEW_PACKET_START -->"
    end = "<!-- V384_FORMAL_SPO_DLA_REVIEW_PACKET_END -->"
    block = f"""
{start}

## Wave v384: Formal SPO/DLA Review Packet

Generated: {status["generated_at_utc"]}

### Objective

v384 executes the formal-method work order by assembling the SPO/DLA review
packet while keeping formal method claims blocked. The packet separates verified
method context and historical audit language from theorem/approval claims.

### Results

- Review packet rows:
  `{status["review_packet_rows_v384"]}`.
- Formal claim readiness rows:
  `{status["formal_claim_readiness_rows_v384"]}`.
- Dependency review rows:
  `{status["dependency_review_rows_v384"]}`.
- Blocked differentiable dependencies:
  `{status["differentiable_dependency_blocked_rows_v384"]}`.
- Historical audit language allowed:
  `{status["historical_audit_language_allowed_v384"]}`.
- Formal SPO+/DLA theorem claim allowed:
  `{status["formal_spo_dla_theorem_claim_allowed_v384"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v384"]}`.

### Interpretation

Paper 4 can describe an historical SPO/DLA audit and bounded solver/formal
context, but not a formal SPO+, DLA optimality, or CRC decision-risk theorem.

### Claim Impact

- Allowed: review packet and bounded historical audit language.
- Still prohibited: formal theorem claims, live/legal/global claims, champion
  replacement and final promotion.

### Quarto Promotion Decision

Keep v384 in the living notebook. v385 should triage the known old full-suite
Quarto registration failure separately from current guardrail health.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v383_status = json.loads((STATUS_DIR / "paper4_v383_status.json").read_text(encoding="utf-8"))
    if v383_status["next_artifact_v383"] != "paper4_v384_formal_spo_dla_review_packet.md":
        raise RuntimeError("v384 expects v383 to route to the formal SPO/DLA review packet.")
    dependency_review = _dependency_review(read_csv("paper4_v32_spo_dependency_blockers.csv"))
    packet = _formal_review_packet()
    readiness = _claim_readiness_matrix()
    blockers = _claim_blockers(dependency_review)
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v384_formal_spo_dla_review_packet.csv", packet)
    write_csv(TABLE_DIR / "paper4_v384_dependency_review_matrix.csv", dependency_review)
    write_csv(TABLE_DIR / "paper4_v384_formal_claim_readiness_matrix.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v384_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v384_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    blocked_dependencies = int(
        dependency_review["decision_v384"].eq("dependency_blocked_for_differentiable_spo").sum()
    )
    oracle_usable_rows = int(
        dependency_review["decision_v384"].eq("usable_for_oracle_or_surrogate_route").sum()
    )
    status = {
        "phase": "v384_formal_spo_dla_review_packet",
        "schema_version": "2026-05-17.384",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_audit_version_v384": PRIOR_AUDIT_VERSION,
        "prior_contract_version_v384": PRIOR_CONTRACT_VERSION,
        "prior_source_log_version_v384": PRIOR_SOURCE_VERSION,
        "dependency_version_v384": DEPENDENCY_VERSION,
        "review_packet_rows_v384": int(len(packet)),
        "formal_claim_readiness_rows_v384": int(len(readiness)),
        "dependency_review_rows_v384": int(len(dependency_review)),
        "differentiable_dependency_blocked_rows_v384": blocked_dependencies,
        "oracle_surrogate_usable_rows_v384": oracle_usable_rows,
        "claim_blocker_rows_v384": int(len(blockers)),
        "claim_matrix_rows_v384": int(len(claim_matrix)),
        "formal_review_packet_created_v384": True,
        "historical_audit_language_allowed_v384": True,
        "bounded_solver_formal_context_allowed_v384": True,
        "formal_spo_plus_claim_allowed_v384": False,
        "formal_dla_optimality_claim_allowed_v384": False,
        "formal_crc_decision_risk_claim_allowed_v384": False,
        "formal_spo_dla_theorem_claim_allowed_v384": False,
        "strict_live_deployment_language_allowed_v384": False,
        "contractual_or_legal_language_allowed_v384": False,
        "global_optimality_language_allowed_v384": False,
        "working_champion_claim_allowed_v384": False,
        "paper1_promotion_allowed_v384": False,
        "paper4_working_champion_changed_v384": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "packet_artifact_v384": (
            "reports/paper_material/paper4/notes/"
            "paper4_v384_formal_spo_dla_review_packet.md"
        ),
        "next_artifact_v384": NEXT_ARTIFACT,
        "claim_boundary": (
            "v384 assembles a formal SPO/DLA review packet; formal theorem, "
            "live/legal/global/final claims remain blocked"
        ),
    }
    PACKET_MD.write_text(_packet_markdown(status, packet), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v384_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v384": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
