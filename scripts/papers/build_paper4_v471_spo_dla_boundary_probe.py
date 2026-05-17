#!/usr/bin/env python3
"""Build Paper 4 v471 SPO-DLA formal boundary probe artifacts."""

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

VERSION = 471
PRIOR_ONLINE_VERSION = 470
NEXT_ARTIFACT = "paper4_v472_ifrs9_proxy_boundary_probe.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v471_spo_dla_boundary_probe.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _boundary_summary() -> pd.DataFrame:
    packet = pd.read_csv(TABLE_DIR / "paper4_v384_formal_spo_dla_review_packet.csv")
    readiness = pd.read_csv(TABLE_DIR / "paper4_v384_formal_claim_readiness_matrix.csv")
    audit = pd.read_csv(TABLE_DIR / "paper4_v298_spo_dla_claim_audit.csv")
    deps = pd.read_csv(TABLE_DIR / "paper4_v57_spo_dependency_probe.csv")
    return pd.DataFrame(
        [
            {
                "summary_id_v471": "spo_dla_boundary_probe",
                "review_packet_rows_v471": len(packet),
                "formal_claim_rows_v471": len(readiness),
                "allowed_formal_claim_rows_v471": int(readiness["allowed_v384"].astype(bool).sum()),
                "blocked_formal_claim_rows_v471": int(
                    (~readiness["allowed_v384"].astype(bool)).sum()
                ),
                "historical_audit_gate_pass_rows_v471": int(
                    audit["gate_pass_v298"].astype(bool).sum()
                ),
                "historical_audit_formal_claim_allowed_rows_v471": int(
                    audit["formal_claim_allowed_v298"].astype(bool).sum()
                ),
                "differentiable_dependency_rows_v471": len(deps),
                "differentiable_dependency_available_rows_v471": int(
                    deps["formal_differentiable_spo_claim_allowed"].astype(bool).sum()
                ),
                "cvxpylayers_available_v471": bool(
                    deps.loc[deps["package"].eq("cvxpylayers"), "available"].iloc[0]
                ),
                "torch_available_v471": bool(
                    deps.loc[deps["package"].eq("torch"), "available"].iloc[0]
                ),
                "formal_spo_plus_claim_allowed_v471": False,
                "formal_dla_theorem_claim_allowed_v471": False,
                "claim_boundary_v471": (
                    "SPO-DLA historical/proxy boundary only; no theorem or formal approval"
                ),
            }
        ]
    )


def _claim_boundary_matrix() -> pd.DataFrame:
    readiness = pd.read_csv(TABLE_DIR / "paper4_v384_formal_claim_readiness_matrix.csv")
    return readiness.rename(
        columns={
            "formal_claim_id_v384": "formal_claim_id_v471",
            "allowed_v384": "allowed_v471",
            "supporting_packet_item_v384": "supporting_packet_item_v471",
            "required_next_artifact_v384": "required_next_artifact_v471",
            "claim_boundary_v384": "claim_boundary_v471",
        }
    )


def _dependency_readiness() -> pd.DataFrame:
    deps = pd.read_csv(TABLE_DIR / "paper4_v57_spo_dependency_probe.csv").copy()
    return deps.rename(
        columns={
            "available": "available_v471",
            "formal_differentiable_spo_claim_allowed": (
                "formal_differentiable_spo_claim_allowed_v471"
            ),
            "claim_boundary": "claim_boundary_v471",
        }
    )


def _blocker_register() -> pd.DataFrame:
    blockers = pd.read_csv(TABLE_DIR / "paper4_v384_claim_blockers.csv")
    rows = []
    for _, row in blockers.iterrows():
        rows.append(
            {
                "blocker_id_v471": row["blocker_id_v384"],
                "blocking_v471": bool(row["blocking_v384"]),
                "evidence_count_v471": int(row["evidence_count_v384"]),
                "required_next_artifact_v471": row["required_next_artifact_v384"],
                "claim_boundary_v471": row["claim_boundary_v384"],
            }
        )
    rows.append(
        {
            "blocker_id_v471": "ifrs9_proxy_boundary_not_refreshed",
            "blocking_v471": True,
            "evidence_count_v471": 1,
            "required_next_artifact_v471": NEXT_ARTIFACT,
            "claim_boundary_v471": "IFRS9 proxy lane remains pending",
        }
    )
    return pd.DataFrame(rows)


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v471_bounded_historical_spo_dla_language",
                "allowed": True,
                "artifact": "paper4_v471_formal_claim_boundary_matrix.csv",
                "boundary": "historical/oracle-surrogate audit only",
            },
            {
                "claim_id": "v471_bounded_cvar_solver_context",
                "allowed": True,
                "artifact": "paper4_v471_formal_claim_boundary_matrix.csv",
                "boundary": "bounded/gap CVaR context only",
            },
            {
                "claim_id": "v471_formal_spo_plus_or_dla_theorem",
                "allowed": False,
                "artifact": "paper4_v471_spo_dla_blocker_register.csv",
                "boundary": "dependencies, proof and review approval missing",
            },
            {
                "claim_id": "v471_crc_or_decision_risk_guarantee",
                "allowed": False,
                "artifact": "paper4_v471_spo_dla_blocker_register.csv",
                "boundary": "implemented and reviewed formal guarantee missing",
            },
            {
                "claim_id": "v471_working_champion_or_final_promotion",
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
                "claim": "v471 permits bounded historical SPO-DLA audit language.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v471_formal_claim_boundary_matrix.csv"
                ),
                "boundary": "Historical/oracle-surrogate audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v471 permits bounded CVaR solver formal context.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v471_formal_claim_boundary_matrix.csv"
                ),
                "boundary": "Bounded/gap context only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v471 proves formal SPO+ or DLA optimality theorem.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v471_spo_dla_blocker_register.csv"
                ),
                "boundary": "Dependencies, proof and formal approval are missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v471 proves formal CRC or decision-risk guarantee.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v471_spo_dla_blocker_register.csv"
                ),
                "boundary": "No implemented and reviewed formal guarantee.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v471 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v471_spo_dla_blocker_register.csv"
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
                "lane": "SPO-DLA",
                "executable_item": "v471 refreshes SPO-DLA claim boundaries.",
                "status": "spo_dla_boundary_probe_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v472 refreshes IFRS9 proxy boundaries",
                "last_wave": "v471",
                "execution_result": "bounded_spo_dla_language_allowed_formal_theorems_blocked",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v471")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 SPO-DLA Boundary Probe v471

Generated: {status["generated_at_utc"]}

## Result

v471 refreshes the formal SPO-DLA boundary. The living lab can use bounded
historical audit/oracle-surrogate language and bounded CVaR solver context, but
formal SPO+, DLA theorem, CRC/decision-risk guarantee, live/legal/global and
final-promotion language remain blocked.

## Counts

- Review packet rows: `{status["review_packet_rows_v471"]}`.
- Formal claim rows: `{status["formal_claim_rows_v471"]}`.
- Allowed formal claim rows: `{status["allowed_formal_claim_rows_v471"]}`.
- Blocked formal claim rows: `{status["blocked_formal_claim_rows_v471"]}`.
- cvxpylayers available: `{status["cvxpylayers_available_v471"]}`.
- torch available: `{status["torch_available_v471"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v471 is a boundary probe only. It does not prove a formal theorem, approve a
DLA model, implement differentiable SPO+, certify CRC/decision-risk guarantees,
replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V471_SPO_DLA_BOUNDARY_PROBE_START -->"
    end = "<!-- V471_SPO_DLA_BOUNDARY_PROBE_END -->"
    block = f"""
{start}

## Wave v471: SPO-DLA Boundary Probe

Generated: {status["generated_at_utc"]}

### Objective

v471 refreshes formal SPO-DLA boundaries after the online monitoring proxy lane.

### Results

- Review packet rows:
  `{status["review_packet_rows_v471"]}`.
- Formal claim rows:
  `{status["formal_claim_rows_v471"]}`.
- Allowed formal claim rows:
  `{status["allowed_formal_claim_rows_v471"]}`.
- Blocked formal claim rows:
  `{status["blocked_formal_claim_rows_v471"]}`.
- Historical audit gate pass rows:
  `{status["historical_audit_gate_pass_rows_v471"]}`.
- Formal SPO+ claim allowed:
  `{status["formal_spo_plus_claim_allowed_v471"]}`.
- Formal DLA theorem claim allowed:
  `{status["formal_dla_theorem_claim_allowed_v471"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v471"]}`.

### Interpretation

The SPO-DLA lane is useful for bounded positioning, not formal theorem claims.
It strengthens the paper by making the method boundary auditable instead of
overstating differentiable SPO+ or DLA optimality.

### Claim Impact

- Allowed: bounded historical SPO-DLA audit language and bounded CVaR solver
  context.
- Still prohibited: formal SPO+/DLA theorem claims, CRC/decision-risk
  guarantees, live/legal/global claims, Paper Estrella replacement and final
  Paper 4 promotion.

### Quarto Promotion Decision

Keep v471 in the living notebook. v472 should refresh IFRS9 proxy boundaries.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v470 = _read_status(PRIOR_ONLINE_VERSION)
    if v470["next_artifact_v470"] != "paper4_v471_spo_dla_boundary_probe.md":
        raise RuntimeError("v471 expects v470 to route to SPO-DLA boundary probe.")

    summary = _boundary_summary()
    boundary_matrix = _claim_boundary_matrix()
    dependency = _dependency_readiness()
    blockers = _blocker_register()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v471_spo_dla_boundary_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v471_formal_claim_boundary_matrix.csv", boundary_matrix)
    write_csv(TABLE_DIR / "paper4_v471_spo_dependency_readiness.csv", dependency)
    write_csv(TABLE_DIR / "paper4_v471_spo_dla_blocker_register.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v471_claim_matrix_delta.csv", claim_matrix)

    row = summary.iloc[0]
    status = {
        "phase": "v471_spo_dla_boundary_probe",
        "schema_version": "2026-05-17.471",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_online_monitoring_version_v471": PRIOR_ONLINE_VERSION,
        "spo_dla_boundary_probe_created_v471": True,
        "review_packet_rows_v471": int(row["review_packet_rows_v471"]),
        "formal_claim_rows_v471": int(row["formal_claim_rows_v471"]),
        "allowed_formal_claim_rows_v471": int(row["allowed_formal_claim_rows_v471"]),
        "blocked_formal_claim_rows_v471": int(row["blocked_formal_claim_rows_v471"]),
        "historical_audit_gate_pass_rows_v471": int(
            row["historical_audit_gate_pass_rows_v471"]
        ),
        "historical_audit_formal_claim_allowed_rows_v471": int(
            row["historical_audit_formal_claim_allowed_rows_v471"]
        ),
        "differentiable_dependency_rows_v471": int(row["differentiable_dependency_rows_v471"]),
        "differentiable_dependency_available_rows_v471": int(
            row["differentiable_dependency_available_rows_v471"]
        ),
        "cvxpylayers_available_v471": bool(row["cvxpylayers_available_v471"]),
        "torch_available_v471": bool(row["torch_available_v471"]),
        "formal_spo_plus_claim_allowed_v471": False,
        "formal_dla_theorem_claim_allowed_v471": False,
        "crc_or_decision_risk_guarantee_allowed_v471": False,
        "working_champion_claim_allowed_v471": False,
        "paper1_promotion_allowed_v471": False,
        "paper4_working_champion_changed_v471": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v471": NEXT_ARTIFACT,
        "claim_boundary": (
            "v471 supports bounded historical SPO-DLA audit language only; formal "
            "theorem, live/legal/global, champion and final promotion claims remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v471 must not create final Paper 4 promotion.")

    PROBE_MD.write_text(_probe_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v471": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
