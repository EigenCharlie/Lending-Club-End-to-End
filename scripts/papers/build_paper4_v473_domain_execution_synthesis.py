#!/usr/bin/env python3
"""Build Paper 4 v473 domain execution synthesis artifacts."""

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

VERSION = 473
PRIOR_IFRS9_VERSION = 472
NEXT_ARTIFACT = "paper4_v474_post_domain_manuscript_delta.md"
SYNTHESIS_MD = NOTEBOOK.parent / "paper4_v473_domain_execution_synthesis.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _claim_counts(version: int) -> tuple[int, int]:
    claims = pd.read_csv(TABLE_DIR / f"paper4_v{version}_claim_matrix_delta.csv")
    allowed = int(claims["allowed"].astype(bool).sum())
    blocked = int((~claims["allowed"].astype(bool)).sum())
    return allowed, blocked


def _domain_synthesis() -> pd.DataFrame:
    v467 = _read_status(467)
    v468 = _read_status(468)
    v469 = _read_status(469)
    v471 = _read_status(471)
    v472 = _read_status(472)
    rows = [
        {
            "domain_lane_v473": "cvar_tail_risk",
            "wave_v473": "v467",
            "primary_allowed_result_v473": (
                f"{v467['local_frontier_candidate_v467']} is the local return/CVaR frontier"
            ),
            "primary_open_blocker_v473": (
                "proxy gap, full-v55 global proof, dynamic and online validation"
            ),
            "next_needed_v473": "post-domain manuscript delta or future v353 validation gates",
            "claim_boundary_v473": "local frontier only; no champion or global proof",
        },
        {
            "domain_lane_v473": "source_governance",
            "wave_v473": "v468",
            "primary_allowed_result_v473": (
                f"{v468['primary_blocker_family_v468']}="
                f"{v468['primary_blocker_source_id_v468']} is primary blocker"
            ),
            "primary_open_blocker_v473": "source cap relaxation and global solver claims",
            "next_needed_v473": "source-aware counterfactual or cap-governance approval",
            "claim_boundary_v473": "diagnostic only; no cap mutation",
        },
        {
            "domain_lane_v473": "dynamic_replay",
            "wave_v473": "v469",
            "primary_allowed_result_v473": (
                f"{v469['latest_dynamic_proxy_candidate_v469']} remains dynamic proxy anchor"
            ),
            "primary_open_blocker_v473": "v353 lacks dynamic replay trace",
            "next_needed_v473": "future v353 dynamic replay build",
            "claim_boundary_v473": "proxy replay inventory only; no live claim",
        },
        {
            "domain_lane_v473": "online_monitoring",
            "wave_v473": "v470",
            "primary_allowed_result_v473": "v9/v10 internal online gates summarized",
            "primary_open_blocker_v473": "v353 online temporal gate and external holdout missing",
            "next_needed_v473": "future v353 online temporal gate",
            "claim_boundary_v473": "internal monitoring proxy only",
        },
        {
            "domain_lane_v473": "spo_dla",
            "wave_v473": "v471",
            "primary_allowed_result_v473": (
                f"{v471['allowed_formal_claim_rows_v471']} bounded formal claims allowed"
            ),
            "primary_open_blocker_v473": "formal SPO+/DLA theorem and CRC guarantee missing",
            "next_needed_v473": "formal review/dependency route before stronger method claims",
            "claim_boundary_v473": "historical/oracle-surrogate boundary only",
        },
        {
            "domain_lane_v473": "ifrs9_proxy",
            "wave_v473": "v472",
            "primary_allowed_result_v473": (
                f"{v472['readiness_missing_requirements_v472']} contractual gaps documented"
            ),
            "primary_open_blocker_v473": "contractual IFRS9 and v353 cashflow gate missing",
            "next_needed_v473": "future contractual servicing/accounting validation",
            "claim_boundary_v473": "IFRS9-inspired proxy only",
        },
    ]
    out = pd.DataFrame(rows)
    counts = {f"v{version}": _claim_counts(version) for version in range(467, 473)}
    out["allowed_claim_rows_v473"] = [counts[wave][0] for wave in out["wave_v473"]]
    out["blocked_claim_rows_v473"] = [counts[wave][1] for wave in out["wave_v473"]]
    return out


def _allowed_claims() -> pd.DataFrame:
    rows = []
    for version in range(467, 473):
        claims = pd.read_csv(TABLE_DIR / f"paper4_v{version}_claim_matrix_delta.csv")
        for _, row in claims.loc[claims["allowed"].astype(bool)].iterrows():
            rows.append(
                {
                    "wave_v473": f"v{version}",
                    "claim_id_v473": row["claim_id"],
                    "artifact_v473": row["artifact"],
                    "boundary_v473": row["boundary"],
                }
            )
    return pd.DataFrame(rows)


def _open_blockers() -> pd.DataFrame:
    blocker_files = {
        "v467": "paper4_v467_cvar_blocker_register.csv",
        "v468": "paper4_v468_source_blocker_register.csv",
        "v469": "paper4_v469_dynamic_blocker_register.csv",
        "v470": "paper4_v470_online_blocker_register.csv",
        "v471": "paper4_v471_spo_dla_blocker_register.csv",
        "v472": "paper4_v472_ifrs9_blocker_register.csv",
    }
    rows = []
    for wave, filename in blocker_files.items():
        version = wave.removeprefix("v")
        data = pd.read_csv(TABLE_DIR / filename)
        for _, row in data.loc[data[f"blocking_v{version}"].astype(bool)].iterrows():
            rows.append(
                {
                    "wave_v473": wave,
                    "blocker_id_v473": row[f"blocker_id_v{version}"],
                    "evidence_count_v473": int(row[f"evidence_count_v{version}"]),
                    "required_next_artifact_v473": row[f"required_next_artifact_v{version}"],
                    "claim_boundary_v473": row[f"claim_boundary_v{version}"],
                }
            )
    return pd.DataFrame(rows)


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v473_domain_execution_sequence_synthesized",
                "allowed": True,
                "artifact": "paper4_v473_domain_execution_synthesis.csv",
                "boundary": "six-lane synthesis only",
            },
            {
                "claim_id": "v473_allowed_claims_and_blockers_indexed",
                "allowed": True,
                "artifact": "paper4_v473_allowed_domain_claims.csv",
                "boundary": "indexing previously bounded claims",
            },
            {
                "claim_id": "v473_domain_lanes_resolved_all_blockers",
                "allowed": False,
                "artifact": "paper4_v473_open_domain_blockers.csv",
                "boundary": "open blockers remain in every domain lane",
            },
            {
                "claim_id": "v473_paper4_working_champion_or_submission_ready",
                "allowed": False,
                "artifact": "paper4_v473_open_domain_blockers.csv",
                "boundary": "champion and submission gates remain blocked",
            },
            {
                "claim_id": "v473_paper_estrella_replacement_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "no final promotion artifact is created",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v473 synthesizes six Paper 4 domain execution lanes.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v473_domain_execution_synthesis.csv"
                ),
                "boundary": "Synthesis only; no blocker is resolved by aggregation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v473 indexes allowed bounded domain claims and open blockers.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v473_allowed_domain_claims.csv"
                ),
                "boundary": "Index of previously bounded claims.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v473 resolves all Paper 4 domain blockers.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v473_open_domain_blockers.csv"
                ),
                "boundary": "Every lane still has explicit open blockers.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v473 authorizes Paper 4 as working champion or submission-ready.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v473_open_domain_blockers.csv"
                ),
                "boundary": "Global, proxy, live, legal and venue gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v473 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v473_open_domain_blockers.csv"
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
                "lane": "Domain Synthesis",
                "executable_item": "v473 synthesizes v466-v472 domain execution.",
                "status": "domain_execution_synthesis_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v474 maps domain execution into manuscript delta",
                "last_wave": "v473",
                "execution_result": "six_domain_lanes_synthesized_with_open_blockers_preserved",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v473")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _synthesis_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Domain Execution Synthesis v473

Generated: {status["generated_at_utc"]}

## Result

v473 synthesizes the six domain lanes executed after v466: CVaR tail risk,
source governance, dynamic replay, online monitoring, SPO-DLA and IFRS9 proxy.
The sequence produced bounded future-paper claims, but every lane still carries
explicit blockers.

## Counts

- Domain lanes synthesized: `{status["domain_lanes_synthesized_v473"]}`.
- Allowed bounded claim rows: `{status["allowed_domain_claim_rows_v473"]}`.
- Open blocker rows: `{status["open_domain_blocker_rows_v473"]}`.
- Domain lanes with open blockers: `{status["domain_lanes_with_open_blockers_v473"]}`.
- Domain blockers resolved: `{status["domain_blockers_resolved_v473"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v473 is a synthesis artifact only. It does not resolve domain blockers, select a
Paper 4 working champion, make the manuscript submission-ready, replace Paper
Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V473_DOMAIN_EXECUTION_SYNTHESIS_START -->"
    end = "<!-- V473_DOMAIN_EXECUTION_SYNTHESIS_END -->"
    block = f"""
{start}

## Wave v473: Domain Execution Synthesis

Generated: {status["generated_at_utc"]}

### Objective

v473 synthesizes the six domain execution lanes completed from v467 through
v472 after the v466 refocus.

### Results

- Domain lanes synthesized:
  `{status["domain_lanes_synthesized_v473"]}`.
- Allowed bounded claim rows:
  `{status["allowed_domain_claim_rows_v473"]}`.
- Open blocker rows:
  `{status["open_domain_blocker_rows_v473"]}`.
- Domain lanes with open blockers:
  `{status["domain_lanes_with_open_blockers_v473"]}`.
- Domain blockers resolved:
  `{status["domain_blockers_resolved_v473"]}`.
- Working champion claim allowed:
  `{status["working_champion_claim_allowed_v473"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v473"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v473"]}`.

### Interpretation

The living lab now has a coherent future-paper evidence bundle: strong bounded
claims exist, and the blocker surface is clear enough to guide manuscript
language without drifting into champion, deployment, legal, accounting or final
promotion claims.

### Claim Impact

- Allowed: six-lane domain synthesis and index of bounded claims.
- Still prohibited: all-blocker resolution, working-champion language,
  submission readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v473 in the living notebook. v474 should map this domain execution into a
post-domain manuscript delta.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v472 = _read_status(PRIOR_IFRS9_VERSION)
    if v472["next_artifact_v472"] != "paper4_v473_domain_execution_synthesis.md":
        raise RuntimeError("v473 expects v472 to route to domain execution synthesis.")

    synthesis = _domain_synthesis()
    allowed = _allowed_claims()
    blockers = _open_blockers()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v473_domain_execution_synthesis.csv", synthesis)
    write_csv(TABLE_DIR / "paper4_v473_allowed_domain_claims.csv", allowed)
    write_csv(TABLE_DIR / "paper4_v473_open_domain_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v473_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v473_domain_execution_synthesis",
        "schema_version": "2026-05-17.473",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_ifrs9_proxy_version_v473": PRIOR_IFRS9_VERSION,
        "domain_execution_synthesis_created_v473": True,
        "domain_lanes_synthesized_v473": len(synthesis),
        "allowed_domain_claim_rows_v473": len(allowed),
        "open_domain_blocker_rows_v473": len(blockers),
        "domain_lanes_with_open_blockers_v473": blockers["wave_v473"].nunique(),
        "domain_blockers_resolved_v473": False,
        "working_champion_claim_allowed_v473": False,
        "submission_ready_claim_allowed_v473": False,
        "paper1_promotion_allowed_v473": False,
        "paper4_working_champion_changed_v473": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v473": NEXT_ARTIFACT,
        "claim_boundary": (
            "v473 synthesizes domain evidence only; open blockers, champion, "
            "submission and final promotion claims remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v473 must not create final Paper 4 promotion.")

    SYNTHESIS_MD.write_text(_synthesis_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v473": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
