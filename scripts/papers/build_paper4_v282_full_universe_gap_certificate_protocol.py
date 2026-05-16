#!/usr/bin/env python3
"""Build Paper 4 v282 full-universe gap-certificate protocol artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime

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

VERSION = 282
PRIOR_RESTRICTED_MILP_VERSION = 281
INCUMBENT_REPAIR_VERSION = 279
TERMINAL_REPRICE_VERSION = 280
NEXT_BOUND_PROBE_VERSION = 283
SCENARIO_COUNT = 128
SOURCE_CONSTRAINT_ROWS = 45
GAP_TOLERANCE = 1e-6


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v282 full-universe gap-certificate protocol.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v282_full_universe_gap_certificate_protocol.csv"
                ),
                "boundary": (
                    "Protocol and model-audit checklist only; no full-universe branch-and-bound "
                    "or branch-and-price certificate has been produced."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v281/v282 prove no improvement inside the expanded top-5000 restricted pool.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v281_restricted_pool_milp_summary.csv"
                ),
                "boundary": "Restricted to the v281 pool and solver tolerance; not the full v55 universe.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v282 proves full-universe global integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v282_claim_blockers.csv"
                ),
                "boundary": "Requires a full-v55 integer dual bound or equivalent branch-and-price gap.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v282 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v282_claim_blockers.csv"
                ),
                "boundary": "No final promotion, dynamic validation, or deployment gate is created.",
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
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v282 converts the v281 top-5000 restricted-pool no-improvement result into "
                    "a full-universe gap-certificate protocol and requirement checklist."
                ),
                "status": "full_universe_gap_certificate_protocol_created",
                "next_artifact": "paper4_v283_full_universe_integer_bound_probe.csv",
                "success_condition": (
                    "a full-v55 model/bound probe either produces a valid integer gap certificate "
                    "or records a reproducible resource-bound blocker"
                ),
                "last_wave": "v282",
                "execution_result": "full_universe_gap_protocol_created_not_certified",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    key_cols = ["last_wave", "lane", "next_artifact"]
    merged_keys = set(map(tuple, additions[key_cols].astype(str).to_numpy()))
    keep = [tuple(row) not in merged_keys for row in current[key_cols].astype(str).to_numpy()]
    write_csv(path, pd.concat([current.loc[keep].copy(), additions], ignore_index=True))


def _update_notebook(status: dict[str, object]) -> None:
    start = "<!-- V282_FULL_UNIVERSE_GAP_CERTIFICATE_PROTOCOL_START -->"
    end = "<!-- V282_FULL_UNIVERSE_GAP_CERTIFICATE_PROTOCOL_END -->"
    block = f"""
{start}

## Wave v282: Full-Universe Gap-Certificate Protocol

Generated: {status["generated_at_utc"]}

### Objective

Translate the v281 top-5000 restricted-pool no-improvement result into an
explicit protocol for a future full-v55 integer optimality/gap certificate.
This records what is already proven, what remains missing, and the next
executable artifact required before any global claim is allowed.

### Results

- v55 universe rows: `{status["universe_rows_v282"]}`.
- v281 restricted pool rows: `{status["restricted_pool_rows_v282"]}`.
- Full omitted candidate rows: `{status["full_omitted_candidate_rows_v282"]}`.
- v281 omitted-pool coverage share:
  `{status["restricted_omitted_coverage_share_v282"]}`.
- v281 MILP gap:
  `{status["prior_restricted_pool_gap_v282"]}`.
- Full-universe certificate available:
  `{status["full_universe_gap_certificate_available_v282"]}`.
- Exact full-universe claim allowed:
  `{status["full_universe_integer_optimality_claim_allowed_v282"]}`.

### Interpretation

v281 is meaningful negative evidence inside an expanded restricted pool, but it
does not cover most v55 candidates and does not provide a full-v55 integer dual
bound. v282 therefore promotes no model; it only specifies the certificate
requirements for the next bound-probe wave.

### Claim Impact

- Allowed: protocol and requirement checklist for a full-universe certificate.
- Still prohibited: full-universe global integer optimality, Paper Estrella
  replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v282 in the living notebook. Promote only after a full-v55 gap certificate,
dynamic validation and promotion gates pass.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    restricted_summary = read_csv("paper4_v281_restricted_pool_milp_summary.csv")
    restricted_allocations = read_parquet("paper4_v281_restricted_pool_milp_allocations.parquet")
    incumbent_summary = read_csv("paper4_v279_restricted_pool_milp_repair_summary.csv")
    reprice_summary = read_csv("paper4_v280_post_restricted_pool_milp_one_swap_summary.csv")
    if (
        universe.empty
        or restricted_summary.empty
        or restricted_allocations.empty
        or incumbent_summary.empty
        or reprice_summary.empty
    ):
        raise RuntimeError("Missing v55, v279, v280, or v281 inputs for v282 protocol.")

    restricted = restricted_summary.iloc[0]
    incumbent = incumbent_summary.iloc[0]
    reprice = reprice_summary.iloc[0]
    universe_rows = int(len(universe))
    selected_rows = int(restricted[f"selected_rows_v{PRIOR_RESTRICTED_MILP_VERSION}"])
    full_omitted_rows = int(universe_rows - selected_rows)
    restricted_pool_rows = int(restricted[f"pool_rows_v{PRIOR_RESTRICTED_MILP_VERSION}"])
    restricted_omitted_rows = int(
        restricted[f"omitted_candidate_rows_v{PRIOR_RESTRICTED_MILP_VERSION}"]
    )
    coverage_share = restricted_pool_rows / universe_rows
    omitted_coverage_share = restricted_omitted_rows / full_omitted_rows
    model_constraint_rows = 1 + 1 + SOURCE_CONSTRAINT_ROWS + 1 + SCENARIO_COUNT
    continuous_vars = 1 + SCENARIO_COUNT

    protocol = pd.DataFrame(
        [
            {
                f"protocol_id_v{VERSION}": "full_v55_integer_gap_certificate_protocol",
                f"prior_restricted_pool_version_v{VERSION}": PRIOR_RESTRICTED_MILP_VERSION,
                f"incumbent_repair_version_v{VERSION}": INCUMBENT_REPAIR_VERSION,
                f"terminal_reprice_version_v{VERSION}": TERMINAL_REPRICE_VERSION,
                f"universe_rows_v{VERSION}": universe_rows,
                f"selected_rows_v{VERSION}": selected_rows,
                f"full_omitted_candidate_rows_v{VERSION}": full_omitted_rows,
                f"restricted_pool_rows_v{VERSION}": restricted_pool_rows,
                f"restricted_omitted_candidate_rows_v{VERSION}": restricted_omitted_rows,
                f"restricted_pool_coverage_share_v{VERSION}": coverage_share,
                f"restricted_omitted_coverage_share_v{VERSION}": omitted_coverage_share,
                f"incumbent_objective_return_v{VERSION}": float(
                    incumbent[f"objective_return_v{INCUMBENT_REPAIR_VERSION}"]
                ),
                f"incumbent_cvar90_v{VERSION}": float(
                    incumbent[f"scenario_loss_cvar90_v{INCUMBENT_REPAIR_VERSION}"]
                ),
                f"terminal_one_swap_cleared_v{VERSION}": bool(
                    reprice[
                        f"post_restricted_pool_milp_one_swap_local_optimality_cleared_v"
                        f"{TERMINAL_REPRICE_VERSION}"
                    ]
                ),
                f"prior_restricted_pool_gap_v{VERSION}": float(
                    restricted[f"milp_gap_v{PRIOR_RESTRICTED_MILP_VERSION}"]
                ),
                f"prior_restricted_pool_improvement_found_v{VERSION}": bool(
                    restricted[
                        f"restricted_pool_improvement_found_v{PRIOR_RESTRICTED_MILP_VERSION}"
                    ]
                ),
                f"estimated_full_binary_variables_v{VERSION}": universe_rows,
                f"estimated_full_continuous_variables_v{VERSION}": continuous_vars,
                f"estimated_full_constraint_rows_v{VERSION}": model_constraint_rows,
                f"full_universe_model_built_v{VERSION}": False,
                f"full_universe_bound_probe_executed_v{VERSION}": False,
                f"full_universe_gap_certificate_available_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_BOUND_PROBE_VERSION}_full_universe_integer_bound_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "protocol only; v281 restricted-pool no-improvement is not a full-v55 gap "
                    "certificate"
                ),
            }
        ]
    )
    requirements = pd.DataFrame(
        [
            {
                f"requirement_id_v{VERSION}": "incumbent_feasibility_audited",
                f"satisfied_by_current_artifacts_v{VERSION}": True,
                f"evidence_artifact_v{VERSION}": (
                    "paper4_v279_restricted_pool_milp_repair_summary.csv;"
                    "paper4_v280_post_restricted_pool_milp_one_swap_summary.csv"
                ),
                f"missing_artifact_v{VERSION}": "",
                f"claim_boundary_v{VERSION}": "incumbent is feasible and one-swap clean only",
            },
            {
                f"requirement_id_v{VERSION}": "restricted_pool_gap_zero",
                f"satisfied_by_current_artifacts_v{VERSION}": True,
                f"evidence_artifact_v{VERSION}": "paper4_v281_restricted_pool_milp_summary.csv",
                f"missing_artifact_v{VERSION}": "",
                f"claim_boundary_v{VERSION}": "only inside v281 restricted pool",
            },
            {
                f"requirement_id_v{VERSION}": "complete_v55_candidate_coverage",
                f"satisfied_by_current_artifacts_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v55_maximal_comparable_universe.parquet",
                f"missing_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_BOUND_PROBE_VERSION}_full_universe_integer_bound_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": "v281 omits most full-v55 candidates",
            },
            {
                f"requirement_id_v{VERSION}": "full_integer_model_or_branch_price_built",
                f"satisfied_by_current_artifacts_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v282_full_universe_gap_certificate_protocol.csv",
                f"missing_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_BOUND_PROBE_VERSION}_full_universe_integer_bound_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": "protocol only; no full-v55 model artifact yet",
            },
            {
                f"requirement_id_v{VERSION}": "valid_global_dual_bound_or_gap",
                f"satisfied_by_current_artifacts_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v282_full_universe_gap_certificate_protocol.csv",
                f"missing_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_BOUND_PROBE_VERSION}_full_universe_integer_bound_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": "requires solver dual bound or equivalent certificate",
            },
            {
                f"requirement_id_v{VERSION}": "promotion_gate_and_dynamic_replay",
                f"satisfied_by_current_artifacts_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v282_claim_blockers.csv",
                f"missing_artifact_v{VERSION}": "future_dynamic_replay_and_promotion_gate",
                f"claim_boundary_v{VERSION}": "no Paper Estrella replacement or final promotion",
            },
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "restricted_pool_not_full_universe",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(full_omitted_rows - restricted_omitted_rows),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_BOUND_PROBE_VERSION}_full_universe_integer_bound_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": "v281 omits most v55 candidate loans",
            },
            {
                f"blocker_id_v{VERSION}": "full_universe_gap_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_BOUND_PROBE_VERSION}_full_universe_integer_bound_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": "no valid full-v55 integer dual bound or gap",
            },
            {
                f"blocker_id_v{VERSION}": "global_integer_optimality_claim_blocked",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_BOUND_PROBE_VERSION}_full_universe_integer_bound_probe.csv"
                ),
                f"claim_boundary_v{VERSION}": "protocol does not prove global optimality",
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
                "claim_id": "v282_full_universe_gap_protocol_executed",
                "allowed": True,
                "artifact": "paper4_v282_full_universe_gap_certificate_protocol.csv",
                "boundary": "protocol/checklist only",
            },
            {
                "claim_id": "v282_v281_restricted_pool_no_improvement",
                "allowed": True,
                "artifact": "paper4_v281_restricted_pool_milp_summary.csv",
                "boundary": "top-5000 restricted pool only",
            },
            {
                "claim_id": "v282_full_universe_gap_certificate",
                "allowed": False,
                "artifact": "paper4_v282_claim_blockers.csv",
                "boundary": "full-v55 bound probe missing",
            },
            {
                "claim_id": "v282_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v282_claim_blockers.csv",
                "boundary": "global gap certificate missing",
            },
            {
                "claim_id": "v282_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v282_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v282_full_universe_gap_certificate_protocol.csv", protocol)
    write_csv(TABLE_DIR / "paper4_v282_gap_certificate_requirements.csv", requirements)
    write_csv(TABLE_DIR / "paper4_v282_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v282_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = protocol.iloc[0]
    status = {
        "phase": "v282_full_universe_gap_certificate_protocol",
        "schema_version": "2026-05-15.282",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_restricted_pool_version_v282": PRIOR_RESTRICTED_MILP_VERSION,
        "incumbent_repair_version_v282": INCUMBENT_REPAIR_VERSION,
        "terminal_reprice_version_v282": TERMINAL_REPRICE_VERSION,
        "universe_rows_v282": int(row["universe_rows_v282"]),
        "selected_rows_v282": int(row["selected_rows_v282"]),
        "full_omitted_candidate_rows_v282": int(row["full_omitted_candidate_rows_v282"]),
        "restricted_pool_rows_v282": int(row["restricted_pool_rows_v282"]),
        "restricted_omitted_candidate_rows_v282": int(
            row["restricted_omitted_candidate_rows_v282"]
        ),
        "restricted_pool_coverage_share_v282": float(row["restricted_pool_coverage_share_v282"]),
        "restricted_omitted_coverage_share_v282": float(
            row["restricted_omitted_coverage_share_v282"]
        ),
        "prior_restricted_pool_gap_v282": float(row["prior_restricted_pool_gap_v282"]),
        "prior_restricted_pool_improvement_found_v282": bool(
            row["prior_restricted_pool_improvement_found_v282"]
        ),
        "estimated_full_binary_variables_v282": int(row["estimated_full_binary_variables_v282"]),
        "estimated_full_continuous_variables_v282": int(
            row["estimated_full_continuous_variables_v282"]
        ),
        "estimated_full_constraint_rows_v282": int(row["estimated_full_constraint_rows_v282"]),
        "requirement_rows_v282": int(len(requirements)),
        "requirements_satisfied_v282": int(
            requirements["satisfied_by_current_artifacts_v282"].astype(bool).sum()
        ),
        "claim_blocker_rows_v282": int(len(blockers)),
        "claim_matrix_rows_v282": int(len(claim_matrix)),
        "full_universe_model_built_v282": False,
        "full_universe_bound_probe_executed_v282": False,
        "full_universe_gap_certificate_available_v282": False,
        "full_universe_integer_optimality_claim_allowed_v282": False,
        "paper1_promotion_allowed_v282": False,
        "paper4_working_champion_changed_v282": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v282": f"paper4_v{NEXT_BOUND_PROBE_VERSION}_full_universe_integer_bound_probe.csv",
        "claim_boundary": (
            "v282 is a full-universe gap-certificate protocol only; exact full-v55 "
            "integer optimality and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v282_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v282": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
