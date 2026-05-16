#!/usr/bin/env python3
"""Build Paper 4 v283 full-universe integer bound/resource probe artifacts."""

from __future__ import annotations

import hashlib
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

VERSION = 283
PROTOCOL_VERSION = 282
INCUMBENT_REPAIR_VERSION = 279
TERMINAL_REPRICE_VERSION = 280
RESTRICTED_POOL_VERSION = 281
NEXT_DECOMPOSITION_VERSION = 284
SCENARIO_COUNT = 128
SOURCE_CONSTRAINT_ROWS = 45
MAX_BINARY_VARS_FOR_DIRECT_MIP = 50_000
GAP_TOLERANCE = 1e-6


def _manifest_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v283 full-universe integer bound/resource probe.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v283_full_universe_integer_bound_probe.csv"
                ),
                "boundary": (
                    "Resource-gated full-v55 model probe only; no direct full-v55 MIP solve "
                    "or valid global gap certificate."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v283 proves full-universe global integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v283_claim_blockers.csv"
                ),
                "boundary": "Direct full-v55 MIP was resource-guarded and no bound was produced.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v283 authorizes Paper Estrella replacement or final Paper 4 promotion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v283_claim_blockers.csv"
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
                    "v283 evaluates whether a direct full-v55 integer bound probe is executable "
                    "and records a reproducible resource blocker when it is not."
                ),
                "status": "full_universe_bound_probe_resource_guarded",
                "next_artifact": "paper4_v284_decomposition_or_branch_price_prototype.csv",
                "success_condition": (
                    "a decomposition/branch-price prototype produces a global bound path or a "
                    "sharper reproducible blocker than direct full-v55 MIP"
                ),
                "last_wave": "v283",
                "execution_result": "direct_full_v55_mip_resource_guarded",
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
    start = "<!-- V283_FULL_UNIVERSE_INTEGER_BOUND_PROBE_START -->"
    end = "<!-- V283_FULL_UNIVERSE_INTEGER_BOUND_PROBE_END -->"
    block = f"""
{start}

## Wave v283: Full-Universe Integer Bound/Resource Probe

Generated: {status["generated_at_utc"]}

### Objective

Turn the v282 protocol into an executable feasibility gate for a direct full-v55
integer bound solve. The probe estimates the exact full-universe formulation
size, compares it with the direct-MIP resource guard, and records whether a
valid global bound was produced.

### Results

- Full-v55 binary variables: `{status["full_binary_variables_v283"]}`.
- Estimated continuous variables: `{status["full_continuous_variables_v283"]}`.
- Estimated constraint rows: `{status["full_constraint_rows_v283"]}`.
- Direct-MIP binary variable guard:
  `{status["max_binary_vars_for_direct_mip_v283"]}`.
- Direct full-v55 MIP attempted:
  `{status["direct_full_mip_attempted_v283"]}`.
- Resource guard reason:
  `{status["resource_guard_reason_v283"]}`.
- Valid full-universe gap certificate available:
  `{status["valid_full_universe_gap_certificate_v283"]}`.

### Interpretation

v283 makes the full-v55 blocker operational. The direct MIP path is too large
for the configured direct-solve guard, so the next executable route is a
decomposition, branch-and-price, or equivalent bound prototype rather than a
promotion claim.

### Claim Impact

- Allowed: resource-gated full-v55 bound probe executed.
- Still prohibited: full-universe global integer optimality, Paper Estrella
  replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v283 in the living notebook. Promote only after a valid global bound,
dynamic validation and promotion gates pass.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    protocol = read_csv("paper4_v282_full_universe_gap_certificate_protocol.csv")
    incumbent = read_csv("paper4_v279_restricted_pool_milp_repair_summary.csv")
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    if universe.empty or protocol.empty or incumbent.empty or source_caps.empty:
        raise RuntimeError("Missing v55, v80, v279, or v282 inputs for v283.")

    protocol_row = protocol.iloc[0]
    incumbent_row = incumbent.iloc[0]
    universe_rows = int(len(universe))
    selected_rows = int(protocol_row[f"selected_rows_v{PROTOCOL_VERSION}"])
    full_omitted_rows = int(universe_rows - selected_rows)
    continuous_vars = 1 + SCENARIO_COUNT
    constraint_rows = 1 + 1 + SOURCE_CONSTRAINT_ROWS + 1 + SCENARIO_COUNT
    direct_attempt = universe_rows <= MAX_BINARY_VARS_FOR_DIRECT_MIP
    resource_guard_reason = (
        "within_direct_mip_guard"
        if direct_attempt
        else "binary_variable_count_exceeds_direct_mip_guard"
    )
    manifest_payload = {
        "version": VERSION,
        "universe_rows": universe_rows,
        "selected_rows": selected_rows,
        "scenario_count": SCENARIO_COUNT,
        "source_constraint_rows": SOURCE_CONSTRAINT_ROWS,
        "constraint_rows": constraint_rows,
        "binary_variables": universe_rows,
        "continuous_variables": continuous_vars,
        "objective": "maximize_mean_return_subject_to_budget_source_cvar_cardinality",
        "incumbent_version": INCUMBENT_REPAIR_VERSION,
        "terminal_reprice_version": TERMINAL_REPRICE_VERSION,
    }
    manifest_digest = _manifest_hash(manifest_payload)

    manifest = pd.DataFrame(
        [
            {
                f"manifest_id_v{VERSION}": "full_v55_integer_model_manifest",
                f"manifest_sha256_v{VERSION}": manifest_digest,
                f"universe_rows_v{VERSION}": universe_rows,
                f"selected_rows_v{VERSION}": selected_rows,
                f"full_omitted_candidate_rows_v{VERSION}": full_omitted_rows,
                f"binary_variables_v{VERSION}": universe_rows,
                f"continuous_variables_v{VERSION}": continuous_vars,
                f"constraint_rows_v{VERSION}": constraint_rows,
                f"scenario_count_v{VERSION}": SCENARIO_COUNT,
                f"source_constraint_rows_v{VERSION}": SOURCE_CONSTRAINT_ROWS,
                f"incumbent_repair_version_v{VERSION}": INCUMBENT_REPAIR_VERSION,
                f"terminal_reprice_version_v{VERSION}": TERMINAL_REPRICE_VERSION,
                f"restricted_pool_version_v{VERSION}": RESTRICTED_POOL_VERSION,
                f"claim_boundary_v{VERSION}": (
                    "full-v55 model manifest only; no solve or global bound certificate"
                ),
            }
        ]
    )
    bound_probe = pd.DataFrame(
        [
            {
                f"probe_id_v{VERSION}": "direct_full_v55_integer_mip_bound_probe",
                f"protocol_version_v{VERSION}": PROTOCOL_VERSION,
                f"full_binary_variables_v{VERSION}": universe_rows,
                f"full_continuous_variables_v{VERSION}": continuous_vars,
                f"full_constraint_rows_v{VERSION}": constraint_rows,
                f"full_omitted_candidate_rows_v{VERSION}": full_omitted_rows,
                f"max_binary_vars_for_direct_mip_v{VERSION}": MAX_BINARY_VARS_FOR_DIRECT_MIP,
                f"direct_full_mip_attempted_v{VERSION}": direct_attempt,
                f"direct_full_mip_attempt_result_v{VERSION}": (
                    "not_executed_resource_guard" if not direct_attempt else "not_implemented"
                ),
                f"resource_guard_reason_v{VERSION}": resource_guard_reason,
                f"incumbent_objective_return_v{VERSION}": float(
                    incumbent_row[f"objective_return_v{INCUMBENT_REPAIR_VERSION}"]
                ),
                f"incumbent_cvar90_v{VERSION}": float(
                    incumbent_row[f"scenario_loss_cvar90_v{INCUMBENT_REPAIR_VERSION}"]
                ),
                f"valid_full_universe_gap_certificate_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_DECOMPOSITION_VERSION}_decomposition_or_branch_price_prototype.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "resource-bound direct full-v55 MIP probe; decomposition or branch-price "
                    "required for global bound"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "direct_full_v55_mip_resource_guarded",
                f"blocking_v{VERSION}": not direct_attempt,
                f"evidence_count_v{VERSION}": universe_rows,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_DECOMPOSITION_VERSION}_decomposition_or_branch_price_prototype.csv"
                ),
                f"claim_boundary_v{VERSION}": "direct full-v55 MIP exceeds configured guard",
            },
            {
                f"blocker_id_v{VERSION}": "global_integer_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_DECOMPOSITION_VERSION}_decomposition_or_branch_price_prototype.csv"
                ),
                f"claim_boundary_v{VERSION}": "no global dual bound or gap certificate produced",
            },
            {
                f"blocker_id_v{VERSION}": "decomposition_or_branch_price_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_DECOMPOSITION_VERSION}_decomposition_or_branch_price_prototype.csv"
                ),
                f"claim_boundary_v{VERSION}": "next route is a decomposed bound prototype",
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
                "claim_id": "v283_full_universe_bound_probe_executed",
                "allowed": True,
                "artifact": "paper4_v283_full_universe_integer_bound_probe.csv",
                "boundary": "resource-gated bound probe only",
            },
            {
                "claim_id": "v283_full_universe_model_manifest_created",
                "allowed": True,
                "artifact": "paper4_v283_full_universe_model_manifest.csv",
                "boundary": "manifest only; no solve",
            },
            {
                "claim_id": "v283_valid_full_universe_gap_certificate",
                "allowed": False,
                "artifact": "paper4_v283_claim_blockers.csv",
                "boundary": "direct MIP resource guarded and no decomposed bound yet",
            },
            {
                "claim_id": "v283_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v283_claim_blockers.csv",
                "boundary": "global bound missing",
            },
            {
                "claim_id": "v283_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v283_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v283_full_universe_integer_bound_probe.csv", bound_probe)
    write_csv(TABLE_DIR / "paper4_v283_full_universe_model_manifest.csv", manifest)
    write_csv(TABLE_DIR / "paper4_v283_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v283_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = bound_probe.iloc[0]
    status = {
        "phase": "v283_full_universe_integer_bound_probe",
        "schema_version": "2026-05-15.283",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "protocol_version_v283": PROTOCOL_VERSION,
        "incumbent_repair_version_v283": INCUMBENT_REPAIR_VERSION,
        "terminal_reprice_version_v283": TERMINAL_REPRICE_VERSION,
        "restricted_pool_version_v283": RESTRICTED_POOL_VERSION,
        "full_binary_variables_v283": int(row["full_binary_variables_v283"]),
        "full_continuous_variables_v283": int(row["full_continuous_variables_v283"]),
        "full_constraint_rows_v283": int(row["full_constraint_rows_v283"]),
        "full_omitted_candidate_rows_v283": int(row["full_omitted_candidate_rows_v283"]),
        "max_binary_vars_for_direct_mip_v283": MAX_BINARY_VARS_FOR_DIRECT_MIP,
        "direct_full_mip_attempted_v283": bool(row["direct_full_mip_attempted_v283"]),
        "resource_guard_reason_v283": str(row["resource_guard_reason_v283"]),
        "manifest_sha256_v283": manifest_digest,
        "valid_full_universe_gap_certificate_v283": False,
        "full_universe_integer_optimality_claim_allowed_v283": False,
        "paper1_promotion_allowed_v283": False,
        "paper4_working_champion_changed_v283": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v283": int(len(blockers)),
        "claim_matrix_rows_v283": int(len(claim_matrix)),
        "next_artifact_v283": (
            f"paper4_v{NEXT_DECOMPOSITION_VERSION}_decomposition_or_branch_price_prototype.csv"
        ),
        "claim_boundary": (
            "v283 is a resource-gated full-v55 integer bound probe only; exact global "
            "integer optimality and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v283_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v283": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
