#!/usr/bin/env python3
"""Build Paper 4 v363 v353 full-dual-bound gap certificate artifacts."""

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

VERSION = 363
BASE_VERSION = 353
PRIOR_DISPOSITION_VERSION = 362
READINESS_VERSION = 356
RESTRICTED_DUAL_VERSION = 71
NEXT_VERSION = 364
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v353_dual_bound_resource_plan.csv"


def _safe_int(payload: dict[str, Any], key: str, default: int = 0) -> int:
    value = payload.get(key, default)
    if value is None or pd.isna(value):
        return default
    return int(value)


def _safe_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if value is None or pd.isna(value):
        return default
    return float(value)


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v363 records a full-dual-bound gap certificate for v353.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v363_v353_full_dual_bound_or_gap_certificate.csv"
                ),
                "boundary": (
                    "Negative/gap certificate only: it records why a valid full-v55 "
                    "termination proof is still unavailable."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v363 proves a valid full-v55 dual-bound termination.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v363_dual_bound_requirement_register.csv"
                ),
                "boundary": (
                    "Direct full MIP guard, all-column pricing termination and integer "
                    "optimality certificate remain unmet."
                ),
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v363 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v363_claim_blockers.csv"
                ),
                "boundary": "Proxy, global, dynamic, online and deployment gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v363 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v363_claim_blockers.csv"
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
                    "v363 converts the v362 fourth-order no-entry memo into a "
                    "full-dual-bound gap certificate and requirement register."
                ),
                "status": "full_dual_bound_gap_certificate_created_no_global_proof",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "resource a full-v55 pricing/dual-bound plan or define a tighter "
                    "published claim that avoids global optimality"
                ),
                "last_wave": "v363",
                "execution_result": "valid_full_v55_dual_bound_still_blocked_with_requirements_recorded",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v363")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V363_V353_FULL_DUAL_BOUND_GAP_CERTIFICATE_START -->"
    end = "<!-- V363_V353_FULL_DUAL_BOUND_GAP_CERTIFICATE_END -->"
    block = f"""
{start}

## Wave v363: v353 Full-Dual-Bound Gap Certificate

Generated: {status["generated_at_utc"]}

### Objective

v362 recorded the no-apply disposition after the v361 fourth-order branch-price
screen found no CVaR-feasible entering row. v363 asks the harder question:
whether the current evidence is enough to claim a valid full-v55 dual-bound or
integer optimality certificate. It is not.

### Results

- Full v55 rows: `{status["full_universe_rows_v363"]}`.
- Full omitted candidates:
  `{status["full_omitted_candidate_rows_v363"]}`.
- Positive source-tight candidate pool used by v361:
  `{status["positive_source_tight_candidate_rows_v363"]}`.
- Bounded candidate-pool share of full omitted universe:
  `{status["bounded_candidate_pool_share_v363"]}`.
- V361 ordered fourth-order rows:
  `{status["v361_ordered_fourth_order_rows_v363"]}`.
- V361 source-exact rows:
  `{status["v361_source_exact_fourth_order_rows_v363"]}`.
- V361 CVaR-feasible entering rows:
  `{status["v361_cvar_feasible_entering_rows_v363"]}`.
- Best source-exact CVaR gap versus v353 cap:
  `{status["best_source_exact_cvar_gap_vs_cap_v363"]}`.
- v71 restricted-master improving omitted columns:
  `{status["v71_improving_omitted_columns_v363"]}`.
- Valid full-v55 dual-bound certificate:
  `{status["valid_full_v55_dual_bound_certificate_v363"]}`.

### Interpretation

v363 is a negative certificate. The living lab now has strong bounded evidence:
fourth-order source-tight pricing screened 371.10M ordered rows and found no
CVaR-feasible entering row. But the bounded pool is still a small subset of the
full omitted universe, and earlier restricted-master dual pricing even detected
negative reduced-cost omitted columns. That combination blocks any global
termination or integer optimality claim.

### Claim Impact

- Allowed: gap certificate explaining why full-v55 dual-bound is not available.
- Allowed: bounded fourth-order no-entry remains valid only within its scope.
- Still prohibited: full-v55 branch-price termination, valid global integer
  optimality, contractual IFRS9, live deployability, Paper Estrella replacement,
  final Paper 4 promotion and working champion claims.

### Quarto Promotion Decision

Keep v363 in the living notebook. The next wave should be a resource plan for a
full-v55 pricing/dual-bound attempt, or a paper-scope decision that removes
global optimality from the publishable claim.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v356_status = json.loads((STATUS_DIR / "paper4_v356_status.json").read_text(encoding="utf-8"))
    v362_status = json.loads((STATUS_DIR / "paper4_v362_status.json").read_text(encoding="utf-8"))
    v71_status = json.loads((STATUS_DIR / "paper4_v71_status.json").read_text(encoding="utf-8"))
    v356_requirements = read_csv("paper4_v356_dual_bound_requirement_register.csv")
    v356_partition = read_csv("paper4_v356_full_universe_candidate_partition.csv")
    v362_scope = read_csv("paper4_v362_no_entry_scope_register.csv")
    if any(df.empty for df in [v356_requirements, v356_partition, v362_scope]):
        raise RuntimeError("Missing v363 full dual-bound gap inputs.")
    if not bool(v362_status["no_apply_disposition_allowed_v362"]):
        raise RuntimeError("v363 expects v362 no-apply disposition evidence.")

    partition = v356_partition.iloc[0]
    full_rows = int(partition["full_universe_rows_v356"])
    selected_rows = int(partition["selected_rows_v356"])
    full_omitted = int(partition["full_omitted_candidate_rows_v356"])
    positive_pool = _safe_int(v362_status["stage_map_snapshot_v362"], "positive_source_tight_fourth_add_candidates")
    bounded_share = positive_pool / max(full_omitted, 1)
    direct_full_mip_guard_met = bool(v356_status["direct_full_mip_guard_met_v356"])
    v71_negative_rc = bool(v71_status["negative_reduced_cost_detected_v71"])
    v71_improving = int(v71_status["improving_omitted_columns_v71"])
    all_column_pricing_terminated = False
    integer_certificate_available = False
    valid_full_certificate = (
        direct_full_mip_guard_met
        and not v71_negative_rc
        and all_column_pricing_terminated
        and integer_certificate_available
    )

    certificate = pd.DataFrame(
        [
            {
                f"certificate_id_v{VERSION}": "v353_full_dual_bound_gap_certificate",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"prior_disposition_version_v{VERSION}": PRIOR_DISPOSITION_VERSION,
                f"readiness_version_v{VERSION}": READINESS_VERSION,
                f"restricted_dual_version_v{VERSION}": RESTRICTED_DUAL_VERSION,
                f"full_universe_rows_v{VERSION}": full_rows,
                f"selected_rows_v{VERSION}": selected_rows,
                f"full_omitted_candidate_rows_v{VERSION}": full_omitted,
                f"positive_source_tight_candidate_rows_v{VERSION}": positive_pool,
                f"bounded_candidate_pool_share_v{VERSION}": bounded_share,
                f"v361_ordered_fourth_order_rows_v{VERSION}": int(
                    v362_status["v361_ordered_fourth_order_rows_screened_v362"]
                ),
                f"v361_source_exact_fourth_order_rows_v{VERSION}": int(
                    v362_status["v361_source_exact_fourth_order_rows_v362"]
                ),
                f"v361_cvar_feasible_entering_rows_v{VERSION}": int(
                    v362_status["v361_cvar_feasible_entering_rows_v362"]
                ),
                f"best_source_exact_cvar_gap_vs_cap_v{VERSION}": float(
                    v362_status["best_source_exact_cvar_gap_vs_cap_v362"]
                ),
                f"direct_full_mip_guard_met_v{VERSION}": direct_full_mip_guard_met,
                f"v71_negative_reduced_cost_detected_v{VERSION}": v71_negative_rc,
                f"v71_improving_omitted_columns_v{VERSION}": v71_improving,
                f"all_column_pricing_terminated_v{VERSION}": all_column_pricing_terminated,
                f"integer_certificate_available_v{VERSION}": integer_certificate_available,
                f"valid_full_v55_dual_bound_certificate_v{VERSION}": valid_full_certificate,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "negative full-dual-bound gap certificate; records blockers only"
                ),
            }
        ]
    )
    requirements = pd.DataFrame(
        [
            {
                f"requirement_id_v{VERSION}": "bounded_fourth_order_no_entry_recorded",
                f"met_v{VERSION}": True,
                f"evidence_artifact_v{VERSION}": (
                    "paper4_v362_v353_apply_fourth_order_candidate_or_bound_memo.csv"
                ),
                f"required_next_artifact_v{VERSION}": "none_for_bounded_scope",
                f"claim_boundary_v{VERSION}": "bounded fourth-order scope only",
            },
            {
                f"requirement_id_v{VERSION}": "full_omitted_universe_priced_or_excluded",
                f"met_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v356_full_universe_candidate_partition.csv",
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "v361 source-tight pool covers only a bounded subset of full omitted universe"
                ),
            },
            {
                f"requirement_id_v{VERSION}": "direct_full_mip_guard_met",
                f"met_v{VERSION}": direct_full_mip_guard_met,
                f"evidence_artifact_v{VERSION}": "paper4_v356_dual_bound_requirement_register.csv",
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "direct full-v55 binary solve remains above guard",
            },
            {
                f"requirement_id_v{VERSION}": "restricted_dual_screen_terminated_without_negative_rc",
                f"met_v{VERSION}": not v71_negative_rc,
                f"evidence_artifact_v{VERSION}": "paper4_v71_reduced_cost_summary.csv",
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "v71 restricted-master dual pricing is not termination",
            },
            {
                f"requirement_id_v{VERSION}": "all_column_pricing_terminated",
                f"met_v{VERSION}": all_column_pricing_terminated,
                f"evidence_artifact_v{VERSION}": "missing",
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "no full-v55 branch-price termination proof exists",
            },
            {
                f"requirement_id_v{VERSION}": "integer_optimality_certificate_available",
                f"met_v{VERSION}": integer_certificate_available,
                f"evidence_artifact_v{VERSION}": "missing",
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "continuous or bounded pricing evidence is not integer proof",
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
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "bounded_pool_not_full_omitted_universe",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": full_omitted - positive_pool,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "positive source-tight pool does not cover full omitted universe",
            },
            {
                f"blocker_id_v{VERSION}": "v71_negative_reduced_cost_persists",
                f"blocking_v{VERSION}": v71_negative_rc,
                f"evidence_count_v{VERSION}": v71_improving,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "restricted-master dual evidence did not terminate",
            },
            {
                f"blocker_id_v{VERSION}": "best_fourth_order_cvar_above_cap",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    round(float(v362_status["best_source_exact_cvar_gap_vs_cap_v362"]))
                ),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "best bounded fourth-order row still violates CVaR cap",
            },
            {
                f"blocker_id_v{VERSION}": "integer_optimality_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "no full integer optimality proof exists",
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
                "claim_id": "v363_full_dual_bound_gap_certificate_created",
                "allowed": True,
                "artifact": "paper4_v363_v353_full_dual_bound_or_gap_certificate.csv",
                "boundary": "negative/gap certificate only",
            },
            {
                "claim_id": "v363_valid_full_v55_dual_bound_certificate",
                "allowed": False,
                "artifact": "paper4_v363_dual_bound_requirement_register.csv",
                "boundary": "requirements unmet",
            },
            {
                "claim_id": "v363_global_integer_optimality_or_working_champion",
                "allowed": False,
                "artifact": "paper4_v363_claim_blockers.csv",
                "boundary": "no integer proof or promotion",
            },
            {
                "claim_id": "v363_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v363_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(
        TABLE_DIR / "paper4_v363_v353_full_dual_bound_or_gap_certificate.csv",
        certificate,
    )
    write_csv(TABLE_DIR / "paper4_v363_dual_bound_requirement_register.csv", requirements)
    write_csv(TABLE_DIR / "paper4_v363_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v363_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = certificate.iloc[0]
    status = {
        "phase": "v363_v353_full_dual_bound_or_gap_certificate",
        "schema_version": "2026-05-17.363",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v363": BASE_VERSION,
        "prior_disposition_version_v363": PRIOR_DISPOSITION_VERSION,
        "readiness_version_v363": READINESS_VERSION,
        "restricted_dual_version_v363": RESTRICTED_DUAL_VERSION,
        "full_universe_rows_v363": int(row[f"full_universe_rows_v{VERSION}"]),
        "selected_rows_v363": int(row[f"selected_rows_v{VERSION}"]),
        "full_omitted_candidate_rows_v363": int(row[f"full_omitted_candidate_rows_v{VERSION}"]),
        "positive_source_tight_candidate_rows_v363": int(
            row[f"positive_source_tight_candidate_rows_v{VERSION}"]
        ),
        "bounded_candidate_pool_share_v363": float(
            row[f"bounded_candidate_pool_share_v{VERSION}"]
        ),
        "v361_ordered_fourth_order_rows_v363": int(
            row[f"v361_ordered_fourth_order_rows_v{VERSION}"]
        ),
        "v361_source_exact_fourth_order_rows_v363": int(
            row[f"v361_source_exact_fourth_order_rows_v{VERSION}"]
        ),
        "v361_cvar_feasible_entering_rows_v363": int(
            row[f"v361_cvar_feasible_entering_rows_v{VERSION}"]
        ),
        "best_source_exact_cvar_gap_vs_cap_v363": float(
            row[f"best_source_exact_cvar_gap_vs_cap_v{VERSION}"]
        ),
        "direct_full_mip_guard_met_v363": bool(row[f"direct_full_mip_guard_met_v{VERSION}"]),
        "v71_negative_reduced_cost_detected_v363": bool(
            row[f"v71_negative_reduced_cost_detected_v{VERSION}"]
        ),
        "v71_improving_omitted_columns_v363": int(
            row[f"v71_improving_omitted_columns_v{VERSION}"]
        ),
        "all_column_pricing_terminated_v363": bool(
            row[f"all_column_pricing_terminated_v{VERSION}"]
        ),
        "integer_certificate_available_v363": bool(
            row[f"integer_certificate_available_v{VERSION}"]
        ),
        "valid_full_v55_dual_bound_certificate_v363": bool(
            row[f"valid_full_v55_dual_bound_certificate_v{VERSION}"]
        ),
        "full_universe_integer_optimality_claim_allowed_v363": False,
        "working_champion_claim_allowed_v363": False,
        "paper1_promotion_allowed_v363": False,
        "paper4_working_champion_changed_v363": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "requirement_rows_v363": int(len(requirements)),
        "requirements_met_v363": int(requirements[f"met_v{VERSION}"].astype(bool).sum()),
        "claim_blocker_rows_v363": int(len(blockers)),
        "claim_matrix_rows_v363": int(len(claim_matrix)),
        "next_artifact_v363": NEXT_ARTIFACT,
        "claim_boundary": (
            "v363 is a negative/gap certificate: full dual-bound, integer optimality, "
            "champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v363_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v363": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
