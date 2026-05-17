#!/usr/bin/env python3
"""Build Paper 4 v365 v353 full-v55 pricing chunk plan artifacts."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
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

VERSION = 365
BASE_VERSION = 353
PRIOR_RESOURCE_VERSION = 364
PRIOR_GAP_VERSION = 363
NEXT_VERSION = 366
CHUNK_ROWS = 10_000
BYTES_PER_FLOAT64 = 8
WORKING_ARRAY_MULTIPLIER = 4
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v353_full_v55_pricing_chunk_prototype.csv"
CHUNK_OUTPUT_DIR = "reports/paper_material/paper4/tables/paper4_v365_pricing_chunks"


def _parquet_meta(name: str) -> dict[str, Any]:
    path = TABLE_DIR / name
    if not path.exists():
        return {"exists": False, "rows": 0, "columns": 0, "row_groups": 0, "size_bytes": 0}
    meta = pq.ParquetFile(path).metadata
    return {
        "exists": True,
        "rows": int(meta.num_rows),
        "columns": int(meta.num_columns),
        "row_groups": int(meta.num_row_groups),
        "size_bytes": int(path.stat().st_size),
    }


def _csv_meta(name: str) -> dict[str, Any]:
    path = TABLE_DIR / name
    if not path.exists():
        return {"exists": False, "rows": 0, "columns": 0, "size_bytes": 0}
    frame = pd.read_csv(path)
    return {
        "exists": True,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "size_bytes": int(path.stat().st_size),
    }


def _mb(bytes_value: float) -> float:
    return bytes_value / (1024**2)


def _chunk_schedule(*, full_omitted: int, path_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    chunk_count = math.ceil(full_omitted / CHUNK_ROWS)
    for chunk_id in range(1, chunk_count + 1):
        start = (chunk_id - 1) * CHUNK_ROWS
        end = min(start + CHUNK_ROWS, full_omitted)
        rows_in_chunk = end - start
        scenario_cells = rows_in_chunk * path_count
        raw_loss_matrix_mb = _mb(scenario_cells * BYTES_PER_FLOAT64)
        working_memory_mb = raw_loss_matrix_mb * WORKING_ARRAY_MULTIPLIER
        rows.append(
            {
                f"chunk_id_v{VERSION}": chunk_id,
                f"start_offset_in_full_omitted_v{VERSION}": start,
                f"end_offset_exclusive_v{VERSION}": end,
                f"chunk_rows_v{VERSION}": rows_in_chunk,
                f"path_count_v{VERSION}": path_count,
                f"scenario_cells_v{VERSION}": scenario_cells,
                f"raw_loss_matrix_mb_v{VERSION}": raw_loss_matrix_mb,
                f"working_memory_estimate_mb_v{VERSION}": working_memory_mb,
                f"output_partition_path_v{VERSION}": (
                    f"{CHUNK_OUTPUT_DIR}/chunk_{chunk_id:04d}.parquet"
                ),
                f"resumable_state_key_v{VERSION}": f"v365_chunk_{chunk_id:04d}",
                f"claim_boundary_v{VERSION}": (
                    "chunk schedule only; no reduced-cost or termination result yet"
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
                "claim": "v365 records a full-v55 pricing chunk plan for v353.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v365_v353_full_v55_pricing_chunk_plan.csv"
                ),
                "boundary": "Planning artifact only; no chunk is priced in v365.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v365 proves full-v55 reduced-cost termination.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v365_claim_blockers.csv"
                ),
                "boundary": "v365 schedules chunks but does not price them.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v365 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v365_claim_blockers.csv"
                ),
                "boundary": "Solver, proxy, dynamic, online and deployment gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v365 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v365_claim_blockers.csv"
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
                    "v365 defines a resumable full-v55 pricing chunk plan after the "
                    "v363/v364 gap and resource registers."
                ),
                "status": "full_v55_pricing_chunk_plan_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v366 prices one deterministic chunk and records runtime, coverage "
                    "and reduced-cost limitations without promotion"
                ),
                "last_wave": "v365",
                "execution_result": "chunk_schedule_input_manifest_and_memory_estimates_recorded",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v365")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V365_V353_FULL_V55_PRICING_CHUNK_PLAN_START -->"
    end = "<!-- V365_V353_FULL_V55_PRICING_CHUNK_PLAN_END -->"
    block = f"""
{start}

## Wave v365: v353 Full-v55 Pricing Chunk Plan

Generated: {status["generated_at_utc"]}

### Objective

v364 identified the next executable step as a full-v55 pricing chunk plan. v365
does not price or solve the full universe. It creates the resource plan,
manifest and resumable schedule needed for a bounded v366 prototype.

### Results

- Full omitted candidate rows:
  `{status["full_omitted_candidate_rows_v365"]}`.
- Scenario path count:
  `{status["path_count_v365"]}`.
- Recommended chunk rows:
  `{status["recommended_chunk_rows_v365"]}`.
- Planned chunks:
  `{status["planned_chunk_count_v365"]}`.
- Last chunk rows:
  `{status["last_chunk_rows_v365"]}`.
- Full raw loss matrix estimate MB:
  `{status["full_raw_loss_matrix_mb_v365"]}`.
- Max chunk working memory estimate MB:
  `{status["max_chunk_working_memory_estimate_mb_v365"]}`.
- Full-v55 pricing executed:
  `{status["full_v55_pricing_executed_v365"]}`.

### Interpretation

v365 makes the full-pricing route executable without overstating it. The plan
turns a large full-v55 obligation into 28 resumable chunks. The next wave can
price a single deterministic chunk and measure whether the route is practical.

### Claim Impact

- Allowed: full-v55 pricing chunk schedule and input manifest.
- Still prohibited: full reduced-cost termination, full-v55 dual-bound,
  integer optimality, working champion, Paper Estrella replacement and final
  promotion.

### Quarto Promotion Decision

Keep v365 in the living notebook. Run v366 as a deterministic chunk prototype
before making any route decision.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v363_status = json.loads((STATUS_DIR / "paper4_v363_status.json").read_text(encoding="utf-8"))
    v364_status = json.loads((STATUS_DIR / "paper4_v364_status.json").read_text(encoding="utf-8"))
    v356_partition = read_csv("paper4_v356_full_universe_candidate_partition.csv")
    v364_pending = read_csv("paper4_v364_executable_pending_register.csv")
    if v356_partition.empty or v364_pending.empty:
        raise RuntimeError("Missing v365 planning inputs.")
    if str(v364_status["recommended_next_wave_v364"]) != "v365_full_v55_pricing_chunk_plan":
        raise RuntimeError("v365 expects v364 to recommend the full-v55 chunk plan.")

    partition = v356_partition.iloc[0]
    full_rows = int(partition["full_universe_rows_v356"])
    selected_rows = int(partition["selected_rows_v356"])
    full_omitted = int(partition["full_omitted_candidate_rows_v356"])
    path_count = int(v70.N_PATHS)
    schedule = _chunk_schedule(full_omitted=full_omitted, path_count=path_count)
    planned_chunks = int(len(schedule))
    last_chunk_rows = int(schedule[f"chunk_rows_v{VERSION}"].iloc[-1])
    full_scenario_cells = full_omitted * path_count
    full_raw_loss_matrix_mb = _mb(full_scenario_cells * BYTES_PER_FLOAT64)
    full_working_memory_mb = full_raw_loss_matrix_mb * WORKING_ARRAY_MULTIPLIER
    max_chunk_working_mb = float(schedule[f"working_memory_estimate_mb_v{VERSION}"].max())

    input_manifest_rows = []
    for artifact, role in [
        ("paper4_v55_maximal_comparable_universe.parquet", "full_v55_pricing_universe"),
        ("paper4_v353_v347_expanded_branch_price_allocations.parquet", "selected_v353_book"),
        ("paper4_v356_full_universe_candidate_partition.csv", "full_omitted_partition"),
        ("paper4_v363_v353_full_dual_bound_or_gap_certificate.csv", "gap_certificate"),
        ("paper4_v364_executable_pending_register.csv", "pending_register"),
        ("paper4_v71_full_universe_reduced_costs.parquet", "prior_restricted_dual_screen"),
        ("paper4_v71_reduced_cost_summary.csv", "prior_restricted_dual_summary"),
    ]:
        meta = _parquet_meta(artifact) if artifact.endswith(".parquet") else _csv_meta(artifact)
        input_manifest_rows.append(
            {
                f"artifact_v{VERSION}": artifact,
                f"role_v{VERSION}": role,
                f"exists_v{VERSION}": bool(meta["exists"]),
                f"rows_v{VERSION}": int(meta["rows"]),
                f"columns_v{VERSION}": int(meta["columns"]),
                f"row_groups_v{VERSION}": int(meta.get("row_groups", 0)),
                f"size_bytes_v{VERSION}": int(meta["size_bytes"]),
                f"claim_boundary_v{VERSION}": "input manifest only",
            }
        )
    input_manifest = pd.DataFrame(input_manifest_rows)

    estimates = pd.DataFrame(
        [
            {
                f"estimate_id_v{VERSION}": "full_raw_loss_matrix",
                f"rows_v{VERSION}": full_omitted,
                f"path_count_v{VERSION}": path_count,
                f"scenario_cells_v{VERSION}": full_scenario_cells,
                f"memory_mb_v{VERSION}": full_raw_loss_matrix_mb,
                f"claim_boundary_v{VERSION}": "loss matrix estimate only",
            },
            {
                f"estimate_id_v{VERSION}": "full_working_arrays",
                f"rows_v{VERSION}": full_omitted,
                f"path_count_v{VERSION}": path_count,
                f"scenario_cells_v{VERSION}": full_scenario_cells,
                f"memory_mb_v{VERSION}": full_working_memory_mb,
                f"claim_boundary_v{VERSION}": "four-array working estimate only",
            },
            {
                f"estimate_id_v{VERSION}": "max_chunk_working_arrays",
                f"rows_v{VERSION}": CHUNK_ROWS,
                f"path_count_v{VERSION}": path_count,
                f"scenario_cells_v{VERSION}": CHUNK_ROWS * path_count,
                f"memory_mb_v{VERSION}": max_chunk_working_mb,
                f"claim_boundary_v{VERSION}": "maximum planned chunk working estimate only",
            },
        ]
    )
    plan = pd.DataFrame(
        [
            {
                f"plan_id_v{VERSION}": "v353_full_v55_pricing_chunk_plan",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"prior_resource_version_v{VERSION}": PRIOR_RESOURCE_VERSION,
                f"prior_gap_version_v{VERSION}": PRIOR_GAP_VERSION,
                f"full_universe_rows_v{VERSION}": full_rows,
                f"selected_rows_v{VERSION}": selected_rows,
                f"full_omitted_candidate_rows_v{VERSION}": full_omitted,
                f"path_count_v{VERSION}": path_count,
                f"recommended_chunk_rows_v{VERSION}": CHUNK_ROWS,
                f"planned_chunk_count_v{VERSION}": planned_chunks,
                f"last_chunk_rows_v{VERSION}": last_chunk_rows,
                f"full_scenario_cells_v{VERSION}": full_scenario_cells,
                f"full_raw_loss_matrix_mb_v{VERSION}": full_raw_loss_matrix_mb,
                f"full_working_memory_estimate_mb_v{VERSION}": full_working_memory_mb,
                f"max_chunk_working_memory_estimate_mb_v{VERSION}": max_chunk_working_mb,
                f"v71_improving_omitted_columns_v{VERSION}": int(
                    v363_status["v71_improving_omitted_columns_v363"]
                ),
                f"bounded_candidate_pool_share_v{VERSION}": float(
                    v363_status["bounded_candidate_pool_share_v363"]
                ),
                f"full_v55_pricing_executed_v{VERSION}": False,
                f"valid_full_v55_dual_bound_certificate_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "chunk plan only; no full-v55 pricing, termination, or integer proof"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "full_v55_pricing_not_executed",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": full_omitted,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "v365 schedules pricing but does not execute it",
            },
            {
                f"blocker_id_v{VERSION}": "v71_negative_reduced_cost_persists",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    v363_status["v71_improving_omitted_columns_v363"]
                ),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "prior restricted-dual screen remains non-terminating",
            },
            {
                f"blocker_id_v{VERSION}": "integer_optimality_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "planning does not provide integer proof",
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
                "claim_id": "v365_full_v55_pricing_chunk_plan_created",
                "allowed": True,
                "artifact": "paper4_v365_v353_full_v55_pricing_chunk_plan.csv",
                "boundary": "planning only",
            },
            {
                "claim_id": "v365_full_v55_pricing_executed",
                "allowed": False,
                "artifact": "paper4_v365_claim_blockers.csv",
                "boundary": "scheduled but not run",
            },
            {
                "claim_id": "v365_valid_full_v55_dual_bound_certificate",
                "allowed": False,
                "artifact": "paper4_v365_claim_blockers.csv",
                "boundary": "termination proof missing",
            },
            {
                "claim_id": "v365_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v365_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v365_v353_full_v55_pricing_chunk_plan.csv", plan)
    write_csv(TABLE_DIR / "paper4_v365_pricing_chunk_schedule.csv", schedule)
    write_csv(TABLE_DIR / "paper4_v365_input_manifest.csv", input_manifest)
    write_csv(TABLE_DIR / "paper4_v365_resource_estimates.csv", estimates)
    write_csv(TABLE_DIR / "paper4_v365_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v365_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = plan.iloc[0]
    status = {
        "phase": "v365_v353_full_v55_pricing_chunk_plan",
        "schema_version": "2026-05-17.365",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v365": BASE_VERSION,
        "prior_resource_version_v365": PRIOR_RESOURCE_VERSION,
        "prior_gap_version_v365": PRIOR_GAP_VERSION,
        "full_universe_rows_v365": int(row[f"full_universe_rows_v{VERSION}"]),
        "selected_rows_v365": int(row[f"selected_rows_v{VERSION}"]),
        "full_omitted_candidate_rows_v365": int(
            row[f"full_omitted_candidate_rows_v{VERSION}"]
        ),
        "path_count_v365": int(row[f"path_count_v{VERSION}"]),
        "recommended_chunk_rows_v365": int(row[f"recommended_chunk_rows_v{VERSION}"]),
        "planned_chunk_count_v365": int(row[f"planned_chunk_count_v{VERSION}"]),
        "last_chunk_rows_v365": int(row[f"last_chunk_rows_v{VERSION}"]),
        "full_scenario_cells_v365": int(row[f"full_scenario_cells_v{VERSION}"]),
        "full_raw_loss_matrix_mb_v365": float(row[f"full_raw_loss_matrix_mb_v{VERSION}"]),
        "full_working_memory_estimate_mb_v365": float(
            row[f"full_working_memory_estimate_mb_v{VERSION}"]
        ),
        "max_chunk_working_memory_estimate_mb_v365": float(
            row[f"max_chunk_working_memory_estimate_mb_v{VERSION}"]
        ),
        "v71_improving_omitted_columns_v365": int(
            row[f"v71_improving_omitted_columns_v{VERSION}"]
        ),
        "bounded_candidate_pool_share_v365": float(
            row[f"bounded_candidate_pool_share_v{VERSION}"]
        ),
        "full_v55_pricing_executed_v365": False,
        "valid_full_v55_dual_bound_certificate_v365": False,
        "full_universe_integer_optimality_claim_allowed_v365": False,
        "working_champion_claim_allowed_v365": False,
        "paper1_promotion_allowed_v365": False,
        "paper4_working_champion_changed_v365": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "input_manifest_rows_v365": int(len(input_manifest)),
        "resource_estimate_rows_v365": int(len(estimates)),
        "claim_blocker_rows_v365": int(len(blockers)),
        "claim_matrix_rows_v365": int(len(claim_matrix)),
        "next_artifact_v365": NEXT_ARTIFACT,
        "claim_boundary": (
            "v365 records a full-v55 pricing chunk plan only; full dual-bound, "
            "integer optimality, champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v365_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v365": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
