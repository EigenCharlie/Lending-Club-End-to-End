#!/usr/bin/env python3
"""Build Paper 4 v372 grade-A source-relief prefilter artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
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

VERSION = 372
PRIOR_DIAGNOSTIC_VERSION = 371
PRIOR_CHUNK_VERSION = 366
NEXT_VERSION = 373
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_full_v55_chunk_002_or_stop_rule.csv"
EXPOSURE_MIN = 842292.375
EXPOSURE_MAX = 850000.0


def _flow_mask(
    *, flow: str, add_grade: np.ndarray, drop_grade: np.ndarray
) -> np.ndarray:
    if flow == "grade_a_relief_drop_a_add_non_a":
        return (drop_grade[None, :] == "A") & (add_grade[:, None] != "A")
    if flow == "grade_a_neutral_drop_a_add_a":
        return (drop_grade[None, :] == "A") & (add_grade[:, None] == "A")
    if flow == "grade_a_pressure_drop_non_a_add_a":
        return (drop_grade[None, :] != "A") & (add_grade[:, None] == "A")
    if flow == "non_a_to_non_a":
        return (drop_grade[None, :] != "A") & (add_grade[:, None] != "A")
    raise ValueError(flow)


def _top_rows(
    *,
    mask: np.ndarray,
    chunk: pd.DataFrame,
    selected: pd.DataFrame,
    return_delta: np.ndarray,
    exposure_after: np.ndarray,
    limit: int = 10,
) -> pd.DataFrame:
    positions = np.argwhere(mask)
    if len(positions) == 0:
        return pd.DataFrame(
            columns=[
                f"rank_v{VERSION}",
                f"added_loan_id_v{VERSION}",
                f"dropped_loan_id_v{VERSION}",
                f"added_grade_v{VERSION}",
                f"dropped_grade_v{VERSION}",
                f"return_delta_v{VERSION}",
                f"exposure_after_swap_v{VERSION}",
                f"claim_boundary_v{VERSION}",
            ]
        )
    values = return_delta[mask]
    order = np.argsort(-values)[:limit]
    rows: list[dict[str, Any]] = []
    for rank, position_index in enumerate(order, start=1):
        add_pos, drop_pos = positions[int(position_index)]
        add_row = chunk.iloc[int(add_pos)]
        drop_row = selected.iloc[int(drop_pos)]
        rows.append(
            {
                f"rank_v{VERSION}": rank,
                f"added_loan_id_v{VERSION}": str(add_row["loan_id"]),
                f"dropped_loan_id_v{VERSION}": str(drop_row["loan_id"]),
                f"added_grade_v{VERSION}": str(add_row["grade"]),
                f"dropped_grade_v{VERSION}": str(drop_row["grade"]),
                f"added_loan_amount_v{VERSION}": float(add_row["loan_amnt"]),
                f"dropped_loan_amount_v{VERSION}": float(drop_row["loan_amnt"]),
                f"return_delta_v{VERSION}": float(
                    return_delta[int(add_pos), int(drop_pos)]
                ),
                f"exposure_after_swap_v{VERSION}": float(
                    exposure_after[int(add_pos), int(drop_pos)]
                ),
                f"claim_boundary_v{VERSION}": (
                    "grade-A relief candidate ranking only; no apply or promotion"
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
                "claim": "v372 tests a grade=A source-relief prefilter for v366 chunk 0001.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v372_grade_a_source_relief_prefilter.csv"
                ),
                "boundary": "Prefilter diagnostic only; no candidate is applied.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v372 finds no return-improving grade=A relief swap in chunk 0001.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v372_grade_a_flow_return_budget.csv"
                ),
                "boundary": "Chunk-0001 one-swap diagnostic only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v372 proves full-v55 reduced-cost termination.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v372_claim_blockers.csv"
                ),
                "boundary": "v372 diagnoses a prefilter; remaining chunks are still unpriced.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v372 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v372_claim_blockers.csv"
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
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v372 tests whether a grade=A source-relief prefilter can recover "
                    "return-improving rows in v366 chunk 0001."
                ),
                "status": "grade_a_relief_prefilter_no_return_improving_rows",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v373 decides whether to run chunk 002, sample chunks, or set a stop rule"
                ),
                "last_wave": "v372",
                "execution_result": "grade_a_relief_is_return_negative_in_chunk_0001",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v372")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V372_GRADE_A_SOURCE_RELIEF_PREFILTER_START -->"
    end = "<!-- V372_GRADE_A_SOURCE_RELIEF_PREFILTER_END -->"
    block = f"""
{start}

## Wave v372: Grade-A Source-Relief Prefilter

Generated: {status["generated_at_utc"]}

### Objective

v371 identified grade=A as the primary source-governance blocker for v366 chunk
0001. v372 tests whether a simple `drop A / add non-A` prefilter can recover
return-improving rows inside that same chunk.

### Results

- Grade-A relief rows:
  `{status["grade_a_relief_rows_v372"]}`.
- Grade-A relief budget-feasible rows:
  `{status["grade_a_relief_budget_rows_v372"]}`.
- Grade-A relief return-improving rows:
  `{status["grade_a_relief_return_improving_rows_v372"]}`.
- Grade-A relief budget+return rows:
  `{status["grade_a_relief_budget_return_rows_v372"]}`.
- Best budget-feasible grade-A relief return delta:
  `{status["best_grade_a_relief_budget_return_delta_v372"]}`.
- Grade-A pressure budget+return rows:
  `{status["grade_a_pressure_budget_return_rows_v372"]}`.
- Recommended route:
  `{status["recommended_route_v372"]}`.
- Next artifact:
  `{status["next_artifact_v372"]}`.

### Interpretation

The grade-A relief prefilter does not recover a candidate in chunk 0001. It
finds many budget-feasible relief swaps, but none improve return; the best
budget-feasible relief row is still negative. That means the immediate issue is
not only source pressure, but a return/source trade-off inside this chunk.

### Claim Impact

- Allowed: grade-A relief prefilter diagnostic for chunk 0001.
- Still prohibited: full-v55 termination, valid global integer optimality,
  working champion, Paper Estrella replacement and final promotion.

### Quarto Promotion Decision

Keep v372 in the living notebook. v373 should decide between chunk 002,
targeted chunk sampling, or a documented stop rule.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v371_status = json.loads((STATUS_DIR / "paper4_v371_status.json").read_text(encoding="utf-8"))
    schedule = read_csv("paper4_v365_pricing_chunk_schedule.csv")
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet(
        "paper4_v353_v347_expanded_branch_price_allocations.parquet"
    ).reset_index(drop=True)
    if any(df.empty for df in [schedule, universe, selected]):
        raise RuntimeError("Missing v372 grade-A prefilter inputs.")
    if v371_status["primary_blocker_family_v371"] != "grade":
        raise RuntimeError("v372 expects v371 to identify grade as the primary blocker.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    for frame in [universe, selected]:
        frame["grade"] = frame["grade"].astype(str)
    selected_ids = set(selected["loan_id"])
    omitted = (
        universe.loc[~universe["loan_id"].isin(selected_ids)]
        .sort_values(["loan_index_v55", "loan_id"])
        .reset_index(drop=True)
    )
    first_chunk = schedule.sort_values("chunk_id_v365").iloc[0]
    chunk = omitted.iloc[
        int(first_chunk["start_offset_in_full_omitted_v365"]) : int(
            first_chunk["end_offset_exclusive_v365"]
        )
    ].copy()
    chunk["grade"] = chunk["grade"].astype(str)

    _losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    add_returns = mean_returns[idx_by_id.loc[chunk["loan_id"]].to_numpy(int)]
    drop_returns = mean_returns[idx_by_id.loc[selected["loan_id"]].to_numpy(int)]
    add_amount = chunk["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    current_exposure = float(selected["loan_amnt"].sum())
    return_delta = add_returns[:, None] - drop_returns[None, :]
    exposure_after = current_exposure + add_amount[:, None] - drop_amount[None, :]
    return_mask = return_delta > 1e-9
    budget_mask = (exposure_after >= EXPOSURE_MIN - 1e-7) & (
        exposure_after <= EXPOSURE_MAX + 1e-7
    )
    budget_return_mask = return_mask & budget_mask
    add_grade = chunk["grade"].to_numpy()
    drop_grade = selected["grade"].to_numpy()

    flow_ids = [
        "grade_a_relief_drop_a_add_non_a",
        "grade_a_neutral_drop_a_add_a",
        "grade_a_pressure_drop_non_a_add_a",
        "non_a_to_non_a",
    ]
    flow_rows: list[dict[str, Any]] = []
    for flow in flow_ids:
        mask = _flow_mask(flow=flow, add_grade=add_grade, drop_grade=drop_grade)
        budget_rows = int((mask & budget_mask).sum())
        return_rows = int((mask & return_mask).sum())
        budget_return_rows = int((mask & budget_return_mask).sum())
        best_budget_return_delta = (
            None if budget_rows == 0 else float(return_delta[mask & budget_mask].max())
        )
        best_return_delta = (
            None if return_rows == 0 else float(return_delta[mask & return_mask].max())
        )
        flow_rows.append(
            {
                f"flow_id_v{VERSION}": flow,
                f"all_rows_v{VERSION}": int(mask.sum()),
                f"budget_feasible_rows_v{VERSION}": budget_rows,
                f"return_improving_rows_v{VERSION}": return_rows,
                f"budget_return_feasible_rows_v{VERSION}": budget_return_rows,
                f"best_budget_feasible_return_delta_v{VERSION}": best_budget_return_delta,
                f"best_return_improving_delta_v{VERSION}": best_return_delta,
                f"share_of_budget_return_rows_v{VERSION}": budget_return_rows
                / max(int(budget_return_mask.sum()), 1),
                f"claim_boundary_v{VERSION}": "grade-flow prefilter diagnostic only",
            }
        )
    flow_summary = pd.DataFrame(flow_rows)
    relief_mask = _flow_mask(
        flow="grade_a_relief_drop_a_add_non_a",
        add_grade=add_grade,
        drop_grade=drop_grade,
    )
    relief_budget_mask = relief_mask & budget_mask
    relief_top = _top_rows(
        mask=relief_budget_mask,
        chunk=chunk,
        selected=selected,
        return_delta=return_delta,
        exposure_after=exposure_after,
    )
    route_options = pd.DataFrame(
        [
            {
                f"route_option_v{VERSION}": "run_grade_a_relief_prefilter_candidate",
                f"recommended_v{VERSION}": False,
                f"evidence_count_v{VERSION}": int((relief_mask & budget_return_mask).sum()),
                f"next_artifact_if_chosen_v{VERSION}": "none_for_chunk_0001",
                f"claim_boundary_v{VERSION}": "no return-improving relief rows exist in chunk 0001",
            },
            {
                f"route_option_v{VERSION}": "chunk_002_or_stop_rule",
                f"recommended_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int((relief_mask & budget_mask).sum()),
                f"next_artifact_if_chosen_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "decide whether new chunks are informative",
            },
            {
                f"route_option_v{VERSION}": "manuscript_limitations_language",
                f"recommended_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int((relief_mask & budget_mask).sum()),
                f"next_artifact_if_chosen_v{VERSION}": "paper4_v372_claim_language_note.md",
                f"claim_boundary_v{VERSION}": "scope language only",
            },
        ]
    )
    relief_row = flow_summary.loc[
        flow_summary[f"flow_id_v{VERSION}"].eq("grade_a_relief_drop_a_add_non_a")
    ].iloc[0]
    pressure_row = flow_summary.loc[
        flow_summary[f"flow_id_v{VERSION}"].eq("grade_a_pressure_drop_non_a_add_a")
    ].iloc[0]
    prefilter = pd.DataFrame(
        [
            {
                f"prefilter_id_v{VERSION}": "v372_grade_a_source_relief_prefilter",
                f"prior_diagnostic_version_v{VERSION}": PRIOR_DIAGNOSTIC_VERSION,
                f"prior_chunk_version_v{VERSION}": PRIOR_CHUNK_VERSION,
                f"selected_grade_a_rows_v{VERSION}": int(selected["grade"].eq("A").sum()),
                f"chunk_grade_a_add_rows_v{VERSION}": int(chunk["grade"].eq("A").sum()),
                f"chunk_non_grade_a_add_rows_v{VERSION}": int(chunk["grade"].ne("A").sum()),
                f"ordered_one_swap_rows_v{VERSION}": int(return_delta.size),
                f"budget_return_feasible_rows_v{VERSION}": int(budget_return_mask.sum()),
                f"grade_a_relief_rows_v{VERSION}": int(relief_row[f"all_rows_v{VERSION}"]),
                f"grade_a_relief_budget_rows_v{VERSION}": int(
                    relief_row[f"budget_feasible_rows_v{VERSION}"]
                ),
                f"grade_a_relief_return_improving_rows_v{VERSION}": int(
                    relief_row[f"return_improving_rows_v{VERSION}"]
                ),
                f"grade_a_relief_budget_return_rows_v{VERSION}": int(
                    relief_row[f"budget_return_feasible_rows_v{VERSION}"]
                ),
                f"best_grade_a_relief_budget_return_delta_v{VERSION}": float(
                    relief_row[f"best_budget_feasible_return_delta_v{VERSION}"]
                ),
                f"grade_a_pressure_budget_return_rows_v{VERSION}": int(
                    pressure_row[f"budget_return_feasible_rows_v{VERSION}"]
                ),
                f"recommended_route_v{VERSION}": "chunk_002_or_stop_rule",
                f"valid_full_v55_dual_bound_certificate_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "grade-A relief prefilter only; no candidate apply or full proof"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "grade_a_relief_has_zero_return_improving_rows",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    relief_row[f"return_improving_rows_v{VERSION}"]
                ),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "chunk 0001 relief prefilter has no candidate",
            },
            {
                f"blocker_id_v{VERSION}": "remaining_chunks_unpriced",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 27,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "full-v55 proof remains open",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_final_promotion_forbidden",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
                f"claim_boundary_v{VERSION}": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v372_grade_a_source_relief_prefilter_created",
                "allowed": True,
                "artifact": "paper4_v372_grade_a_source_relief_prefilter.csv",
                "boundary": "prefilter diagnostic only",
            },
            {
                "claim_id": "v372_no_return_improving_grade_a_relief_in_chunk_0001",
                "allowed": True,
                "artifact": "paper4_v372_grade_a_flow_return_budget.csv",
                "boundary": "chunk-0001 diagnostic only",
            },
            {
                "claim_id": "v372_valid_full_v55_dual_bound_certificate",
                "allowed": False,
                "artifact": "paper4_v372_claim_blockers.csv",
                "boundary": "remaining chunks unpriced",
            },
            {
                "claim_id": "v372_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v372_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v372_grade_a_source_relief_prefilter.csv", prefilter)
    write_csv(TABLE_DIR / "paper4_v372_grade_a_flow_return_budget.csv", flow_summary)
    write_csv(TABLE_DIR / "paper4_v372_top_grade_a_relief_budget_candidates.csv", relief_top)
    write_csv(TABLE_DIR / "paper4_v372_route_options.csv", route_options)
    write_csv(TABLE_DIR / "paper4_v372_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v372_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = prefilter.iloc[0]
    status = {
        "phase": "v372_grade_a_source_relief_prefilter",
        "schema_version": "2026-05-17.372",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_diagnostic_version_v372": PRIOR_DIAGNOSTIC_VERSION,
        "prior_chunk_version_v372": PRIOR_CHUNK_VERSION,
        "selected_grade_a_rows_v372": int(row[f"selected_grade_a_rows_v{VERSION}"]),
        "chunk_grade_a_add_rows_v372": int(row[f"chunk_grade_a_add_rows_v{VERSION}"]),
        "chunk_non_grade_a_add_rows_v372": int(row[f"chunk_non_grade_a_add_rows_v{VERSION}"]),
        "ordered_one_swap_rows_v372": int(row[f"ordered_one_swap_rows_v{VERSION}"]),
        "budget_return_feasible_rows_v372": int(
            row[f"budget_return_feasible_rows_v{VERSION}"]
        ),
        "grade_a_relief_rows_v372": int(row[f"grade_a_relief_rows_v{VERSION}"]),
        "grade_a_relief_budget_rows_v372": int(
            row[f"grade_a_relief_budget_rows_v{VERSION}"]
        ),
        "grade_a_relief_return_improving_rows_v372": int(
            row[f"grade_a_relief_return_improving_rows_v{VERSION}"]
        ),
        "grade_a_relief_budget_return_rows_v372": int(
            row[f"grade_a_relief_budget_return_rows_v{VERSION}"]
        ),
        "best_grade_a_relief_budget_return_delta_v372": float(
            row[f"best_grade_a_relief_budget_return_delta_v{VERSION}"]
        ),
        "grade_a_pressure_budget_return_rows_v372": int(
            row[f"grade_a_pressure_budget_return_rows_v{VERSION}"]
        ),
        "route_option_rows_v372": int(len(route_options)),
        "top_relief_budget_candidate_rows_v372": int(len(relief_top)),
        "claim_blocker_rows_v372": int(len(blockers)),
        "claim_matrix_rows_v372": int(len(claim_matrix)),
        "recommended_route_v372": str(row[f"recommended_route_v{VERSION}"]),
        "valid_full_v55_dual_bound_certificate_v372": False,
        "full_universe_integer_optimality_claim_allowed_v372": False,
        "working_champion_claim_allowed_v372": False,
        "paper1_promotion_allowed_v372": False,
        "paper4_working_champion_changed_v372": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v372": NEXT_ARTIFACT,
        "claim_boundary": (
            "v372 tests grade-A relief prefilter; solver, champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v372_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v372": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
