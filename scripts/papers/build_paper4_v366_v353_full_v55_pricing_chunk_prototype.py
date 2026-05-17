#!/usr/bin/env python3
"""Build Paper 4 v366 v353 full-v55 pricing chunk prototype artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers import build_paper4_v351_v347_branch_price_or_dual_bound_loop as v351
from scripts.papers import build_paper4_v357_v353_branch_price_or_dual_bound_loop as v357
from scripts.papers.paper4_one_swap_living_lab import (
    FAMILIES,
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

VERSION = 366
BASE_VERSION = 353
PRIOR_PLAN_VERSION = 365
NEXT_VERSION = 367
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_route_decision_after_chunk_probe.csv"
EXPOSURE_MIN = 842292.375
EXPOSURE_MAX = 850000.0


def _maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _candidate_columns() -> list[str]:
    return [
        "policy_id",
        f"regime_v{VERSION}",
        f"chunk_id_v{VERSION}",
        f"added_loan_id_v{VERSION}",
        f"dropped_loan_id_v{VERSION}",
        f"action_signature_v{VERSION}",
        f"added_loan_index_v{VERSION}",
        f"added_loan_amount_v{VERSION}",
        f"dropped_loan_amount_v{VERSION}",
        f"return_delta_v{VERSION}",
        f"objective_return_after_swap_v{VERSION}",
        f"exposure_after_swap_v{VERSION}",
        f"loss_mean_after_swap_v{VERSION}",
        f"cvar90_after_swap_v{VERSION}",
        f"source_min_slack_after_swap_v{VERSION}",
        f"max_source_share_after_swap_v{VERSION}",
        f"source_cap_violations_after_swap_v{VERSION}",
        f"first_source_block_family_v{VERSION}",
        f"first_source_block_id_v{VERSION}",
        f"cvar_swap_feasible_v{VERSION}",
        f"chunk_entering_column_v{VERSION}",
        f"claim_boundary_v{VERSION}",
    ]


def _screen_chunk(
    *,
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    chunk: pd.DataFrame,
    source_summary: pd.DataFrame,
    losses: np.ndarray,
    mean_returns: np.ndarray,
    idx_by_id: pd.Series,
    current_losses: np.ndarray,
    current_objective_return: float,
    current_exposure: float,
    cvar_cap: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy(int)
    chunk_idx = idx_by_id.loc[chunk["loan_id"].astype(str)].to_numpy(int)
    selected = selected.copy()
    chunk = chunk.copy()
    selected[f"mean_return_if_dropped_v{VERSION}"] = mean_returns[selected_idx]
    chunk[f"mean_return_if_added_v{VERSION}"] = mean_returns[chunk_idx]

    add_amount = chunk["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    add_return = chunk[f"mean_return_if_added_v{VERSION}"].to_numpy(float)
    drop_return = selected[f"mean_return_if_dropped_v{VERSION}"].to_numpy(float)
    return_delta = add_return[:, None] - drop_return[None, :]
    exposure_after = current_exposure + add_amount[:, None] - drop_amount[None, :]
    return_mask = return_delta > 1e-9
    budget_return_mask = (
        return_mask
        & (exposure_after >= EXPOSURE_MIN - 1e-7)
        & (exposure_after <= EXPOSURE_MAX + 1e-7)
    )

    current_by_family, cap_by_family = v357._source_maps(selected, source_summary)
    cumulative = budget_return_mask.copy()
    family_counts: dict[str, int] = {}
    for family in FAMILIES:
        family_mask = v351._family_mask_after_seed(
            base_mask=budget_return_mask,
            seed_source_maps=current_by_family,
            cap_by_family=cap_by_family,
            second_adds=chunk,
            second_drops=selected,
            seed_exposure=current_exposure,
            family=family,
        )
        family_counts[family] = int(family_mask.sum())
        cumulative &= family_mask

    source_exact_positions = np.argwhere(cumulative)
    rows: list[dict[str, Any]] = []
    best_cvar = np.nan
    best_return = np.nan
    entering_rows = 0
    for add_pos, drop_pos in source_exact_positions:
        add_pos = int(add_pos)
        drop_pos = int(drop_pos)
        add_row = chunk.iloc[add_pos]
        drop_row = selected.iloc[drop_pos]
        new_losses = current_losses + losses[:, chunk_idx[add_pos]] - losses[:, selected_idx[drop_pos]]
        cvar_after = v70._tail_cvar(new_losses)
        total_return_delta = float(return_delta[add_pos, drop_pos])
        best_cvar = cvar_after if pd.isna(best_cvar) else min(best_cvar, cvar_after)
        best_return = total_return_delta if pd.isna(best_return) else max(best_return, total_return_delta)
        source_ok, min_slack, max_share, violations, first_family, first_source = (
            v351._source_metrics_multi(
                add_rows=[add_row],
                drop_rows=[drop_row],
                current_by_family=current_by_family,
                cap_by_family=cap_by_family,
                new_total=float(exposure_after[add_pos, drop_pos]),
            )
        )
        if not source_ok:
            continue
        entering = total_return_delta > 1e-9 and cvar_after <= cvar_cap + 1e-7
        entering_rows += int(entering)
        added_id = str(add_row["loan_id"])
        dropped_id = str(drop_row["loan_id"])
        rows.append(
            {
                "policy_id": "v366_v353_full_v55_chunk_one_swap_prototype",
                f"regime_v{VERSION}": "full_v55_chunk_0001_one_add_one_drop",
                f"chunk_id_v{VERSION}": 1,
                f"added_loan_id_v{VERSION}": added_id,
                f"dropped_loan_id_v{VERSION}": dropped_id,
                f"action_signature_v{VERSION}": f"add={added_id};drop={dropped_id}",
                f"added_loan_index_v{VERSION}": int(add_row["loan_index_v55"]),
                f"added_loan_amount_v{VERSION}": float(add_row["loan_amnt"]),
                f"dropped_loan_amount_v{VERSION}": float(drop_row["loan_amnt"]),
                f"return_delta_v{VERSION}": total_return_delta,
                f"objective_return_after_swap_v{VERSION}": current_objective_return
                + total_return_delta,
                f"exposure_after_swap_v{VERSION}": float(exposure_after[add_pos, drop_pos]),
                f"loss_mean_after_swap_v{VERSION}": float(new_losses.mean()),
                f"cvar90_after_swap_v{VERSION}": cvar_after,
                f"source_min_slack_after_swap_v{VERSION}": min_slack,
                f"max_source_share_after_swap_v{VERSION}": max_share,
                f"source_cap_violations_after_swap_v{VERSION}": violations,
                f"first_source_block_family_v{VERSION}": first_family,
                f"first_source_block_id_v{VERSION}": first_source,
                f"cvar_swap_feasible_v{VERSION}": cvar_after <= cvar_cap + 1e-7,
                f"chunk_entering_column_v{VERSION}": entering,
                f"claim_boundary_v{VERSION}": (
                    "single deterministic full-v55 chunk prototype row; no termination certificate"
                ),
            }
        )

    stage_rows: list[dict[str, Any]] = [
        {
            f"stage_v{VERSION}": "chunk_omitted_add_candidates",
            f"row_count_v{VERSION}": int(len(chunk)),
            f"claim_boundary_v{VERSION}": "first v365 full omitted chunk only",
        },
        {
            f"stage_v{VERSION}": "selected_drop_candidates",
            f"row_count_v{VERSION}": int(len(selected)),
            f"claim_boundary_v{VERSION}": "v353 selected loans",
        },
        {
            f"stage_v{VERSION}": "ordered_one_swap_rows",
            f"row_count_v{VERSION}": int(return_delta.size),
            f"claim_boundary_v{VERSION}": "chunk add/drop combinations only",
        },
        {
            f"stage_v{VERSION}": "return_improving",
            f"row_count_v{VERSION}": int(return_mask.sum()),
            f"claim_boundary_v{VERSION}": "positive one-swap return delta only",
        },
        {
            f"stage_v{VERSION}": "budget_return_feasible",
            f"row_count_v{VERSION}": int(budget_return_mask.sum()),
            f"claim_boundary_v{VERSION}": "budget plus positive return only",
        },
    ]
    stage_rows.extend(
        {
            f"stage_v{VERSION}": f"{family}_source_feasible_alone",
            f"row_count_v{VERSION}": family_counts[family],
            f"claim_boundary_v{VERSION}": f"{family} cap only after budget+return",
        }
        for family in FAMILIES
    )
    stage_rows.extend(
        [
            {
                f"stage_v{VERSION}": "source_exact_feasible",
                f"row_count_v{VERSION}": int(len(rows)),
                f"claim_boundary_v{VERSION}": "all-family source caps in chunk prototype",
            },
            {
                f"stage_v{VERSION}": "cvar_feasible_entering_column",
                f"row_count_v{VERSION}": int(entering_rows),
                f"claim_boundary_v{VERSION}": "source-exact and CVaR-feasible chunk rows",
            },
            {
                f"stage_v{VERSION}": "best_source_exact_return_delta",
                f"row_count_v{VERSION}": _maybe_float(best_return),
                f"claim_boundary_v{VERSION}": "diagnostic scalar encoded in row_count",
            },
            {
                f"stage_v{VERSION}": "best_source_exact_cvar90",
                f"row_count_v{VERSION}": _maybe_float(best_cvar),
                f"claim_boundary_v{VERSION}": "diagnostic scalar encoded in row_count",
            },
        ]
    )
    candidate_screen = pd.DataFrame(rows, columns=_candidate_columns())
    if not candidate_screen.empty:
        candidate_screen = candidate_screen.sort_values(
            [f"chunk_entering_column_v{VERSION}", f"return_delta_v{VERSION}"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return pd.DataFrame(stage_rows), candidate_screen


def _update_claim_boundaries(*, entering_rows: int) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v366 executes a deterministic full-v55 chunk prototype.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v366_v353_full_v55_pricing_chunk_prototype.csv"
                ),
                "boundary": "One chunk only; no full-v55 pricing termination.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v366 finds a bounded chunk entering candidate.",
                "allowed": entering_rows > 0,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v366_entering_candidate_summary.csv"
                ),
                "boundary": "Chunk-local candidate only; route decision required before apply/reprice.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v366 proves full-v55 reduced-cost termination.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v366_claim_blockers.csv"
                ),
                "boundary": "Only chunk 1 is prototyped; remaining chunks are unpriced.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v366 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v366_claim_blockers.csv"
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


def _update_backlog(*, entering_rows: int) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v366 executes the first v365 full-v55 pricing chunk as a "
                    "deterministic one-swap prototype."
                ),
                "status": (
                    "chunk_entering_candidate_found_requires_route_decision"
                    if entering_rows
                    else "chunk_prototype_no_cvar_entering_column"
                ),
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v367 decides whether to apply/reprice a chunk candidate, continue chunking, "
                    "switch bounded search depth, or narrow paper scope"
                ),
                "last_wave": "v366",
                "execution_result": (
                    "bounded_chunk_entering_candidate_found"
                    if entering_rows
                    else "no_chunk_0001_cvar_feasible_entering_column"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v366")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V366_V353_FULL_V55_PRICING_CHUNK_PROTOTYPE_START -->"
    end = "<!-- V366_V353_FULL_V55_PRICING_CHUNK_PROTOTYPE_END -->"
    best_entering = status["best_entering_return_delta_v366"]
    best_entering_text = (
        "not applicable; no chunk entering rows" if best_entering is None else str(best_entering)
    )
    block = f"""
{start}

## Wave v366: v353 Full-v55 Pricing Chunk Prototype

Generated: {status["generated_at_utc"]}

### Objective

v365 created a full-v55 chunk plan. v366 executes the first deterministic chunk
as a one-add/one-drop pricing prototype against the v353 book. This is a
runtime and feasibility probe, not a full-v55 termination proof.

### Results

- Chunk id: `{status["chunk_id_v366"]}`.
- Chunk rows: `{status["chunk_rows_v366"]}`.
- Ordered one-swap rows:
  `{status["ordered_one_swap_rows_v366"]}`.
- Budget+return feasible rows:
  `{status["budget_return_feasible_rows_v366"]}`.
- Source-exact rows:
  `{status["source_exact_rows_v366"]}`.
- CVaR-feasible entering rows:
  `{status["cvar_feasible_entering_rows_v366"]}`.
- Best source-exact return delta:
  `{status["best_source_exact_return_delta_v366"]}`.
- Best source-exact CVaR90:
  `{status["best_source_exact_cvar90_v366"]}`.
- Best entering return delta:
  `{best_entering_text}`.
- Full-v55 termination claim:
  `{status["valid_full_v55_dual_bound_certificate_v366"]}`.

### Interpretation

v366 turns the v365 plan into a measured prototype. The result can reveal a
chunk-local candidate or a no-entry blocker for chunk 1, but it cannot say
anything final about the remaining 27 planned chunks.

### Claim Impact

- Allowed: deterministic chunk-1 prototype evidence.
- Allowed only if positive: chunk-local entering candidate requiring route
  decision before any apply/reprice.
- Still prohibited: full-v55 reduced-cost termination, valid global integer
  optimality, working champion, Paper Estrella replacement and final promotion.

### Quarto Promotion Decision

Keep v366 in the living notebook. v367 must decide the route using this chunk
evidence and the v363/v365 blockers.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v353_status = json.loads((STATUS_DIR / "paper4_v353_status.json").read_text(encoding="utf-8"))
    v365_status = json.loads((STATUS_DIR / "paper4_v365_status.json").read_text(encoding="utf-8"))
    schedule = read_csv("paper4_v365_pricing_chunk_schedule.csv")
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet(
        "paper4_v353_v347_expanded_branch_price_allocations.parquet"
    ).reset_index(drop=True)
    source_summary = read_csv("paper4_v353_v347_expanded_branch_price_source_summary.csv")
    if any(df.empty for df in [schedule, universe, selected, source_summary]):
        raise RuntimeError("Missing v366 chunk prototype inputs.")
    if bool(v365_status["full_v55_pricing_executed_v365"]):
        raise RuntimeError("v366 expects v365 to be a plan, not executed pricing.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    for frame in [universe, selected]:
        for family in FAMILIES:
            frame[family] = frame[family].astype(str)
    selected_ids = set(selected["loan_id"].astype(str))
    omitted = (
        universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)]
        .sort_values(["loan_index_v55", "loan_id"])
        .reset_index(drop=True)
    )
    first_chunk = schedule.sort_values(f"chunk_id_v{PRIOR_PLAN_VERSION}").iloc[0]
    chunk_id = int(first_chunk[f"chunk_id_v{PRIOR_PLAN_VERSION}"])
    chunk_start = int(first_chunk[f"start_offset_in_full_omitted_v{PRIOR_PLAN_VERSION}"])
    chunk_end = int(first_chunk[f"end_offset_exclusive_v{PRIOR_PLAN_VERSION}"])
    chunk = omitted.iloc[chunk_start:chunk_end].copy().reset_index(drop=True)
    if chunk.empty:
        raise RuntimeError("v366 chunk is empty.")

    losses, mean_returns, path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy(int)
    current_losses = losses[:, selected_idx].sum(axis=1)
    current_objective_return = float(v353_status["objective_return_v353"])
    current_exposure = float(selected["loan_amnt"].sum())
    cvar_cap = float(v353_status["scenario_loss_cvar90_v353"])

    stage, candidate_screen = _screen_chunk(
        universe=universe,
        selected=selected,
        chunk=chunk,
        source_summary=source_summary,
        losses=losses,
        mean_returns=mean_returns,
        idx_by_id=idx_by_id,
        current_losses=current_losses,
        current_objective_return=current_objective_return,
        current_exposure=current_exposure,
        cvar_cap=cvar_cap,
    )
    entering_summary = (
        candidate_screen.loc[candidate_screen[f"chunk_entering_column_v{VERSION}"].astype(bool)]
        .sort_values(f"return_delta_v{VERSION}", ascending=False)
        .reset_index(drop=True)
        if not candidate_screen.empty
        else pd.DataFrame(columns=candidate_screen.columns)
    )
    entering_rows = int(len(entering_summary))
    best_by_return = candidate_screen.head(1)
    best_by_cvar = (
        candidate_screen.sort_values(f"cvar90_after_swap_v{VERSION}").head(1)
        if not candidate_screen.empty
        else pd.DataFrame()
    )
    best_entering = entering_summary.head(1)
    stage_map = dict(zip(stage[f"stage_v{VERSION}"], stage[f"row_count_v{VERSION}"], strict=False))
    protocol = pd.DataFrame(
        [
            {
                f"prototype_id_v{VERSION}": "v353_full_v55_chunk_0001_one_swap_prototype",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"prior_plan_version_v{VERSION}": PRIOR_PLAN_VERSION,
                f"chunk_id_v{VERSION}": chunk_id,
                f"chunk_start_offset_v{VERSION}": chunk_start,
                f"chunk_end_offset_exclusive_v{VERSION}": chunk_end,
                f"chunk_rows_v{VERSION}": int(len(chunk)),
                f"path_count_v{VERSION}": int(len(path_ids)),
                f"selected_drop_rows_v{VERSION}": int(len(selected)),
                f"ordered_one_swap_rows_v{VERSION}": int(stage_map["ordered_one_swap_rows"]),
                f"return_improving_rows_v{VERSION}": int(stage_map["return_improving"]),
                f"budget_return_feasible_rows_v{VERSION}": int(stage_map["budget_return_feasible"]),
                f"source_exact_rows_v{VERSION}": int(stage_map["source_exact_feasible"]),
                f"cvar_feasible_entering_rows_v{VERSION}": entering_rows,
                f"best_source_exact_return_delta_v{VERSION}": None
                if best_by_return.empty
                else float(best_by_return[f"return_delta_v{VERSION}"].iloc[0]),
                f"best_source_exact_cvar90_v{VERSION}": None
                if best_by_cvar.empty
                else float(best_by_cvar[f"cvar90_after_swap_v{VERSION}"].iloc[0]),
                f"best_entering_return_delta_v{VERSION}": None
                if best_entering.empty
                else float(best_entering[f"return_delta_v{VERSION}"].iloc[0]),
                f"best_entering_cvar90_v{VERSION}": None
                if best_entering.empty
                else float(best_entering[f"cvar90_after_swap_v{VERSION}"].iloc[0]),
                f"chunk_prototype_executed_v{VERSION}": True,
                f"full_v55_pricing_executed_v{VERSION}": False,
                f"valid_full_v55_dual_bound_certificate_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "chunk 1 one-swap prototype only; no full-v55 termination certificate"
                ),
            }
        ]
    )
    limitations = pd.DataFrame(
        [
            {
                f"limitation_id_v{VERSION}": "remaining_chunks_unpriced",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v365_status["planned_chunk_count_v365"]) - 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "only chunk 1 was prototyped",
            },
            {
                f"limitation_id_v{VERSION}": "one_swap_not_full_column_generation",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "one-add/one-drop prototype is not dual pricing termination",
            },
            {
                f"limitation_id_v{VERSION}": "integer_optimality_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "chunk prototype does not provide integer proof",
            },
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "full_v55_termination_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v365_status["planned_chunk_count_v365"]) - 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "remaining v365 chunks are unpriced",
            },
            {
                f"blocker_id_v{VERSION}": "chunk_candidate_requires_route_decision",
                f"blocking_v{VERSION}": entering_rows > 0,
                f"evidence_count_v{VERSION}": entering_rows,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "positive chunk candidate cannot be applied without route decision",
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
                "claim_id": "v366_chunk_prototype_executed",
                "allowed": True,
                "artifact": "paper4_v366_v353_full_v55_pricing_chunk_prototype.csv",
                "boundary": "chunk 1 only",
            },
            {
                "claim_id": "v366_chunk_entering_candidate_found",
                "allowed": entering_rows > 0,
                "artifact": "paper4_v366_entering_candidate_summary.csv",
                "boundary": "chunk-local candidate only",
            },
            {
                "claim_id": "v366_valid_full_v55_dual_bound_certificate",
                "allowed": False,
                "artifact": "paper4_v366_claim_blockers.csv",
                "boundary": "remaining chunks unpriced",
            },
            {
                "claim_id": "v366_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v366_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v366_v353_full_v55_pricing_chunk_prototype.csv", protocol)
    write_csv(TABLE_DIR / "paper4_v366_chunk_stage_summary.csv", stage)
    write_csv(TABLE_DIR / "paper4_v366_chunk_candidate_screen.csv", candidate_screen)
    write_csv(TABLE_DIR / "paper4_v366_entering_candidate_summary.csv", entering_summary)
    write_csv(TABLE_DIR / "paper4_v366_limitations.csv", limitations)
    write_csv(TABLE_DIR / "paper4_v366_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v366_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries(entering_rows=entering_rows)
    _update_backlog(entering_rows=entering_rows)

    row = protocol.iloc[0]
    status = {
        "phase": "v366_v353_full_v55_pricing_chunk_prototype",
        "schema_version": "2026-05-17.366",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v366": BASE_VERSION,
        "prior_plan_version_v366": PRIOR_PLAN_VERSION,
        "chunk_id_v366": int(row[f"chunk_id_v{VERSION}"]),
        "chunk_start_offset_v366": int(row[f"chunk_start_offset_v{VERSION}"]),
        "chunk_end_offset_exclusive_v366": int(row[f"chunk_end_offset_exclusive_v{VERSION}"]),
        "chunk_rows_v366": int(row[f"chunk_rows_v{VERSION}"]),
        "path_count_v366": int(row[f"path_count_v{VERSION}"]),
        "selected_drop_rows_v366": int(row[f"selected_drop_rows_v{VERSION}"]),
        "ordered_one_swap_rows_v366": int(row[f"ordered_one_swap_rows_v{VERSION}"]),
        "return_improving_rows_v366": int(row[f"return_improving_rows_v{VERSION}"]),
        "budget_return_feasible_rows_v366": int(row[f"budget_return_feasible_rows_v{VERSION}"]),
        "source_exact_rows_v366": int(row[f"source_exact_rows_v{VERSION}"]),
        "cvar_feasible_entering_rows_v366": entering_rows,
        "best_source_exact_return_delta_v366": _maybe_float(
            row[f"best_source_exact_return_delta_v{VERSION}"]
        ),
        "best_source_exact_cvar90_v366": _maybe_float(
            row[f"best_source_exact_cvar90_v{VERSION}"]
        ),
        "best_entering_return_delta_v366": _maybe_float(
            row[f"best_entering_return_delta_v{VERSION}"]
        ),
        "best_entering_cvar90_v366": _maybe_float(row[f"best_entering_cvar90_v{VERSION}"]),
        "chunk_prototype_executed_v366": True,
        "full_v55_pricing_executed_v366": False,
        "valid_full_v55_dual_bound_certificate_v366": False,
        "full_universe_integer_optimality_claim_allowed_v366": False,
        "working_champion_claim_allowed_v366": False,
        "paper1_promotion_allowed_v366": False,
        "paper4_working_champion_changed_v366": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "stage_summary_rows_v366": int(len(stage)),
        "candidate_screen_rows_v366": int(len(candidate_screen)),
        "entering_candidate_summary_rows_v366": int(len(entering_summary)),
        "limitation_rows_v366": int(len(limitations)),
        "claim_blocker_rows_v366": int(len(blockers)),
        "claim_matrix_rows_v366": int(len(claim_matrix)),
        "next_artifact_v366": NEXT_ARTIFACT,
        "claim_boundary": (
            "v366 is a deterministic chunk prototype only; full dual-bound, integer optimality, "
            "champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v366_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v366": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
