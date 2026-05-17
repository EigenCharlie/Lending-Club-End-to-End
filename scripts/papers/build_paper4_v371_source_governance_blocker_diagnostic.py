#!/usr/bin/env python3
"""Build Paper 4 v371 source-governance blocker diagnostic artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
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

VERSION = 371
PRIOR_BACKLOG_VERSION = 370
PRIOR_CHUNK_VERSION = 366
NEXT_VERSION = 372
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_grade_a_source_relief_prefilter.csv"
EXPOSURE_MIN = 842292.375
EXPOSURE_MAX = 850000.0


def _source_caps(source_summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    caps: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        local = source_summary.loc[source_summary["source_family"].astype(str).eq(family)]
        caps[family] = {
            str(row["source_id"]): float(row["cap_share_v353"]) for _, row in local.iterrows()
        }
    return caps


def _family_mask(
    *,
    family: str,
    selected: pd.DataFrame,
    chunk: pd.DataFrame,
    budget_return_mask: np.ndarray,
    exposure_after: np.ndarray,
    caps: dict[str, dict[str, float]],
    current_exposure_by_source: dict[str, float],
) -> np.ndarray:
    out = budget_return_mask.copy()
    add_amount = chunk["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    add_source = chunk[family].astype(str).to_numpy()
    drop_source = selected[family].astype(str).to_numpy()
    for source_id, cap in caps[family].items():
        current = float(current_exposure_by_source.get(f"{family}={source_id}", 0.0))
        new_source = (
            current
            + add_amount[:, None] * (add_source[:, None] == source_id)
            - drop_amount[None, :] * (drop_source[None, :] == source_id)
        )
        out &= (new_source / exposure_after) <= cap + 1e-7
    return out


def _tight_source_slack(
    *,
    family: str,
    source_id: str,
    selected: pd.DataFrame,
    chunk: pd.DataFrame,
    budget_return_mask: np.ndarray,
    exposure_after: np.ndarray,
    cap: float,
    current_exposure: float,
) -> dict[str, Any]:
    add_amount = chunk["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    add_source = chunk[family].astype(str).to_numpy()
    drop_source = selected[family].astype(str).to_numpy()
    new_source = (
        current_exposure
        + add_amount[:, None] * (add_source[:, None] == source_id)
        - drop_amount[None, :] * (drop_source[None, :] == source_id)
    )
    slack = cap - (new_source / exposure_after)
    feasible = budget_return_mask & (slack >= -1e-7)
    budget_slack = slack[budget_return_mask]
    return {
        "tight_source_pass_rows": int(feasible.sum()),
        "tight_source_best_slack": None if len(budget_slack) == 0 else float(np.max(budget_slack)),
        "tight_source_worst_slack": None if len(budget_slack) == 0 else float(np.min(budget_slack)),
        "tight_source_mean_slack": None if len(budget_slack) == 0 else float(np.mean(budget_slack)),
    }


def _pair_flow(
    *,
    family: str,
    source_id: str,
    selected: pd.DataFrame,
    chunk: pd.DataFrame,
    budget_return_mask: np.ndarray,
) -> pd.DataFrame:
    add_source = chunk[family].astype(str).to_numpy()
    drop_source = selected[family].astype(str).to_numpy()
    positions = np.argwhere(budget_return_mask)
    categories = {
        "source_relief_drop_tight_add_other": 0,
        "source_neutral_drop_tight_add_tight": 0,
        "source_pressure_drop_other_add_tight": 0,
        "other_to_other": 0,
    }
    for add_pos, drop_pos in positions:
        add_tight = add_source[int(add_pos)] == source_id
        drop_tight = drop_source[int(drop_pos)] == source_id
        if drop_tight and not add_tight:
            categories["source_relief_drop_tight_add_other"] += 1
        elif drop_tight and add_tight:
            categories["source_neutral_drop_tight_add_tight"] += 1
        elif not drop_tight and add_tight:
            categories["source_pressure_drop_other_add_tight"] += 1
        else:
            categories["other_to_other"] += 1
    total = max(int(budget_return_mask.sum()), 1)
    return pd.DataFrame(
        [
            {
                f"source_family_v{VERSION}": family,
                f"source_id_v{VERSION}": source_id,
                f"flow_category_v{VERSION}": category,
                f"budget_return_rows_v{VERSION}": int(count),
                f"share_of_budget_return_rows_v{VERSION}": count / total,
                f"claim_boundary_v{VERSION}": (
                    "budget+return source-flow diagnostic only; no full-v55 proof"
                ),
            }
            for category, count in categories.items()
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v371 diagnoses source-governance blockers for v366 chunk 0001.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v371_source_governance_blocker_diagnostic.csv"
                ),
                "boundary": "Diagnostic only; no new entering column or full-v55 proof.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v371 identifies grade=A as the primary chunk-0001 source bottleneck.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v371_tight_source_blockers.csv"
                ),
                "boundary": "Chunk-0001 one-swap budget+return diagnostic only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v371 proves full-v55 reduced-cost termination.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v371_claim_blockers.csv"
                ),
                "boundary": "v371 explains a blocker; it does not price remaining chunks.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v371 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v371_claim_blockers.csv"
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
                    "v371 diagnoses why the v366 chunk-0001 budget+return rows collapsed "
                    "to zero source-exact rows."
                ),
                "status": "source_governance_primary_blocker_identified",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v372 tests a grade-A/source-relief prefilter before any new full chunk"
                ),
                "last_wave": "v371",
                "execution_result": "grade_a_primary_blocker_score_decile_secondary",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v371")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V371_SOURCE_GOVERNANCE_BLOCKER_DIAGNOSTIC_START -->"
    end = "<!-- V371_SOURCE_GOVERNANCE_BLOCKER_DIAGNOSTIC_END -->"
    block = f"""
{start}

## Wave v371: Source-Governance Blocker Diagnostic

Generated: {status["generated_at_utc"]}

### Objective

v366 screened the first full-v55 chunk and found zero source-exact rows after
25,223 budget+return-feasible one-swaps. v371 diagnoses the source-governance
bottleneck before spending more compute on chunk continuation.

### Results

- Budget+return feasible rows:
  `{status["budget_return_feasible_rows_v371"]}`.
- Source-exact rows:
  `{status["source_exact_rows_v371"]}`.
- Primary blocker family:
  `{status["primary_blocker_family_v371"]}`.
- Primary blocker source id:
  `{status["primary_blocker_source_id_v371"]}`.
- Primary blocker pass rows:
  `{status["primary_blocker_pass_rows_v371"]}`.
- Secondary blocker family:
  `{status["secondary_blocker_family_v371"]}`.
- Secondary blocker pass rows:
  `{status["secondary_blocker_pass_rows_v371"]}`.
- Fully nonbinding families:
  `{status["fully_nonbinding_family_count_v371"]}`.
- Recommended next artifact:
  `{status["next_artifact_v371"]}`.

### Interpretation

The first chunk did not fail at CVaR; it failed before CVaR because source
governance collapsed the budget+return set. Grade is the binding source family,
with grade=A identified as the tight source. Score decile remains secondary.
That makes a targeted grade-A/source-relief prefilter more informative than
blindly running chunk 002.

### Claim Impact

- Allowed: chunk-0001 source-governance diagnostic.
- Still prohibited: full-v55 termination, valid global integer optimality,
  working champion, Paper Estrella replacement and final promotion.

### Quarto Promotion Decision

Keep v371 in the living notebook. v372 should test a grade-A/source-relief
prefilter or formally justify returning to the original v370 order.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v366_status = json.loads((STATUS_DIR / "paper4_v366_status.json").read_text(encoding="utf-8"))
    schedule = read_csv("paper4_v365_pricing_chunk_schedule.csv")
    stage = read_csv("paper4_v366_chunk_stage_summary.csv")
    source_summary = read_csv("paper4_v353_v347_expanded_branch_price_source_summary.csv")
    hotspots = read_csv("paper4_v356_v353_source_slack_hotspots.csv")
    tight_map = read_csv("paper4_v356_v353_source_tight_candidate_map.csv")
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet(
        "paper4_v353_v347_expanded_branch_price_allocations.parquet"
    ).reset_index(drop=True)
    if any(
        df.empty
        for df in [schedule, stage, source_summary, hotspots, tight_map, universe, selected]
    ):
        raise RuntimeError("Missing v371 source-governance diagnostic inputs.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    for frame in [universe, selected]:
        for family in FAMILIES:
            frame[family] = frame[family].astype(str)
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
    for family in FAMILIES:
        chunk[family] = chunk[family].astype(str)

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
    budget_return_mask = (
        return_mask
        & (exposure_after >= EXPOSURE_MIN - 1e-7)
        & (exposure_after <= EXPOSURE_MAX + 1e-7)
    )

    stage_map = dict(zip(stage["stage_v366"], stage["row_count_v366"], strict=False))
    budget_return_rows = int(stage_map["budget_return_feasible"])
    source_exact_rows = int(stage_map["source_exact_feasible"])
    if int(budget_return_mask.sum()) != budget_return_rows:
        raise RuntimeError("v371 reconstructed budget+return rows do not match v366.")

    caps = _source_caps(source_summary)
    current_by_source: dict[str, float] = {}
    for family in FAMILIES:
        exposure_by_source = selected.groupby(family, dropna=False)["loan_amnt"].sum()
        for source_id, exposure in exposure_by_source.items():
            current_by_source[f"{family}={source_id}"] = float(exposure)

    family_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_mask = _family_mask(
            family=family,
            selected=selected,
            chunk=chunk,
            budget_return_mask=budget_return_mask,
            exposure_after=exposure_after,
            caps=caps,
            current_exposure_by_source=current_by_source,
        )
        stage_count = int(stage_map[f"{family}_source_feasible_alone"])
        if int(family_mask.sum()) != stage_count:
            raise RuntimeError(f"v371 family mask mismatch for {family}.")
        family_rows.append(
            {
                f"source_family_v{VERSION}": family,
                f"budget_return_feasible_rows_v{VERSION}": budget_return_rows,
                f"family_source_feasible_rows_v{VERSION}": stage_count,
                f"family_retention_share_v{VERSION}": stage_count / max(budget_return_rows, 1),
                f"binding_rank_v{VERSION}": 0,
                f"blocker_class_v{VERSION}": "nonbinding"
                if stage_count == budget_return_rows
                else "binding",
                f"claim_boundary_v{VERSION}": "chunk-0001 family-level source diagnostic only",
            }
        )
    family_retention = pd.DataFrame(family_rows).sort_values(
        [f"family_source_feasible_rows_v{VERSION}", f"source_family_v{VERSION}"]
    )
    family_retention[f"binding_rank_v{VERSION}"] = np.arange(1, len(family_retention) + 1)

    tight_rows: list[dict[str, Any]] = []
    for _, tight in tight_map.iterrows():
        family = str(tight["source_family"])
        source_id = str(tight["source_id"])
        cap = float(caps[family][source_id])
        current_source_exposure = current_by_source.get(f"{family}={source_id}", 0.0)
        slack_stats = _tight_source_slack(
            family=family,
            source_id=source_id,
            selected=selected,
            chunk=chunk,
            budget_return_mask=budget_return_mask,
            exposure_after=exposure_after,
            cap=cap,
            current_exposure=current_source_exposure,
        )
        family_stage_count = int(stage_map[f"{family}_source_feasible_alone"])
        hotspot = hotspots.loc[
            hotspots["source_family"].astype(str).eq(family)
            & hotspots["source_id"].astype(str).eq(source_id)
        ].iloc[0]
        tight_rows.append(
            {
                f"pricing_block_id_v{VERSION}": str(tight["pricing_block_id_v356"]),
                f"source_family_v{VERSION}": family,
                f"source_id_v{VERSION}": source_id,
                f"source_slack_v{VERSION}": float(hotspot["source_slack_v356"]),
                f"source_slack_rank_v{VERSION}": int(hotspot["source_slack_rank_v356"]),
                f"v356_candidate_rows_v{VERSION}": int(tight["candidate_rows_v356"]),
                f"v356_positive_return_candidate_rows_v{VERSION}": int(
                    tight["positive_return_candidate_rows_v356"]
                ),
                f"v366_family_feasible_rows_v{VERSION}": family_stage_count,
                f"tight_source_pass_rows_v{VERSION}": int(
                    slack_stats["tight_source_pass_rows"]
                ),
                f"tight_source_best_slack_v{VERSION}": slack_stats["tight_source_best_slack"],
                f"primary_blocker_v{VERSION}": family_stage_count == 0,
                f"claim_boundary_v{VERSION}": "tight source blocker diagnostic only",
            }
        )
    tight_blockers = pd.DataFrame(tight_rows).sort_values(
        [f"v366_family_feasible_rows_v{VERSION}", f"source_slack_rank_v{VERSION}"]
    )

    pair_flow = pd.concat(
        [
            _pair_flow(
                family=str(row[f"source_family_v{VERSION}"]),
                source_id=str(row[f"source_id_v{VERSION}"]),
                selected=selected,
                chunk=chunk,
                budget_return_mask=budget_return_mask,
            )
            for _, row in tight_blockers.iterrows()
        ],
        ignore_index=True,
    )
    recommendations = pd.DataFrame(
        [
            {
                f"recommendation_id_v{VERSION}": "grade_a_source_relief_prefilter",
                f"priority_v{VERSION}": 1,
                f"recommended_v{VERSION}": True,
                f"expected_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"rationale_v{VERSION}": (
                    "grade=A has zero family-feasible rows after budget+return in chunk 0001"
                ),
                f"claim_boundary_v{VERSION}": "next experiment only",
            },
            {
                f"recommendation_id_v{VERSION}": "score_decile_secondary_gate",
                f"priority_v{VERSION}": 2,
                f"recommended_v{VERSION}": True,
                f"expected_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"rationale_v{VERSION}": (
                    "score_decile=0 remains tight but passes 6023 budget+return rows alone"
                ),
                f"claim_boundary_v{VERSION}": "secondary diagnostic only",
            },
            {
                f"recommendation_id_v{VERSION}": "blind_chunk_002",
                f"priority_v{VERSION}": 3,
                f"recommended_v{VERSION}": False,
                f"expected_artifact_v{VERSION}": "paper4_v373_full_v55_chunk_002_or_stop_rule.csv",
                f"rationale_v{VERSION}": (
                    "blind chunking is less informative until the grade=A blocker is addressed"
                ),
                f"claim_boundary_v{VERSION}": "defer until blocker prefilter is tested",
            },
        ]
    )
    diagnostic = pd.DataFrame(
        [
            {
                f"diagnostic_id_v{VERSION}": "v371_v366_chunk_0001_source_governance_blocker",
                f"prior_backlog_version_v{VERSION}": PRIOR_BACKLOG_VERSION,
                f"prior_chunk_version_v{VERSION}": PRIOR_CHUNK_VERSION,
                f"ordered_one_swap_rows_v{VERSION}": int(v366_status["ordered_one_swap_rows_v366"]),
                f"return_improving_rows_v{VERSION}": int(v366_status["return_improving_rows_v366"]),
                f"budget_return_feasible_rows_v{VERSION}": budget_return_rows,
                f"source_exact_rows_v{VERSION}": source_exact_rows,
                f"primary_blocker_family_v{VERSION}": str(
                    tight_blockers.iloc[0][f"source_family_v{VERSION}"]
                ),
                f"primary_blocker_source_id_v{VERSION}": str(
                    tight_blockers.iloc[0][f"source_id_v{VERSION}"]
                ),
                f"primary_blocker_pass_rows_v{VERSION}": int(
                    tight_blockers.iloc[0][f"v366_family_feasible_rows_v{VERSION}"]
                ),
                f"secondary_blocker_family_v{VERSION}": str(
                    tight_blockers.iloc[1][f"source_family_v{VERSION}"]
                ),
                f"secondary_blocker_pass_rows_v{VERSION}": int(
                    tight_blockers.iloc[1][f"v366_family_feasible_rows_v{VERSION}"]
                ),
                f"fully_nonbinding_family_count_v{VERSION}": int(
                    family_retention[f"family_source_feasible_rows_v{VERSION}"]
                    .eq(budget_return_rows)
                    .sum()
                ),
                f"valid_full_v55_dual_bound_certificate_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "source diagnostic only; no chunk continuation proof",
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "grade_a_primary_source_blocker",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    diagnostic.iloc[0][f"primary_blocker_pass_rows_v{VERSION}"]
                ),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "grade=A prefilter needed before blind chunk continuation",
            },
            {
                f"blocker_id_v{VERSION}": "remaining_chunks_unpriced",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 27,
                f"required_next_artifact_v{VERSION}": "paper4_v373_full_v55_chunk_002_or_stop_rule.csv",
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
                "claim_id": "v371_source_governance_blocker_diagnostic_created",
                "allowed": True,
                "artifact": "paper4_v371_source_governance_blocker_diagnostic.csv",
                "boundary": "diagnostic only",
            },
            {
                "claim_id": "v371_grade_a_primary_blocker_identified",
                "allowed": True,
                "artifact": "paper4_v371_tight_source_blockers.csv",
                "boundary": "chunk-0001 source blocker only",
            },
            {
                "claim_id": "v371_valid_full_v55_dual_bound_certificate",
                "allowed": False,
                "artifact": "paper4_v371_claim_blockers.csv",
                "boundary": "remaining chunks unpriced",
            },
            {
                "claim_id": "v371_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v371_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v371_source_governance_blocker_diagnostic.csv", diagnostic)
    write_csv(TABLE_DIR / "paper4_v371_source_family_retention.csv", family_retention)
    write_csv(TABLE_DIR / "paper4_v371_tight_source_blockers.csv", tight_blockers)
    write_csv(TABLE_DIR / "paper4_v371_source_pair_flow_diagnostics.csv", pair_flow)
    write_csv(TABLE_DIR / "paper4_v371_next_experiment_recommendations.csv", recommendations)
    write_csv(TABLE_DIR / "paper4_v371_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v371_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = diagnostic.iloc[0]
    status = {
        "phase": "v371_source_governance_blocker_diagnostic",
        "schema_version": "2026-05-17.371",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_backlog_version_v371": PRIOR_BACKLOG_VERSION,
        "prior_chunk_version_v371": PRIOR_CHUNK_VERSION,
        "ordered_one_swap_rows_v371": int(row[f"ordered_one_swap_rows_v{VERSION}"]),
        "return_improving_rows_v371": int(row[f"return_improving_rows_v{VERSION}"]),
        "budget_return_feasible_rows_v371": int(
            row[f"budget_return_feasible_rows_v{VERSION}"]
        ),
        "source_exact_rows_v371": int(row[f"source_exact_rows_v{VERSION}"]),
        "primary_blocker_family_v371": str(row[f"primary_blocker_family_v{VERSION}"]),
        "primary_blocker_source_id_v371": str(row[f"primary_blocker_source_id_v{VERSION}"]),
        "primary_blocker_pass_rows_v371": int(row[f"primary_blocker_pass_rows_v{VERSION}"]),
        "secondary_blocker_family_v371": str(row[f"secondary_blocker_family_v{VERSION}"]),
        "secondary_blocker_pass_rows_v371": int(
            row[f"secondary_blocker_pass_rows_v{VERSION}"]
        ),
        "fully_nonbinding_family_count_v371": int(
            row[f"fully_nonbinding_family_count_v{VERSION}"]
        ),
        "family_retention_rows_v371": int(len(family_retention)),
        "tight_source_blocker_rows_v371": int(len(tight_blockers)),
        "pair_flow_rows_v371": int(len(pair_flow)),
        "recommendation_rows_v371": int(len(recommendations)),
        "claim_blocker_rows_v371": int(len(blockers)),
        "claim_matrix_rows_v371": int(len(claim_matrix)),
        "valid_full_v55_dual_bound_certificate_v371": False,
        "full_universe_integer_optimality_claim_allowed_v371": False,
        "working_champion_claim_allowed_v371": False,
        "paper1_promotion_allowed_v371": False,
        "paper4_working_champion_changed_v371": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v371": NEXT_ARTIFACT,
        "claim_boundary": (
            "v371 diagnoses chunk-0001 source blockers; solver, champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v371_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v371": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
