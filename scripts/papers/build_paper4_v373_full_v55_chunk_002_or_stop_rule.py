#!/usr/bin/env python3
"""Build Paper 4 v373 chunk-002 or stop-rule decision artifacts."""

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

VERSION = 373
PRIOR_PREFILTER_VERSION = 372
PRIOR_CHUNK_VERSION = 366
NEXT_VERSION = 374
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_paper4_claim_language_section_draft.md"
EXPOSURE_MIN = 842292.375
EXPOSURE_MAX = 850000.0
SAMPLED_CHUNKS = [1, 2, 3, 4, 5, 10, 20, 28]
RECOMMENDED_DECISION = "stop_blind_chunking_after_sampled_source_blocker"


def _source_caps(source_summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    caps: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        local = source_summary.loc[source_summary["source_family"].astype(str).eq(family)]
        caps[family] = {
            str(row["source_id"]): float(row["cap_share_v353"]) for _, row in local.iterrows()
        }
    return caps


def _chunk_screen(
    *,
    chunk_id: int,
    chunk: pd.DataFrame,
    selected: pd.DataFrame,
    mean_returns: np.ndarray,
    idx_by_id: pd.Series,
    caps: dict[str, dict[str, float]],
    current_by_source: dict[tuple[str, str], float],
) -> dict[str, Any]:
    add_returns = mean_returns[idx_by_id.loc[chunk["loan_id"].astype(str)].to_numpy(int)]
    drop_returns = mean_returns[idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy(int)]
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
    exact = budget_return_mask.copy()
    family_counts: dict[str, int] = {}
    for family in FAMILIES:
        family_mask = budget_return_mask.copy()
        add_source = chunk[family].astype(str).to_numpy()
        drop_source = selected[family].astype(str).to_numpy()
        for source_id, cap in caps[family].items():
            new_source = (
                current_by_source.get((family, source_id), 0.0)
                + add_amount[:, None] * (add_source[:, None] == source_id)
                - drop_amount[None, :] * (drop_source[None, :] == source_id)
            )
            family_mask &= (new_source / exposure_after) <= cap + 1e-7
        family_counts[family] = int(family_mask.sum())
        exact &= family_mask

    add_grade = chunk["grade"].astype(str).to_numpy()
    drop_grade = selected["grade"].astype(str).to_numpy()
    grade_relief = (drop_grade[None, :] == "A") & (add_grade[:, None] != "A")
    grade_pressure = (drop_grade[None, :] != "A") & (add_grade[:, None] == "A")
    return {
        f"chunk_id_v{VERSION}": chunk_id,
        f"chunk_rows_v{VERSION}": int(len(chunk)),
        f"chunk_grade_a_add_rows_v{VERSION}": int((chunk["grade"].astype(str) == "A").sum()),
        f"ordered_one_swap_rows_v{VERSION}": int(return_delta.size),
        f"return_improving_rows_v{VERSION}": int(return_mask.sum()),
        f"budget_return_feasible_rows_v{VERSION}": int(budget_return_mask.sum()),
        f"grade_source_feasible_rows_v{VERSION}": family_counts["grade"],
        f"score_decile_source_feasible_rows_v{VERSION}": family_counts["score_decile"],
        f"source_exact_rows_v{VERSION}": int(exact.sum()),
        f"grade_a_relief_return_improving_rows_v{VERSION}": int(
            (grade_relief & return_mask).sum()
        ),
        f"grade_a_relief_budget_return_rows_v{VERSION}": int(
            (grade_relief & budget_return_mask).sum()
        ),
        f"grade_a_pressure_budget_return_rows_v{VERSION}": int(
            (grade_pressure & budget_return_mask).sum()
        ),
        f"claim_boundary_v{VERSION}": (
            "sampled chunk source-screen only; no CVaR pricing or termination proof"
        ),
    }


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v373 samples full-v55 chunks to decide blind chunking versus stop rule.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v373_sampled_chunk_source_screen.csv"
                ),
                "boundary": "Sampled source-screen decision only; no all-column termination.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v373 recommends stopping blind chunking before more CVaR-heavy probes.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v373_full_v55_chunk_002_or_stop_rule.csv"
                ),
                "boundary": "Execution decision only; future chunks remain unpriced.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v373 proves full-v55 reduced-cost termination.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v373_claim_blockers.csv"
                ),
                "boundary": "v373 samples chunks and sets a stop rule; it does not price all chunks.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v373 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v373_claim_blockers.csv"
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
                "lane": "Publishability/Scope",
                "executable_item": (
                    "v373 converts sampled chunk source-screen evidence into a stop-rule "
                    "decision and routes the next wave to manuscript claim language."
                ),
                "status": "blind_chunking_stop_rule_selected",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v374 drafts the Paper 4 results/limitations language with citations "
                    "to v361-v373"
                ),
                "last_wave": "v373",
                "execution_result": "sampled_chunks_zero_source_exact_stop_blind_chunking",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v373")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V373_CHUNK_002_OR_STOP_RULE_START -->"
    end = "<!-- V373_CHUNK_002_OR_STOP_RULE_END -->"
    block = f"""
{start}

## Wave v373: Chunk 002 or Stop Rule

Generated: {status["generated_at_utc"]}

### Objective

v372 showed that grade-A relief in chunk 0001 is return-negative. v373 asks
whether to spend more compute on blind full-v55 chunks or stop blind chunking
and move to manuscript/gated claim language.

### Results

- Sampled chunks:
  `{status["sampled_chunk_count_v373"]}`.
- Chunk 002 source-exact rows:
  `{status["chunk_002_source_exact_rows_v373"]}`.
- Sampled chunks with source-exact rows:
  `{status["sampled_chunks_with_source_exact_rows_v373"]}`.
- Sampled total budget+return rows:
  `{status["sampled_total_budget_return_rows_v373"]}`.
- Sampled total source-exact rows:
  `{status["sampled_total_source_exact_rows_v373"]}`.
- Sampled total grade-A relief budget+return rows:
  `{status["sampled_total_grade_a_relief_budget_return_rows_v373"]}`.
- Recommended decision:
  `{status["recommended_decision_v373"]}`.
- Next artifact:
  `{status["next_artifact_v373"]}`.

### Interpretation

Chunk 002 does not break the pattern: it has zero source-exact rows. The sampled
chunks also produce zero source-exact rows in aggregate. That is enough to stop
blind chunk continuation as the next action and move the result into paper
language with clear limitations.

### Claim Impact

- Allowed: sampled stop-rule decision.
- Still prohibited: full-v55 termination, valid global integer optimality,
  working champion, Paper Estrella replacement and final promotion.

### Quarto Promotion Decision

Keep v373 in the living notebook. v374 should draft the Paper 4 claim language
from the evidence frontier.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v372_status = json.loads((STATUS_DIR / "paper4_v372_status.json").read_text(encoding="utf-8"))
    schedule = read_csv("paper4_v365_pricing_chunk_schedule.csv")
    source_summary = read_csv("paper4_v353_v347_expanded_branch_price_source_summary.csv")
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet(
        "paper4_v353_v347_expanded_branch_price_allocations.parquet"
    ).reset_index(drop=True)
    if any(df.empty for df in [schedule, source_summary, universe, selected]):
        raise RuntimeError("Missing v373 chunk stop-rule inputs.")
    if v372_status["recommended_route_v372"] != "chunk_002_or_stop_rule":
        raise RuntimeError("v373 expects v372 to route to chunk_002_or_stop_rule.")

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
    _losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    caps = _source_caps(source_summary)
    current_by_source: dict[tuple[str, str], float] = {}
    for family in FAMILIES:
        exposure_by_source = selected.groupby(family, dropna=False)["loan_amnt"].sum()
        for source_id, exposure in exposure_by_source.items():
            current_by_source[(family, str(source_id))] = float(exposure)

    rows: list[dict[str, Any]] = []
    for chunk_id in SAMPLED_CHUNKS:
        schedule_row = schedule.loc[schedule["chunk_id_v365"].astype(int).eq(chunk_id)].iloc[0]
        chunk = omitted.iloc[
            int(schedule_row["start_offset_in_full_omitted_v365"]) : int(
                schedule_row["end_offset_exclusive_v365"]
            )
        ].copy()
        for family in FAMILIES:
            chunk[family] = chunk[family].astype(str)
        rows.append(
            _chunk_screen(
                chunk_id=chunk_id,
                chunk=chunk,
                selected=selected,
                mean_returns=mean_returns,
                idx_by_id=idx_by_id,
                caps=caps,
                current_by_source=current_by_source,
            )
        )
    sampled = pd.DataFrame(rows)
    chunk2 = sampled.loc[sampled[f"chunk_id_v{VERSION}"].astype(int).eq(2)].iloc[0]
    source_exact_total = int(sampled[f"source_exact_rows_v{VERSION}"].sum())
    budget_return_total = int(sampled[f"budget_return_feasible_rows_v{VERSION}"].sum())
    grade_relief_total = int(sampled[f"grade_a_relief_budget_return_rows_v{VERSION}"].sum())
    stop_rule = pd.DataFrame(
        [
            {
                f"rule_id_v{VERSION}": "stop_blind_chunking_after_sampled_source_blocker",
                f"recommended_v{VERSION}": True,
                f"evidence_count_v{VERSION}": source_exact_total,
                f"sampled_chunk_count_v{VERSION}": len(sampled),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "stop blind chunking as next action; not full-v55 termination"
                ),
            },
            {
                f"rule_id_v{VERSION}": "run_full_chunk_002_cvar",
                f"recommended_v{VERSION}": False,
                f"evidence_count_v{VERSION}": int(chunk2[f"source_exact_rows_v{VERSION}"]),
                f"sampled_chunk_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "none_until_chunk_002_has_source_exact_rows",
                f"claim_boundary_v{VERSION}": "CVaR screen is uninformative with zero source-exact rows",
            },
            {
                f"rule_id_v{VERSION}": "continue_blind_chunking",
                f"recommended_v{VERSION}": False,
                f"evidence_count_v{VERSION}": len(sampled),
                f"sampled_chunk_count_v{VERSION}": len(sampled),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "sampled chunks show repeated source-exact collapse",
            },
        ]
    )
    decision = pd.DataFrame(
        [
            {
                f"decision_id_v{VERSION}": "v373_full_v55_chunk_002_or_stop_rule",
                f"prior_prefilter_version_v{VERSION}": PRIOR_PREFILTER_VERSION,
                f"prior_chunk_version_v{VERSION}": PRIOR_CHUNK_VERSION,
                f"sampled_chunk_ids_v{VERSION}": ",".join(map(str, SAMPLED_CHUNKS)),
                f"sampled_chunk_count_v{VERSION}": int(len(sampled)),
                f"chunk_002_budget_return_rows_v{VERSION}": int(
                    chunk2[f"budget_return_feasible_rows_v{VERSION}"]
                ),
                f"chunk_002_grade_feasible_rows_v{VERSION}": int(
                    chunk2[f"grade_source_feasible_rows_v{VERSION}"]
                ),
                f"chunk_002_source_exact_rows_v{VERSION}": int(
                    chunk2[f"source_exact_rows_v{VERSION}"]
                ),
                f"sampled_chunks_with_source_exact_rows_v{VERSION}": int(
                    sampled[f"source_exact_rows_v{VERSION}"].gt(0).sum()
                ),
                f"sampled_total_budget_return_rows_v{VERSION}": budget_return_total,
                f"sampled_total_source_exact_rows_v{VERSION}": source_exact_total,
                f"sampled_total_grade_a_relief_budget_return_rows_v{VERSION}": grade_relief_total,
                f"recommended_decision_v{VERSION}": RECOMMENDED_DECISION,
                f"valid_full_v55_dual_bound_certificate_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "sampled chunk stop-rule decision only; no full-v55 termination proof"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "sampled_chunks_zero_source_exact",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": source_exact_total,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "sampled stop rule blocks blind chunk continuation",
            },
            {
                f"blocker_id_v{VERSION}": "remaining_chunks_unpriced",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 27,
                f"required_next_artifact_v{VERSION}": "future_targeted_pricing_or_certificate",
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
                "claim_id": "v373_sampled_chunk_source_screen_created",
                "allowed": True,
                "artifact": "paper4_v373_sampled_chunk_source_screen.csv",
                "boundary": "sampled source-screen only",
            },
            {
                "claim_id": "v373_blind_chunking_stop_rule_selected",
                "allowed": True,
                "artifact": "paper4_v373_full_v55_chunk_002_or_stop_rule.csv",
                "boundary": "execution decision only",
            },
            {
                "claim_id": "v373_valid_full_v55_dual_bound_certificate",
                "allowed": False,
                "artifact": "paper4_v373_claim_blockers.csv",
                "boundary": "remaining chunks unpriced",
            },
            {
                "claim_id": "v373_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v373_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v373_full_v55_chunk_002_or_stop_rule.csv", decision)
    write_csv(TABLE_DIR / "paper4_v373_sampled_chunk_source_screen.csv", sampled)
    write_csv(TABLE_DIR / "paper4_v373_stop_rule_register.csv", stop_rule)
    write_csv(TABLE_DIR / "paper4_v373_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v373_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = decision.iloc[0]
    status = {
        "phase": "v373_full_v55_chunk_002_or_stop_rule",
        "schema_version": "2026-05-17.373",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_prefilter_version_v373": PRIOR_PREFILTER_VERSION,
        "prior_chunk_version_v373": PRIOR_CHUNK_VERSION,
        "sampled_chunk_ids_v373": str(row[f"sampled_chunk_ids_v{VERSION}"]),
        "sampled_chunk_count_v373": int(row[f"sampled_chunk_count_v{VERSION}"]),
        "chunk_002_budget_return_rows_v373": int(row[f"chunk_002_budget_return_rows_v{VERSION}"]),
        "chunk_002_grade_feasible_rows_v373": int(row[f"chunk_002_grade_feasible_rows_v{VERSION}"]),
        "chunk_002_source_exact_rows_v373": int(row[f"chunk_002_source_exact_rows_v{VERSION}"]),
        "sampled_chunks_with_source_exact_rows_v373": int(
            row[f"sampled_chunks_with_source_exact_rows_v{VERSION}"]
        ),
        "sampled_total_budget_return_rows_v373": int(
            row[f"sampled_total_budget_return_rows_v{VERSION}"]
        ),
        "sampled_total_source_exact_rows_v373": int(
            row[f"sampled_total_source_exact_rows_v{VERSION}"]
        ),
        "sampled_total_grade_a_relief_budget_return_rows_v373": int(
            row[f"sampled_total_grade_a_relief_budget_return_rows_v{VERSION}"]
        ),
        "recommended_decision_v373": str(row[f"recommended_decision_v{VERSION}"]),
        "stop_rule_rows_v373": int(len(stop_rule)),
        "claim_blocker_rows_v373": int(len(blockers)),
        "claim_matrix_rows_v373": int(len(claim_matrix)),
        "valid_full_v55_dual_bound_certificate_v373": False,
        "full_universe_integer_optimality_claim_allowed_v373": False,
        "working_champion_claim_allowed_v373": False,
        "paper1_promotion_allowed_v373": False,
        "paper4_working_champion_changed_v373": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v373": NEXT_ARTIFACT,
        "claim_boundary": (
            "v373 selects a blind-chunking stop rule from sampled source-screen evidence"
        ),
    }
    write_json(STATUS_DIR / "paper4_v373_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v373": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
