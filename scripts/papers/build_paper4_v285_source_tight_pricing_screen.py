#!/usr/bin/env python3
"""Build Paper 4 v285 source-tight pricing screen artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers import build_paper4_v82_one_swap_integer_pricing_probe as v82
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

VERSION = 285
DECOMPOSITION_VERSION = 284
INCUMBENT_REPAIR_VERSION = 279
TERMINAL_REPRICE_VERSION = 280
NEXT_VERSION = 286
TIGHT_BLOCKS = [("grade", "A"), ("score_decile", "0")]


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v285 source-tight pricing screen.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v285_source_tight_pricing_screen.csv"
                ),
                "boundary": (
                    "Screen is limited to positive-return candidates in v284 tight source "
                    "blocks and one-drop/one-add pricing."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v285 shows no one-drop/one-add entering column inside positive "
                    "source-tight blocks."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v285_source_tight_pair_stage_summary.csv"
                ),
                "boundary": (
                    "Scope-limited source-tight screen only; does not cover multi-drop, "
                    "multi-add or branch-price global termination."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v285 proves a valid full-universe branch-price bound.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v285_claim_blockers.csv"
                ),
                "boundary": "No dual-bound loop, no multi-column pricing, no termination certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v285 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v285_claim_blockers.csv"
                ),
                "boundary": "No final promotion, dynamic validation or deployment gate is created.",
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
                    "v285 screens positive-return candidates in the v284 source-tight "
                    "blocks and localizes the one-drop/one-add blocker to joint grade A "
                    "and score decile 0 caps."
                ),
                "status": "source_tight_pricing_screen_blocked_by_joint_caps",
                "next_artifact": "paper4_v286_joint_source_relief_pricing_protocol.csv",
                "success_condition": (
                    "multi-source relief protocol determines whether two-drop/multi-add "
                    "columns can free the tight caps without breaking CVaR"
                ),
                "last_wave": "v285",
                "execution_result": "no_source_prefilter_pairs_survive_joint_tight_caps",
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


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V285_SOURCE_TIGHT_PRICING_SCREEN_START -->"
    end = "<!-- V285_SOURCE_TIGHT_PRICING_SCREEN_END -->"
    block = f"""
{start}

## Wave v285: Source-Tight Pricing Screen

Generated: {status["generated_at_utc"]}

### Objective

Convert the v284 decomposition prototype into an executable source-tight
pricing screen. The screen focuses on positive-return omitted candidates inside
the tight grade A and score decile 0 blocks, then tests whether any one-drop /
one-add pricing column survives return, budget and source-cap filters.

### Results

- Tight candidate rows: `{status["tight_candidate_rows_v285"]}`.
- Positive-return tight candidate rows: `{status["positive_tight_candidate_rows_v285"]}`.
- Positive candidates in both grade A and score decile 0:
  `{status["positive_joint_grade_a_score0_candidate_rows_v285"]}`.
- Return-improving pair rows: `{status["return_improving_pair_rows_v285"]}`.
- Budget+return feasible pair rows: `{status["budget_return_feasible_pair_rows_v285"]}`.
- Grade-only source feasible pair rows:
  `{status["grade_source_feasible_pair_rows_v285"]}`.
- Score-decile-only source feasible pair rows:
  `{status["score_decile_source_feasible_pair_rows_v285"]}`.
- All-source prefilter pair rows: `{status["source_prefilter_pair_rows_v285"]}`.
- One-drop/one-add entering columns found:
  `{status["one_drop_one_add_entering_columns_found_v285"]}`.
- Valid branch-price bound produced:
  `{status["valid_branch_price_bound_v285"]}`.

### Interpretation

v285 is valuable because it narrows the v284 decomposition route: many
positive-return columns exist, but the local one-drop/one-add screen collapses
at the joint tight source caps. Grade A alone leaves a tiny set of possible
pairs, score decile 0 alone leaves more, and the intersection leaves none.
That points the next experiment toward explicit multi-source relief rather
than another broad one-swap scan.

### Claim Impact

- Allowed: source-tight pricing screen completed; no one-drop/one-add entering
  column found inside the positive source-tight screen.
- Still prohibited: valid full-universe branch-price bound, global integer
  optimality, Paper Estrella replacement, final Paper 4 promotion and live
  deployment.

### Quarto Promotion Decision

Keep v285 in the living notebook. Promote only after multi-source relief,
global-bound and dynamic-validation gates pass.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def _family_source_mask(
    *,
    base_mask: np.ndarray,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    source_summary: pd.DataFrame,
    current_exposure: float,
    family: str,
) -> np.ndarray:
    add_amount = candidates["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    new_total = current_exposure + add_amount[:, None] - drop_amount[None, :]
    current_by_source = selected.groupby(family, dropna=False)["loan_amnt"].sum()
    cap_by_source = (
        source_summary.loc[source_summary["source_family"].astype(str).eq(family)]
        .set_index("source_id")["cap_share_v80"]
        .astype(float)
    )
    add_source = candidates[family].astype(str).to_numpy()
    drop_source = selected[family].astype(str).to_numpy()
    current_add = np.array([current_by_source.get(x, 0.0) for x in add_source], dtype=float)
    current_drop = np.array([current_by_source.get(x, 0.0) for x in drop_source], dtype=float)
    cap_add = np.array([cap_by_source.get(x, 1.0) for x in add_source], dtype=float)
    cap_drop = np.array([cap_by_source.get(x, 1.0) for x in drop_source], dtype=float)
    same_source = add_source[:, None] == drop_source[None, :]
    add_ok = (
        current_add[:, None] + add_amount[:, None] - drop_amount[None, :] * same_source
    ) <= cap_add[:, None] * new_total + 1e-7
    drop_ok = (
        current_drop[None, :] - drop_amount[None, :] + add_amount[:, None] * same_source
    ) <= cap_drop[None, :] * new_total + 1e-7
    return base_mask & add_ok & drop_ok


def _pricing_block_memberships(candidates: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    block_ids: list[str] = []
    block_counts: list[int] = []
    for _, row in candidates.iterrows():
        memberships = [
            f"{family}={source_id}"
            for family, source_id in TIGHT_BLOCKS
            if str(row[family]) == source_id
        ]
        block_ids.append("|".join(memberships))
        block_counts.append(len(memberships))
    return pd.Series(block_ids, index=candidates.index), pd.Series(
        block_counts, index=candidates.index
    )


def _candidate_diagnostics(
    *,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    return_mask: np.ndarray,
    budget_return_mask: np.ndarray,
    family_masks: dict[str, np.ndarray],
    source_prefilter_mask: np.ndarray,
    return_delta: np.ndarray,
) -> pd.DataFrame:
    block_ids, block_counts = _pricing_block_memberships(candidates)
    budget_return_counts = budget_return_mask.sum(axis=1).astype(int)
    best_drop_ids: list[str] = []
    best_return_deltas: list[float] = []
    best_drop_returns: list[float] = []
    selected_ids = selected["loan_id"].astype(str).to_numpy()
    selected_returns = selected[f"mean_return_if_dropped_v{VERSION}"].to_numpy(float)
    for row_idx in range(len(candidates)):
        feasible = np.flatnonzero(budget_return_mask[row_idx])
        if len(feasible) == 0:
            best_drop_ids.append("")
            best_return_deltas.append(float("nan"))
            best_drop_returns.append(float("nan"))
            continue
        best_pos = feasible[np.argmax(return_delta[row_idx, feasible])]
        best_drop_ids.append(str(selected_ids[best_pos]))
        best_return_deltas.append(float(return_delta[row_idx, best_pos]))
        best_drop_returns.append(float(selected_returns[best_pos]))

    diagnostics = candidates[
        ["loan_id", "loan_amnt", "grade", "score_decile", "income_band", "dti_band", "period"]
    ].copy()
    diagnostics[f"mean_return_v{VERSION}"] = candidates[f"mean_return_v{VERSION}"].to_numpy(float)
    diagnostics[f"pricing_block_ids_v{VERSION}"] = block_ids.to_numpy()
    diagnostics[f"tight_block_count_v{VERSION}"] = block_counts.to_numpy(dtype=int)
    diagnostics[f"return_improving_drop_rows_v{VERSION}"] = return_mask.sum(axis=1).astype(int)
    diagnostics[f"budget_return_drop_rows_v{VERSION}"] = budget_return_counts
    diagnostics[f"grade_source_feasible_drop_rows_v{VERSION}"] = (
        family_masks["grade"].sum(axis=1).astype(int)
    )
    diagnostics[f"score_decile_source_feasible_drop_rows_v{VERSION}"] = (
        family_masks["score_decile"].sum(axis=1).astype(int)
    )
    joint_tight = budget_return_mask & family_masks["grade"] & family_masks["score_decile"]
    diagnostics[f"joint_tight_source_feasible_drop_rows_v{VERSION}"] = joint_tight.sum(
        axis=1
    ).astype(int)
    diagnostics[f"all_source_prefilter_drop_rows_v{VERSION}"] = source_prefilter_mask.sum(
        axis=1
    ).astype(int)
    diagnostics[f"best_budget_return_drop_loan_id_v{VERSION}"] = best_drop_ids
    diagnostics[f"best_budget_return_delta_v{VERSION}"] = best_return_deltas
    diagnostics[f"best_budget_return_drop_mean_return_v{VERSION}"] = best_drop_returns
    diagnostics[f"candidate_screen_role_v{VERSION}"] = "positive_return_source_tight_candidate"
    diagnostics[f"claim_boundary_v{VERSION}"] = (
        "candidate-level source-tight pricing diagnostic; not an entering-column certificate"
    )
    return diagnostics.sort_values(f"mean_return_v{VERSION}", ascending=False).reset_index(
        drop=True
    )


def _exact_and_cvar_pairs(
    *,
    source_prefilter_mask: np.ndarray,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    universe: pd.DataFrame,
    losses: np.ndarray,
    selected_idx: np.ndarray,
    candidate_idx: np.ndarray,
    current_exposure: float,
    current_objective_return: float,
    cvar_cap: float,
    source_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "policy_id",
        f"regime_v{VERSION}",
        f"added_loan_id_v{VERSION}",
        f"dropped_loan_id_v{VERSION}",
        f"added_mean_return_v{VERSION}",
        f"dropped_mean_return_v{VERSION}",
        f"return_delta_v{VERSION}",
        f"objective_return_after_swap_v{VERSION}",
        f"exposure_after_swap_v{VERSION}",
        f"source_min_slack_after_swap_v{VERSION}",
        f"max_source_share_after_swap_v{VERSION}",
        f"source_cap_violations_after_swap_v{VERSION}",
        f"first_source_block_family_v{VERSION}",
        f"first_source_block_id_v{VERSION}",
        f"cvar90_after_swap_v{VERSION}",
        f"cvar_swap_feasible_v{VERSION}",
        f"one_drop_one_add_entering_column_v{VERSION}",
        f"candidate_universe_rows_v{VERSION}",
        f"claim_boundary_v{VERSION}",
    ]
    current_by_family, cap_by_family = v82._source_maps(selected, source_summary)
    current_losses = losses[:, selected_idx].sum(axis=1)
    rows: list[dict[str, Any]] = []
    for candidate_pos, selected_pos in np.argwhere(source_prefilter_mask):
        add_row = candidates.iloc[int(candidate_pos)]
        drop_row = selected.iloc[int(selected_pos)]
        new_total = float(current_exposure + add_row["loan_amnt"] - drop_row["loan_amnt"])
        source_ok, min_slack, max_share, violations, first_family, first_source = (
            v82._exact_source_metrics(
                add_row,
                drop_row,
                current_by_family,
                cap_by_family,
                new_total,
            )
        )
        if not source_ok:
            continue
        swapped_losses = (
            current_losses
            + losses[:, candidate_idx[int(candidate_pos)]]
            - losses[:, selected_idx[int(selected_pos)]]
        )
        cvar_after = v70._tail_cvar(swapped_losses)
        return_delta = float(
            add_row[f"mean_return_v{VERSION}"] - drop_row[f"mean_return_if_dropped_v{VERSION}"]
        )
        rows.append(
            {
                "policy_id": "v285_source_tight_pricing_screen",
                f"regime_v{VERSION}": "post_v279_source_tight_one_drop_one_add",
                f"added_loan_id_v{VERSION}": str(add_row["loan_id"]),
                f"dropped_loan_id_v{VERSION}": str(drop_row["loan_id"]),
                f"added_mean_return_v{VERSION}": float(add_row[f"mean_return_v{VERSION}"]),
                f"dropped_mean_return_v{VERSION}": float(
                    drop_row[f"mean_return_if_dropped_v{VERSION}"]
                ),
                f"return_delta_v{VERSION}": return_delta,
                f"objective_return_after_swap_v{VERSION}": current_objective_return + return_delta,
                f"exposure_after_swap_v{VERSION}": new_total,
                f"source_min_slack_after_swap_v{VERSION}": min_slack,
                f"max_source_share_after_swap_v{VERSION}": max_share,
                f"source_cap_violations_after_swap_v{VERSION}": violations,
                f"first_source_block_family_v{VERSION}": first_family,
                f"first_source_block_id_v{VERSION}": first_source,
                f"cvar90_after_swap_v{VERSION}": cvar_after,
                f"cvar_swap_feasible_v{VERSION}": cvar_after <= cvar_cap + 1e-7,
                f"one_drop_one_add_entering_column_v{VERSION}": (
                    return_delta > 1e-9 and cvar_after <= cvar_cap + 1e-7
                ),
                f"candidate_universe_rows_v{VERSION}": int(len(universe)),
                f"claim_boundary_v{VERSION}": (
                    "exact source and CVaR check for v285 source-tight pairs only"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_stage_summary(
    *,
    total_pairs: int,
    return_pairs: int,
    budget_return_pairs: int,
    family_masks: dict[str, np.ndarray],
    cumulative_counts: list[dict[str, Any]],
    source_exact_pairs: int,
    cvar_feasible_pairs: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            f"stage_v{VERSION}": "all_positive_tight_pairs",
            f"pair_rows_v{VERSION}": total_pairs,
            f"claim_boundary_v{VERSION}": "candidate/drop pair count before filters",
        },
        {
            f"stage_v{VERSION}": "return_improving",
            f"pair_rows_v{VERSION}": return_pairs,
            f"claim_boundary_v{VERSION}": "return delta only",
        },
        {
            f"stage_v{VERSION}": "budget_return_feasible",
            f"pair_rows_v{VERSION}": budget_return_pairs,
            f"claim_boundary_v{VERSION}": "budget plus return only",
        },
    ]
    rows.extend(
        {
            f"stage_v{VERSION}": f"{family}_source_feasible_alone",
            f"pair_rows_v{VERSION}": int(mask.sum()),
            f"claim_boundary_v{VERSION}": f"{family} cap applied alone after budget+return",
        }
        for family, mask in family_masks.items()
    )
    rows.extend(cumulative_counts)
    rows.extend(
        [
            {
                f"stage_v{VERSION}": "source_exact_feasible",
                f"pair_rows_v{VERSION}": source_exact_pairs,
                f"claim_boundary_v{VERSION}": "exact all-family source caps after prefilter",
            },
            {
                f"stage_v{VERSION}": "cvar_feasible_entering_column",
                f"pair_rows_v{VERSION}": cvar_feasible_pairs,
                f"claim_boundary_v{VERSION}": "source-exact plus CVaR feasible entering columns",
            },
        ]
    )
    return pd.DataFrame(rows)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v279_restricted_pool_milp_repair_allocations.parquet")
    repair_summary = read_csv("paper4_v279_restricted_pool_milp_repair_summary.csv")
    source_summary = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_summary = source_summary.loc[
        source_summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    blocks = read_csv("paper4_v284_source_tight_pricing_blocks.csv")
    if (
        universe.empty
        or selected.empty
        or repair_summary.empty
        or source_summary.empty
        or blocks.empty
    ):
        raise RuntimeError("Missing v55, v279, v80 source-cap, or v284 block inputs for v285.")

    selected = selected.reset_index(drop=True)
    repair_row = repair_summary.loc[
        repair_summary[f"portfolio_label_v{INCUMBENT_REPAIR_VERSION}"].eq(
            "restricted_pool_milp_repair_candidate"
        )
    ].iloc[0]
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    selected_ids = set(selected["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    candidates[f"mean_return_v{VERSION}"] = mean_returns[candidates.index.to_numpy()]

    tight_mask = np.zeros(len(candidates), dtype=bool)
    for _, block in blocks.iterrows():
        family = str(block["source_family"])
        source_id = str(block["source_id"])
        tight_mask |= candidates[family].astype(str).eq(source_id).to_numpy()
    tight_candidates = candidates.loc[tight_mask].copy()
    positive = tight_candidates.loc[tight_candidates[f"mean_return_v{VERSION}"].gt(0)].copy()
    positive = positive.reset_index(names=f"universe_index_v{VERSION}")

    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    candidate_idx = positive[f"universe_index_v{VERSION}"].to_numpy(int)
    selected[f"mean_return_if_dropped_v{VERSION}"] = mean_returns[selected_idx]

    current_exposure = float(repair_row[f"portfolio_exposure_v{INCUMBENT_REPAIR_VERSION}"])
    exposure_min = float(repair_row[f"exposure_min_v{INCUMBENT_REPAIR_VERSION}"])
    exposure_max = float(repair_row[f"exposure_max_v{INCUMBENT_REPAIR_VERSION}"])
    cvar_cap = float(repair_row[f"cvar_cap_v{INCUMBENT_REPAIR_VERSION}"])
    current_objective_return = float(repair_row[f"objective_return_v{INCUMBENT_REPAIR_VERSION}"])

    add_amount = positive["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    add_return = positive[f"mean_return_v{VERSION}"].to_numpy(float)
    drop_return = selected[f"mean_return_if_dropped_v{VERSION}"].to_numpy(float)
    return_delta = add_return[:, None] - drop_return[None, :]
    return_mask = return_delta > 1e-9
    exposure_after = current_exposure + add_amount[:, None] - drop_amount[None, :]
    budget_return_mask = (
        return_mask
        & (exposure_after >= exposure_min - 1e-7)
        & (exposure_after <= exposure_max + 1e-7)
    )

    family_masks = {
        family: _family_source_mask(
            base_mask=budget_return_mask,
            candidates=positive,
            selected=selected,
            source_summary=source_summary,
            current_exposure=current_exposure,
            family=family,
        )
        for family in FAMILIES
    }
    cumulative = budget_return_mask.copy()
    cumulative_counts: list[dict[str, Any]] = []
    for family in FAMILIES:
        before = int(cumulative.sum())
        cumulative &= family_masks[family]
        cumulative_counts.append(
            {
                f"stage_v{VERSION}": f"cumulative_source_prefilter_after_{family}",
                f"pair_rows_v{VERSION}": int(cumulative.sum()),
                f"blocked_pair_rows_v{VERSION}": before - int(cumulative.sum()),
                f"claim_boundary_v{VERSION}": (
                    f"cumulative source prefilter through {family}; diagnostic only"
                ),
            }
        )
    source_prefilter_mask = v82._source_prefilter_mask(
        budget_return_mask,
        positive,
        selected,
        source_summary,
        current_exposure,
    )
    pairs = _exact_and_cvar_pairs(
        source_prefilter_mask=source_prefilter_mask,
        candidates=positive,
        selected=selected,
        universe=universe,
        losses=losses,
        selected_idx=selected_idx,
        candidate_idx=candidate_idx,
        current_exposure=current_exposure,
        current_objective_return=current_objective_return,
        cvar_cap=cvar_cap,
        source_summary=source_summary,
    )
    cvar_feasible_pairs = (
        int(pairs[f"one_drop_one_add_entering_column_v{VERSION}"].sum()) if not pairs.empty else 0
    )
    total_pairs = int(return_mask.size)
    return_pairs = int(return_mask.sum())
    budget_return_pairs = int(budget_return_mask.sum())
    source_prefilter_pairs = int(source_prefilter_mask.sum())
    grade_pairs = int(family_masks["grade"].sum())
    score_pairs = int(family_masks["score_decile"].sum())
    joint_grade_score_pairs = int(
        (budget_return_mask & family_masks["grade"] & family_masks["score_decile"]).sum()
    )
    positive_joint = int(
        (positive["grade"].astype(str).eq("A") & positive["score_decile"].astype(str).eq("0")).sum()
    )

    diagnostics = _candidate_diagnostics(
        candidates=positive,
        selected=selected,
        return_mask=return_mask,
        budget_return_mask=budget_return_mask,
        family_masks=family_masks,
        source_prefilter_mask=source_prefilter_mask,
        return_delta=return_delta,
    )
    stage_summary = _build_stage_summary(
        total_pairs=total_pairs,
        return_pairs=return_pairs,
        budget_return_pairs=budget_return_pairs,
        family_masks=family_masks,
        cumulative_counts=cumulative_counts,
        source_exact_pairs=int(len(pairs)),
        cvar_feasible_pairs=cvar_feasible_pairs,
    )
    screen = pd.DataFrame(
        [
            {
                f"screen_id_v{VERSION}": "source_tight_positive_candidate_one_drop_one_add_screen",
                f"decomposition_version_v{VERSION}": DECOMPOSITION_VERSION,
                f"incumbent_repair_version_v{VERSION}": INCUMBENT_REPAIR_VERSION,
                f"terminal_reprice_version_v{VERSION}": TERMINAL_REPRICE_VERSION,
                f"tight_candidate_rows_v{VERSION}": int(len(tight_candidates)),
                f"positive_tight_candidate_rows_v{VERSION}": int(len(positive)),
                f"positive_joint_grade_a_score0_candidate_rows_v{VERSION}": positive_joint,
                f"selected_rows_v{VERSION}": int(len(selected)),
                f"total_pair_rows_screened_v{VERSION}": total_pairs,
                f"return_improving_pair_rows_v{VERSION}": return_pairs,
                f"budget_return_feasible_pair_rows_v{VERSION}": budget_return_pairs,
                f"grade_source_feasible_pair_rows_v{VERSION}": grade_pairs,
                f"score_decile_source_feasible_pair_rows_v{VERSION}": score_pairs,
                f"joint_grade_score_source_feasible_pair_rows_v{VERSION}": joint_grade_score_pairs,
                f"source_prefilter_pair_rows_v{VERSION}": source_prefilter_pairs,
                f"source_exact_pair_rows_v{VERSION}": int(len(pairs)),
                f"cvar_feasible_pair_rows_v{VERSION}": cvar_feasible_pairs,
                f"one_drop_one_add_entering_columns_found_v{VERSION}": cvar_feasible_pairs > 0,
                f"source_tight_pricing_screen_executed_v{VERSION}": True,
                f"valid_branch_price_bound_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_joint_source_relief_pricing_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "source-tight one-drop/one-add pricing screen only; no multi-source "
                    "relief, branch-price bound or global optimality proof"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "joint_tight_source_caps_block_one_drop_pricing",
                f"blocking_v{VERSION}": source_prefilter_pairs == 0,
                f"evidence_count_v{VERSION}": budget_return_pairs,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_joint_source_relief_pricing_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "budget+return pairs collapse to zero after combined source caps"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "source_exact_pricing_pairs_missing",
                f"blocking_v{VERSION}": int(len(pairs)) == 0,
                f"evidence_count_v{VERSION}": int(len(pairs)),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_joint_source_relief_pricing_protocol.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "no source-exact one-drop/one-add pair remains in the tight screen"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "branch_price_dual_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_branch_price_dual_bound_loop",
                f"claim_boundary_v{VERSION}": "no dual-bound loop or termination certificate",
            },
            {
                f"blocker_id_v{VERSION}": "global_integer_optimality_claim_blocked",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_global_gap_or_branch_price_certificate",
                f"claim_boundary_v{VERSION}": "v285 screen does not prove full-universe optimality",
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
                "claim_id": "v285_source_tight_pricing_screen_executed",
                "allowed": True,
                "artifact": "paper4_v285_source_tight_pricing_screen.csv",
                "boundary": "positive-return source-tight one-drop/one-add screen",
            },
            {
                "claim_id": "v285_no_one_drop_one_add_entering_column_in_tight_screen",
                "allowed": cvar_feasible_pairs == 0,
                "artifact": "paper4_v285_source_tight_pair_stage_summary.csv",
                "boundary": "scope-limited to v284 tight blocks and one-drop/one-add pairs",
            },
            {
                "claim_id": "v285_valid_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v285_claim_blockers.csv",
                "boundary": "dual-bound loop missing",
            },
            {
                "claim_id": "v285_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v285_claim_blockers.csv",
                "boundary": "global certificate missing",
            },
            {
                "claim_id": "v285_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v285_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v285_source_tight_pricing_screen.csv", screen)
    write_csv(TABLE_DIR / "paper4_v285_source_tight_pair_stage_summary.csv", stage_summary)
    write_csv(TABLE_DIR / "paper4_v285_source_tight_candidate_diagnostics.csv", diagnostics)
    write_csv(TABLE_DIR / "paper4_v285_source_tight_pricing_pairs.csv", pairs)
    write_csv(TABLE_DIR / "paper4_v285_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v285_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = screen.iloc[0]
    status = {
        "phase": "v285_source_tight_pricing_screen",
        "schema_version": "2026-05-15.285",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "decomposition_version_v285": DECOMPOSITION_VERSION,
        "incumbent_repair_version_v285": INCUMBENT_REPAIR_VERSION,
        "terminal_reprice_version_v285": TERMINAL_REPRICE_VERSION,
        "tight_candidate_rows_v285": int(row[f"tight_candidate_rows_v{VERSION}"]),
        "positive_tight_candidate_rows_v285": int(row[f"positive_tight_candidate_rows_v{VERSION}"]),
        "positive_joint_grade_a_score0_candidate_rows_v285": int(
            row[f"positive_joint_grade_a_score0_candidate_rows_v{VERSION}"]
        ),
        "selected_rows_v285": int(row[f"selected_rows_v{VERSION}"]),
        "total_pair_rows_screened_v285": int(row[f"total_pair_rows_screened_v{VERSION}"]),
        "return_improving_pair_rows_v285": int(row[f"return_improving_pair_rows_v{VERSION}"]),
        "budget_return_feasible_pair_rows_v285": int(
            row[f"budget_return_feasible_pair_rows_v{VERSION}"]
        ),
        "grade_source_feasible_pair_rows_v285": int(
            row[f"grade_source_feasible_pair_rows_v{VERSION}"]
        ),
        "score_decile_source_feasible_pair_rows_v285": int(
            row[f"score_decile_source_feasible_pair_rows_v{VERSION}"]
        ),
        "joint_grade_score_source_feasible_pair_rows_v285": int(
            row[f"joint_grade_score_source_feasible_pair_rows_v{VERSION}"]
        ),
        "source_prefilter_pair_rows_v285": int(row[f"source_prefilter_pair_rows_v{VERSION}"]),
        "source_exact_pair_rows_v285": int(row[f"source_exact_pair_rows_v{VERSION}"]),
        "cvar_feasible_pair_rows_v285": int(row[f"cvar_feasible_pair_rows_v{VERSION}"]),
        "one_drop_one_add_entering_columns_found_v285": bool(
            row[f"one_drop_one_add_entering_columns_found_v{VERSION}"]
        ),
        "source_tight_pricing_screen_executed_v285": True,
        "valid_branch_price_bound_v285": False,
        "full_universe_integer_optimality_claim_allowed_v285": False,
        "paper1_promotion_allowed_v285": False,
        "paper4_working_champion_changed_v285": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "candidate_diagnostic_rows_v285": int(len(diagnostics)),
        "stage_summary_rows_v285": int(len(stage_summary)),
        "claim_blocker_rows_v285": int(len(blockers)),
        "claim_matrix_rows_v285": int(len(claim_matrix)),
        "next_artifact_v285": f"paper4_v{NEXT_VERSION}_joint_source_relief_pricing_protocol.csv",
        "claim_boundary": (
            "v285 clears the source-tight one-drop/one-add pricing screen only; "
            "multi-source relief, branch-price bounds and promotion remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v285_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v285": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
