#!/usr/bin/env python3
"""Build Paper 4 v361 v353 fourth-order branch-price artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers import build_paper4_v351_v347_branch_price_or_dual_bound_loop as v351
from scripts.papers import build_paper4_v352_v347_expand_branch_price_or_dual_bound_loop as v352
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

VERSION = 361
BASE_VERSION = 353
PRIOR_BRANCH_VERSION = 359
READINESS_VERSION = 356
DISPOSITION_VERSION = 360
NEXT_VERSION = 362
EXPOSURE_MIN = 842292.375
EXPOSURE_MAX = 850000.0
NEXT_ARTIFACT = (
    f"paper4_v{NEXT_VERSION}_v353_apply_fourth_order_candidate_or_bound_memo.csv"
)


def _maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _positive_source_tight_pool(
    *,
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    mean_returns: np.ndarray,
    idx_by_id: pd.Series,
    hotspots: pd.DataFrame,
) -> pd.DataFrame:
    selected_ids = set(selected["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    candidate_idx = idx_by_id.loc[candidates["loan_id"].astype(str)].to_numpy(int)
    candidates[f"universe_idx_v{VERSION}"] = candidate_idx
    candidates[f"mean_return_v{VERSION}"] = mean_returns[candidate_idx]
    tight_mask = np.zeros(len(candidates), dtype=bool)
    for _, row in hotspots.loc[
        hotspots[f"source_tight_flag_v{READINESS_VERSION}"].astype(bool)
    ].iterrows():
        family = str(row["source_family"])
        source_id = str(row["source_id"])
        if family in FAMILIES:
            tight_mask |= candidates[family].astype(str).eq(source_id).to_numpy()
    out = (
        candidates.loc[tight_mask & candidates[f"mean_return_v{VERSION}"].gt(0)]
        .sort_values(f"mean_return_v{VERSION}", ascending=False)
        .reset_index(drop=True)
    )
    out[f"fourth_add_rank_v{VERSION}"] = np.arange(1, len(out) + 1)
    return out


def _unique_three_swap_seeds(candidate_screen: pd.DataFrame) -> pd.DataFrame:
    ordered = candidate_screen.copy()
    ordered = ordered.sort_values(
        [
            f"cvar90_after_three_swap_v{PRIOR_BRANCH_VERSION}",
            f"return_delta_v{PRIOR_BRANCH_VERSION}",
        ],
        ascending=[True, False],
    )
    seeds = ordered.drop_duplicates(f"action_signature_v{PRIOR_BRANCH_VERSION}").reset_index(
        drop=True
    )
    seeds[f"three_swap_seed_rank_v{VERSION}"] = np.arange(1, len(seeds) + 1)
    return seeds


def _screen_three_swap_seed(
    *,
    seed_row: pd.Series,
    universe_by_id: pd.DataFrame,
    selected: pd.DataFrame,
    selected_by_id: pd.DataFrame,
    fourth_add_pool: pd.DataFrame,
    losses: np.ndarray,
    idx_by_id: pd.Series,
    current_losses: np.ndarray,
    current_by_family: dict[str, dict[str, float]],
    cap_by_family: dict[str, dict[str, float]],
    current_objective_return: float,
    cvar_cap: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_rank = int(seed_row[f"three_swap_seed_rank_v{VERSION}"])
    add_ids = [
        str(seed_row[f"first_added_loan_id_v{PRIOR_BRANCH_VERSION}"]),
        str(seed_row[f"second_added_loan_id_v{PRIOR_BRANCH_VERSION}"]),
        str(seed_row[f"third_added_loan_id_v{PRIOR_BRANCH_VERSION}"]),
    ]
    drop_ids = [
        str(seed_row[f"first_dropped_loan_id_v{PRIOR_BRANCH_VERSION}"]),
        str(seed_row[f"second_dropped_loan_id_v{PRIOR_BRANCH_VERSION}"]),
        str(seed_row[f"third_dropped_loan_id_v{PRIOR_BRANCH_VERSION}"]),
    ]
    seed_add_rows = [universe_by_id.loc[loan_id] for loan_id in add_ids]
    seed_drop_rows = [selected_by_id.loc[loan_id] for loan_id in drop_ids]
    seed_exposure = float(seed_row[f"exposure_after_three_swap_v{PRIOR_BRANCH_VERSION}"])
    seed_return_delta = float(seed_row[f"return_delta_v{PRIOR_BRANCH_VERSION}"])
    seed_source_maps = v352._source_maps_after_actions(
        add_rows=seed_add_rows,
        drop_rows=seed_drop_rows,
        current_by_family=current_by_family,
    )

    seed_losses = current_losses.copy()
    for loan_id in add_ids:
        seed_losses = seed_losses + losses[:, int(idx_by_id.loc[loan_id])]
    for loan_id in drop_ids:
        seed_losses = seed_losses - losses[:, int(idx_by_id.loc[loan_id])]

    fourth_adds = (
        fourth_add_pool.loc[~fourth_add_pool["loan_id"].astype(str).isin(add_ids)]
        .copy()
        .reset_index(drop=True)
    )
    fourth_drops = (
        selected.loc[~selected["loan_id"].astype(str).isin(drop_ids)].copy().reset_index(drop=True)
    )
    add_amount = fourth_adds["loan_amnt"].to_numpy(float)
    drop_amount = fourth_drops["loan_amnt"].to_numpy(float)
    add_return = fourth_adds[f"mean_return_v{VERSION}"].to_numpy(float)
    drop_return = fourth_drops[f"mean_return_if_dropped_v{VERSION}"].to_numpy(float)
    return_delta = seed_return_delta + add_return[:, None] - drop_return[None, :]
    exposure_after = seed_exposure + add_amount[:, None] - drop_amount[None, :]
    return_mask = return_delta > 1e-9
    budget_return_mask = (
        return_mask
        & (exposure_after >= EXPOSURE_MIN - 1e-7)
        & (exposure_after <= EXPOSURE_MAX + 1e-7)
    )

    family_counts: dict[str, int] = {}
    cumulative = budget_return_mask.copy()
    for family in FAMILIES:
        family_mask = v351._family_mask_after_seed(
            base_mask=budget_return_mask,
            seed_source_maps=seed_source_maps,
            cap_by_family=cap_by_family,
            second_adds=fourth_adds,
            second_drops=fourth_drops,
            seed_exposure=seed_exposure,
            family=family,
        )
        family_counts[family] = int(family_mask.sum())
        cumulative &= family_mask

    source_exact_positions = np.argwhere(cumulative)
    candidate_rows: list[dict[str, Any]] = []
    fourth_add_idx = fourth_adds[f"universe_idx_v{VERSION}"].to_numpy(int)
    fourth_drop_idx = idx_by_id.loc[fourth_drops["loan_id"].astype(str)].to_numpy(int)
    best_cvar = np.nan
    best_return = np.nan
    for add_pos, drop_pos in source_exact_positions:
        add_pos = int(add_pos)
        drop_pos = int(drop_pos)
        fourth_add = fourth_adds.iloc[add_pos]
        fourth_drop = fourth_drops.iloc[drop_pos]
        new_losses = (
            seed_losses
            + losses[:, fourth_add_idx[add_pos]]
            - losses[:, fourth_drop_idx[drop_pos]]
        )
        cvar_after = v70._tail_cvar(new_losses)
        total_return_delta = float(return_delta[add_pos, drop_pos])
        best_cvar = cvar_after if pd.isna(best_cvar) else min(best_cvar, cvar_after)
        best_return = (
            total_return_delta if pd.isna(best_return) else max(best_return, total_return_delta)
        )
        fourth_add_id = str(fourth_add["loan_id"])
        fourth_drop_id = str(fourth_drop["loan_id"])
        source_ok, min_slack, max_share, violations, first_family, first_source = (
            v351._source_metrics_multi(
                add_rows=[*seed_add_rows, fourth_add],
                drop_rows=[*seed_drop_rows, fourth_drop],
                current_by_family=current_by_family,
                cap_by_family=cap_by_family,
                new_total=float(exposure_after[add_pos, drop_pos]),
            )
        )
        if not source_ok:
            continue
        entering = total_return_delta > 1e-9 and cvar_after <= cvar_cap + 1e-7
        all_add_ids = [*add_ids, fourth_add_id]
        all_drop_ids = [*drop_ids, fourth_drop_id]
        candidate_rows.append(
            {
                "policy_id": "v361_v353_fourth_order_branch_price_loop",
                f"regime_v{VERSION}": "post_v353_seeded_four_add_four_drop",
                f"three_swap_seed_rank_v{VERSION}": seed_rank,
                f"seed_action_signature_v{VERSION}": seed_row[
                    f"action_signature_v{PRIOR_BRANCH_VERSION}"
                ],
                f"first_added_loan_id_v{VERSION}": add_ids[0],
                f"second_added_loan_id_v{VERSION}": add_ids[1],
                f"third_added_loan_id_v{VERSION}": add_ids[2],
                f"fourth_added_loan_id_v{VERSION}": fourth_add_id,
                f"first_dropped_loan_id_v{VERSION}": drop_ids[0],
                f"second_dropped_loan_id_v{VERSION}": drop_ids[1],
                f"third_dropped_loan_id_v{VERSION}": drop_ids[2],
                f"fourth_dropped_loan_id_v{VERSION}": fourth_drop_id,
                f"action_signature_v{VERSION}": v352._action_signature(
                    all_add_ids, all_drop_ids
                ),
                f"seed_return_delta_v{VERSION}": seed_return_delta,
                f"return_delta_v{VERSION}": total_return_delta,
                f"objective_return_after_four_swap_v{VERSION}": (
                    current_objective_return + total_return_delta
                ),
                f"exposure_after_four_swap_v{VERSION}": float(exposure_after[add_pos, drop_pos]),
                f"loss_mean_after_four_swap_v{VERSION}": float(new_losses.mean()),
                f"cvar90_after_four_swap_v{VERSION}": cvar_after,
                f"source_min_slack_after_four_swap_v{VERSION}": min_slack,
                f"max_source_share_after_four_swap_v{VERSION}": max_share,
                f"source_cap_violations_after_four_swap_v{VERSION}": violations,
                f"first_source_block_family_v{VERSION}": first_family,
                f"first_source_block_id_v{VERSION}": first_source,
                f"cvar_four_swap_feasible_v{VERSION}": cvar_after <= cvar_cap + 1e-7,
                f"four_add_four_drop_entering_column_v{VERSION}": entering,
                f"claim_boundary_v{VERSION}": (
                    "bounded post-v353 fourth-order branch-price row; no dual-bound certificate"
                ),
            }
        )

    entering_rows = sum(
        row[f"four_add_four_drop_entering_column_v{VERSION}"] for row in candidate_rows
    )
    stage_row = {
        f"three_swap_seed_rank_v{VERSION}": seed_rank,
        f"seed_action_signature_v{VERSION}": seed_row[
            f"action_signature_v{PRIOR_BRANCH_VERSION}"
        ],
        f"seed_return_delta_v{VERSION}": seed_return_delta,
        f"seed_cvar90_v{VERSION}": float(
            seed_row[f"cvar90_after_three_swap_v{PRIOR_BRANCH_VERSION}"]
        ),
        f"fourth_add_candidate_rows_v{VERSION}": int(len(fourth_adds)),
        f"fourth_drop_candidate_rows_v{VERSION}": int(len(fourth_drops)),
        f"ordered_fourth_order_rows_v{VERSION}": int(return_delta.size),
        f"return_improving_rows_v{VERSION}": int(return_mask.sum()),
        f"budget_return_rows_v{VERSION}": int(budget_return_mask.sum()),
        f"source_exact_rows_v{VERSION}": int(len(candidate_rows)),
        f"cvar_feasible_entering_rows_v{VERSION}": int(entering_rows),
        f"best_source_exact_return_delta_v{VERSION}": _maybe_float(best_return),
        f"best_source_exact_cvar90_v{VERSION}": _maybe_float(best_cvar),
        f"claim_boundary_v{VERSION}": (
            "one unique v359 source-exact three-swap row used as a fourth-order seed"
        ),
    }
    for family in FAMILIES:
        stage_row[f"{family}_source_feasible_alone_rows_v{VERSION}"] = family_counts[family]
    return stage_row, candidate_rows


def _stage_summary(
    *,
    seed_stage: pd.DataFrame,
    source_exact_rows: int,
    entering_rows: int,
    fourth_add_pool_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            f"stage_v{VERSION}": "v359_unique_source_exact_three_swap_seeds",
            f"row_count_v{VERSION}": int(len(seed_stage)),
            f"claim_boundary_v{VERSION}": "unique v359 source-exact three-swap actions",
        },
        {
            f"stage_v{VERSION}": "positive_source_tight_fourth_add_candidates",
            f"row_count_v{VERSION}": fourth_add_pool_rows,
            f"claim_boundary_v{VERSION}": "positive omitted candidates touching v356 tight blocks",
        },
        {
            f"stage_v{VERSION}": "ordered_fourth_order_rows",
            f"row_count_v{VERSION}": int(seed_stage[f"ordered_fourth_order_rows_v{VERSION}"].sum()),
            f"claim_boundary_v{VERSION}": "ordered three-swap seed/add/drop combinations",
        },
        {
            f"stage_v{VERSION}": "return_improving",
            f"row_count_v{VERSION}": int(seed_stage[f"return_improving_rows_v{VERSION}"].sum()),
            f"claim_boundary_v{VERSION}": "positive total four-swap return delta only",
        },
        {
            f"stage_v{VERSION}": "budget_return_feasible",
            f"row_count_v{VERSION}": int(seed_stage[f"budget_return_rows_v{VERSION}"].sum()),
            f"claim_boundary_v{VERSION}": "budget plus positive return only",
        },
    ]
    rows.extend(
        {
            f"stage_v{VERSION}": f"{family}_source_feasible_alone",
            f"row_count_v{VERSION}": int(
                seed_stage[f"{family}_source_feasible_alone_rows_v{VERSION}"].sum()
            ),
            f"claim_boundary_v{VERSION}": f"{family} cap only after budget+return",
        }
        for family in FAMILIES
    )
    rows.extend(
        [
            {
                f"stage_v{VERSION}": "source_exact_feasible",
                f"row_count_v{VERSION}": source_exact_rows,
                f"claim_boundary_v{VERSION}": "all-family source caps after fourth-order screen",
            },
            {
                f"stage_v{VERSION}": "cvar_feasible_entering_column",
                f"row_count_v{VERSION}": entering_rows,
                f"claim_boundary_v{VERSION}": (
                    "source-exact plus CVaR-feasible return-improving rows"
                ),
            },
        ]
    )
    return pd.DataFrame(rows)


def _update_claim_boundaries(*, entering_rows: int) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    no_entry_allowed = entering_rows == 0
    additions = pd.DataFrame(
        [
            {
                "claim": "v361 executes a bounded post-v353 fourth-order branch-price loop.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v361_v353_fourth_order_or_full_dual_bound.csv"
                ),
                "boundary": (
                    "Unique v359 three-swap seeds plus one extra source-tight add/drop; "
                    "no full-v55 dual-bound certificate."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v361 finds bounded post-v353 CVaR-feasible entering candidates.",
                "allowed": entering_rows > 0,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v361_entering_candidate_summary.csv"
                ),
                "boundary": (
                    "Bounded v359-seeded fourth-order scope only; apply/reprice required "
                    "and no full-universe certificate."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v361 finds no bounded fourth-order CVaR-feasible entering column.",
                "allowed": no_entry_allowed,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v361_fourth_order_branch_price_stage_summary.csv"
                ),
                "boundary": (
                    "Bounded fourth-order scope only; not full branch-price termination."
                ),
                "prohibited_claim_flag": not no_entry_allowed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v361 proves a valid full-universe branch-price bound.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v361_claim_blockers.csv"
                ),
                "boundary": "No terminating full-v55 dual-bound loop or global certificate exists.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v361 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v361_claim_blockers.csv"
                ),
                "boundary": "Proxy, global, dynamic, online and deployment gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v361 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v361_claim_blockers.csv"
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
    status = (
        "fourth_order_post_v353_branch_price_entering_candidate_found_requires_apply_reprice"
        if entering_rows
        else "fourth_order_post_v353_branch_price_loop_no_cvar_entering_column"
    )
    result = (
        "bounded_post_v353_fourth_order_cvar_feasible_entering_column_found"
        if entering_rows
        else "no_bounded_post_v353_fourth_order_cvar_feasible_entering_column"
    )
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v361 expands the v359 bounded branch-price loop from unique "
                    "source-exact three-swap seeds to a fourth source-tight add/drop."
                ),
                "status": status,
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "apply any bounded fourth-order entering candidate and reprice, or "
                    "convert the deeper no-entry evidence into a valid dual-bound blocker"
                ),
                "last_wave": "v361",
                "execution_result": result,
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v361")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V361_V353_FOURTH_ORDER_OR_FULL_DUAL_BOUND_START -->"
    end = "<!-- V361_V353_FOURTH_ORDER_OR_FULL_DUAL_BOUND_END -->"
    best_entering = status["best_entering_return_delta_v361"]
    best_entering_text = (
        "not applicable; no bounded entering rows" if best_entering is None else str(best_entering)
    )
    block = f"""
{start}

## Wave v361: v353 Fourth-Order Branch-Price Loop

Generated: {status["generated_at_utc"]}

### Objective

v360 recorded the no-apply disposition after v359 found no CVaR-feasible
third-order entering row. v361 expands the bounded evidence one level deeper:
unique v359 source-exact three-swap rows become seeds, then one additional
source-tight add/drop is tested.

### Results

- Unique v359 three-swap seed rows:
  `{status["three_swap_seed_rows_v361"]}`.
- Positive source-tight fourth-add candidates:
  `{status["positive_source_tight_candidate_rows_v361"]}`.
- Ordered fourth-order rows screened:
  `{status["ordered_fourth_order_rows_screened_v361"]}`.
- Budget+return feasible rows:
  `{status["budget_return_feasible_rows_v361"]}`.
- Source-exact fourth-order rows:
  `{status["source_exact_fourth_order_rows_v361"]}`.
- Unique source-exact action signatures:
  `{status["unique_source_exact_action_signatures_v361"]}`.
- CVaR-feasible entering rows:
  `{status["cvar_feasible_entering_rows_v361"]}`.
- Best entering return delta:
  `{best_entering_text}`.
- Best entering CVaR90:
  `{status["best_entering_cvar90_v361"]}`.
- Valid branch-price bound:
  `{status["valid_branch_price_bound_v361"]}`.

### Interpretation

v361 tests the practical fourth-order frontier after the v359/v360 no-entry
evidence. A positive entering count is actionable only for an apply-and-reprice
wave; a zero entering count strengthens the local blocker. Either result remains
bounded branch-price evidence, not a full-v55 dual-bound termination
certificate.

### Claim Impact

- Allowed: bounded fourth-order branch-price loop executed.
- Allowed only within this bounded scope: bounded fourth-order entering
  candidates when the count is positive, or no-entry evidence when the count is
  zero.
- Still prohibited: full-universe branch-price termination, valid global
  integer optimality, contractual IFRS9, live deployability, Paper Estrella
  replacement, final Paper 4 promotion and working champion claims.

### Quarto Promotion Decision

Keep v361 in the living notebook. The next wave should apply any bounded
fourth-order entering candidate if one exists; otherwise it should convert the
deeper no-entry evidence into a stronger gap/dual-bound memo without promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet(
        "paper4_v353_v347_expanded_branch_price_allocations.parquet"
    ).reset_index(drop=True)
    source_summary = read_csv("paper4_v353_v347_expanded_branch_price_source_summary.csv")
    v353_summary = read_csv("paper4_v353_v347_apply_expanded_branch_price_candidate.csv")
    v356_hotspots = read_csv("paper4_v356_v353_source_slack_hotspots.csv")
    v359_candidates = read_csv("paper4_v359_branch_price_candidate_screen.csv")
    v360_memo = read_csv("paper4_v360_v353_apply_expanded_branch_price_candidate_or_bound_memo.csv")
    v360_status = json.loads((STATUS_DIR / "paper4_v360_status.json").read_text(encoding="utf-8"))
    if any(
        df.empty
        for df in [
            universe,
            selected,
            source_summary,
            v353_summary,
            v356_hotspots,
            v359_candidates,
            v360_memo,
        ]
    ):
        raise RuntimeError("Missing v361 fourth-order branch-price inputs.")
    if not bool(v360_status["no_apply_disposition_allowed_v360"]):
        raise RuntimeError("v361 expects v360 no-apply disposition after v359 no-entry evidence.")
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    for frame in [universe, selected]:
        for family in FAMILIES:
            frame[family] = frame[family].astype(str)

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    universe_by_id = universe.set_index("loan_id", drop=False)
    selected_by_id = selected.set_index("loan_id", drop=False)
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy(int)
    selected[f"mean_return_if_dropped_v{VERSION}"] = mean_returns[selected_idx]
    current_losses = losses[:, selected_idx].sum(axis=1)
    fourth_add_pool = _positive_source_tight_pool(
        universe=universe,
        selected=selected,
        mean_returns=mean_returns,
        idx_by_id=idx_by_id,
        hotspots=v356_hotspots,
    )
    three_swap_seeds = _unique_three_swap_seeds(v359_candidates)
    v353_row = v353_summary.iloc[0]
    current_objective_return = float(v353_row["objective_return_v353"])
    cvar_cap = float(v353_row["scenario_loss_cvar90_v353"])
    current_by_family, cap_by_family = v357._source_maps(selected, source_summary)

    seed_stage_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for _, seed_row in three_swap_seeds.iterrows():
        stage_row, rows = _screen_three_swap_seed(
            seed_row=seed_row,
            universe_by_id=universe_by_id,
            selected=selected,
            selected_by_id=selected_by_id,
            fourth_add_pool=fourth_add_pool,
            losses=losses,
            idx_by_id=idx_by_id,
            current_losses=current_losses,
            current_by_family=current_by_family,
            cap_by_family=cap_by_family,
            current_objective_return=current_objective_return,
            cvar_cap=cvar_cap,
        )
        seed_stage_rows.append(stage_row)
        candidate_rows.extend(rows)

    seed_stage = pd.DataFrame(seed_stage_rows)
    candidate_screen = pd.DataFrame(candidate_rows)
    if not candidate_screen.empty:
        candidate_screen = candidate_screen.sort_values(
            [f"four_add_four_drop_entering_column_v{VERSION}", f"return_delta_v{VERSION}"],
            ascending=[False, False],
        ).reset_index(drop=True)
    entering_summary = (
        candidate_screen.loc[
            candidate_screen[f"four_add_four_drop_entering_column_v{VERSION}"].astype(bool)
        ]
        .sort_values(f"return_delta_v{VERSION}", ascending=False)
        .reset_index(drop=True)
        if not candidate_screen.empty
        else pd.DataFrame()
    )
    entering_rows = int(len(entering_summary))
    unique_source_exact = (
        int(candidate_screen[f"action_signature_v{VERSION}"].nunique())
        if not candidate_screen.empty
        else 0
    )
    stage_summary = _stage_summary(
        seed_stage=seed_stage,
        source_exact_rows=int(len(candidate_screen)),
        entering_rows=entering_rows,
        fourth_add_pool_rows=int(len(fourth_add_pool)),
    )
    best_by_return = candidate_screen.head(1)
    best_by_cvar = (
        candidate_screen.sort_values(f"cvar90_after_four_swap_v{VERSION}").head(1)
        if not candidate_screen.empty
        else pd.DataFrame()
    )
    best_entering = entering_summary.head(1)
    protocol = pd.DataFrame(
        [
            {
                f"protocol_id_v{VERSION}": "v353_fourth_order_branch_price_loop",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"prior_branch_version_v{VERSION}": PRIOR_BRANCH_VERSION,
                f"readiness_version_v{VERSION}": READINESS_VERSION,
                f"disposition_version_v{VERSION}": DISPOSITION_VERSION,
                f"selected_rows_v{VERSION}": int(len(selected)),
                f"three_swap_seed_rows_v{VERSION}": int(len(three_swap_seeds)),
                f"positive_source_tight_candidate_rows_v{VERSION}": int(len(fourth_add_pool)),
                f"ordered_fourth_order_rows_screened_v{VERSION}": int(
                    seed_stage[f"ordered_fourth_order_rows_v{VERSION}"].sum()
                ),
                f"return_improving_rows_v{VERSION}": int(
                    seed_stage[f"return_improving_rows_v{VERSION}"].sum()
                ),
                f"budget_return_feasible_rows_v{VERSION}": int(
                    seed_stage[f"budget_return_rows_v{VERSION}"].sum()
                ),
                f"grade_source_feasible_rows_v{VERSION}": int(
                    seed_stage[f"grade_source_feasible_alone_rows_v{VERSION}"].sum()
                ),
                f"score_decile_source_feasible_rows_v{VERSION}": int(
                    seed_stage[f"score_decile_source_feasible_alone_rows_v{VERSION}"].sum()
                ),
                f"source_exact_fourth_order_rows_v{VERSION}": int(len(candidate_screen)),
                f"unique_source_exact_action_signatures_v{VERSION}": unique_source_exact,
                f"cvar_feasible_entering_rows_v{VERSION}": entering_rows,
                f"entering_candidate_summary_rows_v{VERSION}": int(len(entering_summary)),
                f"best_source_exact_return_delta_v{VERSION}": None
                if best_by_return.empty
                else float(best_by_return[f"return_delta_v{VERSION}"].iloc[0]),
                f"best_source_exact_cvar90_v{VERSION}": None
                if best_by_cvar.empty
                else float(best_by_cvar[f"cvar90_after_four_swap_v{VERSION}"].iloc[0]),
                f"best_entering_return_delta_v{VERSION}": None
                if best_entering.empty
                else float(best_entering[f"return_delta_v{VERSION}"].iloc[0]),
                f"best_entering_cvar90_v{VERSION}": None
                if best_entering.empty
                else float(best_entering[f"cvar90_after_four_swap_v{VERSION}"].iloc[0]),
                f"branch_price_loop_executed_v{VERSION}": True,
                f"valid_branch_price_bound_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "bounded fourth-order branch-price loop only; "
                    "no full-v55 dual-bound termination"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "fourth_order_entering_column_missing",
                f"blocking_v{VERSION}": entering_rows == 0,
                f"evidence_count_v{VERSION}": entering_rows,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "bounded entering rows found; apply and reprice before any claim expansion"
                    if entering_rows
                    else "no CVaR-feasible entering row in bounded fourth-order scope"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "valid_branch_price_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "bounded fourth-order pricing is not a termination proof",
            },
            {
                f"blocker_id_v{VERSION}": "proxy_gap_persists",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v353_row["missing_proxy_rows_v353"]),
                f"required_next_artifact_v{VERSION}": "future_proxy_or_ifrs9_gate",
                f"claim_boundary_v{VERSION}": "v353 candidate still contains missing proxy rows",
            },
            {
                f"blocker_id_v{VERSION}": "full_v55_dual_bound_loop_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(universe)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "full-v55 dual-bound loop remains missing",
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
                "claim_id": "v361_fourth_order_branch_price_loop_executed",
                "allowed": True,
                "artifact": "paper4_v361_v353_fourth_order_or_full_dual_bound.csv",
                "boundary": "bounded fourth-order scope",
            },
            {
                "claim_id": "v361_bounded_entering_candidate_found",
                "allowed": entering_rows > 0,
                "artifact": "paper4_v361_entering_candidate_summary.csv",
                "boundary": "bounded candidate only; apply/reprice required",
            },
            {
                "claim_id": "v361_no_fourth_order_entering_column",
                "allowed": entering_rows == 0,
                "artifact": "paper4_v361_fourth_order_branch_price_stage_summary.csv",
                "boundary": "not a global branch-price termination",
            },
            {
                "claim_id": "v361_valid_full_universe_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v361_claim_blockers.csv",
                "boundary": "dual-bound loop missing",
            },
            {
                "claim_id": "v361_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v361_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v361_v353_fourth_order_or_full_dual_bound.csv", protocol)
    write_csv(TABLE_DIR / "paper4_v361_fourth_order_seed_summary.csv", seed_stage)
    write_csv(TABLE_DIR / "paper4_v361_fourth_order_branch_price_stage_summary.csv", stage_summary)
    write_csv(TABLE_DIR / "paper4_v361_branch_price_candidate_screen.csv", candidate_screen)
    write_csv(TABLE_DIR / "paper4_v361_entering_candidate_summary.csv", entering_summary)
    write_csv(TABLE_DIR / "paper4_v361_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v361_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries(entering_rows=entering_rows)
    _update_backlog(entering_rows=entering_rows)

    row = protocol.iloc[0]
    status = {
        "phase": "v361_v353_fourth_order_or_full_dual_bound",
        "schema_version": "2026-05-17.361",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v361": BASE_VERSION,
        "prior_branch_version_v361": PRIOR_BRANCH_VERSION,
        "readiness_version_v361": READINESS_VERSION,
        "disposition_version_v361": DISPOSITION_VERSION,
        "selected_rows_v361": int(row[f"selected_rows_v{VERSION}"]),
        "three_swap_seed_rows_v361": int(row[f"three_swap_seed_rows_v{VERSION}"]),
        "positive_source_tight_candidate_rows_v361": int(
            row[f"positive_source_tight_candidate_rows_v{VERSION}"]
        ),
        "ordered_fourth_order_rows_screened_v361": int(
            row[f"ordered_fourth_order_rows_screened_v{VERSION}"]
        ),
        "return_improving_rows_v361": int(row[f"return_improving_rows_v{VERSION}"]),
        "budget_return_feasible_rows_v361": int(row[f"budget_return_feasible_rows_v{VERSION}"]),
        "grade_source_feasible_rows_v361": int(row[f"grade_source_feasible_rows_v{VERSION}"]),
        "score_decile_source_feasible_rows_v361": int(
            row[f"score_decile_source_feasible_rows_v{VERSION}"]
        ),
        "source_exact_fourth_order_rows_v361": int(
            row[f"source_exact_fourth_order_rows_v{VERSION}"]
        ),
        "unique_source_exact_action_signatures_v361": unique_source_exact,
        "cvar_feasible_entering_rows_v361": entering_rows,
        "entering_candidate_summary_rows_v361": int(
            row[f"entering_candidate_summary_rows_v{VERSION}"]
        ),
        "best_source_exact_return_delta_v361": _maybe_float(
            row[f"best_source_exact_return_delta_v{VERSION}"]
        ),
        "best_source_exact_cvar90_v361": _maybe_float(row[f"best_source_exact_cvar90_v{VERSION}"]),
        "best_entering_return_delta_v361": _maybe_float(
            row[f"best_entering_return_delta_v{VERSION}"]
        ),
        "best_entering_cvar90_v361": _maybe_float(row[f"best_entering_cvar90_v{VERSION}"]),
        "branch_price_loop_executed_v361": True,
        "valid_branch_price_bound_v361": False,
        "full_universe_integer_optimality_claim_allowed_v361": False,
        "working_champion_claim_allowed_v361": False,
        "paper1_promotion_allowed_v361": False,
        "paper4_working_champion_changed_v361": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v361": int(len(blockers)),
        "claim_matrix_rows_v361": int(len(claim_matrix)),
        "next_artifact_v361": NEXT_ARTIFACT,
        "claim_boundary": (
            "v361 expands the bounded branch-price loop to fourth order; full dual-bound, "
            "proxy, live, champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v361_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v361": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
