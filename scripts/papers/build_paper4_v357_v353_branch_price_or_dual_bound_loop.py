#!/usr/bin/env python3
"""Build Paper 4 v357 v353 bounded branch-price / dual-bound loop artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers import build_paper4_v351_v347_branch_price_or_dual_bound_loop as v351
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

VERSION = 357
BASE_VERSION = 353
REPRICE_VERSION = 354
READINESS_VERSION = 356
NEXT_VERSION = 358
EXPOSURE_MIN = 842292.375
EXPOSURE_MAX = 850000.0
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v353_apply_branch_price_candidate_or_bound_memo.csv"


def _source_maps(
    selected: pd.DataFrame,
    source_summary: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    current_by_family: dict[str, dict[str, float]] = {}
    cap_by_family: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        current_by_family[family] = (
            selected.groupby(family, dropna=False)["loan_amnt"].sum().astype(float).to_dict()
        )
        cap_by_family[family] = (
            source_summary.loc[source_summary["source_family"].astype(str).eq(family)]
            .set_index("source_id")[f"cap_share_v{BASE_VERSION}"]
            .astype(float)
            .to_dict()
        )
    return current_by_family, cap_by_family


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
    out[f"second_add_rank_v{VERSION}"] = np.arange(1, len(out) + 1)
    return out


def _screen_seed_pair(
    *,
    seed_rank: int,
    seed_pair: pd.Series,
    universe_by_id: pd.DataFrame,
    selected: pd.DataFrame,
    second_add_pool: pd.DataFrame,
    losses: np.ndarray,
    selected_idx: np.ndarray,
    idx_by_id: pd.Series,
    current_losses: np.ndarray,
    current_by_family: dict[str, dict[str, float]],
    cap_by_family: dict[str, dict[str, float]],
    current_objective_return: float,
    cvar_cap: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_add_id = str(seed_pair[f"added_loan_id_v{REPRICE_VERSION}"])
    seed_drop_id = str(seed_pair[f"dropped_loan_id_v{REPRICE_VERSION}"])
    seed_add = universe_by_id.loc[seed_add_id]
    seed_drop = selected.loc[selected["loan_id"].astype(str).eq(seed_drop_id)].iloc[0]
    seed_exposure = float(seed_pair[f"exposure_after_swap_v{REPRICE_VERSION}"])
    seed_return_delta = float(seed_pair[f"return_delta_v{REPRICE_VERSION}"])
    seed_source_maps = v351._seed_source_maps(
        seed_add=seed_add,
        seed_drop=seed_drop,
        current_by_family=current_by_family,
    )
    seed_losses = (
        current_losses
        + losses[:, int(idx_by_id.loc[seed_add_id])]
        - losses[:, int(idx_by_id.loc[seed_drop_id])]
    )
    second_adds = (
        second_add_pool.loc[~second_add_pool["loan_id"].astype(str).eq(seed_add_id)]
        .copy()
        .reset_index(drop=True)
    )
    second_drops = (
        selected.loc[~selected["loan_id"].astype(str).eq(seed_drop_id)]
        .copy()
        .reset_index(drop=True)
    )
    add_amount = second_adds["loan_amnt"].to_numpy(float)
    drop_amount = second_drops["loan_amnt"].to_numpy(float)
    add_return = second_adds[f"mean_return_v{VERSION}"].to_numpy(float)
    drop_return = second_drops[f"mean_return_if_dropped_v{VERSION}"].to_numpy(float)
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
            second_adds=second_adds,
            second_drops=second_drops,
            seed_exposure=seed_exposure,
            family=family,
        )
        family_counts[family] = int(family_mask.sum())
        cumulative &= family_mask

    source_exact_positions = np.argwhere(cumulative)
    candidate_rows: list[dict[str, Any]] = []
    second_add_idx = second_adds[f"universe_idx_v{VERSION}"].to_numpy(int)
    second_drop_idx = selected_idx[selected["loan_id"].astype(str).to_numpy() != seed_drop_id]
    best_cvar = np.nan
    best_return = np.nan
    for add_pos, drop_pos in source_exact_positions:
        add_pos = int(add_pos)
        drop_pos = int(drop_pos)
        second_add = second_adds.iloc[add_pos]
        second_drop = second_drops.iloc[drop_pos]
        new_losses = (
            seed_losses + losses[:, second_add_idx[add_pos]] - losses[:, second_drop_idx[drop_pos]]
        )
        cvar_after = v70._tail_cvar(new_losses)
        total_return_delta = float(return_delta[add_pos, drop_pos])
        best_cvar = cvar_after if pd.isna(best_cvar) else min(best_cvar, cvar_after)
        best_return = (
            total_return_delta if pd.isna(best_return) else max(best_return, total_return_delta)
        )
        source_ok, min_slack, max_share, violations, first_family, first_source = (
            v351._source_metrics_multi(
                add_rows=[seed_add, second_add],
                drop_rows=[seed_drop, second_drop],
                current_by_family=current_by_family,
                cap_by_family=cap_by_family,
                new_total=float(exposure_after[add_pos, drop_pos]),
            )
        )
        if not source_ok:
            continue
        entering = total_return_delta > 1e-9 and cvar_after <= cvar_cap + 1e-7
        second_add_id = str(second_add["loan_id"])
        second_drop_id = str(second_drop["loan_id"])
        candidate_rows.append(
            {
                "policy_id": "v357_v353_bounded_branch_price_loop",
                f"regime_v{VERSION}": "post_v353_seeded_two_add_two_drop",
                f"seed_pair_rank_v{VERSION}": seed_rank,
                f"seed_added_loan_id_v{VERSION}": seed_add_id,
                f"seed_dropped_loan_id_v{VERSION}": seed_drop_id,
                f"second_added_loan_id_v{VERSION}": second_add_id,
                f"second_dropped_loan_id_v{VERSION}": second_drop_id,
                f"action_signature_v{VERSION}": v351._action_signature(
                    seed_add_id, seed_drop_id, second_add_id, second_drop_id
                ),
                f"seed_return_delta_v{VERSION}": seed_return_delta,
                f"return_delta_v{VERSION}": total_return_delta,
                f"objective_return_after_two_swap_v{VERSION}": (
                    current_objective_return + total_return_delta
                ),
                f"exposure_after_two_swap_v{VERSION}": float(exposure_after[add_pos, drop_pos]),
                f"loss_mean_after_two_swap_v{VERSION}": float(new_losses.mean()),
                f"cvar90_after_two_swap_v{VERSION}": cvar_after,
                f"source_min_slack_after_two_swap_v{VERSION}": min_slack,
                f"max_source_share_after_two_swap_v{VERSION}": max_share,
                f"source_cap_violations_after_two_swap_v{VERSION}": violations,
                f"first_source_block_family_v{VERSION}": first_family,
                f"first_source_block_id_v{VERSION}": first_source,
                f"cvar_two_swap_feasible_v{VERSION}": cvar_after <= cvar_cap + 1e-7,
                f"two_add_two_drop_entering_column_v{VERSION}": entering,
                f"claim_boundary_v{VERSION}": (
                    "bounded post-v353 second-order branch-price row; no dual-bound certificate"
                ),
            }
        )

    entering_rows = sum(
        row[f"two_add_two_drop_entering_column_v{VERSION}"] for row in candidate_rows
    )
    stage_row = {
        f"seed_pair_rank_v{VERSION}": seed_rank,
        f"seed_added_loan_id_v{VERSION}": seed_add_id,
        f"seed_dropped_loan_id_v{VERSION}": seed_drop_id,
        f"ordered_second_order_rows_v{VERSION}": int(return_delta.size),
        f"return_improving_rows_v{VERSION}": int(return_mask.sum()),
        f"budget_return_rows_v{VERSION}": int(budget_return_mask.sum()),
        f"source_exact_rows_v{VERSION}": int(len(candidate_rows)),
        f"cvar_feasible_entering_rows_v{VERSION}": int(entering_rows),
        f"best_source_exact_return_delta_v{VERSION}": None
        if pd.isna(best_return)
        else float(best_return),
        f"best_source_exact_cvar90_v{VERSION}": None if pd.isna(best_cvar) else float(best_cvar),
        f"claim_boundary_v{VERSION}": (
            "one v354 source-exact row used as seed for second-order branch-price screen"
        ),
    }
    for family in FAMILIES:
        stage_row[f"{family}_source_feasible_alone_rows_v{VERSION}"] = family_counts[family]
    return stage_row, candidate_rows


def _stage_summary(
    *,
    seed_stage: pd.DataFrame,
    second_add_rows: int,
    selected_rows: int,
    source_exact_rows: int,
    entering_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            f"stage_v{VERSION}": "v354_source_exact_seed_pairs",
            f"row_count_v{VERSION}": int(len(seed_stage)),
            f"claim_boundary_v{VERSION}": "v354 source-exact one-add/one-drop rows",
        },
        {
            f"stage_v{VERSION}": "positive_source_tight_second_add_candidates",
            f"row_count_v{VERSION}": second_add_rows,
            f"claim_boundary_v{VERSION}": "positive omitted candidates touching v356 tight blocks",
        },
        {
            f"stage_v{VERSION}": "second_drop_candidates",
            f"row_count_v{VERSION}": selected_rows,
            f"claim_boundary_v{VERSION}": "all selected v353 loans, excluding the seed drop per seed",
        },
        {
            f"stage_v{VERSION}": "ordered_second_order_rows",
            f"row_count_v{VERSION}": int(seed_stage[f"ordered_second_order_rows_v{VERSION}"].sum()),
            f"claim_boundary_v{VERSION}": "ordered seed/add/drop combinations before filters",
        },
        {
            f"stage_v{VERSION}": "return_improving",
            f"row_count_v{VERSION}": int(seed_stage[f"return_improving_rows_v{VERSION}"].sum()),
            f"claim_boundary_v{VERSION}": "positive total two-swap return delta only",
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
                f"claim_boundary_v{VERSION}": "all-family source caps after second-order screen",
            },
            {
                f"stage_v{VERSION}": "cvar_feasible_entering_column",
                f"row_count_v{VERSION}": entering_rows,
                f"claim_boundary_v{VERSION}": "source-exact plus CVaR-feasible return-improving rows",
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
                "claim": "v357 executes a bounded post-v353 branch-price loop.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v357_v353_branch_price_or_dual_bound_loop.csv"
                ),
                "boundary": "Seeded second-order branch-price screen only; no dual-bound certificate.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v357 finds bounded post-v353 CVaR-feasible entering candidates.",
                "allowed": entering_rows > 0,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v357_branch_price_candidate_screen.csv"
                ),
                "boundary": (
                    "Bounded v354-seeded second-order scope only; apply/reprice required "
                    "and no full-universe certificate."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v357 finds no bounded post-v353 CVaR-feasible entering column.",
                "allowed": no_entry_allowed,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v357_second_order_branch_price_stage_summary.csv"
                ),
                "boundary": "Bounded v354-seeded second-order scope only.",
                "prohibited_claim_flag": not no_entry_allowed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v357 proves a valid full-universe branch-price bound.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v357_claim_blockers.csv"
                ),
                "boundary": "No terminating dual-bound loop or global certificate exists.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v357 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v357_claim_blockers.csv"
                ),
                "boundary": "Proxy, global, dynamic, online and deployment gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v357 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v357_claim_blockers.csv"
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
        "bounded_post_v353_branch_price_entering_candidate_found_requires_apply_reprice"
        if entering_rows
        else "bounded_post_v353_branch_price_loop_no_cvar_entering_column"
    )
    result = (
        "bounded_post_v353_second_order_cvar_feasible_entering_column_found"
        if entering_rows
        else "no_bounded_post_v353_second_order_cvar_feasible_entering_column"
    )
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v357 seeds a bounded post-v353 second-order branch-price loop from the "
                    "v354 source-exact frontier over the v356 source-tight positive candidates."
                ),
                "status": status,
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "apply any bounded entering candidate and reprice, or convert the "
                    "deeper no-entry evidence into a valid dual-bound blocker without promotion"
                ),
                "last_wave": "v357",
                "execution_result": result,
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v357")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V357_V353_BRANCH_PRICE_OR_DUAL_BOUND_LOOP_START -->"
    end = "<!-- V357_V353_BRANCH_PRICE_OR_DUAL_BOUND_LOOP_END -->"
    best_entering = status["best_entering_return_delta_v357"]
    best_entering_text = (
        "not applicable; no bounded entering rows" if best_entering is None else str(best_entering)
    )
    block = f"""
{start}

## Wave v357: v353 Bounded Branch-Price Loop

Generated: {status["generated_at_utc"]}

### Objective

v356 converted the v354/v355 evidence into a v353 dual-bound readiness register.
v357 executes the next bounded pricing loop: seed from the v354 source-exact
one-swap rows, then search a second source-tight add/drop over the 4,385
positive v356 source-tight candidates.

### Results

- Seed pair rows: `{status["seed_pair_rows_v357"]}`.
- Positive source-tight second-add candidates:
  `{status["positive_source_tight_candidate_rows_v357"]}`.
- Ordered second-order rows screened:
  `{status["ordered_second_order_rows_screened_v357"]}`.
- Budget+return feasible rows:
  `{status["budget_return_feasible_rows_v357"]}`.
- Source-exact second-order rows:
  `{status["source_exact_second_order_rows_v357"]}`.
- Unique source-exact action signatures:
  `{status["unique_source_exact_action_signatures_v357"]}`.
- CVaR-feasible entering rows:
  `{status["cvar_feasible_entering_rows_v357"]}`.
- Best entering return delta:
  `{best_entering_text}`.
- Valid branch-price bound:
  `{status["valid_branch_price_bound_v357"]}`.

### Interpretation

v357 advances beyond readiness by running a bounded second-order pricing loop
for the v353 candidate. It can identify actionable bounded entering columns, or
it can sharpen the no-entry blocker, but it still does not produce a terminating
full-v55 branch-price or dual-bound certificate.

### Claim Impact

- Allowed: bounded v354-seeded second-order branch-price loop executed.
- Allowed only within this bounded scope: bounded entering candidates when the
  count is positive, or no-entry evidence when the count is zero.
- Still prohibited: full-universe branch-price termination, valid global
  integer optimality, contractual IFRS9, live deployability, Paper Estrella
  replacement, final Paper 4 promotion and working champion claims.

### Quarto Promotion Decision

Keep v357 in the living notebook. The next wave should apply any bounded
entering candidate if one exists; otherwise it should convert the deeper
no-entry evidence into a stronger dual-bound or gap-blocker memo without
promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def _maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet(
        "paper4_v353_v347_expanded_branch_price_allocations.parquet"
    ).reset_index(drop=True)
    source_summary = read_csv("paper4_v353_v347_expanded_branch_price_source_summary.csv")
    v353_summary = read_csv("paper4_v353_v347_apply_expanded_branch_price_candidate.csv")
    v354_pairs = read_csv("paper4_v354_post_v353_one_swap_reprice.csv")
    v356_hotspots = read_csv("paper4_v356_v353_source_slack_hotspots.csv")
    v356_status = json.loads((STATUS_DIR / "paper4_v356_status.json").read_text(encoding="utf-8"))
    if any(
        df.empty
        for df in [universe, selected, source_summary, v353_summary, v354_pairs, v356_hotspots]
    ):
        raise RuntimeError("Missing v357 branch-price loop inputs.")
    if bool(v356_status["valid_branch_price_bound_v356"]):
        raise RuntimeError("v357 expects v356 to be readiness only, not a valid bound.")
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
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy(int)
    selected[f"mean_return_if_dropped_v{VERSION}"] = mean_returns[selected_idx]
    current_losses = losses[:, selected_idx].sum(axis=1)
    second_add_pool = _positive_source_tight_pool(
        universe=universe,
        selected=selected,
        mean_returns=mean_returns,
        idx_by_id=idx_by_id,
        hotspots=v356_hotspots,
    )
    seed_pairs = v354_pairs.sort_values(f"return_delta_v{REPRICE_VERSION}", ascending=False)
    v353_row = v353_summary.iloc[0]
    current_objective_return = float(v353_row["objective_return_v353"])
    cvar_cap = float(v353_row["scenario_loss_cvar90_v353"])
    current_by_family, cap_by_family = _source_maps(selected, source_summary)

    seed_stage_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for seed_rank, seed_pair in enumerate(seed_pairs.itertuples(index=False), start=1):
        stage_row, rows = _screen_seed_pair(
            seed_rank=seed_rank,
            seed_pair=pd.Series(seed_pair._asdict()),
            universe_by_id=universe_by_id,
            selected=selected,
            second_add_pool=second_add_pool,
            losses=losses,
            selected_idx=selected_idx,
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
            [f"two_add_two_drop_entering_column_v{VERSION}", f"return_delta_v{VERSION}"],
            ascending=[False, False],
        ).reset_index(drop=True)
    entering_summary = (
        candidate_screen.loc[
            candidate_screen[f"two_add_two_drop_entering_column_v{VERSION}"].astype(bool)
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
        second_add_rows=int(len(second_add_pool)),
        selected_rows=int(len(selected)),
        source_exact_rows=int(len(candidate_screen)),
        entering_rows=entering_rows,
    )
    best_by_return = candidate_screen.head(1)
    best_by_cvar = (
        candidate_screen.sort_values(f"cvar90_after_two_swap_v{VERSION}").head(1)
        if not candidate_screen.empty
        else pd.DataFrame()
    )
    best_entering = entering_summary.head(1)
    protocol = pd.DataFrame(
        [
            {
                f"protocol_id_v{VERSION}": "v353_seeded_second_order_branch_price_loop",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"reprice_version_v{VERSION}": REPRICE_VERSION,
                f"readiness_version_v{VERSION}": READINESS_VERSION,
                f"selected_rows_v{VERSION}": int(len(selected)),
                f"seed_pair_rows_v{VERSION}": int(len(seed_pairs)),
                f"positive_source_tight_candidate_rows_v{VERSION}": int(len(second_add_pool)),
                f"ordered_second_order_rows_screened_v{VERSION}": int(
                    seed_stage[f"ordered_second_order_rows_v{VERSION}"].sum()
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
                f"source_exact_second_order_rows_v{VERSION}": int(len(candidate_screen)),
                f"unique_source_exact_action_signatures_v{VERSION}": unique_source_exact,
                f"cvar_feasible_entering_rows_v{VERSION}": entering_rows,
                f"entering_candidate_summary_rows_v{VERSION}": int(len(entering_summary)),
                f"best_source_exact_return_delta_v{VERSION}": None
                if best_by_return.empty
                else float(best_by_return[f"return_delta_v{VERSION}"].iloc[0]),
                f"best_source_exact_cvar90_v{VERSION}": None
                if best_by_cvar.empty
                else float(best_by_cvar[f"cvar90_after_two_swap_v{VERSION}"].iloc[0]),
                f"best_entering_return_delta_v{VERSION}": None
                if best_entering.empty
                else float(best_entering[f"return_delta_v{VERSION}"].iloc[0]),
                f"best_entering_cvar90_v{VERSION}": None
                if best_entering.empty
                else float(best_entering[f"cvar90_after_two_swap_v{VERSION}"].iloc[0]),
                f"branch_price_loop_executed_v{VERSION}": True,
                f"valid_branch_price_bound_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "bounded second-order branch-price loop only; no dual-bound termination"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "bounded_second_order_entering_column_missing",
                f"blocking_v{VERSION}": entering_rows == 0,
                f"evidence_count_v{VERSION}": entering_rows,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "bounded entering rows found; apply and reprice before any claim expansion"
                    if entering_rows
                    else "no CVaR-feasible entering row in bounded second-order scope"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "valid_branch_price_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "bounded pricing loop is not a termination proof",
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
                "claim_id": "v357_bounded_branch_price_loop_executed",
                "allowed": True,
                "artifact": "paper4_v357_v353_branch_price_or_dual_bound_loop.csv",
                "boundary": "bounded seeded second-order scope",
            },
            {
                "claim_id": "v357_bounded_entering_candidate_found",
                "allowed": entering_rows > 0,
                "artifact": "paper4_v357_branch_price_candidate_screen.csv",
                "boundary": "bounded candidate only; apply/reprice required",
            },
            {
                "claim_id": "v357_no_bounded_second_order_entering_column",
                "allowed": entering_rows == 0,
                "artifact": "paper4_v357_second_order_branch_price_stage_summary.csv",
                "boundary": "not a global branch-price termination",
            },
            {
                "claim_id": "v357_valid_full_universe_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v357_claim_blockers.csv",
                "boundary": "dual-bound loop missing",
            },
            {
                "claim_id": "v357_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v357_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v357_v353_branch_price_or_dual_bound_loop.csv", protocol)
    write_csv(TABLE_DIR / "paper4_v357_second_order_seed_pair_summary.csv", seed_stage)
    write_csv(TABLE_DIR / "paper4_v357_second_order_branch_price_stage_summary.csv", stage_summary)
    write_csv(TABLE_DIR / "paper4_v357_branch_price_candidate_screen.csv", candidate_screen)
    write_csv(TABLE_DIR / "paper4_v357_entering_candidate_summary.csv", entering_summary)
    write_csv(TABLE_DIR / "paper4_v357_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v357_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries(entering_rows=entering_rows)
    _update_backlog(entering_rows=entering_rows)

    row = protocol.iloc[0]
    status = {
        "phase": "v357_v353_branch_price_or_dual_bound_loop",
        "schema_version": "2026-05-17.357",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v357": BASE_VERSION,
        "reprice_version_v357": REPRICE_VERSION,
        "readiness_version_v357": READINESS_VERSION,
        "selected_rows_v357": int(row[f"selected_rows_v{VERSION}"]),
        "seed_pair_rows_v357": int(row[f"seed_pair_rows_v{VERSION}"]),
        "positive_source_tight_candidate_rows_v357": int(
            row[f"positive_source_tight_candidate_rows_v{VERSION}"]
        ),
        "ordered_second_order_rows_screened_v357": int(
            row[f"ordered_second_order_rows_screened_v{VERSION}"]
        ),
        "return_improving_rows_v357": int(row[f"return_improving_rows_v{VERSION}"]),
        "budget_return_feasible_rows_v357": int(row[f"budget_return_feasible_rows_v{VERSION}"]),
        "grade_source_feasible_rows_v357": int(row[f"grade_source_feasible_rows_v{VERSION}"]),
        "score_decile_source_feasible_rows_v357": int(
            row[f"score_decile_source_feasible_rows_v{VERSION}"]
        ),
        "source_exact_second_order_rows_v357": int(
            row[f"source_exact_second_order_rows_v{VERSION}"]
        ),
        "unique_source_exact_action_signatures_v357": unique_source_exact,
        "cvar_feasible_entering_rows_v357": entering_rows,
        "entering_candidate_summary_rows_v357": int(
            row[f"entering_candidate_summary_rows_v{VERSION}"]
        ),
        "best_source_exact_return_delta_v357": _maybe_float(
            row[f"best_source_exact_return_delta_v{VERSION}"]
        ),
        "best_source_exact_cvar90_v357": _maybe_float(row[f"best_source_exact_cvar90_v{VERSION}"]),
        "best_entering_return_delta_v357": _maybe_float(
            row[f"best_entering_return_delta_v{VERSION}"]
        ),
        "best_entering_cvar90_v357": _maybe_float(row[f"best_entering_cvar90_v{VERSION}"]),
        "branch_price_loop_executed_v357": True,
        "valid_branch_price_bound_v357": False,
        "full_universe_integer_optimality_claim_allowed_v357": False,
        "working_champion_claim_allowed_v357": False,
        "paper1_promotion_allowed_v357": False,
        "paper4_working_champion_changed_v357": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v357": int(len(blockers)),
        "claim_matrix_rows_v357": int(len(claim_matrix)),
        "next_artifact_v357": NEXT_ARTIFACT,
        "claim_boundary": (
            "v357 runs a bounded second-order branch-price loop; full dual-bound, "
            "proxy, live, champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v357_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v357": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
