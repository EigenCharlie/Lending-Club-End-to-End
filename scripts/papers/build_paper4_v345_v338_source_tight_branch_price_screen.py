#!/usr/bin/env python3
"""Build Paper 4 v345 v338 source-tight branch-price screen artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
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

VERSION = 345
BASE_VERSION = 338
READINESS_VERSION = 344
NEXT_VERSION = 346
TOP_DIAGNOSTIC_ROWS = 250
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v338_multi_source_relief_or_bound_protocol.csv"
EXPOSURE_MIN = 842292.375
EXPOSURE_MAX = 850000.0


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
        .set_index("source_id")[f"cap_share_v{BASE_VERSION}"]
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


def _exact_source_metrics(
    add_row: pd.Series,
    drop_row: pd.Series,
    current_by_family: dict[str, dict[str, float]],
    cap_by_family: dict[str, dict[str, float]],
    new_total: float,
) -> tuple[bool, float, float, int, str, str]:
    min_slack = np.inf
    max_share = 0.0
    violations = 0
    first_family = ""
    first_source = ""
    add_amount = float(add_row["loan_amnt"])
    drop_amount = float(drop_row["loan_amnt"])
    for family in FAMILIES:
        add_source = str(add_row[family])
        drop_source = str(drop_row[family])
        sources = set(current_by_family[family]) | {add_source, drop_source}
        caps = cap_by_family[family]
        for source_id in sources:
            exposure = current_by_family[family].get(source_id, 0.0)
            if source_id == add_source:
                exposure += add_amount
            if source_id == drop_source:
                exposure -= drop_amount
            share = exposure / max(new_total, 1.0)
            cap = caps.get(source_id, 1.0)
            slack = cap - share
            min_slack = min(min_slack, slack)
            max_share = max(max_share, share)
            if share > cap + 1e-7:
                violations += 1
                if not first_family:
                    first_family = family
                    first_source = source_id
    return (
        violations == 0,
        float(min_slack),
        float(max_share),
        violations,
        first_family,
        first_source,
    )


def _candidate_block_memberships(
    candidates: pd.DataFrame,
    tight_blocks: list[tuple[str, str]],
) -> tuple[pd.Series, pd.Series]:
    block_ids: list[str] = []
    block_counts: list[int] = []
    for _, row in candidates.iterrows():
        memberships = [
            f"{family}={source_id}"
            for family, source_id in tight_blocks
            if str(row[family]) == str(source_id)
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
    tight_blocks: list[tuple[str, str]],
    return_mask: np.ndarray,
    budget_return_mask: np.ndarray,
    family_masks: dict[str, np.ndarray],
    source_prefilter_mask: np.ndarray,
    return_delta: np.ndarray,
) -> pd.DataFrame:
    block_ids, block_counts = _candidate_block_memberships(candidates, tight_blocks)
    selected_ids = selected["loan_id"].astype(str).to_numpy()
    selected_returns = selected[f"mean_return_if_dropped_v{VERSION}"].to_numpy(float)
    best_drop_ids: list[str] = []
    best_return_deltas: list[float] = []
    best_drop_returns: list[float] = []
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
        [
            "loan_id",
            "loan_amnt",
            "grade",
            "score_decile",
            "income_band",
            "dti_band",
            "period",
            "state_top20",
        ]
    ].copy()
    diagnostics[f"mean_return_v{VERSION}"] = candidates[f"mean_return_v{VERSION}"].to_numpy(float)
    diagnostics[f"pricing_block_ids_v{VERSION}"] = block_ids.to_numpy()
    diagnostics[f"tight_block_count_v{VERSION}"] = block_counts.to_numpy(dtype=int)
    diagnostics[f"return_improving_drop_rows_v{VERSION}"] = return_mask.sum(axis=1).astype(int)
    diagnostics[f"budget_return_drop_rows_v{VERSION}"] = budget_return_mask.sum(axis=1).astype(int)
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
    diagnostics[f"claim_boundary_v{VERSION}"] = (
        "candidate-level v338 source-tight pricing diagnostic; not a branch-price certificate"
    )
    return (
        diagnostics.sort_values(f"mean_return_v{VERSION}", ascending=False)
        .head(TOP_DIAGNOSTIC_ROWS)
        .reset_index(drop=True)
    )


def _exact_and_cvar_pairs(
    *,
    source_prefilter_mask: np.ndarray,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
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
        f"claim_boundary_v{VERSION}",
    ]
    current_by_family, cap_by_family = _source_maps(selected, source_summary)
    current_losses = losses[:, selected_idx].sum(axis=1)
    rows: list[dict[str, Any]] = []
    for candidate_pos, selected_pos in np.argwhere(source_prefilter_mask):
        add_row = candidates.iloc[int(candidate_pos)]
        drop_row = selected.iloc[int(selected_pos)]
        new_total = float(current_exposure + add_row["loan_amnt"] - drop_row["loan_amnt"])
        source_ok, min_slack, max_share, violations, first_family, first_source = (
            _exact_source_metrics(
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
                "policy_id": "v345_v338_source_tight_branch_price_screen",
                f"regime_v{VERSION}": "post_v338_source_tight_one_drop_one_add",
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
                f"claim_boundary_v{VERSION}": (
                    "v338 source-tight exact source/CVaR pair only; no global bound"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _stage_summary(
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


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v345 v338 source-tight branch-price screen.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v345_v338_source_tight_branch_price_screen.csv"
                ),
                "boundary": (
                    "Scope-limited one-drop/one-add screen inside v344 source-tight candidates."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v345 finds no CVaR-feasible one-drop/one-add source-tight entering column.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v345_source_tight_pair_stage_summary.csv"
                ),
                "boundary": (
                    "Local source-tight screen only; multi-source relief and global bound remain open."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v345 proves a valid full-universe branch-price termination certificate.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v345_claim_blockers.csv"
                ),
                "boundary": "No dual-bound loop or multi-column branch-price termination exists.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v345 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v345_claim_blockers.csv"
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
                    "v345 executes a source-tight one-drop/one-add branch-price screen for "
                    "v338 after the v344 readiness gate."
                ),
                "status": "source_tight_branch_price_screen_no_entering_column",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "multi-source relief or a valid dual-bound loop resolves the remaining "
                    "source-tight blocker without promoting"
                ),
                "last_wave": "v345",
                "execution_result": "source_tight_one_drop_one_add_cvar_entering_columns_zero",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v345")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V345_V338_SOURCE_TIGHT_BRANCH_PRICE_SCREEN_START -->"
    end = "<!-- V345_V338_SOURCE_TIGHT_BRANCH_PRICE_SCREEN_END -->"
    block = f"""
{start}

## Wave v345: v338 Source-Tight Branch-Price Screen

Generated: {status["generated_at_utc"]}

### Objective

v344 identified grade A and score decile 0 as the tight v338 governance blocks.
v345 executes the next local pricing screen: positive-return source-tight
candidates against all v338 drops, with budget, source and CVaR filters.

### Results

- Unique source-tight candidate rows: `{status["unique_source_tight_candidate_rows_v345"]}`.
- Positive source-tight candidate rows:
  `{status["positive_source_tight_candidate_rows_v345"]}`.
- Total pair rows screened:
  `{status["total_pair_rows_screened_v345"]}`.
- Return-improving pair rows:
  `{status["return_improving_pair_rows_v345"]}`.
- Budget+return feasible pair rows:
  `{status["budget_return_feasible_pair_rows_v345"]}`.
- Source prefilter pair rows:
  `{status["source_prefilter_pair_rows_v345"]}`.
- Source-exact pair rows:
  `{status["source_exact_pair_rows_v345"]}`.
- CVaR-feasible entering columns:
  `{status["cvar_feasible_entering_columns_v345"]}`.
- Valid branch-price bound:
  `{status["valid_branch_price_bound_v345"]}`.

### Interpretation

v345 sharpens the v344 blocker. There are many positive source-tight candidates,
but the one-drop/one-add branch-price screen still finds no CVaR-feasible
entering column. This is stronger than a readiness checklist, but it remains a
local pricing result, not a branch-price termination certificate.

### Claim Impact

- Allowed: v338 source-tight one-drop/one-add screen completed and no
  CVaR-feasible entering column found in that local scope.
- Still prohibited: full-universe branch-price termination, contractual IFRS9,
  live deployability, Paper Estrella replacement, final Paper 4 promotion and
  working champion claims.

### Quarto Promotion Decision

Keep v345 in the living notebook. The next wave should test multi-source relief
or a formal dual-bound loop without promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v338_post_v336_swap_allocations.parquet").reset_index(drop=True)
    source_summary = read_csv("paper4_v338_post_v336_swap_source_summary.csv")
    v338_summary = read_csv("paper4_v338_apply_next_post_v336_swap.csv")
    v344_hotspots = read_csv("paper4_v344_v338_source_slack_hotspots.csv")
    v344_status = json.loads((STATUS_DIR / "paper4_v344_status.json").read_text(encoding="utf-8"))
    if any(df.empty for df in [universe, selected, source_summary, v338_summary, v344_hotspots]):
        raise RuntimeError("Missing v345 source-tight screen inputs.")
    if bool(v344_status["valid_full_universe_gap_certificate_v344"]):
        raise RuntimeError("v345 expects v344 global gap certificate to remain missing.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    for family in FAMILIES:
        universe[family] = universe[family].astype(str)
        selected[family] = selected[family].astype(str)

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy(int)
    selected[f"mean_return_if_dropped_v{VERSION}"] = mean_returns[selected_idx]

    selected_ids = set(selected["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].isin(selected_ids)].copy()
    candidate_idx_all = idx_by_id.loc[candidates["loan_id"].astype(str)].to_numpy(int)
    candidates[f"universe_idx_v{VERSION}"] = candidate_idx_all
    candidates[f"mean_return_v{VERSION}"] = mean_returns[candidate_idx_all]

    tight_blocks = [
        (str(row["source_family"]), str(row["source_id"]))
        for _, row in v344_hotspots.loc[
            v344_hotspots["source_tight_flag_v344"].astype(bool)
        ].iterrows()
        if str(row["source_family"]) in FAMILIES
    ]
    tight_mask = np.zeros(len(candidates), dtype=bool)
    for family, source_id in tight_blocks:
        tight_mask |= candidates[family].astype(str).eq(source_id).to_numpy()
    source_tight = candidates.loc[tight_mask].copy().reset_index(drop=True)
    positive = source_tight.loc[source_tight[f"mean_return_v{VERSION}"].gt(0)].copy()
    positive = positive.reset_index(drop=True)
    candidate_idx = positive[f"universe_idx_v{VERSION}"].to_numpy(int)

    v338_row = v338_summary.iloc[0]
    current_exposure = float(v338_row["portfolio_exposure_v338"])
    current_objective_return = float(v338_row["objective_return_v338"])
    cvar_cap = float(v338_row["scenario_loss_cvar90_v338"])
    add_amount = positive["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    add_return = positive[f"mean_return_v{VERSION}"].to_numpy(float)
    drop_return = selected[f"mean_return_if_dropped_v{VERSION}"].to_numpy(float)
    return_delta = add_return[:, None] - drop_return[None, :]
    return_mask = return_delta > 1e-9
    exposure_after = current_exposure + add_amount[:, None] - drop_amount[None, :]
    budget_return_mask = (
        return_mask
        & (exposure_after >= EXPOSURE_MIN - 1e-7)
        & (exposure_after <= EXPOSURE_MAX + 1e-7)
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
                    f"cumulative v338 source prefilter through {family}; diagnostic only"
                ),
            }
        )
    source_prefilter_mask = cumulative
    pairs = _exact_and_cvar_pairs(
        source_prefilter_mask=source_prefilter_mask,
        candidates=positive,
        selected=selected,
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
    stage_summary = _stage_summary(
        total_pairs=total_pairs,
        return_pairs=return_pairs,
        budget_return_pairs=budget_return_pairs,
        family_masks=family_masks,
        cumulative_counts=cumulative_counts,
        source_exact_pairs=int(len(pairs)),
        cvar_feasible_pairs=cvar_feasible_pairs,
    )
    diagnostics = _candidate_diagnostics(
        candidates=positive,
        selected=selected,
        tight_blocks=tight_blocks,
        return_mask=return_mask,
        budget_return_mask=budget_return_mask,
        family_masks=family_masks,
        source_prefilter_mask=source_prefilter_mask,
        return_delta=return_delta,
    )

    valid_branch_price_bound = False
    screen = pd.DataFrame(
        [
            {
                f"screen_id_v{VERSION}": "v338_source_tight_one_drop_one_add_branch_price_screen",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"readiness_version_v{VERSION}": READINESS_VERSION,
                f"selected_rows_v{VERSION}": int(len(selected)),
                f"tight_source_rows_v{VERSION}": int(len(tight_blocks)),
                f"unique_source_tight_candidate_rows_v{VERSION}": int(len(source_tight)),
                f"positive_source_tight_candidate_rows_v{VERSION}": int(len(positive)),
                f"total_pair_rows_screened_v{VERSION}": total_pairs,
                f"return_improving_pair_rows_v{VERSION}": return_pairs,
                f"budget_return_feasible_pair_rows_v{VERSION}": budget_return_pairs,
                f"grade_source_feasible_pair_rows_v{VERSION}": int(family_masks["grade"].sum()),
                f"score_decile_source_feasible_pair_rows_v{VERSION}": int(
                    family_masks["score_decile"].sum()
                ),
                f"source_prefilter_pair_rows_v{VERSION}": source_prefilter_pairs,
                f"source_exact_pair_rows_v{VERSION}": int(len(pairs)),
                f"cvar_feasible_entering_columns_v{VERSION}": cvar_feasible_pairs,
                f"best_source_exact_return_delta_v{VERSION}": None
                if pairs.empty
                else float(pairs[f"return_delta_v{VERSION}"].max()),
                f"best_source_exact_cvar90_v{VERSION}": None
                if pairs.empty
                else float(
                    pairs.sort_values(f"return_delta_v{VERSION}", ascending=False).iloc[0][
                        f"cvar90_after_swap_v{VERSION}"
                    ]
                ),
                f"valid_branch_price_bound_v{VERSION}": valid_branch_price_bound,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "source-tight one-drop/one-add screen only; no branch-price termination"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "cvar_feasible_entering_column_missing",
                f"blocking_v{VERSION}": cvar_feasible_pairs == 0,
                f"evidence_count_v{VERSION}": cvar_feasible_pairs,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "no CVaR-feasible source-tight entering column found",
            },
            {
                f"blocker_id_v{VERSION}": "multi_source_relief_or_dual_loop_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(tight_blocks)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "multi-source relief or formal dual loop still required",
            },
            {
                f"blocker_id_v{VERSION}": "valid_branch_price_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "local pricing screen is not a termination certificate",
            },
            {
                f"blocker_id_v{VERSION}": "contractual_ifrs9_and_live_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 74,
                f"required_next_artifact_v{VERSION}": "future_contractual_or_live_holdout_gate",
                f"claim_boundary_v{VERSION}": "v338 imputed proxy and internal online blockers remain",
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
                "claim_id": "v345_source_tight_branch_price_screen_executed",
                "allowed": True,
                "artifact": "paper4_v345_v338_source_tight_branch_price_screen.csv",
                "boundary": "local one-drop/one-add source-tight screen only",
            },
            {
                "claim_id": "v345_no_cvar_feasible_source_tight_entering_column",
                "allowed": cvar_feasible_pairs == 0,
                "artifact": "paper4_v345_source_tight_pair_stage_summary.csv",
                "boundary": "scope-limited local pricing result only",
            },
            {
                "claim_id": "v345_valid_full_universe_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v345_claim_blockers.csv",
                "boundary": "formal dual loop missing",
            },
            {
                "claim_id": "v345_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v345_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v345_v338_source_tight_branch_price_screen.csv", screen)
    write_csv(TABLE_DIR / "paper4_v345_source_tight_pair_stage_summary.csv", stage_summary)
    write_csv(TABLE_DIR / "paper4_v345_source_tight_candidate_diagnostics.csv", diagnostics)
    write_csv(TABLE_DIR / "paper4_v345_source_tight_pricing_pairs.csv", pairs)
    write_csv(TABLE_DIR / "paper4_v345_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v345_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = screen.iloc[0]
    status = {
        "phase": "v345_v338_source_tight_branch_price_screen",
        "schema_version": "2026-05-16.345",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v345": BASE_VERSION,
        "readiness_version_v345": READINESS_VERSION,
        "selected_rows_v345": int(row[f"selected_rows_v{VERSION}"]),
        "tight_source_rows_v345": int(row[f"tight_source_rows_v{VERSION}"]),
        "unique_source_tight_candidate_rows_v345": int(
            row[f"unique_source_tight_candidate_rows_v{VERSION}"]
        ),
        "positive_source_tight_candidate_rows_v345": int(
            row[f"positive_source_tight_candidate_rows_v{VERSION}"]
        ),
        "total_pair_rows_screened_v345": total_pairs,
        "return_improving_pair_rows_v345": return_pairs,
        "budget_return_feasible_pair_rows_v345": budget_return_pairs,
        "grade_source_feasible_pair_rows_v345": int(
            row[f"grade_source_feasible_pair_rows_v{VERSION}"]
        ),
        "score_decile_source_feasible_pair_rows_v345": int(
            row[f"score_decile_source_feasible_pair_rows_v{VERSION}"]
        ),
        "source_prefilter_pair_rows_v345": source_prefilter_pairs,
        "source_exact_pair_rows_v345": int(len(pairs)),
        "cvar_feasible_entering_columns_v345": cvar_feasible_pairs,
        "best_source_exact_return_delta_v345": None
        if pairs.empty
        else float(pairs[f"return_delta_v{VERSION}"].max()),
        "best_source_exact_cvar90_v345": None
        if pairs.empty
        else float(
            pairs.sort_values(f"return_delta_v{VERSION}", ascending=False).iloc[0][
                f"cvar90_after_swap_v{VERSION}"
            ]
        ),
        "valid_branch_price_bound_v345": valid_branch_price_bound,
        "full_universe_integer_optimality_claim_allowed_v345": False,
        "working_champion_claim_allowed_v345": False,
        "paper1_promotion_allowed_v345": False,
        "paper4_working_champion_changed_v345": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v345": int(len(blockers)),
        "claim_matrix_rows_v345": int(len(claim_matrix)),
        "next_artifact_v345": NEXT_ARTIFACT,
        "claim_boundary": (
            "v345 screens source-tight one-drop/one-add pricing; full branch-price, "
            "IFRS9, live, champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v345_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v345": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
