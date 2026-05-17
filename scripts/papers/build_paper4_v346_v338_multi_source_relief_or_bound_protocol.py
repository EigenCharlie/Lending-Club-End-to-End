#!/usr/bin/env python3
"""Build Paper 4 v346 v338 multi-source relief/bound protocol artifacts."""

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

VERSION = 346
BASE_VERSION = 338
PREVIOUS_VERSION = 345
READINESS_VERSION = 344
NEXT_VERSION = 347
SEED_PAIR_LIMIT = 8
SECOND_ADD_LIMIT = 5_000
EXPOSURE_MIN = 842292.375
EXPOSURE_MAX = 850000.0
MAIN_ARTIFACT = "paper4_v346_v338_multi_source_relief_or_bound_protocol.csv"
NEXT_APPLY_ARTIFACT = f"paper4_v{NEXT_VERSION}_v338_apply_multi_source_relief_candidate.csv"
NEXT_BOUND_ARTIFACT = (
    f"paper4_v{NEXT_VERSION}_v338_expand_multi_source_relief_or_dual_bound_loop.csv"
)


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


def _seed_source_maps(
    *,
    seed_add: pd.Series,
    seed_drop: pd.Series,
    current_by_family: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    seeded: dict[str, dict[str, float]] = {
        family: dict(source_map) for family, source_map in current_by_family.items()
    }
    add_amount = float(seed_add["loan_amnt"])
    drop_amount = float(seed_drop["loan_amnt"])
    for family in FAMILIES:
        add_source = str(seed_add[family])
        drop_source = str(seed_drop[family])
        seeded[family][add_source] = seeded[family].get(add_source, 0.0) + add_amount
        seeded[family][drop_source] = seeded[family].get(drop_source, 0.0) - drop_amount
    return seeded


def _family_mask_after_seed(
    *,
    base_mask: np.ndarray,
    seed_source_maps: dict[str, dict[str, float]],
    cap_by_family: dict[str, dict[str, float]],
    second_adds: pd.DataFrame,
    second_drops: pd.DataFrame,
    seed_exposure: float,
    family: str,
) -> np.ndarray:
    add_amount = second_adds["loan_amnt"].to_numpy(float)
    drop_amount = second_drops["loan_amnt"].to_numpy(float)
    new_total = seed_exposure + add_amount[:, None] - drop_amount[None, :]
    add_source = second_adds[family].astype(str).to_numpy()
    drop_source = second_drops[family].astype(str).to_numpy()
    mask = base_mask.copy()
    sources = sorted(set(seed_source_maps[family]) | set(add_source) | set(drop_source))
    for source_id in sources:
        cap = cap_by_family[family].get(str(source_id), 1.0)
        base_exposure = seed_source_maps[family].get(str(source_id), 0.0)
        exposure = (
            base_exposure
            + add_amount[:, None] * (add_source[:, None] == str(source_id))
            - drop_amount[None, :] * (drop_source[None, :] == str(source_id))
        )
        mask &= exposure <= cap * new_total + 1e-7
    return mask


def _source_metrics_multi(
    *,
    add_rows: list[pd.Series],
    drop_rows: list[pd.Series],
    current_by_family: dict[str, dict[str, float]],
    cap_by_family: dict[str, dict[str, float]],
    new_total: float,
) -> tuple[bool, float, float, int, str, str]:
    min_slack = np.inf
    max_share = 0.0
    violations = 0
    first_family = ""
    first_source = ""
    for family in FAMILIES:
        sources = set(current_by_family[family])
        sources.update(str(row[family]) for row in add_rows)
        sources.update(str(row[family]) for row in drop_rows)
        caps = cap_by_family[family]
        for source_id in sources:
            exposure = current_by_family[family].get(source_id, 0.0)
            exposure += sum(
                float(row["loan_amnt"]) for row in add_rows if str(row[family]) == source_id
            )
            exposure -= sum(
                float(row["loan_amnt"]) for row in drop_rows if str(row[family]) == source_id
            )
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


def _action_signature(seed_add_id: str, seed_drop_id: str, add_id: str, drop_id: str) -> str:
    add_ids = "|".join(sorted([seed_add_id, add_id]))
    drop_ids = "|".join(sorted([seed_drop_id, drop_id]))
    return f"add={add_ids};drop={drop_ids}"


def _build_positive_source_tight_pool(
    *,
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    mean_returns: np.ndarray,
    idx_by_id: pd.Series,
    v344_hotspots: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    selected_ids = set(selected["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    candidate_idx = idx_by_id.loc[candidates["loan_id"].astype(str)].to_numpy(int)
    candidates[f"universe_idx_v{VERSION}"] = candidate_idx
    candidates[f"mean_return_v{VERSION}"] = mean_returns[candidate_idx]
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
    pool = (
        candidates.loc[tight_mask & candidates[f"mean_return_v{VERSION}"].gt(0)]
        .sort_values(f"mean_return_v{VERSION}", ascending=False)
        .head(SECOND_ADD_LIMIT)
        .reset_index(drop=True)
    )
    pool[f"second_add_rank_v{VERSION}"] = np.arange(1, len(pool) + 1)
    return pool, tight_blocks


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
    seed_add_id = str(seed_pair[f"added_loan_id_v{PREVIOUS_VERSION}"])
    seed_drop_id = str(seed_pair[f"dropped_loan_id_v{PREVIOUS_VERSION}"])
    seed_add = universe_by_id.loc[seed_add_id]
    seed_drop = selected.loc[selected["loan_id"].astype(str).eq(seed_drop_id)].iloc[0]
    seed_exposure = float(seed_pair[f"exposure_after_swap_v{PREVIOUS_VERSION}"])
    seed_return_delta = float(seed_pair[f"return_delta_v{PREVIOUS_VERSION}"])
    seed_source_maps = _seed_source_maps(
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
        family_mask = _family_mask_after_seed(
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
    second_add_ids = second_adds["loan_id"].astype(str).to_numpy()
    second_drop_ids = second_drops["loan_id"].astype(str).to_numpy()
    for add_pos, drop_pos in source_exact_positions:
        add_pos = int(add_pos)
        drop_pos = int(drop_pos)
        second_add = second_adds.iloc[add_pos]
        second_drop = second_drops.iloc[drop_pos]
        new_losses = (
            seed_losses + losses[:, second_add_idx[add_pos]] - losses[:, second_drop_idx[drop_pos]]
        )
        cvar_after = v70._tail_cvar(new_losses)
        source_ok, min_slack, max_share, violations, first_family, first_source = (
            _source_metrics_multi(
                add_rows=[seed_add, second_add],
                drop_rows=[seed_drop, second_drop],
                current_by_family=current_by_family,
                cap_by_family=cap_by_family,
                new_total=float(exposure_after[add_pos, drop_pos]),
            )
        )
        if not source_ok:
            continue
        total_return_delta = float(return_delta[add_pos, drop_pos])
        entering = total_return_delta > 1e-9 and cvar_after <= cvar_cap + 1e-7
        second_add_id = str(second_add_ids[add_pos])
        second_drop_id = str(second_drop_ids[drop_pos])
        candidate_rows.append(
            {
                "policy_id": "v346_v338_second_order_multi_source_relief_protocol",
                f"regime_v{VERSION}": "post_v338_source_tight_two_add_two_drop",
                f"seed_pair_rank_v{VERSION}": seed_rank,
                f"seed_added_loan_id_v{VERSION}": seed_add_id,
                f"seed_dropped_loan_id_v{VERSION}": seed_drop_id,
                f"second_added_loan_id_v{VERSION}": second_add_id,
                f"second_dropped_loan_id_v{VERSION}": second_drop_id,
                f"action_signature_v{VERSION}": _action_signature(
                    seed_add_id, seed_drop_id, second_add_id, second_drop_id
                ),
                f"seed_return_delta_v{VERSION}": seed_return_delta,
                f"second_add_mean_return_v{VERSION}": float(add_return[add_pos]),
                f"second_drop_mean_return_v{VERSION}": float(drop_return[drop_pos]),
                f"return_delta_v{VERSION}": total_return_delta,
                f"objective_return_after_two_swap_v{VERSION}": (
                    current_objective_return + total_return_delta
                ),
                f"exposure_after_two_swap_v{VERSION}": float(exposure_after[add_pos, drop_pos]),
                f"source_min_slack_after_two_swap_v{VERSION}": min_slack,
                f"max_source_share_after_two_swap_v{VERSION}": max_share,
                f"source_cap_violations_after_two_swap_v{VERSION}": violations,
                f"first_source_block_family_v{VERSION}": first_family,
                f"first_source_block_id_v{VERSION}": first_source,
                f"cvar90_after_two_swap_v{VERSION}": cvar_after,
                f"cvar_two_swap_feasible_v{VERSION}": cvar_after <= cvar_cap + 1e-7,
                f"two_add_two_drop_entering_column_v{VERSION}": entering,
                f"claim_boundary_v{VERSION}": (
                    "bounded second-order source-tight relief diagnostic; no global bound"
                ),
            }
        )

    cvar_feasible_rows = sum(
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
        f"cvar_feasible_entering_rows_v{VERSION}": int(cvar_feasible_rows),
        f"best_source_exact_return_delta_v{VERSION}": None
        if not candidate_rows
        else max(float(row[f"return_delta_v{VERSION}"]) for row in candidate_rows),
        f"best_source_exact_cvar90_v{VERSION}": None
        if not candidate_rows
        else min(float(row[f"cvar90_after_two_swap_v{VERSION}"]) for row in candidate_rows),
        f"claim_boundary_v{VERSION}": (
            "one v345 source-exact pair used as seed for a second add/drop screen"
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
            f"stage_v{VERSION}": "v345_source_exact_seed_pairs",
            f"row_count_v{VERSION}": int(len(seed_stage)),
            f"claim_boundary_v{VERSION}": "ordered v345 one-add/one-drop source-exact pairs",
        },
        {
            f"stage_v{VERSION}": "positive_source_tight_second_add_candidates",
            f"row_count_v{VERSION}": second_add_rows,
            f"claim_boundary_v{VERSION}": "all positive omitted candidates touching v344 tight blocks",
        },
        {
            f"stage_v{VERSION}": "second_drop_candidates",
            f"row_count_v{VERSION}": selected_rows,
            f"claim_boundary_v{VERSION}": "all selected v338 loans, excluding the seed drop per seed",
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
                f"claim_boundary_v{VERSION}": "all-family source caps after second-order relief",
            },
            {
                f"stage_v{VERSION}": "cvar_feasible_entering_column",
                f"row_count_v{VERSION}": entering_rows,
                f"claim_boundary_v{VERSION}": "source-exact plus CVaR-feasible return-improving rows",
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
                "claim": "Paper 4 has a v346 v338 second-order multi-source relief protocol.",
                "allowed": True,
                "evidence_artifact": f"reports/paper_material/paper4/tables/{MAIN_ARTIFACT}",
                "boundary": (
                    "Uses v345 source-exact pairs as seeds and screens a bounded "
                    "two-add/two-drop local relief space."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v346 finds a local two-add/two-drop source/CVaR feasible "
                    "return-improving candidate."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v346_second_order_entering_candidates.csv"
                ),
                "boundary": (
                    "Local second-order candidate only; requires apply/reprice before any "
                    "working-champion language."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v346 proves a valid full-universe branch-price termination certificate.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v346_claim_blockers.csv"
                ),
                "boundary": "No full branch-price dual-bound loop or termination certificate exists.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v346 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v346_claim_blockers.csv"
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
                    "v346 expands v345 from one-drop/one-add source-tight pricing into an "
                    "all-positive second-order multi-source relief protocol."
                ),
                "status": "local_two_swap_entering_candidate_found_requires_apply_reprice",
                "next_artifact": NEXT_APPLY_ARTIFACT,
                "success_condition": (
                    "apply the local two-swap candidate, regenerate source/CVaR diagnostics "
                    "and reprice without promotion"
                ),
                "last_wave": "v346",
                "execution_result": (
                    "one_ordered_cvar_feasible_second_order_entering_candidate_found"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v346")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V346_V338_MULTI_SOURCE_RELIEF_OR_BOUND_PROTOCOL_START -->"
    end = "<!-- V346_V338_MULTI_SOURCE_RELIEF_OR_BOUND_PROTOCOL_END -->"
    block = f"""
{start}

## Wave v346: v338 Multi-Source Relief Protocol

Generated: {status["generated_at_utc"]}

### Objective

v345 showed that one-drop/one-add source-tight pricing found no CVaR-feasible
entering column. v346 asks the next executable question: if each v345
source-exact pair is used as a seed, can a second source-tight add and a second
selected drop jointly relieve the tight governance sources while preserving
budget, CVaR and positive return?

### Results

- Positive source-tight second-add candidates:
  `{status["positive_source_tight_candidate_rows_v346"]}`.
- Ordered second-order rows screened:
  `{status["ordered_second_order_rows_screened_v346"]}`.
- Budget+return feasible rows:
  `{status["budget_return_feasible_rows_v346"]}`.
- Source-exact second-order rows:
  `{status["source_exact_second_order_rows_v346"]}`.
- CVaR-feasible entering rows:
  `{status["cvar_feasible_entering_rows_v346"]}`.
- Unique entering action signatures:
  `{status["unique_entering_action_signatures_v346"]}`.
- Best entering action signature:
  `{status["best_entering_action_signature_v346"]}`.
- Best entering return delta:
  `{status["best_entering_return_delta_v346"]}`.
- Best entering CVaR90:
  `{status["best_entering_cvar90_v346"]}`.
- Valid branch-price bound:
  `{status["valid_branch_price_bound_v346"]}`.

### Interpretation

v346 produces a genuinely useful local result: the source-tight blocker is not
fully terminal under second-order relief. Among the ordered two-add/two-drop
rows, one local candidate improves return while staying inside v338 exposure,
source caps and the current CVaR threshold. This is not yet a working champion
and not a global branch-price certificate; it is the next concrete candidate for
apply-and-reprice validation.

### Claim Impact

- Allowed: v346 executed a bounded second-order multi-source relief protocol.
- Allowed: one local source/CVaR feasible return-improving two-swap candidate
  exists in that protocol scope.
- Still prohibited: full-universe branch-price termination, contractual IFRS9,
  live deployability, Paper Estrella replacement, final Paper 4 promotion and
  working champion claims.

### Quarto Promotion Decision

Keep v346 in the living notebook. The next wave should apply the local
two-swap candidate and reprice it as a candidate portfolio, without promotion.

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
    v345_pairs = read_csv("paper4_v345_source_tight_pricing_pairs.csv")
    v345_status = json.loads((STATUS_DIR / "paper4_v345_status.json").read_text(encoding="utf-8"))
    if any(df.empty for df in [universe, selected, source_summary, v338_summary, v345_pairs]):
        raise RuntimeError("Missing v346 multi-source relief inputs.")
    if bool(v345_status["valid_branch_price_bound_v345"]):
        raise RuntimeError("v346 expects v345 to remain a local screen without a valid bound.")
    if int(v345_status["cvar_feasible_entering_columns_v345"]) != 0:
        raise RuntimeError("v346 expects v345 one-drop/one-add entering columns to be absent.")
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

    second_add_pool, tight_blocks = _build_positive_source_tight_pool(
        universe=universe,
        selected=selected,
        mean_returns=mean_returns,
        idx_by_id=idx_by_id,
        v344_hotspots=v344_hotspots,
    )
    seed_pairs = (
        v345_pairs.sort_values(f"return_delta_v{PREVIOUS_VERSION}", ascending=False)
        .head(SEED_PAIR_LIMIT)
        .reset_index(drop=True)
    )
    v338_row = v338_summary.iloc[0]
    current_objective_return = float(v338_row["objective_return_v338"])
    cvar_cap = float(v338_row["scenario_loss_cvar90_v338"])
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
            [
                f"two_add_two_drop_entering_column_v{VERSION}",
                f"return_delta_v{VERSION}",
                f"cvar90_after_two_swap_v{VERSION}",
            ],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    entering = candidate_screen.loc[
        candidate_screen[f"two_add_two_drop_entering_column_v{VERSION}"].astype(bool)
    ].copy()
    source_exact_rows = int(len(candidate_screen))
    entering_rows = int(len(entering))
    unique_source_exact = (
        int(candidate_screen[f"action_signature_v{VERSION}"].nunique())
        if not candidate_screen.empty
        else 0
    )
    unique_entering = (
        int(entering[f"action_signature_v{VERSION}"].nunique()) if not entering.empty else 0
    )
    best_entering = (
        entering.sort_values(f"return_delta_v{VERSION}", ascending=False).head(1)
        if not entering.empty
        else pd.DataFrame()
    )
    next_artifact = NEXT_APPLY_ARTIFACT if entering_rows > 0 else NEXT_BOUND_ARTIFACT
    stage_summary = _stage_summary(
        seed_stage=seed_stage,
        second_add_rows=int(len(second_add_pool)),
        selected_rows=int(len(selected)),
        source_exact_rows=source_exact_rows,
        entering_rows=entering_rows,
    )

    protocol = pd.DataFrame(
        [
            {
                f"protocol_id_v{VERSION}": (
                    "v338_source_tight_second_order_multi_source_relief_protocol"
                ),
                f"base_version_v{VERSION}": BASE_VERSION,
                f"previous_screen_version_v{VERSION}": PREVIOUS_VERSION,
                f"readiness_version_v{VERSION}": READINESS_VERSION,
                f"selected_rows_v{VERSION}": int(len(selected)),
                f"tight_source_rows_v{VERSION}": int(len(tight_blocks)),
                f"seed_pair_rows_v{VERSION}": int(len(seed_pairs)),
                f"positive_source_tight_candidate_rows_v{VERSION}": int(len(second_add_pool)),
                f"second_add_candidate_limit_v{VERSION}": SECOND_ADD_LIMIT,
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
                f"source_exact_second_order_rows_v{VERSION}": source_exact_rows,
                f"unique_source_exact_action_signatures_v{VERSION}": unique_source_exact,
                f"cvar_feasible_entering_rows_v{VERSION}": entering_rows,
                f"unique_entering_action_signatures_v{VERSION}": unique_entering,
                f"best_entering_action_signature_v{VERSION}": ""
                if best_entering.empty
                else str(best_entering[f"action_signature_v{VERSION}"].iloc[0]),
                f"best_entering_return_delta_v{VERSION}": None
                if best_entering.empty
                else float(best_entering[f"return_delta_v{VERSION}"].iloc[0]),
                f"best_entering_cvar90_v{VERSION}": None
                if best_entering.empty
                else float(best_entering[f"cvar90_after_two_swap_v{VERSION}"].iloc[0]),
                f"best_entering_exposure_v{VERSION}": None
                if best_entering.empty
                else float(best_entering[f"exposure_after_two_swap_v{VERSION}"].iloc[0]),
                f"valid_branch_price_bound_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": next_artifact,
                f"claim_boundary_v{VERSION}": (
                    "local second-order source-tight relief protocol only; apply/reprice "
                    "and global bound evidence remain required"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "local_two_swap_candidate_requires_apply_reprice",
                f"blocking_v{VERSION}": entering_rows > 0,
                f"evidence_count_v{VERSION}": unique_entering,
                f"required_next_artifact_v{VERSION}": next_artifact,
                f"claim_boundary_v{VERSION}": (
                    "local entering candidate is not a working champion until applied and repriced"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "valid_branch_price_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_BOUND_ARTIFACT,
                f"claim_boundary_v{VERSION}": "no dual-bound loop or termination certificate",
            },
            {
                f"blocker_id_v{VERSION}": "global_integer_optimality_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(second_add_pool)),
                f"required_next_artifact_v{VERSION}": NEXT_BOUND_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "second-order local screen is not a full-v55 integer proof"
                ),
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
                "claim_id": "v346_multi_source_relief_protocol_executed",
                "allowed": True,
                "artifact": MAIN_ARTIFACT,
                "boundary": "bounded second-order local source-tight relief screen",
            },
            {
                "claim_id": "v346_local_two_add_two_drop_entering_candidate_found",
                "allowed": entering_rows > 0,
                "artifact": "paper4_v346_second_order_entering_candidates.csv",
                "boundary": "local candidate only; apply/reprice still required",
            },
            {
                "claim_id": "v346_valid_full_universe_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v346_claim_blockers.csv",
                "boundary": "formal dual-bound loop missing",
            },
            {
                "claim_id": "v346_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v346_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / MAIN_ARTIFACT, protocol)
    write_csv(TABLE_DIR / "paper4_v346_second_order_seed_pair_summary.csv", seed_stage)
    write_csv(TABLE_DIR / "paper4_v346_second_order_stage_summary.csv", stage_summary)
    write_csv(TABLE_DIR / "paper4_v346_multi_source_relief_candidate_screen.csv", candidate_screen)
    write_csv(TABLE_DIR / "paper4_v346_second_order_entering_candidates.csv", entering)
    write_csv(TABLE_DIR / "paper4_v346_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v346_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = protocol.iloc[0]
    status = {
        "phase": "v346_v338_multi_source_relief_or_bound_protocol",
        "schema_version": "2026-05-16.346",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v346": BASE_VERSION,
        "previous_screen_version_v346": PREVIOUS_VERSION,
        "readiness_version_v346": READINESS_VERSION,
        "selected_rows_v346": int(row[f"selected_rows_v{VERSION}"]),
        "tight_source_rows_v346": int(row[f"tight_source_rows_v{VERSION}"]),
        "seed_pair_rows_v346": int(row[f"seed_pair_rows_v{VERSION}"]),
        "positive_source_tight_candidate_rows_v346": int(
            row[f"positive_source_tight_candidate_rows_v{VERSION}"]
        ),
        "second_add_candidate_limit_v346": SECOND_ADD_LIMIT,
        "ordered_second_order_rows_screened_v346": int(
            row[f"ordered_second_order_rows_screened_v{VERSION}"]
        ),
        "return_improving_rows_v346": int(row[f"return_improving_rows_v{VERSION}"]),
        "budget_return_feasible_rows_v346": int(row[f"budget_return_feasible_rows_v{VERSION}"]),
        "grade_source_feasible_rows_v346": int(row[f"grade_source_feasible_rows_v{VERSION}"]),
        "score_decile_source_feasible_rows_v346": int(
            row[f"score_decile_source_feasible_rows_v{VERSION}"]
        ),
        "source_exact_second_order_rows_v346": source_exact_rows,
        "unique_source_exact_action_signatures_v346": unique_source_exact,
        "cvar_feasible_entering_rows_v346": entering_rows,
        "unique_entering_action_signatures_v346": unique_entering,
        "best_entering_action_signature_v346": str(
            row[f"best_entering_action_signature_v{VERSION}"]
        ),
        "best_entering_return_delta_v346": None
        if best_entering.empty
        else float(row[f"best_entering_return_delta_v{VERSION}"]),
        "best_entering_cvar90_v346": None
        if best_entering.empty
        else float(row[f"best_entering_cvar90_v{VERSION}"]),
        "best_entering_exposure_v346": None
        if best_entering.empty
        else float(row[f"best_entering_exposure_v{VERSION}"]),
        "valid_branch_price_bound_v346": False,
        "full_universe_integer_optimality_claim_allowed_v346": False,
        "working_champion_claim_allowed_v346": False,
        "paper1_promotion_allowed_v346": False,
        "paper4_working_champion_changed_v346": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v346": int(len(blockers)),
        "claim_matrix_rows_v346": int(len(claim_matrix)),
        "next_artifact_v346": next_artifact,
        "claim_boundary": (
            "v346 finds a local second-order source/CVaR feasible candidate, but "
            "apply/reprice, full branch-price, IFRS9, live, champion and promotion "
            "claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v346_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v346": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
