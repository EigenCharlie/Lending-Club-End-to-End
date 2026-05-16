#!/usr/bin/env python3
"""Build Paper 4 v301 source-tight/imputation-repair pricing artifacts."""

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

VERSION = 301
SOURCE_CANDIDATE_VERSION = 295
PROTOCOL_VERSION = 300
NEXT_VERSION = 302
TOP_ROWS = 200
TOP_PROFILE_ROWS = 50


def _observed_proxy_candidate_pool(
    *,
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    v47_panel: pd.DataFrame,
    idx_by_id: pd.Series,
    mean_returns: np.ndarray,
) -> pd.DataFrame:
    selected_ids = set(selected["loan_id"].astype(str))
    observed_ids = set(v47_panel["loan_id"].astype(str)) & set(universe["loan_id"].astype(str))
    candidates = universe.loc[
        universe["loan_id"].astype(str).isin(observed_ids - selected_ids)
    ].copy()
    candidates = candidates.reset_index(drop=True)
    candidates["loan_id"] = candidates["loan_id"].astype(str)
    candidates[f"universe_idx_v{VERSION}"] = idx_by_id.loc[
        candidates["loan_id"].astype(str)
    ].to_numpy()
    candidates[f"mean_return_v{VERSION}"] = mean_returns[
        candidates[f"universe_idx_v{VERSION}"].to_numpy()
    ]
    candidates[f"proxy_source_v{VERSION}"] = "observed_v47_proxy"
    return candidates


def _imputed_drop_pool(
    *,
    selected: pd.DataFrame,
    v299_panel: pd.DataFrame,
    idx_by_id: pd.Series,
    mean_returns: np.ndarray,
) -> pd.DataFrame:
    loan_level = v299_panel.sort_values(["loan_id", "month_index"]).drop_duplicates("loan_id")
    loan_level["loan_id"] = loan_level["loan_id"].astype(str)
    imputed = loan_level.loc[
        loan_level["proxy_source_v299"].astype(str).str.startswith("imputed"),
        ["loan_id", "proxy_source_v299"],
    ].copy()
    drops = selected.merge(imputed, on="loan_id", how="inner").reset_index(drop=True)
    drops["loan_id"] = drops["loan_id"].astype(str)
    drops[f"universe_idx_v{VERSION}"] = idx_by_id.loc[drops["loan_id"].astype(str)].to_numpy()
    drops[f"mean_return_v{VERSION}"] = mean_returns[drops[f"universe_idx_v{VERSION}"].to_numpy()]
    return drops


def _family_source_mask(
    *,
    base_mask: np.ndarray,
    candidates: pd.DataFrame,
    drops: pd.DataFrame,
    selected: pd.DataFrame,
    source_summary: pd.DataFrame,
    new_exposure: np.ndarray,
    family: str,
) -> np.ndarray:
    add_amount = candidates["loan_amnt"].to_numpy(float)
    drop_amount = drops["loan_amnt"].to_numpy(float)
    current_by_source = selected.groupby(family, dropna=False)["loan_amnt"].sum()
    cap_by_source = (
        source_summary.loc[source_summary["source_family"].astype(str).eq(family)]
        .set_index("source_id")[f"cap_share_v{SOURCE_CANDIDATE_VERSION}"]
        .astype(float)
    )
    add_source = candidates[family].astype(str).to_numpy()
    drop_source = drops[family].astype(str).to_numpy()
    family_mask = base_mask.copy()
    for source_id, cap in cap_by_source.items():
        source_id = str(source_id)
        current_source_exposure = float(current_by_source.get(source_id, 0.0))
        add_to_source = add_source == source_id
        drop_from_source = drop_source == source_id
        new_source_exposure = (
            current_source_exposure
            + add_amount[:, None] * add_to_source[:, None]
            - drop_amount[None, :] * drop_from_source[None, :]
        )
        family_mask &= new_source_exposure <= float(cap) * new_exposure + 1e-7
    return family_mask


def _relief_profile(add: pd.Series, drop: pd.Series) -> tuple[str, float, float]:
    grade_a_delta = (float(add["loan_amnt"]) if str(add["grade"]) == "A" else 0.0) - (
        float(drop["loan_amnt"]) if str(drop["grade"]) == "A" else 0.0
    )
    score0_delta = (float(add["loan_amnt"]) if str(add["score_decile"]) == "0" else 0.0) - (
        float(drop["loan_amnt"]) if str(drop["score_decile"]) == "0" else 0.0
    )
    if grade_a_delta < 0 and score0_delta < 0:
        profile = "relieves_grade_A_and_score0"
    elif grade_a_delta < 0:
        profile = "relieves_grade_A_only"
    elif score0_delta < 0:
        profile = "relieves_score0_only"
    elif grade_a_delta > 0 or score0_delta > 0:
        profile = "tight_exposure_increase"
    else:
        profile = "no_tight_relief"
    return profile, grade_a_delta, score0_delta


def _repair_pairs(
    *,
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    drops: pd.DataFrame,
    source_summary: pd.DataFrame,
    losses: np.ndarray,
    idx_by_id: pd.Series,
    objective_return: float,
    exposure_min: float,
    exposure_max: float,
    cvar_cap: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    add_amount = candidates["loan_amnt"].to_numpy(float)
    drop_amount = drops["loan_amnt"].to_numpy(float)
    add_return = candidates[f"mean_return_v{VERSION}"].to_numpy(float)
    drop_return = drops[f"mean_return_v{VERSION}"].to_numpy(float)
    current_exposure = float(selected["loan_amnt"].sum())
    new_exposure = current_exposure + add_amount[:, None] - drop_amount[None, :]
    return_delta = add_return[:, None] - drop_return[None, :]
    budget_mask = (new_exposure >= exposure_min - 1e-7) & (new_exposure <= exposure_max + 1e-7)
    source_mask = budget_mask.copy()
    family_counts: dict[str, int] = {}
    for family in FAMILIES:
        family_mask = _family_source_mask(
            base_mask=budget_mask,
            candidates=candidates,
            drops=drops,
            selected=selected,
            source_summary=source_summary,
            new_exposure=new_exposure,
            family=family,
        )
        family_counts[family] = int(family_mask.sum())
        source_mask &= family_mask

    add_pos, drop_pos = np.where(source_mask)
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    current_losses = losses[:, selected_idx].sum(axis=1)
    cvars = np.array([], dtype=float)
    if len(add_pos):
        candidate_idx = candidates[f"universe_idx_v{VERSION}"].to_numpy()[add_pos]
        drop_idx = drops[f"universe_idx_v{VERSION}"].to_numpy()[drop_pos]
        repaired_losses = current_losses[:, None] + losses[:, candidate_idx] - losses[:, drop_idx]
        cvars = np.array(
            [v70._tail_cvar(repaired_losses[:, col]) for col in range(repaired_losses.shape[1])],
            dtype=float,
        )
    cvar_keep = cvars <= cvar_cap + 1e-7
    feasible_add_pos = add_pos[cvar_keep]
    feasible_drop_pos = drop_pos[cvar_keep]
    feasible_cvars = cvars[cvar_keep]

    rows: list[dict[str, Any]] = []
    for rank_source, (add_idx, drop_idx, cvar_after) in enumerate(
        zip(feasible_add_pos, feasible_drop_pos, feasible_cvars, strict=False),
        start=1,
    ):
        add = candidates.iloc[int(add_idx)]
        drop = drops.iloc[int(drop_idx)]
        profile, grade_a_delta, score0_delta = _relief_profile(add, drop)
        delta = float(return_delta[int(add_idx), int(drop_idx)])
        rows.append(
            {
                f"candidate_pair_index_v{VERSION}": rank_source,
                f"repair_profile_v{VERSION}": profile,
                f"added_loan_id_v{VERSION}": str(add["loan_id"]),
                f"dropped_loan_id_v{VERSION}": str(drop["loan_id"]),
                f"added_loan_amount_v{VERSION}": float(add["loan_amnt"]),
                f"dropped_loan_amount_v{VERSION}": float(drop["loan_amnt"]),
                f"added_mean_return_v{VERSION}": float(add[f"mean_return_v{VERSION}"]),
                f"dropped_mean_return_v{VERSION}": float(drop[f"mean_return_v{VERSION}"]),
                f"return_delta_v{VERSION}": delta,
                f"objective_return_after_repair_v{VERSION}": objective_return + delta,
                f"exposure_after_repair_v{VERSION}": float(
                    new_exposure[int(add_idx), int(drop_idx)]
                ),
                f"cvar90_after_repair_v{VERSION}": float(cvar_after),
                f"imputation_reduction_v{VERSION}": 1,
                f"imputed_proxy_loan_rows_after_v{VERSION}": int(len(drops) - 1),
                f"grade_A_exposure_delta_v{VERSION}": grade_a_delta,
                f"score0_exposure_delta_v{VERSION}": score0_delta,
                f"return_improving_repair_v{VERSION}": delta > 0,
                f"budget_feasible_v{VERSION}": True,
                f"source_feasible_v{VERSION}": True,
                f"cvar_feasible_v{VERSION}": True,
                f"added_proxy_source_v{VERSION}": "observed_v47_proxy",
                f"dropped_proxy_source_v{VERSION}": str(drop["proxy_source_v299"]),
                f"added_grade_v{VERSION}": str(add["grade"]),
                f"dropped_grade_v{VERSION}": str(drop["grade"]),
                f"added_score_decile_v{VERSION}": str(add["score_decile"]),
                f"dropped_score_decile_v{VERSION}": str(drop["score_decile"]),
                f"added_period_v{VERSION}": str(add["period"]),
                f"dropped_period_v{VERSION}": str(drop["period"]),
                f"claim_boundary_v{VERSION}": (
                    "one-swap imputation/source relief candidate only; no branch-price termination"
                ),
            }
        )
    repair_pairs = pd.DataFrame(rows)
    if repair_pairs.empty:
        top_pairs = repair_pairs
    else:
        top_overall = repair_pairs.sort_values(f"return_delta_v{VERSION}", ascending=False).head(
            TOP_ROWS
        )
        top_by_profile = (
            repair_pairs.sort_values(f"return_delta_v{VERSION}", ascending=False)
            .groupby(f"repair_profile_v{VERSION}", group_keys=False)
            .head(TOP_PROFILE_ROWS)
        )
        top_pairs = (
            pd.concat([top_overall, top_by_profile], ignore_index=True)
            .drop_duplicates(
                [f"added_loan_id_v{VERSION}", f"dropped_loan_id_v{VERSION}"],
                keep="first",
            )
            .sort_values(f"return_delta_v{VERSION}", ascending=False)
            .reset_index(drop=True)
        )
        top_pairs[f"repair_rank_v{VERSION}"] = range(1, len(top_pairs) + 1)
    stage_summary = pd.DataFrame(
        [
            {
                f"stage_v{VERSION}": "observed_proxy_candidate_rows",
                f"pair_rows_v{VERSION}": int(len(candidates)),
                f"claim_boundary_v{VERSION}": "observed-v47 proxy additions outside v295 only",
            },
            {
                f"stage_v{VERSION}": "imputed_selected_drop_rows",
                f"pair_rows_v{VERSION}": int(len(drops)),
                f"claim_boundary_v{VERSION}": "v295 selected loans with imputed proxy source",
            },
            {
                f"stage_v{VERSION}": "all_observed_drop_imputed_pair_rows",
                f"pair_rows_v{VERSION}": int(len(candidates) * len(drops)),
                f"claim_boundary_v{VERSION}": "one-drop/one-add imputation repair scope",
            },
            {
                f"stage_v{VERSION}": "budget_feasible_pair_rows",
                f"pair_rows_v{VERSION}": int(budget_mask.sum()),
                f"claim_boundary_v{VERSION}": "budget range only",
            },
            {
                f"stage_v{VERSION}": "source_exact_pair_rows",
                f"pair_rows_v{VERSION}": int(source_mask.sum()),
                f"claim_boundary_v{VERSION}": "all v295 source caps exact screen",
            },
            {
                f"stage_v{VERSION}": "cvar_feasible_repair_pair_rows",
                f"pair_rows_v{VERSION}": int(len(repair_pairs)),
                f"claim_boundary_v{VERSION}": "CVaR90 no-worse-than-v295 screen",
            },
            {
                f"stage_v{VERSION}": "return_improving_repair_pair_rows",
                f"pair_rows_v{VERSION}": int(
                    repair_pairs[f"return_improving_repair_v{VERSION}"].sum()
                )
                if not repair_pairs.empty
                else 0,
                f"claim_boundary_v{VERSION}": "return-improving imputation repair candidates",
            },
            {
                f"stage_v{VERSION}": "tight_relief_feasible_pair_rows",
                f"pair_rows_v{VERSION}": int(
                    repair_pairs[f"repair_profile_v{VERSION}"]
                    .isin(
                        [
                            "relieves_grade_A_and_score0",
                            "relieves_grade_A_only",
                            "relieves_score0_only",
                        ]
                    )
                    .sum()
                )
                if not repair_pairs.empty
                else 0,
                f"claim_boundary_v{VERSION}": "feasible repairs that reduce at least one tight source",
            },
        ]
    )
    for family, count in family_counts.items():
        stage_summary = pd.concat(
            [
                stage_summary,
                pd.DataFrame(
                    [
                        {
                            f"stage_v{VERSION}": f"{family}_source_feasible_pair_rows",
                            f"pair_rows_v{VERSION}": count,
                            f"claim_boundary_v{VERSION}": f"{family} cap screen only",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return repair_pairs, top_pairs, stage_summary


def _profile_summary(repair_pairs: pd.DataFrame) -> pd.DataFrame:
    if repair_pairs.empty:
        return pd.DataFrame(
            columns=[
                f"repair_profile_v{VERSION}",
                f"pair_rows_v{VERSION}",
                f"return_improving_pair_rows_v{VERSION}",
                f"best_return_delta_v{VERSION}",
                f"worst_return_delta_v{VERSION}",
                f"min_cvar90_after_repair_v{VERSION}",
                f"claim_boundary_v{VERSION}",
            ]
        )
    out = (
        repair_pairs.groupby(f"repair_profile_v{VERSION}", dropna=False)
        .agg(
            **{
                f"pair_rows_v{VERSION}": (f"repair_profile_v{VERSION}", "size"),
                f"return_improving_pair_rows_v{VERSION}": (
                    f"return_improving_repair_v{VERSION}",
                    "sum",
                ),
                f"best_return_delta_v{VERSION}": (f"return_delta_v{VERSION}", "max"),
                f"worst_return_delta_v{VERSION}": (f"return_delta_v{VERSION}", "min"),
                f"min_cvar90_after_repair_v{VERSION}": (f"cvar90_after_repair_v{VERSION}", "min"),
            }
        )
        .reset_index()
    )
    out[f"claim_boundary_v{VERSION}"] = (
        "profile-level one-swap repair summary; no branch-price/global certificate"
    )
    return out.sort_values(f"best_return_delta_v{VERSION}", ascending=False).reset_index(drop=True)


def _candidate_rows(top_pairs: pd.DataFrame) -> pd.DataFrame:
    if top_pairs.empty:
        return pd.DataFrame()
    best_return = top_pairs.sort_values(f"return_delta_v{VERSION}", ascending=False).head(1).copy()
    tight = top_pairs.loc[
        top_pairs[f"repair_profile_v{VERSION}"].isin(
            ["relieves_grade_A_and_score0", "relieves_grade_A_only", "relieves_score0_only"]
        )
    ]
    frames = []
    best_return[f"repair_candidate_id_v{VERSION}"] = "best_return_imputation_repair"
    frames.append(best_return)
    if not tight.empty:
        best_tight = tight.sort_values(f"return_delta_v{VERSION}", ascending=False).head(1).copy()
        best_tight[f"repair_candidate_id_v{VERSION}"] = "best_tight_source_relief_repair"
        frames.append(best_tight)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(f"repair_candidate_id_v{VERSION}", keep="first")


def _candidate_allocations(
    *,
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    drops: pd.DataFrame,
    candidate_rows: pd.DataFrame,
) -> pd.DataFrame:
    allocation_frames: list[pd.DataFrame] = []
    keep_cols = ["loan_id", "loan_amnt", *FAMILIES]
    candidate_by_id = candidates.set_index("loan_id", drop=False)
    drop_by_id = drops.set_index("loan_id", drop=False)
    for row in candidate_rows.itertuples(index=False):
        candidate_id = getattr(row, f"repair_candidate_id_v{VERSION}")
        add_id = str(getattr(row, f"added_loan_id_v{VERSION}"))
        drop_id = str(getattr(row, f"dropped_loan_id_v{VERSION}"))
        repaired = selected.loc[~selected["loan_id"].astype(str).eq(drop_id), keep_cols].copy()
        add = candidate_by_id.loc[add_id, keep_cols].to_frame().T
        repaired = pd.concat([repaired, add], ignore_index=True)
        repaired[f"repair_candidate_id_v{VERSION}"] = candidate_id
        repaired[f"selected_v{VERSION}"] = True
        repaired[f"repair_action_v{VERSION}"] = "kept_from_v295"
        repaired.loc[repaired["loan_id"].astype(str).eq(add_id), f"repair_action_v{VERSION}"] = (
            f"added_observed_proxy_for_{drop_id}"
        )
        repaired[f"portfolio_label_v{VERSION}"] = candidate_id
        repaired[f"claim_boundary_v{VERSION}"] = (
            "v301 one-swap repair candidate allocation; not a promoted portfolio"
        )
        repaired[f"dropped_loan_id_v{VERSION}"] = drop_id
        repaired[f"dropped_proxy_source_v{VERSION}"] = str(
            drop_by_id.loc[drop_id, "proxy_source_v299"]
        )
        allocation_frames.append(repaired)
    return pd.concat(allocation_frames, ignore_index=True) if allocation_frames else pd.DataFrame()


def _candidate_source_summary(
    *,
    allocations: pd.DataFrame,
    universe: pd.DataFrame,
    source_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if allocations.empty:
        return pd.DataFrame()
    cap_lookup = {}
    for family in FAMILIES:
        family_caps = source_summary.loc[source_summary["source_family"].astype(str).eq(family)]
        cap_lookup[family] = {
            str(row["source_id"]): float(row[f"cap_share_v{SOURCE_CANDIDATE_VERSION}"])
            for _, row in family_caps.iterrows()
        }
    for candidate_id, portfolio in allocations.groupby(f"repair_candidate_id_v{VERSION}"):
        exposure = float(portfolio["loan_amnt"].sum())
        for family in FAMILIES:
            by_source = portfolio.groupby(family, dropna=False)["loan_amnt"].sum()
            for source_id in sorted(universe[family].dropna().astype(str).unique()):
                source_exposure = float(by_source.get(source_id, 0.0))
                share = source_exposure / max(exposure, 1.0)
                cap = float(cap_lookup[family].get(source_id, 1.0))
                rows.append(
                    {
                        f"repair_candidate_id_v{VERSION}": candidate_id,
                        "source_family": family,
                        "source_id": source_id,
                        f"cap_share_v{VERSION}": cap,
                        f"source_exposure_v{VERSION}": source_exposure,
                        f"source_share_v{VERSION}": share,
                        f"source_slack_v{VERSION}": cap - share,
                        f"source_cap_violated_v{VERSION}": share > cap + 1e-7,
                        f"claim_boundary_v{VERSION}": (
                            "v301 repair source diagnostic only; no branch-price proof"
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
                "claim": "Paper 4 has a v301 observed-proxy imputation repair screen.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v301_source_tight_branch_price_pricing_or_imputation_repair.csv"
                ),
                "boundary": "One-drop/one-add repair screen only; no promotion.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v301 finds feasible one-swap imputation repair candidates.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v301_observed_proxy_imputation_repair_top_candidates.csv"
                ),
                "boundary": (
                    "Candidates reduce one imputed proxy row under exact budget/source/CVaR screens; "
                    "best exact-source repair has a small return cost."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v301 finds a return-improving one-swap imputation repair candidate.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v301_claim_blockers.csv"
                ),
                "boundary": "Exact all-source caps remove all return-improving one-swap repairs.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v301 finds feasible tight-source relief candidates.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v301_tight_source_relief_profile.csv"
                ),
                "boundary": "Feasible source-relief screen only; best tight relief has return cost.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v301 proves a valid full-universe branch-price bound.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v301_claim_blockers.csv"
                ),
                "boundary": "No dual-bound loop or branch-price termination certificate exists.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v301 resolves contractual IFRS9 or live deployability for v295.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v301_claim_blockers.csv"
                ),
                "boundary": "Best repair still leaves imputed rows and no external online holdout.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v301 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v301_claim_blockers.csv"
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
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v301 screens observed-proxy one-swap replacements for v295 imputed loans "
                    "and separates pure imputation repair from tight-source relief."
                ),
                "status": "observed_proxy_imputation_repair_screen_executed",
                "next_artifact": (
                    f"paper4_v{NEXT_VERSION}_apply_v301_repair_or_multi_swap_imputation_frontier.csv"
                ),
                "success_condition": (
                    "apply the best bounded repair or expand to a multi-swap imputation frontier "
                    "without promoting"
                ),
                "last_wave": "v301",
                "execution_result": "feasible_imputation_repair_found_with_small_return_cost_tight_relief_feasible",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v301")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V301_SOURCE_TIGHT_IMPUTATION_REPAIR_START -->"
    end = "<!-- V301_SOURCE_TIGHT_IMPUTATION_REPAIR_END -->"
    block = f"""
{start}

## Wave v301: Source-Tight / Imputation Repair Pricing

Generated: {status["generated_at_utc"]}

### Objective

v300 localized two blockers: v295 still needs source-tight branch-price evidence
and has 76 selected loans with imputed IFRS9-inspired cashflow proxies. v301
executes a bounded one-drop/one-add repair screen: drop one imputed selected
loan, add one observed-v47-proxy loan, and require budget, source and CVaR
feasibility.

### Results

- Observed proxy add candidates: `{status["observed_proxy_candidate_rows_v301"]}`.
- Imputed selected drop rows: `{status["imputed_selected_drop_rows_v301"]}`.
- Pair rows screened: `{status["total_pair_rows_v301"]}`.
- Budget-feasible pair rows: `{status["budget_feasible_pair_rows_v301"]}`.
- Source-exact pair rows: `{status["source_exact_pair_rows_v301"]}`.
- CVaR-feasible repair rows: `{status["cvar_feasible_repair_pair_rows_v301"]}`.
- Return-improving repair rows: `{status["return_improving_repair_pair_rows_v301"]}`.
- Tight-relief feasible rows: `{status["tight_relief_feasible_pair_rows_v301"]}`.
- Best repair return delta: `{status["best_repair_return_delta_v301"]}`.
- Best repair imputed rows after one swap:
  `{status["best_repair_imputed_proxy_loan_rows_after_v301"]}`.
- Best tight-relief return delta: `{status["best_tight_relief_return_delta_v301"]}`.
- Valid branch-price bound: `{status["valid_branch_price_bound_v301"]}`.

### Interpretation

v301 is the first candidate-specific data-quality repair signal after v299:
there are feasible swaps that reduce imputed cashflow dependence while
respecting budget, all exact source caps and the v295 CVaR cap. Once the exact
source caps are enforced, no one-swap repair improves return; the best repair
has only a small return cost. Tight-source relief is also feasible, but still
not return-improving, so this is evidence for a future frontier, not a promotion.

### Claim Impact

- Allowed: observed-proxy imputation repair screen; bounded one-swap feasible
  repair candidates; feasible tight-source relief profile.
- Still prohibited: return-improving one-swap repair claim under exact source
  caps.
- Still prohibited: full-universe branch-price bound, contractual IFRS9, live
  deployability, Paper Estrella replacement, final Paper 4 promotion and
  working champion claims.

### Quarto Promotion Decision

Keep v301 in the living notebook. The next wave should apply the bounded repair
or expand to a multi-swap imputation frontier while preserving claim guards.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v295_broader_multi_swap_allocations.parquet").reset_index(
        drop=True
    )
    selected["loan_id"] = selected["loan_id"].astype(str)
    source_summary = read_csv("paper4_v295_broader_source_summary.csv")
    v295_summary = read_csv("paper4_v295_broader_multi_swap_or_global_gap_probe.csv")
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    v299_panel = read_parquet("paper4_v299_v295_cashflow_proxy_panel.parquet")
    v300_status = json.loads((STATUS_DIR / "paper4_v300_status.json").read_text(encoding="utf-8"))
    if any(
        df.empty for df in [universe, selected, source_summary, v295_summary, v47_panel, v299_panel]
    ):
        raise RuntimeError("Missing v55, v295, v47, v299 or source inputs for v301.")
    if int(v300_status["source_tight_rows_v300"]) != 2:
        raise RuntimeError("v301 expects v300 to identify two source-tight rows.")

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    candidates = _observed_proxy_candidate_pool(
        universe=universe,
        selected=selected,
        v47_panel=v47_panel,
        idx_by_id=idx_by_id,
        mean_returns=mean_returns,
    )
    drops = _imputed_drop_pool(
        selected=selected,
        v299_panel=v299_panel,
        idx_by_id=idx_by_id,
        mean_returns=mean_returns,
    )
    v295_row = v295_summary.iloc[0]
    objective_return = float(v295_row[f"objective_return_v{SOURCE_CANDIDATE_VERSION}"])
    cvar_cap = float(v295_row[f"scenario_loss_cvar90_v{SOURCE_CANDIDATE_VERSION}"])
    exposure_min = float(v295_row[f"exposure_min_v{SOURCE_CANDIDATE_VERSION}"])
    exposure_max = float(v295_row[f"exposure_max_v{SOURCE_CANDIDATE_VERSION}"])
    repair_pairs, top_pairs, stage_summary = _repair_pairs(
        universe=universe,
        selected=selected,
        candidates=candidates,
        drops=drops,
        source_summary=source_summary,
        losses=losses,
        idx_by_id=idx_by_id,
        objective_return=objective_return,
        exposure_min=exposure_min,
        exposure_max=exposure_max,
        cvar_cap=cvar_cap,
    )
    profile_summary = _profile_summary(repair_pairs)
    candidate_rows = _candidate_rows(top_pairs)
    allocations = _candidate_allocations(
        selected=selected,
        candidates=candidates,
        drops=drops,
        candidate_rows=candidate_rows,
    )
    repair_source_summary = _candidate_source_summary(
        allocations=allocations,
        universe=universe,
        source_summary=source_summary,
    )

    best = candidate_rows.loc[
        candidate_rows[f"repair_candidate_id_v{VERSION}"].eq("best_return_imputation_repair")
    ].iloc[0]
    tight_candidates = candidate_rows.loc[
        candidate_rows[f"repair_candidate_id_v{VERSION}"].eq("best_tight_source_relief_repair")
    ]
    best_tight = tight_candidates.iloc[0] if not tight_candidates.empty else best
    tight_profiles = [
        "relieves_grade_A_and_score0",
        "relieves_grade_A_only",
        "relieves_score0_only",
    ]
    tight_relief_rows = int(repair_pairs[f"repair_profile_v{VERSION}"].isin(tight_profiles).sum())
    return_improving_tight_rows = int(
        repair_pairs.loc[
            repair_pairs[f"repair_profile_v{VERSION}"].isin(tight_profiles),
            f"return_improving_repair_v{VERSION}",
        ].sum()
    )
    best_repair_imputed_after = int(best[f"imputed_proxy_loan_rows_after_v{VERSION}"])
    valid_branch_price_bound = False
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "full_universe_gap_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v300_status["full_binary_variables_v300"]),
                f"required_next_artifact_v{VERSION}": "future_branch_price_dual_bound_loop",
                f"claim_boundary_v{VERSION}": "v301 is a bounded repair screen, not a global bound",
            },
            {
                f"blocker_id_v{VERSION}": "branch_price_termination_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_branch_price_termination_certificate",
                f"claim_boundary_v{VERSION}": "no dual-priced column-generation termination exists",
            },
            {
                f"blocker_id_v{VERSION}": "residual_cashflow_imputation_after_best_repair",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": best_repair_imputed_after,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_apply_v301_repair_or_multi_swap_imputation_frontier.csv"
                ),
                f"claim_boundary_v{VERSION}": "best one-swap repair still leaves imputed proxy rows",
            },
            {
                f"blocker_id_v{VERSION}": "return_improving_imputation_repair_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    repair_pairs[f"return_improving_repair_v{VERSION}"].sum()
                ),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_apply_v301_repair_or_multi_swap_imputation_frontier.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "no return-improving imputation repair survives exact all-source caps"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "return_improving_tight_relief_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": return_improving_tight_rows,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_apply_v301_repair_or_multi_swap_imputation_frontier.csv"
                ),
                f"claim_boundary_v{VERSION}": "tight-source relief exists but not with positive return",
            },
            {
                f"blocker_id_v{VERSION}": "external_online_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": "future_external_online_holdout",
                f"claim_boundary_v{VERSION}": "v301 does not create external online evidence",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_working_champion_gate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_global_dynamic_promotion_gate",
                f"claim_boundary_v{VERSION}": "working champion replacement remains blocked",
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
                "claim_id": "v301_observed_proxy_imputation_repair_screen_executed",
                "allowed": True,
                "artifact": "paper4_v301_source_tight_branch_price_pricing_or_imputation_repair.csv",
                "boundary": "bounded one-swap repair screen",
            },
            {
                "claim_id": "v301_feasible_imputation_repair_found",
                "allowed": True,
                "artifact": "paper4_v301_observed_proxy_imputation_repair_top_candidates.csv",
                "boundary": "one-swap feasible candidate only",
            },
            {
                "claim_id": "v301_return_improving_imputation_repair_found",
                "allowed": False,
                "artifact": "paper4_v301_claim_blockers.csv",
                "boundary": "no exact-source return-improving one-swap repair",
            },
            {
                "claim_id": "v301_feasible_tight_source_relief_found",
                "allowed": True,
                "artifact": "paper4_v301_tight_source_relief_profile.csv",
                "boundary": "tight relief feasible but not return-improving",
            },
            {
                "claim_id": "v301_valid_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v301_claim_blockers.csv",
                "boundary": "dual-bound/termination missing",
            },
            {
                "claim_id": "v301_contractual_ifrs9_or_live_deployability",
                "allowed": False,
                "artifact": "paper4_v301_claim_blockers.csv",
                "boundary": "residual imputation and no external holdout",
            },
            {
                "claim_id": "v301_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v301_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    summary = pd.DataFrame(
        [
            {
                f"screen_id_v{VERSION}": "source_tight_branch_price_pricing_or_imputation_repair",
                f"source_candidate_version_v{VERSION}": SOURCE_CANDIDATE_VERSION,
                f"protocol_version_v{VERSION}": PROTOCOL_VERSION,
                f"observed_proxy_candidate_rows_v{VERSION}": int(len(candidates)),
                f"imputed_selected_drop_rows_v{VERSION}": int(len(drops)),
                f"total_pair_rows_v{VERSION}": int(len(candidates) * len(drops)),
                f"budget_feasible_pair_rows_v{VERSION}": int(
                    stage_summary.loc[
                        stage_summary[f"stage_v{VERSION}"].eq("budget_feasible_pair_rows"),
                        f"pair_rows_v{VERSION}",
                    ].iloc[0]
                ),
                f"source_exact_pair_rows_v{VERSION}": int(
                    stage_summary.loc[
                        stage_summary[f"stage_v{VERSION}"].eq("source_exact_pair_rows"),
                        f"pair_rows_v{VERSION}",
                    ].iloc[0]
                ),
                f"cvar_feasible_repair_pair_rows_v{VERSION}": int(len(repair_pairs)),
                f"return_improving_repair_pair_rows_v{VERSION}": int(
                    repair_pairs[f"return_improving_repair_v{VERSION}"].sum()
                ),
                f"tight_relief_feasible_pair_rows_v{VERSION}": tight_relief_rows,
                f"return_improving_tight_relief_pair_rows_v{VERSION}": return_improving_tight_rows,
                f"best_repair_added_loan_id_v{VERSION}": str(best[f"added_loan_id_v{VERSION}"]),
                f"best_repair_dropped_loan_id_v{VERSION}": str(best[f"dropped_loan_id_v{VERSION}"]),
                f"best_repair_return_delta_v{VERSION}": float(best[f"return_delta_v{VERSION}"]),
                f"best_repair_objective_return_after_v{VERSION}": float(
                    best[f"objective_return_after_repair_v{VERSION}"]
                ),
                f"best_repair_cvar90_after_v{VERSION}": float(
                    best[f"cvar90_after_repair_v{VERSION}"]
                ),
                f"best_repair_imputed_proxy_loan_rows_after_v{VERSION}": best_repair_imputed_after,
                f"best_tight_relief_added_loan_id_v{VERSION}": str(
                    best_tight[f"added_loan_id_v{VERSION}"]
                ),
                f"best_tight_relief_dropped_loan_id_v{VERSION}": str(
                    best_tight[f"dropped_loan_id_v{VERSION}"]
                ),
                f"best_tight_relief_return_delta_v{VERSION}": float(
                    best_tight[f"return_delta_v{VERSION}"]
                ),
                f"best_tight_relief_cvar90_after_v{VERSION}": float(
                    best_tight[f"cvar90_after_repair_v{VERSION}"]
                ),
                f"best_tight_relief_grade_A_exposure_delta_v{VERSION}": float(
                    best_tight[f"grade_A_exposure_delta_v{VERSION}"]
                ),
                f"best_tight_relief_score0_exposure_delta_v{VERSION}": float(
                    best_tight[f"score0_exposure_delta_v{VERSION}"]
                ),
                f"feasible_imputation_repair_found_v{VERSION}": not repair_pairs.empty,
                f"return_improving_imputation_repair_found_v{VERSION}": bool(
                    repair_pairs[f"return_improving_repair_v{VERSION}"].any()
                ),
                f"source_tight_relief_repair_found_v{VERSION}": tight_relief_rows > 0,
                f"valid_branch_price_bound_v{VERSION}": valid_branch_price_bound,
                f"contractual_ifrs9_claim_allowed_v{VERSION}": False,
                f"strict_live_deployability_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_apply_v301_repair_or_multi_swap_imputation_frontier.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "bounded one-swap repair screen only; no branch-price, IFRS9, live or promotion claim"
                ),
            }
        ]
    )

    write_csv(
        TABLE_DIR / "paper4_v301_source_tight_branch_price_pricing_or_imputation_repair.csv",
        summary,
    )
    write_csv(TABLE_DIR / "paper4_v301_imputation_repair_stage_summary.csv", stage_summary)
    write_csv(
        TABLE_DIR / "paper4_v301_observed_proxy_imputation_repair_top_candidates.csv",
        top_pairs,
    )
    write_csv(TABLE_DIR / "paper4_v301_tight_source_relief_profile.csv", profile_summary)
    write_csv(TABLE_DIR / "paper4_v301_repair_candidate_summary.csv", candidate_rows)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v301_repair_candidate_allocations.parquet", index=False
    )
    write_csv(TABLE_DIR / "paper4_v301_repair_source_summary.csv", repair_source_summary)
    write_csv(TABLE_DIR / "paper4_v301_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v301_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    summary_row = summary.iloc[0]
    status = {
        "phase": "v301_source_tight_branch_price_pricing_or_imputation_repair",
        "schema_version": "2026-05-15.301",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "source_candidate_version_v301": SOURCE_CANDIDATE_VERSION,
        "protocol_version_v301": PROTOCOL_VERSION,
        "observed_proxy_candidate_rows_v301": int(
            summary_row[f"observed_proxy_candidate_rows_v{VERSION}"]
        ),
        "imputed_selected_drop_rows_v301": int(
            summary_row[f"imputed_selected_drop_rows_v{VERSION}"]
        ),
        "total_pair_rows_v301": int(summary_row[f"total_pair_rows_v{VERSION}"]),
        "budget_feasible_pair_rows_v301": int(summary_row[f"budget_feasible_pair_rows_v{VERSION}"]),
        "source_exact_pair_rows_v301": int(summary_row[f"source_exact_pair_rows_v{VERSION}"]),
        "cvar_feasible_repair_pair_rows_v301": int(
            summary_row[f"cvar_feasible_repair_pair_rows_v{VERSION}"]
        ),
        "return_improving_repair_pair_rows_v301": int(
            summary_row[f"return_improving_repair_pair_rows_v{VERSION}"]
        ),
        "tight_relief_feasible_pair_rows_v301": int(
            summary_row[f"tight_relief_feasible_pair_rows_v{VERSION}"]
        ),
        "return_improving_tight_relief_pair_rows_v301": int(
            summary_row[f"return_improving_tight_relief_pair_rows_v{VERSION}"]
        ),
        "top_candidate_rows_v301": int(len(top_pairs)),
        "repair_candidate_summary_rows_v301": int(len(candidate_rows)),
        "repair_candidate_allocation_rows_v301": int(len(allocations)),
        "repair_source_summary_rows_v301": int(len(repair_source_summary)),
        "best_repair_added_loan_id_v301": str(summary_row[f"best_repair_added_loan_id_v{VERSION}"]),
        "best_repair_dropped_loan_id_v301": str(
            summary_row[f"best_repair_dropped_loan_id_v{VERSION}"]
        ),
        "best_repair_return_delta_v301": float(summary_row[f"best_repair_return_delta_v{VERSION}"]),
        "best_repair_objective_return_after_v301": float(
            summary_row[f"best_repair_objective_return_after_v{VERSION}"]
        ),
        "best_repair_cvar90_after_v301": float(summary_row[f"best_repair_cvar90_after_v{VERSION}"]),
        "best_repair_imputed_proxy_loan_rows_after_v301": int(
            summary_row[f"best_repair_imputed_proxy_loan_rows_after_v{VERSION}"]
        ),
        "best_tight_relief_added_loan_id_v301": str(
            summary_row[f"best_tight_relief_added_loan_id_v{VERSION}"]
        ),
        "best_tight_relief_dropped_loan_id_v301": str(
            summary_row[f"best_tight_relief_dropped_loan_id_v{VERSION}"]
        ),
        "best_tight_relief_return_delta_v301": float(
            summary_row[f"best_tight_relief_return_delta_v{VERSION}"]
        ),
        "best_tight_relief_cvar90_after_v301": float(
            summary_row[f"best_tight_relief_cvar90_after_v{VERSION}"]
        ),
        "best_tight_relief_grade_A_exposure_delta_v301": float(
            summary_row[f"best_tight_relief_grade_A_exposure_delta_v{VERSION}"]
        ),
        "best_tight_relief_score0_exposure_delta_v301": float(
            summary_row[f"best_tight_relief_score0_exposure_delta_v{VERSION}"]
        ),
        "feasible_imputation_repair_found_v301": bool(
            summary_row[f"feasible_imputation_repair_found_v{VERSION}"]
        ),
        "return_improving_imputation_repair_found_v301": bool(
            summary_row[f"return_improving_imputation_repair_found_v{VERSION}"]
        ),
        "source_tight_relief_repair_found_v301": bool(
            summary_row[f"source_tight_relief_repair_found_v{VERSION}"]
        ),
        "valid_branch_price_bound_v301": valid_branch_price_bound,
        "strict_live_deployability_claim_allowed_v301": False,
        "contractual_ifrs9_claim_allowed_v301": False,
        "working_champion_claim_allowed_v301": False,
        "paper1_promotion_allowed_v301": False,
        "paper4_working_champion_changed_v301": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v301": int(len(blockers)),
        "claim_matrix_rows_v301": int(len(claim_matrix)),
        "next_artifact_v301": (
            f"paper4_v{NEXT_VERSION}_apply_v301_repair_or_multi_swap_imputation_frontier.csv"
        ),
        "claim_boundary": (
            "v301 finds bounded repair candidates but leaves branch-price, IFRS9, live deployment, "
            "working champion and promotion claims blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v301_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v301": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
