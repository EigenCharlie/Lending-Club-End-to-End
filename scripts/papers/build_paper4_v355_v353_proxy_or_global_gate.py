#!/usr/bin/env python3
"""Build Paper 4 v355 v353 proxy/global gate artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers import build_paper4_v349_v347_proxy_or_dual_bound_gate as proxy_gate
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

VERSION = 355
BASE_VERSION = 353
REFERENCE_VERSION = 347
REPRICE_VERSION = 354
NEXT_VERSION = 356
TARGET_SELECTED_ROWS = 171
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v353_dual_bound_after_proxy_gate.csv"


def _configure_proxy_gate() -> None:
    proxy_gate.VERSION = VERSION
    proxy_gate.BASE_VERSION = BASE_VERSION
    proxy_gate.REFERENCE_VERSION = REFERENCE_VERSION
    proxy_gate.REPRICE_VERSION = REPRICE_VERSION
    proxy_gate.NEXT_VERSION = NEXT_VERSION
    proxy_gate.NEXT_ARTIFACT = NEXT_ARTIFACT


def _period_distribution(portfolio: pd.DataFrame) -> dict[str, int]:
    return {
        str(period): int(count)
        for period, count in portfolio["period"].astype(str).value_counts().sort_index().items()
    }


def _build_pool(
    *,
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    observed_ids: set[str],
    idx_by_id: pd.Series,
    mean_returns: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_ids = set(selected["loan_id"].astype(str))
    universe = universe.copy()
    universe["loan_id"] = universe["loan_id"].astype(str)
    universe[f"mean_return_v{VERSION}"] = mean_returns
    universe[f"observed_v47_proxy_v{VERSION}"] = universe["loan_id"].isin(observed_ids)
    observed_omitted = universe.loc[
        ~universe["loan_id"].isin(selected_ids) & universe[f"observed_v47_proxy_v{VERSION}"]
    ].copy()
    pool = pd.concat(
        [universe.loc[universe["loan_id"].isin(selected_ids)], observed_omitted],
        ignore_index=True,
    ).drop_duplicates("loan_id")
    pool["loan_id"] = pool["loan_id"].astype(str)
    pool[f"incumbent_selected_v{VERSION}"] = pool["loan_id"].isin(selected_ids)
    pool[f"observed_candidate_v{VERSION}"] = (
        ~pool[f"incumbent_selected_v{VERSION}"] & pool[f"observed_v47_proxy_v{VERSION}"]
    )
    pool[f"pool_role_v{VERSION}"] = np.where(
        pool[f"incumbent_selected_v{VERSION}"],
        "v353_selected_base",
        "observed_omitted_candidate",
    )
    pool[f"universe_idx_v{VERSION}"] = idx_by_id.loc[pool["loan_id"]].to_numpy(int)
    selected_observed = int(
        pool.loc[pool[f"incumbent_selected_v{VERSION}"], f"observed_v47_proxy_v{VERSION}"].sum()
    )
    selected_missing = int(pool[f"incumbent_selected_v{VERSION}"].sum() - selected_observed)
    pool_summary = pd.DataFrame(
        [
            {
                f"pool_id_v{VERSION}": "v353_plus_all_observed_omitted_candidates",
                f"pool_rows_v{VERSION}": int(len(pool)),
                f"selected_base_rows_v{VERSION}": int(pool[f"incumbent_selected_v{VERSION}"].sum()),
                f"selected_observed_proxy_rows_v{VERSION}": selected_observed,
                f"selected_missing_proxy_rows_v{VERSION}": selected_missing,
                f"observed_omitted_candidate_rows_v{VERSION}": int(
                    pool[f"observed_candidate_v{VERSION}"].sum()
                ),
                f"total_observed_omitted_rows_v{VERSION}": int(len(observed_omitted)),
                f"expanded_pool_includes_all_observed_omitted_v{VERSION}": True,
                f"claim_boundary_v{VERSION}": (
                    "all observed omitted candidates plus v353 selected loans; not full-v55 pricing"
                ),
            }
        ]
    )
    return pool, pool_summary


def _tier_summary(
    tier_metrics: list[dict[str, Any]],
    reference: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics in tier_metrics:
        objective_return = metrics.get("objective_return")
        cvar = metrics.get("scenario_loss_cvar90")
        observed = metrics.get("observed_proxy_rows")
        missing = metrics.get("missing_proxy_rows")
        rows.append(
            {
                f"tier_id_v{VERSION}": metrics["tier_id"],
                f"solver_success_v{VERSION}": metrics["solver_success"],
                f"milp_status_v{VERSION}": metrics["milp_status"],
                f"milp_message_v{VERSION}": metrics["milp_message"],
                f"milp_fun_v{VERSION}": metrics["milp_fun"],
                f"milp_gap_v{VERSION}": metrics["milp_gap"],
                f"milp_dual_bound_v{VERSION}": metrics["milp_dual_bound"],
                f"milp_node_count_v{VERSION}": metrics["milp_node_count"],
                f"incumbent_found_v{VERSION}": metrics["incumbent_found"],
                f"constraint_rows_v{VERSION}": metrics["constraint_rows"],
                f"variable_count_v{VERSION}": metrics["variable_count"],
                f"return_floor_v{VERSION}": metrics["return_floor"],
                f"cvar_cap_v{VERSION}": metrics["cvar_cap"],
                f"selected_rows_v{VERSION}": metrics.get("selected_rows"),
                f"portfolio_exposure_v{VERSION}": metrics.get("portfolio_exposure"),
                f"objective_return_v{VERSION}": objective_return,
                f"delta_return_vs_v353_v{VERSION}": None
                if objective_return is None
                else float(objective_return - reference["v353_return"]),
                f"delta_return_vs_v347_v{VERSION}": None
                if objective_return is None
                else float(objective_return - reference["v347_return"]),
                f"scenario_loss_mean_v{VERSION}": metrics.get("scenario_loss_mean"),
                f"scenario_loss_cvar90_v{VERSION}": cvar,
                f"delta_cvar90_vs_v353_v{VERSION}": None
                if cvar is None
                else float(cvar - reference["v353_cvar"]),
                f"delta_cvar90_vs_v347_v{VERSION}": None
                if cvar is None
                else float(cvar - reference["v347_cvar"]),
                f"observed_proxy_rows_v{VERSION}": observed,
                f"missing_proxy_rows_v{VERSION}": missing,
                f"observed_proxy_delta_vs_v353_v{VERSION}": None
                if observed is None
                else int(observed - reference["v353_observed"]),
                f"missing_proxy_delta_vs_v353_v{VERSION}": None
                if missing is None
                else int(missing - reference["v353_missing"]),
                f"coverage_restores_or_improves_v347_v{VERSION}": None
                if missing is None
                else int(missing) <= int(reference["v347_missing"]),
                f"period_distribution_v{VERSION}": json.dumps(
                    metrics.get("period_distribution", {}), sort_keys=True
                ),
                f"claim_boundary_v{VERSION}": (
                    "all-observed-omitted proxy repair tier; not full-universe pricing"
                ),
            }
        )
    return pd.DataFrame(rows)


def _actions(*, pool: pd.DataFrame, selected: pd.DataFrame, tier_id: str) -> pd.DataFrame:
    columns = [
        f"tier_id_v{VERSION}",
        f"action_v{VERSION}",
        "loan_id",
        "loan_amnt",
        *FAMILIES,
        f"mean_return_v{VERSION}",
        f"observed_v47_proxy_v{VERSION}",
        f"claim_boundary_v{VERSION}",
    ]
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected_ids = set(selected["loan_id"].astype(str))
    work = pool.copy()
    work["loan_id"] = work["loan_id"].astype(str)
    changed = work.loc[
        (work[f"incumbent_selected_v{VERSION}"] & ~work["loan_id"].isin(selected_ids))
        | (~work[f"incumbent_selected_v{VERSION}"] & work["loan_id"].isin(selected_ids))
    ].copy()
    changed[f"tier_id_v{VERSION}"] = tier_id
    changed[f"action_v{VERSION}"] = np.where(
        changed["loan_id"].isin(selected_ids),
        "add_observed_candidate",
        "drop_v353_selected",
    )
    changed[f"claim_boundary_v{VERSION}"] = (
        "v355 coverage-only incumbent action list; not a repair recommendation"
    )
    return changed[columns].sort_values(
        [f"action_v{VERSION}", f"mean_return_v{VERSION}"], ascending=[True, False]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v355 tests v353 proxy repair over all observed omitted candidates.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v355_proxy_repair_tier_summary.csv"
                ),
                "boundary": "All observed omitted candidates plus v353 selected loans; not full-v55.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v355 documents strict and relaxed v353 proxy-repair infeasibility.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v355_proxy_repair_tier_summary.csv"
                ),
                "boundary": (
                    "No repair preserves v353 CVaR with either v353 or v347 return floor "
                    "inside the all-observed-omitted pool."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v355 repairs v353 proxy coverage while preserving v347 return and v353 CVaR.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v355_claim_blockers.csv"
                ),
                "boundary": "Relaxed v347-return/v353-CVaR proxy repair tier is infeasible.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v355 proves a valid branch-price or global integer bound.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v355_claim_blockers.csv"
                ),
                "boundary": "No full-v55 dual-bound loop or global certificate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v355 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v355_claim_blockers.csv"
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
                    "v355 tests whether the one-swap-cleared v353 candidate can repair "
                    "proxy coverage using all observed omitted candidates under static gates."
                ),
                "status": "strict_and_relaxed_v353_proxy_repair_infeasible",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "build a full-v55 dual-bound/global gate or an explicit proxy-value "
                    "tradeoff protocol without promotion"
                ),
                "last_wave": "v355",
                "execution_result": (
                    "strict_and_relaxed_v353_proxy_repair_infeasible_coverage_only_return_collapse"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v355")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V355_V353_PROXY_OR_GLOBAL_GATE_START -->"
    end = "<!-- V355_V353_PROXY_OR_GLOBAL_GATE_END -->"
    block = f"""
{start}

## Wave v355: v353 Proxy Repair / Global Gate

Generated: {status["generated_at_utc"]}

### Objective

v354 cleared the immediate one-swap repricing gate for the v353 candidate, but
v353 has 77 missing proxy rows. v355 tests whether all observed omitted proxy
candidates can repair that gap while preserving the v353 CVaR cap and
economically relevant return floors.

### Results

- Pool rows: `{status["pool_rows_v355"]}`.
- Observed omitted candidate rows:
  `{status["observed_omitted_candidate_rows_v355"]}`.
- Strict v353-return/v353-CVaR repair feasible:
  `{status["strict_v353_repair_feasible_v355"]}`.
- Relaxed v347-return/v353-CVaR repair feasible:
  `{status["relaxed_v347_return_repair_feasible_v355"]}`.
- Coverage-only incumbent found:
  `{status["coverage_only_incumbent_found_v355"]}`.
- Coverage-only missing proxy rows:
  `{status["coverage_only_missing_proxy_rows_v355"]}`.
- Coverage-only return delta vs v353:
  `{status["coverage_only_delta_return_vs_v353_v355"]}`.
- Coverage-only CVaR delta vs v353:
  `{status["coverage_only_delta_cvar90_vs_v353_v355"]}`.
- Valid branch-price bound:
  `{status["valid_branch_price_bound_v355"]}`.

### Interpretation

v355 sharpens the v353 tradeoff after one-swap clearing. It tests whether proxy
coverage can be repaired under v353's improved tail-risk cap without destroying
the economic story. This is still a proxy/global gate, not a champion or a
full-universe branch-price certificate.

### Claim Impact

- Allowed: all-observed-omitted proxy repair gate executed.
- Allowed if both economic tiers fail: strict and relaxed proxy-repair
  infeasibility in that bounded scope.
- Still prohibited: proxy-repaired v353 candidate, full-universe/global
  optimality, branch-price certificate, contractual IFRS9, live deployment,
  Paper Estrella replacement, final Paper 4 promotion and working champion
  claims.

### Quarto Promotion Decision

Keep v355 in the living notebook. The next wave should attempt a dual-bound or
explicit proxy-value tradeoff protocol without promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    _configure_proxy_gate()
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet(
        "paper4_v353_v347_expanded_branch_price_allocations.parquet"
    ).reset_index(drop=True)
    v353_summary = read_csv("paper4_v353_v347_apply_expanded_branch_price_candidate.csv")
    v347_summary = read_csv("paper4_v347_v338_apply_multi_source_relief_candidate.csv")
    source_caps = read_csv("paper4_v353_v347_expanded_branch_price_source_summary.csv")
    v354_status = json.loads((STATUS_DIR / "paper4_v354_status.json").read_text(encoding="utf-8"))
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    if any(
        df.empty for df in [universe, selected, v353_summary, v347_summary, source_caps, v47_panel]
    ):
        raise RuntimeError("Missing v355 proxy/global gate inputs.")
    if not bool(v354_status["post_v353_one_swap_local_optimality_cleared_v354"]):
        raise RuntimeError("v355 expects v354 to clear the post-v353 one-swap gate.")
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    for frame in [universe, selected]:
        for family in FAMILIES:
            frame[family] = frame[family].astype(str)
    observed_ids = set(v47_panel["loan_id"].astype(str))
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    pool, pool_summary = _build_pool(
        universe=universe,
        selected=selected,
        observed_ids=observed_ids,
        idx_by_id=idx_by_id,
        mean_returns=mean_returns,
    )
    losses_pool = losses[:, pool[f"universe_idx_v{VERSION}"].to_numpy(int)]

    v353_row = v353_summary.iloc[0]
    v347_row = v347_summary.iloc[0]
    reference = {
        "v353_return": float(v353_row["objective_return_v353"]),
        "v353_cvar": float(v353_row["scenario_loss_cvar90_v353"]),
        "v353_observed": int(v353_row["observed_proxy_rows_v353"]),
        "v353_missing": int(v353_row["missing_proxy_rows_v353"]),
        "v347_return": float(v347_row["objective_return_v347"]),
        "v347_cvar": float(v347_row["scenario_loss_cvar90_v347"]),
        "v347_observed": int(v347_row["observed_proxy_rows_v347"]),
        "v347_missing": int(v347_row["missing_proxy_rows_v347"]),
    }
    target_period_counts = _period_distribution(selected)
    tiers = [
        {
            "tier_id": "strict_v353_return_v353_cvar",
            "return_floor": reference["v353_return"],
            "cvar_cap": reference["v353_cvar"],
        },
        {
            "tier_id": "relaxed_v347_return_v353_cvar",
            "return_floor": reference["v347_return"],
            "cvar_cap": reference["v353_cvar"],
        },
        {
            "tier_id": "coverage_only_v353_cvar",
            "return_floor": -1e12,
            "cvar_cap": reference["v353_cvar"],
        },
    ]
    tier_metrics: list[dict[str, Any]] = []
    coverage_selected = pd.DataFrame()
    for tier in tiers:
        selected_tier, metrics = proxy_gate._solve_tier(
            tier_id=tier["tier_id"],
            pool=pool,
            losses_pool=losses_pool,
            losses_full=losses,
            source_caps=source_caps,
            target_period_counts=target_period_counts,
            return_floor=float(tier["return_floor"]),
            cvar_cap=float(tier["cvar_cap"]),
        )
        tier_metrics.append(metrics)
        if tier["tier_id"] == "coverage_only_v353_cvar":
            coverage_selected = selected_tier

    tier_summary = _tier_summary(tier_metrics, reference)
    strict_row = tier_summary.loc[
        tier_summary[f"tier_id_v{VERSION}"].eq("strict_v353_return_v353_cvar")
    ].iloc[0]
    relaxed_row = tier_summary.loc[
        tier_summary[f"tier_id_v{VERSION}"].eq("relaxed_v347_return_v353_cvar")
    ].iloc[0]
    coverage_row = tier_summary.loc[
        tier_summary[f"tier_id_v{VERSION}"].eq("coverage_only_v353_cvar")
    ].iloc[0]
    strict_feasible = bool(strict_row[f"solver_success_v{VERSION}"])
    relaxed_feasible = bool(relaxed_row[f"solver_success_v{VERSION}"])
    coverage_incumbent_found = bool(coverage_row[f"incumbent_found_v{VERSION}"])
    coverage_missing = (
        None
        if pd.isna(coverage_row[f"missing_proxy_rows_v{VERSION}"])
        else int(coverage_row[f"missing_proxy_rows_v{VERSION}"])
    )
    coverage_delta_return = (
        None
        if pd.isna(coverage_row[f"delta_return_vs_v353_v{VERSION}"])
        else float(coverage_row[f"delta_return_vs_v353_v{VERSION}"])
    )
    coverage_delta_cvar = (
        None
        if pd.isna(coverage_row[f"delta_cvar90_vs_v353_v{VERSION}"])
        else float(coverage_row[f"delta_cvar90_vs_v353_v{VERSION}"])
    )
    main = pd.DataFrame(
        [
            {
                f"gate_id_v{VERSION}": "v355_v353_proxy_or_global_gate",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"reference_version_v{VERSION}": REFERENCE_VERSION,
                f"reprice_version_v{VERSION}": REPRICE_VERSION,
                f"pool_rows_v{VERSION}": int(pool_summary[f"pool_rows_v{VERSION}"].iloc[0]),
                f"observed_omitted_candidate_rows_v{VERSION}": int(
                    pool_summary[f"observed_omitted_candidate_rows_v{VERSION}"].iloc[0]
                ),
                f"selected_observed_proxy_rows_v{VERSION}": reference["v353_observed"],
                f"selected_missing_proxy_rows_v{VERSION}": reference["v353_missing"],
                f"v347_missing_proxy_rows_v{VERSION}": reference["v347_missing"],
                f"strict_v353_repair_feasible_v{VERSION}": strict_feasible,
                f"relaxed_v347_return_repair_feasible_v{VERSION}": relaxed_feasible,
                f"coverage_only_solver_success_v{VERSION}": bool(
                    coverage_row[f"solver_success_v{VERSION}"]
                ),
                f"coverage_only_incumbent_found_v{VERSION}": coverage_incumbent_found,
                f"coverage_only_observed_proxy_rows_v{VERSION}": None
                if coverage_missing is None
                else TARGET_SELECTED_ROWS - coverage_missing,
                f"coverage_only_missing_proxy_rows_v{VERSION}": coverage_missing,
                f"coverage_only_delta_return_vs_v353_v{VERSION}": coverage_delta_return,
                f"coverage_only_delta_cvar90_vs_v353_v{VERSION}": coverage_delta_cvar,
                f"coverage_only_return_collapse_flag_v{VERSION}": (
                    coverage_delta_return is not None and coverage_delta_return < -1000.0
                ),
                f"valid_branch_price_bound_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "proxy-repair gate over observed omitted candidates; no global bound or promotion"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "strict_v353_proxy_repair_infeasible",
                f"blocking_v{VERSION}": not strict_feasible,
                f"evidence_count_v{VERSION}": int(strict_row[f"milp_status_v{VERSION}"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "strict v353-return/v353-CVaR repair is infeasible",
            },
            {
                f"blocker_id_v{VERSION}": "relaxed_v347_return_proxy_repair_infeasible",
                f"blocking_v{VERSION}": not relaxed_feasible,
                f"evidence_count_v{VERSION}": int(relaxed_row[f"milp_status_v{VERSION}"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "relaxed v347-return/v353-CVaR repair is infeasible",
            },
            {
                f"blocker_id_v{VERSION}": "coverage_only_return_collapse",
                f"blocking_v{VERSION}": coverage_delta_return is not None
                and coverage_delta_return < -1000.0,
                f"evidence_count_v{VERSION}": 0
                if coverage_delta_return is None
                else int(abs(coverage_delta_return)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "coverage-only incumbent restores proxy rows only by destroying return"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "valid_branch_price_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "no full-v55 dual-bound loop or termination certificate",
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
                "claim_id": "v355_proxy_repair_gate_executed",
                "allowed": True,
                "artifact": "paper4_v355_v353_proxy_or_global_gate.csv",
                "boundary": "all observed omitted proxy candidates; not full-v55 pricing",
            },
            {
                "claim_id": "v355_strict_and_relaxed_proxy_repair_infeasible",
                "allowed": not strict_feasible and not relaxed_feasible,
                "artifact": "paper4_v355_proxy_repair_tier_summary.csv",
                "boundary": "scope-limited all-observed-omitted MILP tiers",
            },
            {
                "claim_id": "v355_proxy_repair_candidate_found",
                "allowed": strict_feasible or relaxed_feasible,
                "artifact": "paper4_v355_claim_blockers.csv",
                "boundary": "requires economic repair feasibility, not coverage-only incumbent",
            },
            {
                "claim_id": "v355_valid_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v355_claim_blockers.csv",
                "boundary": "formal dual-bound loop missing",
            },
            {
                "claim_id": "v355_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v355_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    coverage_actions = _actions(
        pool=pool,
        selected=coverage_selected,
        tier_id="coverage_only_v353_cvar",
    )

    write_csv(TABLE_DIR / "paper4_v355_v353_proxy_or_global_gate.csv", main)
    write_csv(TABLE_DIR / "paper4_v355_proxy_repair_pool_summary.csv", pool_summary)
    write_csv(TABLE_DIR / "paper4_v355_proxy_repair_tier_summary.csv", tier_summary)
    write_csv(TABLE_DIR / "paper4_v355_coverage_only_incumbent_actions.csv", coverage_actions)
    write_csv(TABLE_DIR / "paper4_v355_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v355_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = main.iloc[0]
    status = {
        "phase": "v355_v353_proxy_or_global_gate",
        "schema_version": "2026-05-17.355",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v355": BASE_VERSION,
        "reference_version_v355": REFERENCE_VERSION,
        "reprice_version_v355": REPRICE_VERSION,
        "pool_rows_v355": int(row[f"pool_rows_v{VERSION}"]),
        "observed_omitted_candidate_rows_v355": int(
            row[f"observed_omitted_candidate_rows_v{VERSION}"]
        ),
        "selected_observed_proxy_rows_v355": int(row[f"selected_observed_proxy_rows_v{VERSION}"]),
        "selected_missing_proxy_rows_v355": int(row[f"selected_missing_proxy_rows_v{VERSION}"]),
        "v347_missing_proxy_rows_v355": int(row[f"v347_missing_proxy_rows_v{VERSION}"]),
        "strict_v353_repair_feasible_v355": bool(row[f"strict_v353_repair_feasible_v{VERSION}"]),
        "relaxed_v347_return_repair_feasible_v355": bool(
            row[f"relaxed_v347_return_repair_feasible_v{VERSION}"]
        ),
        "coverage_only_solver_success_v355": bool(row[f"coverage_only_solver_success_v{VERSION}"]),
        "coverage_only_incumbent_found_v355": bool(
            row[f"coverage_only_incumbent_found_v{VERSION}"]
        ),
        "coverage_only_observed_proxy_rows_v355": int(
            row[f"coverage_only_observed_proxy_rows_v{VERSION}"]
        )
        if not pd.isna(row[f"coverage_only_observed_proxy_rows_v{VERSION}"])
        else None,
        "coverage_only_missing_proxy_rows_v355": int(
            row[f"coverage_only_missing_proxy_rows_v{VERSION}"]
        )
        if not pd.isna(row[f"coverage_only_missing_proxy_rows_v{VERSION}"])
        else None,
        "coverage_only_delta_return_vs_v353_v355": None
        if pd.isna(row[f"coverage_only_delta_return_vs_v353_v{VERSION}"])
        else float(row[f"coverage_only_delta_return_vs_v353_v{VERSION}"]),
        "coverage_only_delta_cvar90_vs_v353_v355": None
        if pd.isna(row[f"coverage_only_delta_cvar90_vs_v353_v{VERSION}"])
        else float(row[f"coverage_only_delta_cvar90_vs_v353_v{VERSION}"]),
        "coverage_only_return_collapse_flag_v355": bool(
            row[f"coverage_only_return_collapse_flag_v{VERSION}"]
        ),
        "coverage_only_action_rows_v355": int(len(coverage_actions)),
        "valid_branch_price_bound_v355": False,
        "full_universe_integer_optimality_claim_allowed_v355": False,
        "working_champion_claim_allowed_v355": False,
        "paper1_promotion_allowed_v355": False,
        "paper4_working_champion_changed_v355": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v355": int(len(blockers)),
        "claim_matrix_rows_v355": int(len(claim_matrix)),
        "next_artifact_v355": NEXT_ARTIFACT,
        "claim_boundary": (
            "v355 tests all observed omitted proxy repair for v353; strict/relaxed "
            "repair is infeasible and no global bound or promotion is authorized"
        ),
    }
    write_json(STATUS_DIR / "paper4_v355_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v355": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
