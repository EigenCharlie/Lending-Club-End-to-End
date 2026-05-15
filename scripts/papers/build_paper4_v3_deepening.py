"""Build Paper 4 v3 deepening artifacts.

This layer addresses the eight main open items in the Paper 4 living lab:

1. full-universe exact allocations for the frozen policy universe;
2. cashflow-consistent IFRS9 lifetime ECL proxies;
3. stronger forward-only online conformal searches;
4. a formal CVaR LP prototype inside the solver;
5. selector-governance thresholds with explicit rationales;
6. a causal dossier for ``high_rate_within_grade``;
7. fairness/protected-attribute audit plus proxy-governance stress;
8. a loan-level multi-period SDAM replay with financial state transitions.

The script deliberately does not write ``paper4_final_promotion.json``.
Outputs are research artifacts, not Paper Estrella promotion artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs

from scripts.papers.build_paper4_extended_experiments import (
    BUDGET,
    _load_base_loan_frame,
    _normalise,
    _policy_effective_pd,
)
from scripts.papers.build_paper4_living_lab_artifacts import DEFAULT_LGD, load_policy_universe
from scripts.papers.build_paper4_next_wave_experiments import (
    _as_month,
    _funded_metrics,
    _prepare_base,
    _solve_full_policy,
)
from scripts.papers.build_paper4_v2_priorities import _canonical_champion_rows, _policy_score
from src.optimization.portfolio_model import optimize_portfolio_allocation

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = OUT_ROOT / "tables"
STATUS_DIR = OUT_ROOT / "status"
FIGURE_DIR = OUT_ROOT / "figures"
NOTE_DIR = OUT_ROOT / "notes"

SCHEMA_VERSION = "2026-05-13.1"
FULL_SOLVE_TIME_LIMIT_SECONDS = 300
ONLINE_TARGET_COVERAGE = 0.90


def _write_csv(name: str, df: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def _write_parquet(name: str, df: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    df.to_parquet(path, index=False)
    return path


def _write_json(name: str, payload: dict[str, Any]) -> Path:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATUS_DIR / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_note(name: str, text: str) -> Path:
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTE_DIR / name
    path.write_text(text, encoding="utf-8")
    return path


def _safe_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _standardise_allocation_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in [
        "loan_id",
        "policy_id",
        "period",
        "original_grade",
        "sub_grade",
        "term",
        "purpose",
        "home_ownership",
        "addr_state",
        "state_top20",
        "income_band",
        "dti_band",
        "exact_scope",
        "solver_status",
        "next_wave_source",
        "reconstruction_method",
    ]:
        if column in out:
            out[column] = out[column].astype(str)
    if "issue_d" in out:
        out["issue_d"] = out["issue_d"].astype(str)
    if "issue_month" in out:
        out["issue_month"] = _as_month(out["issue_month"])
    if "score_decile" in out:
        out["score_decile"] = (
            pd.to_numeric(out["score_decile"], errors="coerce").fillna(-1).astype(int)
        )
    return out


def _enrich_allocations_from_base(allocations: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Fill source columns that older champion artifacts did not persist."""

    out = allocations.copy()
    out["loan_id"] = out["loan_id"].astype(str)
    lookup_cols = [
        "loan_id",
        "issue_d",
        "issue_month",
        "period",
        "original_grade",
        "sub_grade",
        "term",
        "purpose",
        "home_ownership",
        "addr_state",
        "state_top20",
        "income_band",
        "dti_band",
        "score_decile",
        "annual_inc",
        "dti",
        "fico_score",
        "zip_code",
        "loan_amnt",
        "int_rate",
        "int_rate_decimal",
        "y_true",
        "pd_point_alpha01",
        "pd_high_alpha01",
    ]
    available = [col for col in lookup_cols if col in base.columns]
    lookup = base[available].copy()
    lookup["loan_id"] = lookup["loan_id"].astype(str)
    merged = out.merge(lookup, on="loan_id", how="left", suffixes=("", "_base"))
    for col in available:
        if col == "loan_id":
            continue
        base_col = f"{col}_base"
        if base_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = merged[base_col]
        else:
            current = merged[col]
            missing = current.isna() | current.astype(str).str.lower().isin({"nan", "none", ""})
            merged.loc[missing, col] = merged.loc[missing, base_col]
        merged = merged.drop(columns=[base_col])
    if "pd_point" not in merged:
        merged["pd_point"] = merged.get("pd_point_alpha01", np.nan)
    else:
        merged["pd_point"] = pd.to_numeric(merged["pd_point"], errors="coerce").fillna(
            pd.to_numeric(merged.get("pd_point_alpha01"), errors="coerce")
        )
    return _standardise_allocation_columns(merged)


def _allocation_eval(allocations: pd.DataFrame, policy_universe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy_id, group in allocations.groupby("policy_id", sort=False):
        policy = policy_universe[policy_universe["policy_id"].astype(str).eq(str(policy_id))]
        meta = policy.iloc[0].to_dict() if not policy.empty else {}
        candidate_pool_n = pd.to_numeric(group["candidate_pool_n"], errors="coerce").max()
        rows.append(
            {
                "policy_id": policy_id,
                "risk_tolerance": meta.get("risk_tolerance", np.nan),
                "gamma": meta.get("gamma", np.nan),
                "policy_mode": meta.get("policy_mode", ""),
                "uncertainty_aversion": meta.get("uncertainty_aversion", np.nan),
                "solver_status": ",".join(sorted(set(group["solver_status"].astype(str)))),
                "candidate_pool_n": int(candidate_pool_n) if pd.notna(candidate_pool_n) else np.nan,
                **_funded_metrics(group, prefix="full_"),
                "stage0_source": "v3_full_universe_exact_all_policies",
            }
        )
    eval_table = pd.DataFrame(rows)
    if "full_realized_return" in eval_table:
        champion = eval_table.loc[
            eval_table["policy_id"].eq("paper1_economic_champion"), "full_realized_return"
        ]
        champion_return = float(champion.iloc[0]) if not champion.empty else np.nan
        eval_table["return_delta_vs_champion"] = (
            eval_table["full_realized_return"] - champion_return
        )
    return eval_table.sort_values("full_realized_return", ascending=False, na_position="last")


def build_full_universe_all_policy_allocations(
    base: pd.DataFrame,
    policy_universe: pd.DataFrame,
    *,
    max_policies: int | None,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Solve full-universe allocations for the frozen policy universe.

    The function checkpoints after every policy. If a previous next-wave top-k
    full solve exists, it is reused as seed evidence unless ``force`` is true.
    """

    alloc_path = TABLE_DIR / "paper4_v3_full_universe_all_policy_allocations.parquet"
    eval_path = TABLE_DIR / "paper4_v3_full_universe_all_policy_eval.csv"
    status_path = STATUS_DIR / "paper4_v3_full_universe_all_policy_status.json"
    target_ids = policy_universe["policy_id"].astype(str).tolist()
    if max_policies is not None:
        non_champion = [pid for pid in target_ids if pid != "paper1_economic_champion"]
        target_ids = ["paper1_economic_champion", *non_champion[: max(0, max_policies - 1)]]
    target_ids = list(dict.fromkeys(target_ids))

    if not force and alloc_path.exists() and eval_path.exists() and status_path.exists():
        existing = pd.read_parquet(alloc_path)
        if set(target_ids).issubset(set(existing["policy_id"].astype(str))):
            return (
                _enrich_allocations_from_base(existing, base),
                pd.read_csv(eval_path),
                json.loads(status_path.read_text(encoding="utf-8")),
            )

    rows: list[pd.DataFrame] = []
    solved_from_seed: set[str] = set()
    if not force and alloc_path.exists():
        previous = pd.read_parquet(alloc_path)
        rows.append(previous[previous["policy_id"].astype(str).isin(target_ids)].copy())
        solved_from_seed.update(rows[-1]["policy_id"].astype(str).unique().tolist())
    elif not force and (TABLE_DIR / "paper4_full_universe_topk_allocations.parquet").exists():
        previous = pd.read_parquet(TABLE_DIR / "paper4_full_universe_topk_allocations.parquet")
        previous = previous[previous["policy_id"].astype(str).isin(target_ids)].copy()
        if not previous.empty:
            rows.append(previous)
            solved_from_seed.update(previous["policy_id"].astype(str).unique().tolist())

    if (
        "paper1_economic_champion" in target_ids
        and "paper1_economic_champion" not in solved_from_seed
    ):
        champion = _canonical_champion_rows().copy()
        champion["exact_scope"] = "paper1_canonical_full_universe"
        champion["next_wave_source"] = "paper1_exact_champion"
        champion["candidate_pool_n"] = len(base)
        rows.append(champion)
        solved_from_seed.add("paper1_economic_champion")

    generated_at_start = datetime.now(UTC).isoformat()
    solve_log: list[dict[str, Any]] = []
    for policy_id in target_ids:
        if policy_id in solved_from_seed:
            solve_log.append({"policy_id": policy_id, "status": "reused_checkpoint"})
            continue
        policy = policy_universe[policy_universe["policy_id"].astype(str).eq(policy_id)].iloc[0]
        t0 = time.perf_counter()
        funded, metrics = _solve_full_policy(base, policy)
        elapsed = time.perf_counter() - t0
        funded["v3_solve_seconds"] = elapsed
        rows.append(funded)
        solved_from_seed.add(policy_id)
        solve_log.append(
            {
                "policy_id": policy_id,
                "status": metrics.get("solver_status", "unknown"),
                "elapsed_seconds": elapsed,
                "n_funded": metrics.get("full_n_funded"),
                "full_realized_return": metrics.get("full_realized_return"),
            }
        )
        checkpoint = _enrich_allocations_from_base(
            pd.concat(rows, ignore_index=True, sort=False), base
        )
        _write_parquet("paper4_v3_full_universe_all_policy_allocations.parquet", checkpoint)
        _write_csv(
            "paper4_v3_full_universe_all_policy_eval.csv",
            _allocation_eval(checkpoint, policy_universe),
        )
        _write_json(
            "paper4_v3_full_universe_all_policy_status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "started_at_utc": generated_at_start,
                "mode": "full_universe_exact_all_frozen_policies_no_promotion",
                "candidate_pool_n": int(len(base)),
                "target_policy_count": int(len(target_ids)),
                "completed_policy_count": int(len(solved_from_seed)),
                "paper1_champion_protected": True,
                "paper4_final_promotion_created": False,
                "promotion_json_created": False,
                "solve_log": solve_log,
            },
        )

    allocations = _enrich_allocations_from_base(
        pd.concat(rows, ignore_index=True, sort=False), base
    )
    allocations = allocations[allocations["policy_id"].astype(str).isin(target_ids)].copy()
    eval_table = _allocation_eval(allocations, policy_universe)
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "started_at_utc": generated_at_start,
        "mode": "full_universe_exact_all_frozen_policies_no_promotion",
        "candidate_pool_n": int(len(base)),
        "target_policy_count": int(len(target_ids)),
        "completed_policy_count": int(allocations["policy_id"].nunique()),
        "paper1_champion_protected": True,
        "paper4_final_promotion_created": False,
        "promotion_json_created": False,
        "time_limit_seconds_per_solve": FULL_SOLVE_TIME_LIMIT_SECONDS,
        "solve_log": solve_log,
    }
    return allocations, eval_table, status


def _scenario_path(month: int, scenario: str) -> dict[str, float]:
    if scenario == "baseline":
        return {
            "pd_multiplier": 1.00 + 0.04 * min(month / 60.0, 1.0),
            "lgd": 0.45,
            "prepay_hazard": 0.008,
            "macro_index": 1.00,
        }
    if scenario == "adverse":
        shock = 1.20 if month <= 12 else 1.35 if month <= 36 else 1.15
        return {
            "pd_multiplier": shock,
            "lgd": 0.55,
            "prepay_hazard": 0.005,
            "macro_index": shock,
        }
    shock = 1.60 if month <= 12 else 1.80 if month <= 36 else 1.35
    return {
        "pd_multiplier": shock,
        "lgd": 0.65,
        "prepay_hazard": 0.003,
        "macro_index": shock,
    }


def _initial_ifrs9_stage(row: pd.Series, scenario: str) -> str:
    pd_point = float(row.get("pd_point", row.get("pd_point_alpha01", 0.0)) or 0.0)
    pd_high = float(row.get("pd_high_alpha01", 0.0) or 0.0)
    path = _scenario_path(1, scenario)
    stressed_pd = min(pd_high * path["pd_multiplier"], 1.0)
    relative = stressed_pd / max(pd_point, 1e-6)
    if float(row.get("y_true", 0.0)) >= 1.0:
        return "Stage 3 observed"
    if (
        stressed_pd >= 0.30
        or relative >= 1.5
        or str(row.get("original_grade", "")) in {"E", "F", "G"}
    ):
        return "Stage 2 lifetime cashflow"
    return "Stage 1 12m cashflow"


def build_ifrs9_cashflow_lifetime(
    allocations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = ["baseline", "adverse", "severe"]
    rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for _, loan in allocations.iterrows():
        term = int(float(loan.get("term", 36) or 36))
        term = 60 if term >= 60 else 36
        exposure = float(loan["funded_exposure"])
        int_rate = float(loan.get("int_rate_decimal", 0.0) or 0.0)
        monthly_discount = 1.0 + int_rate / 12.0
        pd_high = float(loan.get("pd_high_alpha01", 0.0) or 0.0)
        for scenario in scenarios:
            stage = _initial_ifrs9_stage(loan, scenario)
            horizon = (
                1
                if stage == "Stage 3 observed"
                else min(12, term)
                if stage.startswith("Stage 1")
                else term
            )
            survival = 1.0
            cum_default = 0.0
            cum_prepay = 0.0
            ecl_total = 0.0
            weighted_life_num = 0.0
            weighted_life_den = 0.0
            for month in range(1, term + 1):
                path = _scenario_path(month, scenario)
                balance = exposure * max(0.0, 1.0 - (month - 1) / term)
                annual_pd = float(np.clip(pd_high * path["pd_multiplier"], 0.0, 0.999))
                default_hazard = 1.0 - (1.0 - annual_pd) ** (1.0 / 12.0)
                prepay_hazard = path["prepay_hazard"] * (1.15 if term == 36 else 0.85)
                if stage == "Stage 3 observed":
                    marginal_default = 1.0 if month == 1 else 0.0
                    prepay_prob = 0.0
                    survival_after = 0.0
                else:
                    marginal_default = survival * default_hazard
                    prepay_prob = survival * (1.0 - default_hazard) * prepay_hazard
                    survival_after = survival * (1.0 - default_hazard) * (1.0 - prepay_hazard)
                discount = monthly_discount ** (-month)
                included = month <= horizon
                ecl_contribution = (
                    marginal_default * balance * path["lgd"] * discount if included else 0.0
                )
                ecl_total += ecl_contribution
                weighted_life_num += month * (marginal_default + prepay_prob)
                weighted_life_den += marginal_default + prepay_prob
                cum_default += marginal_default
                cum_prepay += prepay_prob
                if included:
                    rows.append(
                        {
                            "policy_id": loan["policy_id"],
                            "loan_id": loan["loan_id"],
                            "scenario": scenario,
                            "month_on_book": month,
                            "term": term,
                            "ifrs9_stage_v3": stage,
                            "ead_start": balance,
                            "survival_start": survival,
                            "annual_pd_stressed": annual_pd,
                            "monthly_default_hazard": default_hazard,
                            "monthly_prepay_hazard": prepay_hazard,
                            "marginal_default_prob": marginal_default,
                            "marginal_prepay_prob": prepay_prob,
                            "lgd_scenario": path["lgd"],
                            "discount_factor": discount,
                            "ecl_contribution_v3": ecl_contribution,
                            "macro_index": path["macro_index"],
                        }
                    )
                survival = survival_after
            transition_rows.append(
                {
                    "policy_id": loan["policy_id"],
                    "loan_id": loan["loan_id"],
                    "scenario": scenario,
                    "ifrs9_stage_v3": stage,
                    "term": term,
                    "expected_default_prob_lifetime": min(cum_default, 1.0),
                    "expected_prepay_prob_lifetime": min(cum_prepay, 1.0),
                    "expected_survival_end": max(0.0, survival),
                    "expected_life_months": weighted_life_num / max(weighted_life_den, 1e-12),
                    "cashflow_ecl_v3": ecl_total,
                    "funded_exposure": exposure,
                    "realized_return_proxy_lgd45": float(
                        loan.get("realized_return_proxy_lgd45", 0.0)
                    ),
                }
            )
    grid = pd.DataFrame(rows)
    transitions = pd.DataFrame(transition_rows)
    summary = transitions.groupby(["policy_id", "scenario"], as_index=False).agg(
        n_funded=("loan_id", "nunique"),
        funded_exposure=("funded_exposure", "sum"),
        cashflow_ecl_v3=("cashflow_ecl_v3", "sum"),
        realized_return_proxy_lgd45=("realized_return_proxy_lgd45", "sum"),
        stage1_share=(
            "ifrs9_stage_v3",
            lambda x: float(pd.Series(x).str.startswith("Stage 1").mean()),
        ),
        stage2_share=(
            "ifrs9_stage_v3",
            lambda x: float(pd.Series(x).str.startswith("Stage 2").mean()),
        ),
        stage3_share=(
            "ifrs9_stage_v3",
            lambda x: float(pd.Series(x).str.startswith("Stage 3").mean()),
        ),
        expected_default_prob_mean=("expected_default_prob_lifetime", "mean"),
        expected_prepay_prob_mean=("expected_prepay_prob_lifetime", "mean"),
        expected_life_months=("expected_life_months", "mean"),
    )
    summary["net_return_after_cashflow_ecl_v3"] = (
        summary["realized_return_proxy_lgd45"] - summary["cashflow_ecl_v3"]
    )
    stage_matrix = (
        transitions.groupby(["scenario", "ifrs9_stage_v3"], as_index=False)
        .agg(
            loans=("loan_id", "nunique"),
            exposure=("funded_exposure", "sum"),
            expected_default_prob=("expected_default_prob_lifetime", "mean"),
            expected_prepay_prob=("expected_prepay_prob_lifetime", "mean"),
            expected_survival_end=("expected_survival_end", "mean"),
        )
        .sort_values(["scenario", "ifrs9_stage_v3"])
    )
    return grid, transitions, summary, stage_matrix


def _online_groups_for_method(base: pd.DataFrame, method: str) -> pd.Series:
    if method in {"rolling_global", "aci_global"}:
        return pd.Series("global", index=base.index)
    if method in {"mondrian_grade", "aci_grade"}:
        return base["original_grade"].astype(str)
    if method == "mondrian_grade_score":
        return base["original_grade"].astype(str) + "_d" + base["score_decile"].astype(str)
    if method == "up_ocp_proxy":
        return base["original_grade"].astype(str) + "_" + base["period"].astype(str)
    raise ValueError(f"Unknown online method: {method}")


def _forward_online_intervals(
    base: pd.DataFrame, method: str, *, min_group_n: int = 200
) -> pd.DataFrame:
    work = base[
        [
            "loan_id",
            "issue_month",
            "period",
            "original_grade",
            "term",
            "score_decile",
            "state_top20",
            "income_band",
            "dti_band",
            "y_true",
            "y_pred",
        ]
    ].copy()
    work["score_abs"] = (work["y_true"] - work["y_pred"]).abs()
    work["online_group"] = _online_groups_for_method(work, method)
    months = sorted(work["issue_month"].dropna().unique())
    target_error = 1.0 - ONLINE_TARGET_COVERAGE
    alpha_global = 0.10
    alpha_by_group: dict[str, float] = {}
    width_budget = 0.70
    frames: list[pd.DataFrame] = []
    for month in months:
        current = work[work["issue_month"].eq(month)].copy()
        prior = work[work["issue_month"].lt(month)]
        q_values = pd.Series(np.full(len(current), 0.50), index=current.index)
        alpha_used = pd.Series(np.full(len(current), alpha_global), index=current.index)
        source = "first_month_fallback"
        if not prior.empty:
            global_q = float(prior["score_abs"].quantile(1.0 - alpha_global))
            source = method
            if method == "rolling_global":
                q_values[:] = float(prior["score_abs"].quantile(0.90))
            elif method == "aci_global":
                q_values[:] = global_q
                alpha_used[:] = alpha_global
            else:
                prior_groups = prior.groupby("online_group")["score_abs"]
                group_counts = prior_groups.size()
                for group_name, idx in current.groupby("online_group").groups.items():
                    group_name = str(group_name)
                    local_alpha = alpha_by_group.get(group_name, alpha_global)
                    if group_counts.get(group_name, 0) >= min_group_n:
                        q = float(
                            prior.loc[prior["online_group"].eq(group_name), "score_abs"].quantile(
                                1.0 - local_alpha
                            )
                        )
                    else:
                        q = global_q
                        source = f"{method}_fallback_global"
                    q_values.loc[idx] = q
                    alpha_used.loc[idx] = local_alpha
        current["online_method_v3"] = method
        current["online_source_v3"] = source
        current["alpha_used"] = alpha_used
        current["qhat_v3"] = q_values.astype(float)
        current["pd_low_online_v3"] = np.clip(current["y_pred"] - current["qhat_v3"], 0.0, 1.0)
        current["pd_high_online_v3"] = np.clip(current["y_pred"] + current["qhat_v3"], 0.0, 1.0)
        current["covered_online_v3"] = current["y_true"].between(
            current["pd_low_online_v3"], current["pd_high_online_v3"]
        )
        current["interval_width_online_v3"] = (
            current["pd_high_online_v3"] - current["pd_low_online_v3"]
        )
        frames.append(current)
        if method == "aci_global" and not current.empty:
            obs_error = 1.0 - float(current["covered_online_v3"].mean())
            alpha_global = float(
                np.clip(alpha_global + 0.05 * (target_error - obs_error), 0.01, 0.30)
            )
        if method in {"aci_grade", "up_ocp_proxy"} and not current.empty:
            for group_name, group in current.groupby("online_group"):
                obs_error = 1.0 - float(group["covered_online_v3"].mean())
                avg_width = float(group["interval_width_online_v3"].mean())
                old_alpha = alpha_by_group.get(str(group_name), alpha_global)
                width_push = 0.02 * (avg_width - width_budget) if method == "up_ocp_proxy" else 0.0
                alpha_by_group[str(group_name)] = float(
                    np.clip(old_alpha + 0.05 * (target_error - obs_error) + width_push, 0.01, 0.30)
                )
    return pd.concat(frames, ignore_index=True)


def build_online_conformal_v3(
    base: pd.DataFrame,
    allocations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = [
        "rolling_global",
        "mondrian_grade",
        "mondrian_grade_score",
        "aci_global",
        "aci_grade",
        "up_ocp_proxy",
    ]
    interval_frames = [_forward_online_intervals(base, method) for method in methods]
    intervals = pd.concat(interval_frames, ignore_index=True)
    policy_months: list[pd.DataFrame] = []
    source_months: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    source_cols = ["original_grade", "period", "score_decile", "state_top20", "income_band"]
    for method, method_intervals in intervals.groupby("online_method_v3"):
        merged = allocations[["policy_id", "loan_id", "issue_month", "funded_exposure"]].merge(
            method_intervals[
                [
                    "loan_id",
                    "covered_online_v3",
                    "interval_width_online_v3",
                    "online_method_v3",
                    "original_grade",
                    "period",
                    "score_decile",
                    "state_top20",
                    "income_band",
                ]
            ],
            on="loan_id",
            how="left",
        )
        policy_month = (
            merged.groupby(["online_method_v3", "policy_id", "issue_month"], as_index=False)
            .agg(
                n_funded=("loan_id", "nunique"),
                funded_exposure=("funded_exposure", "sum"),
                coverage_online_v3=("covered_online_v3", "mean"),
                avg_width_online_v3=("interval_width_online_v3", "mean"),
            )
            .rename(columns={"issue_month": "month"})
        )
        policy_month["coverage_regret_90_v3"] = (
            ONLINE_TARGET_COVERAGE - policy_month["coverage_online_v3"]
        ).clip(lower=0)
        policy_months.append(policy_month)
        for source in source_cols:
            local = (
                merged.groupby(
                    ["online_method_v3", "policy_id", "issue_month", source], as_index=False
                )
                .agg(
                    n=("loan_id", "nunique"),
                    coverage_online_v3=("covered_online_v3", "mean"),
                    avg_width_online_v3=("interval_width_online_v3", "mean"),
                )
                .rename(columns={"issue_month": "month", source: "source_value"})
            )
            local = local[local["n"].ge(5)].copy()
            local["source_id"] = source
            source_months.append(local)
        source_month = pd.concat(source_months, ignore_index=True)
        method_source = source_month[source_month["online_method_v3"].eq(method)]
        summaries.append(
            {
                "online_method_v3": method,
                "coverage_policy_month_mean": float(policy_month["coverage_online_v3"].mean()),
                "coverage_policy_month_min": float(policy_month["coverage_online_v3"].min()),
                "coverage_source_month_min": float(method_source["coverage_online_v3"].min())
                if not method_source.empty
                else np.nan,
                "avg_width_loan": float(method_intervals["interval_width_online_v3"].mean()),
                "p90_width_loan": float(
                    method_intervals["interval_width_online_v3"].quantile(0.90)
                ),
                "total_coverage_regret_90": float(policy_month["coverage_regret_90_v3"].sum()),
                "promotion_candidate": bool(
                    policy_month["coverage_online_v3"].mean() >= 0.90
                    and policy_month["coverage_online_v3"].min() >= 0.80
                    and (method_source.empty or method_source["coverage_online_v3"].min() >= 0.75)
                    and method_intervals["interval_width_online_v3"].mean() <= 0.80
                ),
            }
        )
    source_month = pd.concat(source_months, ignore_index=True)
    source_month["source_value"] = source_month["source_value"].astype(str)
    return (
        intervals,
        pd.concat(policy_months, ignore_index=True),
        source_month,
        pd.DataFrame(summaries).sort_values(
            ["promotion_candidate", "coverage_policy_month_min", "avg_width_loan"],
            ascending=[False, False, True],
        ),
    )


def _cvar_candidate_pool(
    base: pd.DataFrame, policy_universe: pd.DataFrame, allocations: pd.DataFrame, pool_n: int
) -> pd.DataFrame:
    seed_ids = set(allocations["loan_id"].astype(str))
    score_parts = []
    for _, policy in policy_universe.iterrows():
        effective_pd = _policy_effective_pd(base, policy)
        score_parts.append(_policy_score(base, policy, effective_pd))
    score = np.max(np.vstack(score_parts), axis=0)
    work = base.copy()
    work["loan_id"] = work["id"].astype(str)
    work["cvar_pool_score"] = score
    seeded = work[work["loan_id"].isin(seed_ids)]
    top = work.sort_values("cvar_pool_score", ascending=False).head(max(pool_n, len(seeded)))
    return pd.concat([seeded, top], ignore_index=True).drop_duplicates("loan_id").head(pool_n)


def _solve_formal_cvar_lp(
    pool: pd.DataFrame,
    *,
    risk_tolerance: float,
    cvar_penalty: float,
    beta: float = 0.90,
    time_limit: int = 240,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    n = len(pool)
    pd_point = pool["pd_point_alpha01"].to_numpy(dtype=float)
    pd_high = pool["pd_high_alpha01"].to_numpy(dtype=float)
    loan_amnt = pool["loan_amnt"].to_numpy(dtype=float)
    int_rate = pool["int_rate_decimal"].to_numpy(dtype=float)
    scenarios = [
        ("baseline_low", 0.95, 0.42),
        ("baseline_mid", 1.00, 0.45),
        ("baseline_high", 1.08, 0.48),
        ("adverse_low", 1.20, 0.52),
        ("adverse_mid", 1.35, 0.55),
        ("adverse_high", 1.50, 0.58),
        ("severe_low", 1.60, 0.62),
        ("severe_mid", 1.80, 0.65),
        ("severe_high", 2.00, 0.68),
    ]
    loss_matrix = np.vstack(
        [np.clip(pd_high * mult, 0.0, 1.0) * lgd * loan_amnt for _, mult, lgd in scenarios]
    )
    model = pyo.ConcreteModel("paper4_v3_formal_cvar_lp")
    model.I = pyo.RangeSet(0, n - 1)
    model.S = pyo.RangeSet(0, len(scenarios) - 1)
    model.x = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.eta = pyo.Var(domain=pyo.NonNegativeReals)
    model.u = pyo.Var(model.S, domain=pyo.NonNegativeReals)

    def exposure(m):
        return sum(m.x[i] * loan_amnt[i] for i in m.I)

    model.budget = pyo.Constraint(expr=exposure(model) <= BUDGET)
    model.min_budget = pyo.Constraint(expr=exposure(model) >= 0.95 * BUDGET)
    model.pd_cap = pyo.Constraint(
        expr=sum(model.x[i] * loan_amnt[i] * pd_high[i] for i in model.I)
        <= risk_tolerance * (exposure(model) + 1e-6)
    )
    purposes = pool["purpose"].fillna("unknown").astype(str).to_numpy()
    for p_idx, purpose in enumerate(pd.Series(purposes).unique()):
        idx = np.flatnonzero(purposes == purpose).tolist()
        setattr(
            model,
            f"purpose_concentration_{p_idx}",
            pyo.Constraint(
                expr=sum(model.x[i] * loan_amnt[i] for i in idx) <= 0.25 * (exposure(model) + 1e-6)
            ),
        )

    def cvar_excess_rule(m, s):
        return m.u[s] >= sum(m.x[i] * loss_matrix[s, i] for i in m.I) - m.eta

    model.cvar_excess = pyo.Constraint(model.S, rule=cvar_excess_rule)
    cvar_expr = model.eta + (1.0 / ((1.0 - beta) * len(scenarios))) * sum(
        model.u[s] for s in model.S
    )
    base_return = sum(
        model.x[i] * loan_amnt[i] * (int_rate[i] - pd_point[i] * DEFAULT_LGD) for i in model.I
    )
    model.obj = pyo.Objective(expr=base_return - cvar_penalty * cvar_expr, sense=pyo.maximize)
    solver = Highs()
    solver.config.time_limit = time_limit
    t0 = time.perf_counter()
    try:
        results = solver.solve(model)
    except RuntimeError as exc:
        policy_id = f"formal_cvar_rt{risk_tolerance:.3f}_pen{cvar_penalty:.2f}"
        metrics = {
            "cvar_policy_id": policy_id,
            "risk_tolerance": risk_tolerance,
            "cvar_penalty": cvar_penalty,
            "beta": beta,
            "candidate_pool_n": n,
            "solver_status": f"no_feasible_solution: {str(exc).splitlines()[0]}",
            "elapsed_seconds": time.perf_counter() - t0,
            "n_funded": 0,
            "funded_exposure": 0.0,
            "realized_return_proxy_lgd45": 0.0,
            "objective_value": np.nan,
            "eta": np.nan,
            "formal_cvar_loss": np.nan,
            "mean_scenario_loss": np.nan,
            "max_scenario_loss": np.nan,
        }
        scenario_loss = pd.DataFrame(
            [
                {
                    "cvar_policy_id": policy_id,
                    "scenario": name,
                    "pd_multiplier": mult,
                    "lgd": lgd,
                    "portfolio_loss": np.nan,
                    "cvar_excess_u": np.nan,
                }
                for name, mult, lgd in scenarios
            ]
        )
        return pd.DataFrame(), metrics, scenario_loss
    elapsed = time.perf_counter() - t0
    allocation = np.array([float(pyo.value(model.x[i])) for i in model.I])
    mask = allocation > 1e-8
    funded = pool.loc[mask].copy()
    funded["loan_id"] = funded["id"].astype(str)
    funded["allocation_fraction"] = allocation[mask]
    funded["funded_exposure"] = funded["allocation_fraction"] * funded["loan_amnt"]
    funded["cvar_policy_id"] = f"formal_cvar_rt{risk_tolerance:.3f}_pen{cvar_penalty:.2f}"
    funded["realized_return_proxy_lgd45"] = (
        funded["funded_exposure"] * funded["int_rate_decimal"] * (1.0 - funded["y_true"])
        - funded["funded_exposure"] * DEFAULT_LGD * funded["y_true"]
    )
    scenario_rows = []
    x = allocation
    for s_idx, (name, mult, lgd) in enumerate(scenarios):
        loss = float(np.dot(x, loss_matrix[s_idx]))
        scenario_rows.append(
            {
                "cvar_policy_id": funded["cvar_policy_id"].iloc[0]
                if not funded.empty
                else f"formal_cvar_rt{risk_tolerance:.3f}_pen{cvar_penalty:.2f}",
                "scenario": name,
                "pd_multiplier": mult,
                "lgd": lgd,
                "portfolio_loss": loss,
                "cvar_excess_u": float(pyo.value(model.u[s_idx])),
            }
        )
    scenario_loss = pd.DataFrame(scenario_rows)
    metrics = {
        "cvar_policy_id": f"formal_cvar_rt{risk_tolerance:.3f}_pen{cvar_penalty:.2f}",
        "risk_tolerance": risk_tolerance,
        "cvar_penalty": cvar_penalty,
        "beta": beta,
        "candidate_pool_n": n,
        "solver_status": str(getattr(results, "termination_condition", "unknown")),
        "elapsed_seconds": elapsed,
        "n_funded": int(funded["loan_id"].nunique()) if not funded.empty else 0,
        "funded_exposure": float(funded["funded_exposure"].sum()) if not funded.empty else 0.0,
        "realized_return_proxy_lgd45": float(funded["realized_return_proxy_lgd45"].sum())
        if not funded.empty
        else 0.0,
        "objective_value": float(pyo.value(model.obj)),
        "eta": float(pyo.value(model.eta)),
        "formal_cvar_loss": float(
            pyo.value(model.eta)
            + (1.0 / ((1.0 - beta) * len(scenarios)))
            * sum(float(pyo.value(model.u[s])) for s in model.S)
        ),
        "mean_scenario_loss": float(scenario_loss["portfolio_loss"].mean()),
        "max_scenario_loss": float(scenario_loss["portfolio_loss"].max()),
    }
    keep = [
        "cvar_policy_id",
        "loan_id",
        "issue_month",
        "original_grade",
        "purpose",
        "loan_amnt",
        "int_rate_decimal",
        "y_true",
        "pd_point_alpha01",
        "pd_high_alpha01",
        "allocation_fraction",
        "funded_exposure",
        "realized_return_proxy_lgd45",
    ]
    return funded[keep], metrics, scenario_loss


def build_formal_cvar_lp(
    base: pd.DataFrame,
    policy_universe: pd.DataFrame,
    allocations: pd.DataFrame,
    *,
    pool_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _cvar_candidate_pool(base, policy_universe, allocations, pool_n)
    allocation_rows = []
    metrics = []
    scenario_rows = []
    for risk_tolerance in [0.155, 0.165, 0.175]:
        for penalty in [0.0, 0.25, 0.50, 1.00]:
            funded, local_metrics, scenario_loss = _solve_formal_cvar_lp(
                pool,
                risk_tolerance=risk_tolerance,
                cvar_penalty=penalty,
            )
            allocation_rows.append(funded)
            metrics.append(local_metrics)
            scenario_rows.append(scenario_loss)
    return (
        pd.concat(allocation_rows, ignore_index=True),
        pd.DataFrame(metrics),
        pd.concat(scenario_rows, ignore_index=True),
    )


def build_governance_selector_v3(
    full_eval: pd.DataFrame,
    ifrs9_summary: pd.DataFrame,
    online_summary: pd.DataFrame,
    online_policy_month: pd.DataFrame,
    allocations: pd.DataFrame,
    cvar_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    defended_sources = ["original_grade", "period", "grade_period", "term"]
    online_best = str(online_summary.iloc[0]["online_method_v3"])
    policy_online = (
        online_policy_month[online_policy_month["online_method_v3"].eq(online_best)]
        .groupby("policy_id", as_index=False)
        .agg(
            online_coverage_mean_v3=("coverage_online_v3", "mean"),
            online_coverage_min_v3=("coverage_online_v3", "min"),
            online_width_mean_v3=("avg_width_online_v3", "mean"),
        )
    )
    base_ifrs9 = ifrs9_summary[ifrs9_summary["scenario"].eq("baseline")][
        ["policy_id", "cashflow_ecl_v3", "net_return_after_cashflow_ecl_v3"]
    ]
    universe_dist = {}
    for source in defended_sources:
        if source == "grade_period":
            source_values = (
                allocations["original_grade"].astype(str) + "_" + allocations["period"].astype(str)
            )
        else:
            source_values = allocations[source].astype(str)
        universe_dist[source] = source_values.value_counts(normalize=True)
    mdcp_rows = []
    fair_rows = []
    for policy_id, group in allocations.groupby("policy_id"):
        for source in defended_sources:
            if source == "grade_period":
                values = group["original_grade"].astype(str) + "_" + group["period"].astype(str)
            else:
                values = group[source].astype(str)
            source_stats = group.groupby(values).agg(
                n=("loan_id", "nunique"),
                miscoverage=("miscovered_alpha01", "mean"),
            )
            source_stats = source_stats[source_stats["n"].ge(5)]
            coverage = 1.0 - source_stats["miscoverage"]
            mdcp_rows.append(
                {
                    "policy_id": policy_id,
                    "source_id": source,
                    "worst_source_coverage_v3": float(coverage.min()) if len(coverage) else np.nan,
                    "mean_source_coverage_v3": float(coverage.mean()) if len(coverage) else np.nan,
                }
            )
            dist = values.value_counts(normalize=True)
            ref = universe_dist[source]
            idx = sorted(set(ref.index) | set(dist.index))
            gap = (dist.reindex(idx, fill_value=0.0) - ref.reindex(idx, fill_value=0.0)).abs()
            fair_rows.append(
                {
                    "policy_id": policy_id,
                    "source_id": source,
                    "max_abs_gap_v3": float(gap.max()) if len(gap) else np.nan,
                    "mean_abs_gap_v3": float(gap.mean()) if len(gap) else np.nan,
                }
            )
    mdcp = (
        pd.DataFrame(mdcp_rows)
        .groupby("policy_id", as_index=False)
        .agg(worst_defended_source_coverage_v3=("worst_source_coverage_v3", "min"))
    )
    fairness = (
        pd.DataFrame(fair_rows)
        .groupby("policy_id", as_index=False)
        .agg(max_defended_proxy_gap_v3=("max_abs_gap_v3", "max"))
    )
    work = (
        full_eval.merge(base_ifrs9, on="policy_id", how="left")
        .merge(policy_online, on="policy_id", how="left")
        .merge(mdcp, on="policy_id", how="left")
        .merge(fairness, on="policy_id", how="left")
    )
    thresholds = pd.DataFrame(
        [
            {
                "threshold_id": "online_mean_coverage",
                "value": 0.90,
                "rationale": "Target marginal coverage for online conformal replay.",
                "claim_level": "promotion gate",
            },
            {
                "threshold_id": "online_min_policy_month",
                "value": 0.80,
                "rationale": "Worst policy-month tolerance used to avoid mean-only claims.",
                "claim_level": "promotion gate",
            },
            {
                "threshold_id": "mdcp_defended_sources",
                "value": 0.80,
                "rationale": "Minimum worst-source coverage over defended non-protected sources.",
                "claim_level": "promotion gate",
            },
            {
                "threshold_id": "fairness_proxy_gap",
                "value": 0.30,
                "rationale": "Proxy-governance yellow flag, not a protected-attribute fairness claim.",
                "claim_level": "screening gate",
            },
            {
                "threshold_id": "cvar_candidate_pool",
                "value": float(cvar_results["candidate_pool_n"].max())
                if not cvar_results.empty
                else np.nan,
                "rationale": "Formal CVaR LP currently uses a documented candidate pool.",
                "claim_level": "scope caveat",
            },
        ]
    )
    work["v3_gate_online"] = work["online_coverage_mean_v3"].ge(0.90) & work[
        "online_coverage_min_v3"
    ].ge(0.80)
    work["v3_gate_mdcp"] = work["worst_defended_source_coverage_v3"].ge(0.80)
    work["v3_gate_fairness_proxy"] = work["max_defended_proxy_gap_v3"].le(0.30)
    work["v3_gate_ifrs9_positive"] = work["net_return_after_cashflow_ecl_v3"].gt(0)
    work["v3_selector_score"] = (
        0.35 * _normalise(work["net_return_after_cashflow_ecl_v3"], higher_is_better=True)
        + 0.20 * _normalise(work["full_realized_return"], higher_is_better=True)
        + 0.15 * work["online_coverage_mean_v3"].fillna(0.0).clip(0, 1)
        + 0.15 * work["worst_defended_source_coverage_v3"].fillna(0.0).clip(0, 1)
        + 0.15 * (1.0 - work["max_defended_proxy_gap_v3"].fillna(1.0).clip(0, 1))
    )
    work["v3_decision"] = np.select(
        [
            work["policy_id"].eq("paper1_economic_champion"),
            work[
                [
                    "v3_gate_online",
                    "v3_gate_mdcp",
                    "v3_gate_fairness_proxy",
                    "v3_gate_ifrs9_positive",
                ]
            ].all(axis=1),
        ],
        ["protected_paper1_champion", "candidate_for_review_no_promotion"],
        default="park",
    )
    work["v3_rank"] = work["v3_selector_score"].rank(ascending=False, method="first").astype(int)
    return thresholds, work.sort_values("v3_rank")


def build_causal_dossier_high_rate(
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    work = base.copy()
    work["treatment_high_rate_within_grade"] = work["int_rate_decimal"] > work.groupby(
        "original_grade"
    )["int_rate_decimal"].transform("median")
    work["outcome_default"] = work["y_true"].astype(float)
    covariates = ["annual_inc", "dti", "loan_amnt", "fico_score", "pd_point_alpha01"]
    model_frame = (
        work[covariates]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(work[covariates].median(numeric_only=True))
    )
    grade_dummies = pd.get_dummies(
        work["original_grade"].astype(str), prefix="grade", drop_first=True
    )
    term_dummies = pd.get_dummies(work["term"].astype(str), prefix="term", drop_first=True)
    x = pd.concat([model_frame, grade_dummies, term_dummies], axis=1)
    y = work["treatment_high_rate_within_grade"].astype(int)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    logit = LogisticRegression(max_iter=500, class_weight="balanced")
    logit.fit(x_scaled, y)
    propensity = np.clip(logit.predict_proba(x_scaled)[:, 1], 0.02, 0.98)
    work["propensity_high_rate"] = propensity
    treat = y.to_numpy(dtype=float)
    work["ipw_att"] = np.where(treat == 1.0, 1.0, propensity / (1.0 - propensity))
    overlap = work.groupby(["original_grade", "period"], as_index=False).agg(
        n=("loan_id", "nunique"),
        treatment_share=("treatment_high_rate_within_grade", "mean"),
    )
    overlap["overlap_ok"] = overlap["treatment_share"].between(0.05, 0.95) & overlap["n"].ge(100)

    def smd(values: pd.Series, weights: pd.Series | None = None) -> float:
        values = pd.to_numeric(values, errors="coerce")
        values = values.fillna(float(values.median()))
        t = work["treatment_high_rate_within_grade"].astype(bool)
        if weights is None:
            mt, mc = values[t].mean(), values[~t].mean()
            vt, vc = values[t].var(), values[~t].var()
        else:
            w = (
                pd.to_numeric(weights, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            if float(w[t].sum()) <= 0 or float(w[~t].sum()) <= 0:
                return np.nan
            mt = np.average(values[t], weights=w[t])
            mc = np.average(values[~t], weights=w[~t])
            vt = np.average((values[t] - mt) ** 2, weights=w[t])
            vc = np.average((values[~t] - mc) ** 2, weights=w[~t])
        return float(abs(mt - mc) / max(math.sqrt((vt + vc) / 2.0), 1e-12))

    balance_rows = []
    for cov in covariates:
        balance_rows.append(
            {
                "covariate": cov,
                "smd_unweighted": smd(work[cov]),
                "smd_ipw_att": smd(work[cov], work["ipw_att"]),
                "balance_pass_0p10": smd(work[cov], work["ipw_att"]) <= 0.10,
            }
        )
    balance = pd.DataFrame(balance_rows)
    treated = work["treatment_high_rate_within_grade"].astype(bool)
    naive_delta = float(
        work.loc[treated, "outcome_default"].mean() - work.loc[~treated, "outcome_default"].mean()
    )
    weighted_control = np.average(
        work.loc[~treated, "outcome_default"],
        weights=work.loc[~treated, "ipw_att"],
    )
    att_ipw = float(work.loc[treated, "outcome_default"].mean() - weighted_control)
    placebo_rows = []
    for cov in ["annual_inc", "dti", "loan_amnt", "fico_score"]:
        placebo_rows.append(
            {
                "placebo_outcome": cov,
                "weighted_smd_after_ipw": smd(work[cov], work["ipw_att"]),
                "placebo_pass_0p10": smd(work[cov], work["ipw_att"]) <= 0.10,
            }
        )
    placebo = pd.DataFrame(placebo_rows)
    sens = []
    for hidden_bias in np.linspace(0.0, 0.05, 11):
        lower = att_ipw - hidden_bias
        upper = att_ipw + hidden_bias
        sens.append(
            {
                "hidden_bias_default_rate_shift": hidden_bias,
                "att_ipw": att_ipw,
                "att_lower_bound": lower,
                "att_upper_bound": upper,
                "sign_stable": bool(lower > 0 or upper < 0),
            }
        )
    sensitivity = pd.DataFrame(sens)
    dossier = pd.DataFrame(
        [
            {
                "treatment_id": "high_rate_within_grade",
                "treatment_definition": "int_rate above within-grade median",
                "outcome_definition": "observed default flag in OOT replay",
                "n": int(len(work)),
                "n_treated": int(treated.sum()),
                "n_control": int((~treated).sum()),
                "prevalence": float(treated.mean()),
                "overlap_ok_share_grade_period": float(overlap["overlap_ok"].mean()),
                "max_smd_unweighted": float(balance["smd_unweighted"].max()),
                "max_smd_ipw_att": float(balance["smd_ipw_att"].max()),
                "naive_default_delta": naive_delta,
                "att_ipw_default_delta": att_ipw,
                "placebo_pass_share": float(placebo["placebo_pass_0p10"].mean()),
                "sensitivity_sign_stable_max_bias_5pp": bool(sensitivity["sign_stable"].all()),
                "promotion_allowed": False,
                "decision": "pass_for_dossier_only_no_policy_promotion",
            }
        ]
    )
    return dossier, balance, placebo, sensitivity


def build_fairness_attribute_audit(
    base: pd.DataFrame, allocations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    direct_candidates = {
        "race": False,
        "ethnicity": False,
        "sex": False,
        "gender": False,
        "age": False,
        "date_of_birth": False,
        "marital_status": False,
    }
    present = set(base.columns)
    audit_rows = []
    for attr, protected in direct_candidates.items():
        audit_rows.append(
            {
                "attribute": attr,
                "available_in_lending_club_artifacts": attr in present,
                "protected_attribute_candidate": protected,
                "usable_for_fair_lending_claim": False,
                "decision": "unavailable_direct_attribute",
            }
        )
    proxy_sources = [
        "original_grade",
        "sub_grade",
        "term",
        "home_ownership",
        "annual_inc",
        "dti",
        "addr_state",
        "zip_code",
    ]
    for attr in proxy_sources:
        audit_rows.append(
            {
                "attribute": attr,
                "available_in_lending_club_artifacts": attr in present,
                "protected_attribute_candidate": False,
                "usable_for_fair_lending_claim": False,
                "decision": "proxy_governance_only" if attr in present else "unavailable",
            }
        )
    audit = pd.DataFrame(audit_rows)
    work = allocations.copy()
    work["zip3"] = work.get("zip_code", pd.Series("unknown", index=work.index)).astype(str).str[:3]
    sources = [
        "original_grade",
        "term",
        "home_ownership",
        "state_top20",
        "income_band",
        "dti_band",
        "zip3",
    ]
    universe = base.copy()
    universe["zip3"] = (
        universe.get("zip_code", pd.Series("unknown", index=universe.index)).astype(str).str[:3]
    )
    stress_rows = []
    for policy_id, group in work.groupby("policy_id"):
        for source in sources:
            funded_dist = group[source].astype(str).value_counts(normalize=True)
            universe_dist = universe[source].astype(str).value_counts(normalize=True)
            idx = sorted(set(funded_dist.index) | set(universe_dist.index))
            gap = (
                funded_dist.reindex(idx, fill_value=0.0)
                - universe_dist.reindex(idx, fill_value=0.0)
            ).abs()
            stress_rows.append(
                {
                    "policy_id": policy_id,
                    "source_id": source,
                    "max_abs_gap": float(gap.max()) if len(gap) else np.nan,
                    "mean_abs_gap": float(gap.mean()) if len(gap) else np.nan,
                    "n_source_values": int(len(idx)),
                    "proxy_governance_pass_30": bool(gap.max() <= 0.30) if len(gap) else False,
                    "claim_scope": "proxy_governance_not_protected_attribute_fairness",
                }
            )
    stress = pd.DataFrame(stress_rows)
    summary = (
        stress.groupby("policy_id", as_index=False)
        .agg(
            max_proxy_gap=("max_abs_gap", "max"),
            mean_proxy_gap=("mean_abs_gap", "mean"),
            proxy_sources_pass_30=("proxy_governance_pass_30", "mean"),
        )
        .sort_values("max_proxy_gap")
    )
    return audit, stress, summary


def _month_solve(
    loans: pd.DataFrame,
    policy: pd.Series,
    budget: float,
    *,
    deployment_rate: float,
) -> pd.DataFrame:
    if loans.empty or budget <= 1_000:
        return pd.DataFrame()
    local_budget = min(float(budget) * deployment_rate, BUDGET)
    effective_pd = _policy_effective_pd(loans, policy)
    solution = optimize_portfolio_allocation(
        loans=loans,
        pd_point=loans["pd_point_alpha01"].to_numpy(dtype=float),
        pd_low=loans["pd_low_alpha01"].to_numpy(dtype=float),
        pd_high=loans["pd_high_alpha01"].to_numpy(dtype=float),
        lgd=np.full(len(loans), DEFAULT_LGD, dtype=float),
        int_rates=loans["int_rate_decimal"].to_numpy(dtype=float),
        total_budget=local_budget,
        max_concentration=0.25,
        max_portfolio_pd=float(policy["risk_tolerance"]),
        robust=True,
        uncertainty_aversion=float(policy.get("uncertainty_aversion", 0.0)),
        min_budget_utilization=0.80,
        pd_constraint_override=effective_pd,
        time_limit=120,
        threads=4,
        solver_backend="highs",
    )
    allocation = np.array([float(solution["allocation"].get(i, 0.0)) for i in range(len(loans))])
    mask = allocation > 1e-8
    funded = loans.loc[mask].copy()
    funded["allocation_fraction"] = allocation[mask]
    funded["funded_exposure"] = funded["allocation_fraction"] * funded["loan_amnt"]
    funded["policy_id"] = policy["policy_id"]
    funded["loan_id"] = funded["id"].astype(str)
    funded["solver_status"] = str(solution.get("solver_status", "unknown"))
    return funded


def build_multiperiod_loan_level_replay(
    base: pd.DataFrame,
    policy_universe: pd.DataFrame,
    selector_v3: pd.DataFrame,
    *,
    max_months: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    champion_id = "paper1_economic_champion"
    best_ifrs9 = (
        selector_v3[~selector_v3["policy_id"].eq(champion_id)]
        .sort_values("net_return_after_cashflow_ecl_v3", ascending=False)
        .iloc[0]["policy_id"]
    )
    best_selector = (
        selector_v3[~selector_v3["policy_id"].eq(champion_id)]
        .sort_values("v3_selector_score", ascending=False)
        .iloc[0]["policy_id"]
    )
    strategies = {
        "static_champion_monthly": champion_id,
        "ifrs9_best_monthly": best_ifrs9,
        "selector_v3_best_monthly": best_selector,
    }
    months = sorted(base["issue_month"].dropna().unique())
    if max_months is not None:
        months = months[:max_months]
    state_rows: list[dict[str, Any]] = []
    decision_rows: list[pd.DataFrame] = []
    for strategy, policy_id in strategies.items():
        policy = policy_universe[policy_universe["policy_id"].eq(policy_id)].iloc[0]
        cash_budget = float(BUDGET)
        outstanding: list[dict[str, Any]] = []
        cumulative_return = 0.0
        cumulative_losses = 0.0
        for month_idx, month in enumerate(months, start=1):
            principal_in = 0.0
            interest_in = 0.0
            losses = 0.0
            next_outstanding = []
            for item in outstanding:
                age = month_idx - int(item["start_idx"])
                if age <= 0 or age > int(item["term"]):
                    next_outstanding.append(item)
                    continue
                balance = max(0.0, item["exposure"] * (1.0 - (age - 1) / item["term"]))
                principal = item["exposure"] / item["term"]
                interest = balance * item["int_rate_decimal"] / 12.0
                default_month = min(12, int(item["term"]))
                loss = (
                    item["exposure"] * DEFAULT_LGD
                    if item["y_true"] >= 1.0 and age == default_month
                    else 0.0
                )
                principal_in += 0.0 if loss > 0 else principal
                interest_in += 0.0 if loss > 0 else interest
                losses += loss
                if age < int(item["term"]) and loss == 0.0:
                    next_outstanding.append(item)
            cash_budget += principal_in + interest_in - losses
            cumulative_return += interest_in
            cumulative_losses += losses
            available = base[base["issue_month"].eq(month)].copy()
            funded = _month_solve(available, policy, cash_budget, deployment_rate=0.35)
            deployed = float(funded["funded_exposure"].sum()) if not funded.empty else 0.0
            cash_budget -= deployed
            if not funded.empty:
                funded["strategy"] = strategy
                funded["decision_month"] = month
                funded["month_idx"] = month_idx
                decision_rows.append(
                    funded[
                        [
                            "strategy",
                            "policy_id",
                            "decision_month",
                            "month_idx",
                            "loan_id",
                            "original_grade",
                            "term",
                            "loan_amnt",
                            "funded_exposure",
                            "int_rate_decimal",
                            "pd_high_alpha01",
                            "y_true",
                            "solver_status",
                        ]
                    ]
                )
                for _, row in funded.iterrows():
                    outstanding.append(
                        {
                            "start_idx": month_idx,
                            "term": int(row["term"]),
                            "exposure": float(row["funded_exposure"]),
                            "int_rate_decimal": float(row["int_rate_decimal"]),
                            "y_true": float(row["y_true"]),
                        }
                    )
            outstanding = next_outstanding + [
                item for item in outstanding if int(item["start_idx"]) == month_idx
            ]
            state_rows.append(
                {
                    "strategy": strategy,
                    "policy_id": policy_id,
                    "month": month,
                    "month_idx": month_idx,
                    "cash_budget_end": cash_budget,
                    "principal_in": principal_in,
                    "interest_in": interest_in,
                    "realized_losses": losses,
                    "deployed_new": deployed,
                    "outstanding_exposure_end": sum(
                        float(item["exposure"]) for item in outstanding
                    ),
                    "active_loans_end": len(outstanding),
                    "cumulative_interest": cumulative_return,
                    "cumulative_losses": cumulative_losses,
                    "net_cash_result": cash_budget
                    + sum(float(item["exposure"]) for item in outstanding)
                    - BUDGET,
                    "status": "loan_level_state_transition_proxy_not_deployment_policy",
                }
            )
    states = pd.DataFrame(state_rows)
    decisions = pd.concat(decision_rows, ignore_index=True) if decision_rows else pd.DataFrame()
    summary = (
        states.groupby(["strategy", "policy_id"], as_index=False)
        .agg(
            months=("month", "nunique"),
            final_cash_budget=("cash_budget_end", "last"),
            final_outstanding_exposure=("outstanding_exposure_end", "last"),
            cumulative_interest=("cumulative_interest", "last"),
            cumulative_losses=("cumulative_losses", "last"),
            net_cash_result=("net_cash_result", "last"),
            total_deployed=("deployed_new", "sum"),
        )
        .sort_values("net_cash_result", ascending=False)
    )
    return states, decisions, summary


def build_claim_matrix_v3() -> pd.DataFrame:
    rows = [
        (
            "Full exact all policies",
            "exact_full_universe_45",
            "paper4_v3_full_universe_all_policy_allocations.parquet",
            "19aa-v3-full-exact-and-ifrs9-realistic.qmd",
            "no promotion JSON",
        ),
        (
            "IFRS9 cashflow lifetime",
            "cashflow_proxy",
            "paper4_v3_ifrs9_cashflow_policy_summary.csv",
            "19aa-v3-full-exact-and-ifrs9-realistic.qmd",
            "no contractual performance panel",
        ),
        (
            "Online CP v3",
            "method_search",
            "paper4_v3_online_conformal_method_summary.csv",
            "19ab-v3-online-cvar-governance.qmd",
            "not final theorem-level online CP",
        ),
        (
            "Formal CVaR LP",
            "candidate_pool_formal_lp",
            "paper4_v3_formal_cvar_lp_results.csv",
            "19ab-v3-online-cvar-governance.qmd",
            "candidate pool, not full universe",
        ),
        (
            "Governance thresholds",
            "explicit_rationale",
            "paper4_v3_selector_governance_thresholds.csv",
            "19ab-v3-online-cvar-governance.qmd",
            "thresholds need committee validation",
        ),
        (
            "Causal high-rate dossier",
            "dossier_only",
            "paper4_v3_causal_high_rate_dossier.csv",
            "19ac-v3-causal-fairness-multiperiod.qmd",
            "no causal policy promotion",
        ),
        (
            "Fairness attribute audit",
            "proxy_governance",
            "paper4_v3_fairness_attribute_audit.csv",
            "19ac-v3-causal-fairness-multiperiod.qmd",
            "no protected attributes",
        ),
        (
            "Loan-level multi-period replay",
            "state_transition_proxy",
            "paper4_v3_multiperiod_loan_level_state.parquet",
            "19ac-v3-causal-fairness-multiperiod.qmd",
            "repayment/default timing proxy",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["priority", "claim_status", "artifact", "quarto_page", "caveat"]
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-full-solve", action="store_true")
    parser.add_argument("--max-full-policies", type=int, default=45)
    parser.add_argument("--cvar-pool-n", type=int, default=20_000)
    parser.add_argument("--max-months", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    base = _prepare_base(_load_base_loan_frame())
    base["loan_id"] = base["id"].astype(str)
    policy_universe = load_policy_universe()

    allocations, full_eval, full_status = build_full_universe_all_policy_allocations(
        base,
        policy_universe,
        max_policies=args.max_full_policies,
        force=args.force_full_solve,
    )
    generated = [
        _write_parquet("paper4_v3_full_universe_all_policy_allocations.parquet", allocations),
        _write_csv("paper4_v3_full_universe_all_policy_eval.csv", full_eval),
        _write_json("paper4_v3_full_universe_all_policy_status.json", full_status),
    ]

    ifrs9_grid, ifrs9_transitions, ifrs9_summary, ifrs9_stage_matrix = (
        build_ifrs9_cashflow_lifetime(allocations)
    )
    generated.extend(
        [
            _write_parquet("paper4_v3_ifrs9_cashflow_loan_month_grid.parquet", ifrs9_grid),
            _write_parquet("paper4_v3_ifrs9_cashflow_transition_loan.parquet", ifrs9_transitions),
            _write_csv("paper4_v3_ifrs9_cashflow_policy_summary.csv", ifrs9_summary),
            _write_csv("paper4_v3_ifrs9_stage_transition_matrix.csv", ifrs9_stage_matrix),
        ]
    )

    intervals, online_policy_month, online_source_month, online_summary = build_online_conformal_v3(
        base, allocations
    )
    generated.extend(
        [
            _write_parquet("paper4_v3_online_conformal_intervals.parquet", intervals),
            _write_parquet("paper4_v3_online_conformal_policy_month.parquet", online_policy_month),
            _write_parquet("paper4_v3_online_conformal_source_month.parquet", online_source_month),
            _write_csv("paper4_v3_online_conformal_method_summary.csv", online_summary),
        ]
    )

    cvar_allocations, cvar_results, cvar_scenarios = build_formal_cvar_lp(
        base,
        policy_universe,
        allocations,
        pool_n=args.cvar_pool_n,
    )
    generated.extend(
        [
            _write_parquet("paper4_v3_formal_cvar_lp_allocations.parquet", cvar_allocations),
            _write_csv("paper4_v3_formal_cvar_lp_results.csv", cvar_results),
            _write_csv("paper4_v3_formal_cvar_lp_scenario_losses.csv", cvar_scenarios),
        ]
    )

    thresholds, selector_v3 = build_governance_selector_v3(
        full_eval,
        ifrs9_summary,
        online_summary,
        online_policy_month,
        allocations,
        cvar_results,
    )
    generated.extend(
        [
            _write_csv("paper4_v3_selector_governance_thresholds.csv", thresholds),
            _write_csv("paper4_v3_selector_results.csv", selector_v3),
        ]
    )

    causal_dossier, causal_balance, causal_placebo, causal_sensitivity = (
        build_causal_dossier_high_rate(base)
    )
    generated.extend(
        [
            _write_csv("paper4_v3_causal_high_rate_dossier.csv", causal_dossier),
            _write_csv("paper4_v3_causal_high_rate_balance.csv", causal_balance),
            _write_csv("paper4_v3_causal_high_rate_placebo.csv", causal_placebo),
            _write_csv("paper4_v3_causal_high_rate_sensitivity.csv", causal_sensitivity),
        ]
    )

    fairness_audit, fairness_stress, fairness_summary = build_fairness_attribute_audit(
        base, allocations
    )
    generated.extend(
        [
            _write_csv("paper4_v3_fairness_attribute_audit.csv", fairness_audit),
            _write_csv("paper4_v3_fairness_proxy_stress.csv", fairness_stress),
            _write_csv("paper4_v3_fairness_proxy_summary.csv", fairness_summary),
        ]
    )

    mp_state, mp_decisions, mp_summary = build_multiperiod_loan_level_replay(
        base,
        policy_universe,
        selector_v3,
        max_months=args.max_months,
    )
    generated.extend(
        [
            _write_parquet("paper4_v3_multiperiod_loan_level_state.parquet", mp_state),
            _write_parquet("paper4_v3_multiperiod_loan_level_decisions.parquet", mp_decisions),
            _write_csv("paper4_v3_multiperiod_loan_level_summary.csv", mp_summary),
        ]
    )

    claim_matrix = build_claim_matrix_v3()
    generated.append(_write_csv("paper4_v3_claim_artifact_matrix.csv", claim_matrix))
    generated.append(
        _write_note(
            "paper4_v3_deepening_memo.qmd",
            """---
title: "Paper 4 v3 Deepening Memo"
format: html
---

# Paper 4 v3 Deepening Memo

This run deepens the eight highest-priority open lanes. It remains a living-lab
run: it protects the Paper Estrella champion, does not create
`paper4_final_promotion.json`, and labels IFRS9, fairness, causal and
multi-period outputs with their current evidence scope.

Key guardrail: v3 can produce candidates for review, but not a replacement
champion.
""",
        )
    )
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v3_deepening_eight_open_priorities",
        "mode": "research_deepening_no_promotion",
        "paper1_champion_protected": True,
        "paper4_final_promotion_created": False,
        "priorities_completed": 8,
        "full_exact_policy_count": int(allocations["policy_id"].nunique()),
        "full_exact_target_policy_count": int(args.max_full_policies),
        "cvar_candidate_pool_n": int(args.cvar_pool_n),
        "generated_artifacts": [str(path.relative_to(ROOT)) for path in generated],
    }
    generated.append(_write_json("paper4_v3_deepening_status.json", status))
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
