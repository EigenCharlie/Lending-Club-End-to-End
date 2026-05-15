"""Build Paper 4 v4 artifacts for the eleven active open priorities.

This layer keeps Paper Estrella frozen.  It may name Paper 4 working champions,
but it never writes or updates ``models/final_project_promotion.json`` and it
never creates ``paper4_final_promotion.json``.  Every comparison against Paper
Estrella is a reference benchmark, not a request to replace the thesis/journal
champion.
"""

from __future__ import annotations

import argparse
import hashlib
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
    _safe_read_csv,
    _safe_read_parquet,
    _weighted_average,
)
from scripts.papers.build_paper4_living_lab_artifacts import DEFAULT_LGD
from scripts.papers.build_paper4_next_wave_experiments import _as_month, _prepare_base
from src.optimization.portfolio_model import optimize_portfolio_allocation

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = OUT_ROOT / "tables"
STATUS_DIR = OUT_ROOT / "status"
NOTE_DIR = OUT_ROOT / "notes"
FIGURE_DIR = OUT_ROOT / "figures"

SCHEMA_VERSION = "2026-05-13.2"
FROZEN_PAPER1_CHAMPION = "paper1_economic_champion"
GROSS_CHALLENGER = "crpto_rt0p175_g0p45_u0p00_alpha0p01_conservative_proxy_rs42"
RNG_SEED = 20260513


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


def _policy_id(rt: float, gamma: float, u: float) -> str:
    return f"paper4_v4_local_rt{rt:.4f}_g{gamma:.3f}_u{u:.3f}".replace(".", "p")


def _grid_policy_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rt in [0.1700, 0.1725, 0.1750, 0.1775]:
        for gamma in [0.400, 0.425, 0.450, 0.475, 0.500]:
            for u in [0.000, 0.025, 0.050, 0.075, 0.100]:
                rows.append(
                    {
                        "policy_id": _policy_id(rt, gamma, u),
                        "risk_tolerance": rt,
                        "gamma": gamma,
                        "policy_mode": "blended_uncertainty",
                        "uncertainty_aversion": u,
                        "source": "paper4_v4_challenger_local_grid",
                    }
                )
    return pd.DataFrame(rows)


def _load_v3_allocations() -> pd.DataFrame:
    allocations = _safe_read_parquet(
        TABLE_DIR / "paper4_v3_full_universe_all_policy_allocations.parquet"
    )
    if allocations.empty:
        raise FileNotFoundError(
            "Run v3 first: paper4_v3_full_universe_all_policy_allocations.parquet"
        )
    allocations = allocations.copy()
    allocations["loan_id"] = allocations["loan_id"].astype(str)
    allocations["issue_month"] = _as_month(allocations["issue_month"])
    return allocations


def _load_v3_eval() -> pd.DataFrame:
    eval_table = _safe_read_csv(TABLE_DIR / "paper4_v3_full_universe_all_policy_eval.csv")
    if eval_table.empty:
        raise FileNotFoundError("Run v3 first: paper4_v3_full_universe_all_policy_eval.csv")
    return eval_table


def _load_performance_reference() -> pd.DataFrame:
    """Load loan status/LGD/EAD columns when available."""

    candidates = [
        ROOT / "data" / "processed" / "ead_dataset.parquet",
        ROOT / "data" / "processed" / "loan_master.parquet",
    ]
    frames = []
    keep = [
        "id",
        "loan_status",
        "installment",
        "funded_amnt",
        "total_pymnt",
        "total_rec_prncp",
        "total_rec_int",
        "recoveries",
        "collection_recovery_fee",
        "out_prncp",
        "last_pymnt_d",
        "next_pymnt_d",
        "lgd",
        "lgd_months_since_issue",
        "lgd_is_mature_24m",
        "default_flag",
    ]
    for path in candidates:
        frame = _safe_read_parquet(path)
        if frame.empty or "id" not in frame.columns:
            continue
        cols = [col for col in keep if col in frame.columns]
        local = frame[cols].copy()
        local["loan_id"] = local["id"].astype(str)
        frames.append(local.drop_duplicates("loan_id"))
    if not frames:
        return pd.DataFrame(columns=["loan_id"])
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="loan_id", how="left", suffixes=("", "_alt"))
        for col in frame.columns:
            if col in {"loan_id", "id"}:
                continue
            alt = f"{col}_alt"
            if alt in out.columns:
                out[col] = out[col].where(out[col].notna(), out[alt])
                out = out.drop(columns=[alt])
    return out.drop_duplicates("loan_id")


def _build_candidate_pool(
    base: pd.DataFrame,
    policies: pd.DataFrame,
    seed_allocations: pd.DataFrame,
    *,
    candidate_pool_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    seed_ids = set(seed_allocations["loan_id"].astype(str))
    score_matrix = []
    effective_by_policy: dict[str, np.ndarray] = {}
    for _, policy in policies.iterrows():
        effective = _policy_effective_pd(base, policy)
        effective_by_policy[str(policy["policy_id"])] = effective
        point = base["pd_point_alpha01"].to_numpy(dtype=float)
        high = base["pd_high_alpha01"].to_numpy(dtype=float)
        rate = base["int_rate_decimal"].to_numpy(dtype=float)
        u = float(policy["uncertainty_aversion"])
        rt = float(policy["risk_tolerance"])
        score = (
            rate
            - point * DEFAULT_LGD
            - u * np.clip(high - point, 0.0, 1.0) * DEFAULT_LGD
            - 0.50 * np.maximum(effective - rt, 0.0)
        )
        score_matrix.append(score)
    max_score = np.max(np.vstack(score_matrix), axis=0)
    min_effective = np.min(np.vstack(list(effective_by_policy.values())), axis=0)
    work = base.copy()
    work["loan_id"] = work["id"].astype(str)
    work["v4_pool_score"] = max_score
    work["v4_min_effective_pd"] = min_effective
    seeded = work[work["loan_id"].isin(seed_ids)].copy()
    low_risk = work[work["v4_min_effective_pd"].le(0.24)].sort_values(
        "v4_pool_score", ascending=False
    )
    filler = work.sort_values("v4_pool_score", ascending=False)
    pool = pd.concat([seeded, low_risk, filler], ignore_index=True)
    pool = pool.drop_duplicates("loan_id").head(candidate_pool_n).copy()
    status = {
        "candidate_pool_n": int(len(pool)),
        "seeded_policy_loan_count": int(len(seeded)),
        "seeded_unique_loan_count": int(seeded["loan_id"].nunique()),
        "pool_source": "v3 exact allocations + max local-grid objective score + low-risk fill",
    }
    return pool.reset_index(drop=True), status


def _solve_policy_on_pool(
    pool: pd.DataFrame, policy: pd.Series
) -> tuple[pd.DataFrame, dict[str, Any]]:
    policy_id = str(policy["policy_id"])
    effective = _policy_effective_pd(pool, policy)
    t0 = time.perf_counter()
    min_budget = 0.95
    try:
        solution = optimize_portfolio_allocation(
            loans=pool,
            pd_point=pool["pd_point_alpha01"].to_numpy(dtype=float),
            pd_low=pool["pd_low_alpha01"].to_numpy(dtype=float),
            pd_high=pool["pd_high_alpha01"].to_numpy(dtype=float),
            lgd=np.full(len(pool), DEFAULT_LGD, dtype=float),
            int_rates=pool["int_rate_decimal"].to_numpy(dtype=float),
            total_budget=BUDGET,
            max_concentration=0.25,
            max_portfolio_pd=float(policy["risk_tolerance"]),
            robust=True,
            uncertainty_aversion=float(policy["uncertainty_aversion"]),
            min_budget_utilization=min_budget,
            pd_constraint_override=effective,
            time_limit=90,
            threads=4,
            solver_backend="highs",
        )
    except RuntimeError:
        min_budget = 0.80
        solution = optimize_portfolio_allocation(
            loans=pool,
            pd_point=pool["pd_point_alpha01"].to_numpy(dtype=float),
            pd_low=pool["pd_low_alpha01"].to_numpy(dtype=float),
            pd_high=pool["pd_high_alpha01"].to_numpy(dtype=float),
            lgd=np.full(len(pool), DEFAULT_LGD, dtype=float),
            int_rates=pool["int_rate_decimal"].to_numpy(dtype=float),
            total_budget=BUDGET,
            max_concentration=0.25,
            max_portfolio_pd=float(policy["risk_tolerance"]),
            robust=True,
            uncertainty_aversion=float(policy["uncertainty_aversion"]),
            min_budget_utilization=min_budget,
            pd_constraint_override=effective,
            time_limit=90,
            threads=4,
            solver_backend="highs",
        )
    elapsed = time.perf_counter() - t0
    allocation = np.array([float(solution["allocation"].get(i, 0.0)) for i in range(len(pool))])
    mask = allocation > 1e-8
    funded = pool.loc[mask].copy()
    funded["policy_id"] = policy_id
    funded["loan_id"] = funded["id"].astype(str)
    funded["allocation_fraction"] = allocation[mask]
    funded["funded_exposure"] = funded["allocation_fraction"] * funded["loan_amnt"]
    funded["portfolio_weight"] = funded["funded_exposure"] / max(
        float(funded["funded_exposure"].sum()), 1e-12
    )
    funded["pd_point"] = funded["pd_point_alpha01"]
    funded["effective_pd_alpha01"] = effective[mask]
    funded["miscovered_alpha01"] = funded["y_true"].gt(funded["pd_high_alpha01"])
    funded["realized_return_proxy_lgd45"] = (
        funded["funded_exposure"] * funded["int_rate_decimal"] * (1.0 - funded["y_true"])
        - funded["funded_exposure"] * DEFAULT_LGD * funded["y_true"]
    )
    funded["ecl_baseline_lgd45"] = (
        funded["pd_high_alpha01"] * DEFAULT_LGD * funded["funded_exposure"]
    )
    funded["solver_status"] = str(solution.get("solver_status", "unknown"))
    funded["candidate_pool_n"] = int(len(pool))
    funded["solve_scope"] = "paper4_v4_local_candidate_pool_exact_lp"
    funded["v4_solve_seconds"] = elapsed
    funded["min_budget_utilization_used"] = min_budget
    metrics = {
        "policy_id": policy_id,
        "risk_tolerance": float(policy["risk_tolerance"]),
        "gamma": float(policy["gamma"]),
        "policy_mode": str(policy["policy_mode"]),
        "uncertainty_aversion": float(policy["uncertainty_aversion"]),
        "solver_status": str(solution.get("solver_status", "unknown")),
        "solve_scope": "paper4_v4_local_candidate_pool_exact_lp",
        "candidate_pool_n": int(len(pool)),
        "min_budget_utilization_used": min_budget,
        "solve_seconds": elapsed,
        "n_funded": int(funded["loan_id"].nunique()),
        "funded_exposure": float(funded["funded_exposure"].sum()),
        "realized_return_proxy_lgd45": float(funded["realized_return_proxy_lgd45"].sum()),
        "ecl_baseline_lgd45": float(funded["ecl_baseline_lgd45"].sum()),
        "weighted_pd_point": _weighted_average(
            funded["pd_point_alpha01"], funded["funded_exposure"]
        ),
        "weighted_pd_high": _weighted_average(funded["pd_high_alpha01"], funded["funded_exposure"]),
        "weighted_effective_pd": _weighted_average(
            funded["effective_pd_alpha01"], funded["funded_exposure"]
        ),
        "coverage_alpha01": float(1.0 - funded["miscovered_alpha01"].mean()),
        "default_rate_observed": float(funded["y_true"].mean()),
    }
    keep = [
        "policy_id",
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
        "loan_amnt",
        "int_rate",
        "int_rate_decimal",
        "annual_inc",
        "dti",
        "fico_score",
        "zip_code",
        "y_true",
        "allocation_fraction",
        "funded_exposure",
        "portfolio_weight",
        "pd_point_alpha01",
        "pd_low_alpha01",
        "pd_high_alpha01",
        "pd_point",
        "effective_pd_alpha01",
        "miscovered_alpha01",
        "realized_return_proxy_lgd45",
        "ecl_baseline_lgd45",
        "solver_status",
        "candidate_pool_n",
        "solve_scope",
        "v4_solve_seconds",
        "min_budget_utilization_used",
    ]
    return funded[[col for col in keep if col in funded.columns]], metrics


def build_challenger_local_search(
    base: pd.DataFrame,
    v3_allocations: pd.DataFrame,
    v3_eval: pd.DataFrame,
    *,
    candidate_pool_n: int,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    out_csv = TABLE_DIR / "paper4_challenger_local_search.csv"
    out_alloc = TABLE_DIR / "paper4_challenger_local_allocations.parquet"
    out_pool = TABLE_DIR / "paper4_challenger_local_candidate_pool.parquet"
    status_path = STATUS_DIR / "paper4_challenger_local_search_status.json"
    if (
        not force
        and out_csv.exists()
        and out_alloc.exists()
        and out_pool.exists()
        and status_path.exists()
    ):
        return (
            pd.read_csv(out_csv),
            pd.read_parquet(out_alloc),
            pd.read_parquet(out_pool),
            json.loads(status_path.read_text(encoding="utf-8")),
        )

    policies = _grid_policy_frame()
    seed = v3_allocations[
        v3_allocations["policy_id"].isin([FROZEN_PAPER1_CHAMPION, GROSS_CHALLENGER])
    ].copy()
    pool, pool_status = _build_candidate_pool(
        base, policies, seed, candidate_pool_n=candidate_pool_n
    )
    allocation_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    started = datetime.now(UTC).isoformat()
    for idx, policy in policies.iterrows():
        funded, metrics = _solve_policy_on_pool(pool, policy)
        allocation_frames.append(funded)
        metric_rows.append(metrics)
        if (idx + 1) % 10 == 0:
            partial_alloc = pd.concat(allocation_frames, ignore_index=True)
            partial_eval = pd.DataFrame(metric_rows)
            _write_parquet("paper4_challenger_local_allocations.parquet", partial_alloc)
            _write_csv("paper4_challenger_local_search.csv", partial_eval)
    allocations = pd.concat(allocation_frames, ignore_index=True)
    search = pd.DataFrame(metric_rows)
    frozen_return = float(
        v3_eval.loc[v3_eval["policy_id"].eq(FROZEN_PAPER1_CHAMPION), "full_realized_return"].iloc[0]
    )
    gross_return = float(
        v3_eval.loc[v3_eval["policy_id"].eq(GROSS_CHALLENGER), "full_realized_return"].iloc[0]
    )
    search["delta_vs_paper1_frozen_full_exact"] = (
        search["realized_return_proxy_lgd45"] - frozen_return
    )
    search["delta_vs_v3_gross_challenger_full_exact"] = (
        search["realized_return_proxy_lgd45"] - gross_return
    )
    search["paper4_local_rank_return"] = (
        search["realized_return_proxy_lgd45"].rank(ascending=False, method="first").astype(int)
    )
    search["grid_family"] = "uncertainty_aversion_local_search"
    search = search.sort_values("paper4_local_rank_return")
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "started_at_utc": started,
        "mode": "paper4_v4_local_challenger_search_no_paper1_changes",
        "paper1_frozen_reference_policy": FROZEN_PAPER1_CHAMPION,
        "paper1_artifacts_modified": False,
        "paper4_final_promotion_created": False,
        "grid_policy_count": int(len(policies)),
        **pool_status,
        "best_policy_id": str(search.iloc[0]["policy_id"]),
        "best_local_return": float(search.iloc[0]["realized_return_proxy_lgd45"]),
    }
    _write_parquet("paper4_challenger_local_allocations.parquet", allocations)
    _write_csv("paper4_challenger_local_search.csv", search)
    _write_parquet("paper4_challenger_local_candidate_pool.parquet", pool)
    _write_json("paper4_challenger_local_search_status.json", status)
    return search, allocations, pool, status


def _hash_uniform(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def build_monthly_bootstrap(
    search_allocations: pd.DataFrame, v3_allocations: pd.DataFrame
) -> pd.DataFrame:
    reference = v3_allocations[v3_allocations["policy_id"].eq(FROZEN_PAPER1_CHAMPION)].copy()
    reference["policy_id"] = FROZEN_PAPER1_CHAMPION
    work = pd.concat([search_allocations, reference], ignore_index=True, sort=False)
    monthly = (
        work.groupby(["policy_id", "issue_month"], as_index=False)
        .agg(
            month_return=("realized_return_proxy_lgd45", "sum"),
            funded_exposure=("funded_exposure", "sum"),
        )
        .rename(columns={"issue_month": "month"})
    )
    pivot = monthly.pivot_table(
        index="month", columns="policy_id", values="month_return", fill_value=0.0
    )
    ref = (
        pivot[FROZEN_PAPER1_CHAMPION]
        if FROZEN_PAPER1_CHAMPION in pivot
        else pd.Series(0.0, index=pivot.index)
    )
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    months = np.array(pivot.index)
    n_months = len(months)
    for policy_id in pivot.columns:
        if policy_id == FROZEN_PAPER1_CHAMPION:
            continue
        diff = (pivot[policy_id] - ref).to_numpy(dtype=float)
        draws = []
        for _ in range(500):
            idx = rng.integers(0, n_months, n_months)
            draws.append(float(diff[idx].sum()))
        rows.append(
            {
                "policy_id": policy_id,
                "reference_policy_id": FROZEN_PAPER1_CHAMPION,
                "n_months": int(n_months),
                "paired_monthly_diff": float(diff.sum()),
                "bootstrap_p05": float(np.quantile(draws, 0.05)),
                "bootstrap_p50": float(np.quantile(draws, 0.50)),
                "bootstrap_p95": float(np.quantile(draws, 0.95)),
                "prob_diff_positive": float(np.mean(np.asarray(draws) > 0.0)),
                "bootstrap_scope": "common observed issue-month resampling",
            }
        )
    return pd.DataFrame(rows).sort_values("paired_monthly_diff", ascending=False)


def _hierarchical_q(
    prior: pd.DataFrame, current: pd.DataFrame, method: str
) -> tuple[pd.Series, pd.Series]:
    if prior.empty:
        return pd.Series(0.55, index=current.index), pd.Series(
            "first_month_guard", index=current.index
        )
    global_q = float(prior["score_abs"].quantile(0.92))
    q = pd.Series(global_q, index=current.index)
    source = pd.Series("global_prior", index=current.index)
    if method == "hierarchical_mondrian_shrinkage":
        levels = [
            ["original_grade", "score_decile"],
            ["original_grade", "period"],
            ["original_grade"],
        ]
        for cols in levels:
            key_prior = prior[cols].astype(str).agg("|".join, axis=1)
            key_current = current[cols].astype(str).agg("|".join, axis=1)
            stats = (
                prior.assign(_key=key_prior).groupby("_key")["score_abs"].agg(["count", "quantile"])
            )
            stats = (
                prior.assign(_key=key_prior)
                .groupby("_key")["score_abs"]
                .quantile(0.90)
                .to_frame("local_q")
                .join(
                    prior.assign(_key=key_prior).groupby("_key")["score_abs"].size().to_frame("n")
                )
            )
            for key, idx in key_current.groupby(key_current).groups.items():
                if (
                    key in stats.index
                    and stats.loc[key, "n"] >= 80
                    and (q.loc[idx] == global_q).all()
                ):
                    n = float(stats.loc[key, "n"])
                    shrink = n / (n + 250.0)
                    q.loc[idx] = (
                        shrink * float(stats.loc[key, "local_q"]) + (1.0 - shrink) * global_q
                    )
                    source.loc[idx] = f"shrink_{'+'.join(cols)}"
    elif method == "minimum_cell_pooling":
        key_prior = prior["original_grade"].astype(str) + "|" + prior["period"].astype(str)
        key_current = current["original_grade"].astype(str) + "|" + current["period"].astype(str)
        stats = (
            prior.assign(_key=key_prior)
            .groupby("_key")["score_abs"]
            .quantile(0.94)
            .to_frame("q")
            .join(prior.assign(_key=key_prior).groupby("_key")["score_abs"].size().to_frame("n"))
        )
        grade_stats = prior.groupby("original_grade")["score_abs"].quantile(0.93)
        for key, idx in key_current.groupby(key_current).groups.items():
            grade = str(key).split("|")[0]
            if key in stats.index and stats.loc[key, "n"] >= 120:
                q.loc[idx] = float(stats.loc[key, "q"])
                source.loc[idx] = "grade_period_min120"
            elif grade in grade_stats.index:
                q.loc[idx] = max(global_q, float(grade_stats.loc[grade]))
                source.loc[idx] = "grade_pool_fallback"
    elif method == "aci_local_guard":
        key_prior = prior["original_grade"].astype(str)
        key_current = current["original_grade"].astype(str)
        grade_stats = prior.assign(_key=key_prior).groupby("_key")["score_abs"].quantile(0.95)
        for key, idx in key_current.groupby(key_current).groups.items():
            q.loc[idx] = max(global_q, float(grade_stats.get(key, global_q)))
            source.loc[idx] = "grade_aci_guard"
    elif method == "source_aware_guarded":
        q[:] = max(global_q, float(prior["score_abs"].quantile(0.96)))
        source[:] = "source_aware_base_guard"
    else:
        raise ValueError(method)
    return q.clip(0.0, 1.0), source


def build_online_conformal_v4(
    base: pd.DataFrame,
    allocations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    months = sorted(work["issue_month"].dropna().unique())
    methods = [
        "hierarchical_mondrian_shrinkage",
        "minimum_cell_pooling",
        "aci_local_guard",
        "source_aware_guarded",
    ]
    interval_frames: list[pd.DataFrame] = []
    for method in methods:
        for month in months:
            current = work[work["issue_month"].eq(month)].copy()
            prior = work[work["issue_month"].lt(month)]
            q, source = _hierarchical_q(prior, current, method)
            current["online_method_v4"] = method
            current["online_source_v4"] = source
            current["qhat_v4"] = q
            current["pd_low_online_v4"] = np.clip(current["y_pred"] - current["qhat_v4"], 0.0, 1.0)
            current["pd_high_online_v4"] = np.clip(current["y_pred"] + current["qhat_v4"], 0.0, 1.0)
            current["covered_online_v4"] = current["y_true"].between(
                current["pd_low_online_v4"], current["pd_high_online_v4"]
            )
            current["interval_width_online_v4"] = (
                current["pd_high_online_v4"] - current["pd_low_online_v4"]
            )
            interval_frames.append(current)
    intervals = pd.concat(interval_frames, ignore_index=True)
    base_cols = [
        "loan_id",
        "covered_online_v4",
        "interval_width_online_v4",
        "online_method_v4",
        "original_grade",
        "period",
        "score_decile",
        "state_top20",
        "income_band",
    ]
    policy_month_frames: list[pd.DataFrame] = []
    source_month_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        method_intervals = intervals[intervals["online_method_v4"].eq(method)]
        merged = allocations[["policy_id", "loan_id", "issue_month", "funded_exposure"]].merge(
            method_intervals[base_cols], on="loan_id", how="left"
        )
        if method == "source_aware_guarded":
            pm_size = merged.groupby(["policy_id", "issue_month"])["loan_id"].transform("nunique")
            grade_size = merged.groupby(["policy_id", "issue_month", "original_grade"])[
                "loan_id"
            ].transform("nunique")
            guard = pm_size.lt(10) | grade_size.lt(8)
            merged.loc[guard, "covered_online_v4"] = True
            merged.loc[guard, "interval_width_online_v4"] = 1.0
        policy_month = (
            merged.groupby(["online_method_v4", "policy_id", "issue_month"], as_index=False)
            .agg(
                n_funded=("loan_id", "nunique"),
                funded_exposure=("funded_exposure", "sum"),
                coverage_online_v4=("covered_online_v4", "mean"),
                avg_width_online_v4=("interval_width_online_v4", "mean"),
            )
            .rename(columns={"issue_month": "month"})
        )
        policy_month["coverage_regret_90_v4"] = (0.90 - policy_month["coverage_online_v4"]).clip(
            lower=0
        )
        policy_month_frames.append(policy_month)
        for source in ["original_grade", "period", "score_decile", "state_top20", "income_band"]:
            local = (
                merged.groupby(
                    ["online_method_v4", "policy_id", "issue_month", source], as_index=False
                )
                .agg(
                    n=("loan_id", "nunique"),
                    coverage_online_v4=("covered_online_v4", "mean"),
                    avg_width_online_v4=("interval_width_online_v4", "mean"),
                )
                .rename(columns={"issue_month": "month", source: "source_value"})
            )
            local = local[local["n"].ge(5)].copy()
            local["source_id"] = source
            source_month_frames.append(local)
        method_source = pd.concat(source_month_frames, ignore_index=True)
        method_source = method_source[method_source["online_method_v4"].eq(method)]
        summary_rows.append(
            {
                "online_method_v4": method,
                "coverage_policy_month_mean": float(policy_month["coverage_online_v4"].mean()),
                "coverage_policy_month_min": float(policy_month["coverage_online_v4"].min()),
                "coverage_source_month_min": float(method_source["coverage_online_v4"].min())
                if not method_source.empty
                else np.nan,
                "avg_width_policy_month": float(policy_month["avg_width_online_v4"].mean()),
                "avg_width_loan": float(method_intervals["interval_width_online_v4"].mean()),
                "total_coverage_regret_90": float(policy_month["coverage_regret_90_v4"].sum()),
                "gate_pass": bool(
                    policy_month["coverage_online_v4"].min() >= 0.80
                    and (method_source.empty or method_source["coverage_online_v4"].min() >= 0.75)
                    and policy_month["avg_width_online_v4"].mean() <= 0.95
                ),
            }
        )
    policy_month_all = pd.concat(policy_month_frames, ignore_index=True)
    source_month_all = pd.concat(source_month_frames, ignore_index=True)
    source_month_all["source_value"] = source_month_all["source_value"].astype(str)
    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "gate_pass",
            "coverage_policy_month_min",
            "coverage_source_month_min",
            "avg_width_policy_month",
        ],
        ascending=[False, False, False, True],
    )
    return intervals, policy_month_all, source_month_all, summary


def _grade_cif_lookup() -> pd.DataFrame:
    cif_impact = _safe_read_parquet(ROOT / "data" / "processed" / "cif_ecl_impact.parquet")
    if not cif_impact.empty and {"grade", "cif_t60m", "pd_12m_km"}.issubset(cif_impact.columns):
        out = cif_impact[["grade", "cif_t60m", "pd_12m_km", "pd_lifetime_km", "cf_lifetime"]].copy()
        out = out.rename(columns={"grade": "original_grade"})
        return out
    return pd.DataFrame(
        {
            "original_grade": list("ABCDEFG"),
            "cif_t60m": [0.04, 0.09, 0.14, 0.20, 0.28, 0.36, 0.44],
            "pd_12m_km": [0.03, 0.05, 0.08, 0.11, 0.16, 0.22, 0.28],
            "pd_lifetime_km": [0.18, 0.27, 0.36, 0.46, 0.55, 0.64, 0.72],
            "cf_lifetime": [0.73] * 7,
        }
    )


def _scenario_multiplier_table() -> pd.DataFrame:
    ts = _safe_read_parquet(ROOT / "data" / "processed" / "ts_ecl_intervals.parquet")
    if not ts.empty and {"pd_mult_point", "pd_mult_adverse", "pd_mult_optimistic"}.issubset(
        ts.columns
    ):
        return pd.DataFrame(
            [
                {
                    "scenario": "optimistic",
                    "pd_multiplier": float(ts["pd_mult_optimistic"].mean()),
                    "macro_source": "ts_ecl_intervals",
                },
                {
                    "scenario": "baseline",
                    "pd_multiplier": float(ts["pd_mult_point"].mean()),
                    "macro_source": "ts_ecl_intervals",
                },
                {
                    "scenario": "adverse",
                    "pd_multiplier": float(ts["pd_mult_adverse"].mean()),
                    "macro_source": "ts_ecl_intervals",
                },
                {
                    "scenario": "severe",
                    "pd_multiplier": float(max(ts["pd_mult_adverse"].quantile(0.90), 1.8)),
                    "macro_source": "ts_ecl_intervals_p90_adverse_floor",
                },
            ]
        )
    return pd.DataFrame(
        [
            {"scenario": "optimistic", "pd_multiplier": 0.80, "macro_source": "fallback"},
            {"scenario": "baseline", "pd_multiplier": 1.00, "macro_source": "fallback"},
            {"scenario": "adverse", "pd_multiplier": 1.45, "macro_source": "fallback"},
            {"scenario": "severe", "pd_multiplier": 1.90, "macro_source": "fallback"},
        ]
    )


def build_ifrs9_v4_contractual(
    allocations: pd.DataFrame,
    performance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cif = _grade_cif_lookup()
    scenarios = _scenario_multiplier_table()
    work = allocations.merge(performance, on="loan_id", how="left", suffixes=("", "_perf"))
    work["loan_status"] = (
        work.get("loan_status", pd.Series("unknown", index=work.index))
        .fillna("unknown")
        .astype(str)
    )
    work["actual_lgd"] = pd.to_numeric(work.get("lgd"), errors="coerce").clip(0, 1)
    work["actual_lgd"] = work["actual_lgd"].fillna(DEFAULT_LGD)
    work["installment"] = pd.to_numeric(work.get("installment"), errors="coerce")
    work["installment"] = work["installment"].fillna(
        work["funded_exposure"] / work["term"].astype(float).clip(lower=1)
    )
    work = work.merge(cif, on="original_grade", how="left")
    work["cif_t60m"] = work["cif_t60m"].fillna(work["pd_high_alpha01"])
    work["pd_12m_km"] = work["pd_12m_km"].fillna(work["pd_point_alpha01"])
    rows = []
    stage_rows = []
    for _, sc in scenarios.iterrows():
        scenario = str(sc["scenario"])
        mult = float(sc["pd_multiplier"])
        local = work.copy()
        stressed_lifetime_pd = np.clip(
            0.55 * local["pd_high_alpha01"].astype(float) * mult
            + 0.45 * local["cif_t60m"].astype(float) * mult,
            0,
            1,
        )
        stressed_12m_pd = np.clip(
            0.60 * local["pd_point_alpha01"].astype(float) * mult
            + 0.40 * local["pd_12m_km"].astype(float) * mult,
            0,
            1,
        )
        default_observed = local["loan_status"].str.contains(
            "Charged Off|Default", case=False, regex=True
        )
        relative_sicr = stressed_lifetime_pd / np.maximum(
            local["pd_point_alpha01"].astype(float), 1e-4
        )
        stage = np.select(
            [
                default_observed,
                relative_sicr >= 1.50,
                stressed_lifetime_pd >= 0.30,
                local["original_grade"].astype(str).isin(["E", "F", "G"]),
            ],
            [
                "Stage 3 observed/default",
                "Stage 2 SICR relative",
                "Stage 2 SICR absolute",
                "Stage 2 grade risk",
            ],
            default="Stage 1 12m",
        )
        local["ifrs9_stage_v4"] = stage
        term = pd.to_numeric(local["term"], errors="coerce").fillna(36).clip(lower=1)
        age = pd.to_numeric(local.get("lgd_months_since_issue"), errors="coerce").fillna(term)
        contractual_ead_factor = np.where(
            default_observed,
            np.clip(1.0 - np.minimum(age, term) / (term * 1.15), 0.10, 1.0),
            np.clip(1.0 - np.minimum(12, term) / (term * 1.25), 0.35, 1.0),
        )
        ead_12m = local["funded_exposure"].astype(float) * contractual_ead_factor
        ead_lifetime = local["funded_exposure"].astype(float) * np.clip(
            0.55 + 0.30 * (term / 60.0), 0.55, 0.85
        )
        lgd_scenario = np.clip(local["actual_lgd"].astype(float) * (0.90 + 0.25 * mult), 0.05, 0.95)
        ecl = np.select(
            [
                local["ifrs9_stage_v4"].astype(str).str.startswith("Stage 3"),
                local["ifrs9_stage_v4"].astype(str).str.startswith("Stage 2"),
            ],
            [
                local["funded_exposure"].astype(float) * lgd_scenario,
                ead_lifetime * stressed_lifetime_pd * lgd_scenario,
            ],
            default=ead_12m * stressed_12m_pd * lgd_scenario,
        )
        local["scenario"] = scenario
        local["macro_pd_multiplier_v4"] = mult
        local["macro_source"] = sc["macro_source"]
        local["ead_12m_v4"] = ead_12m
        local["ead_lifetime_v4"] = ead_lifetime
        local["stressed_12m_pd_v4"] = stressed_12m_pd
        local["stressed_lifetime_pd_v4"] = stressed_lifetime_pd
        local["lgd_scenario_v4"] = lgd_scenario
        local["contractual_ecl_v4"] = ecl
        local["net_return_after_contractual_ecl_v4"] = (
            local["realized_return_proxy_lgd45"].astype(float) - local["contractual_ecl_v4"]
        )
        rows.append(
            local[
                [
                    "policy_id",
                    "loan_id",
                    "scenario",
                    "ifrs9_stage_v4",
                    "funded_exposure",
                    "loan_status",
                    "actual_lgd",
                    "ead_12m_v4",
                    "ead_lifetime_v4",
                    "stressed_12m_pd_v4",
                    "stressed_lifetime_pd_v4",
                    "lgd_scenario_v4",
                    "contractual_ecl_v4",
                    "realized_return_proxy_lgd45",
                    "net_return_after_contractual_ecl_v4",
                    "macro_pd_multiplier_v4",
                    "macro_source",
                ]
            ]
        )
        stage_rows.append(
            local.groupby(["scenario", "ifrs9_stage_v4"], as_index=False).agg(
                loans=("loan_id", "nunique"),
                exposure=("funded_exposure", "sum"),
                ecl=("contractual_ecl_v4", "sum"),
            )
        )
    loan_level = pd.concat(rows, ignore_index=True)
    summary = loan_level.groupby(["policy_id", "scenario"], as_index=False).agg(
        n_funded=("loan_id", "nunique"),
        funded_exposure=("funded_exposure", "sum"),
        contractual_ecl_v4=("contractual_ecl_v4", "sum"),
        realized_return_proxy_lgd45=("realized_return_proxy_lgd45", "sum"),
        stage1_share=(
            "ifrs9_stage_v4",
            lambda x: float(pd.Series(x).str.startswith("Stage 1").mean()),
        ),
        stage2_share=(
            "ifrs9_stage_v4",
            lambda x: float(pd.Series(x).str.startswith("Stage 2").mean()),
        ),
        stage3_share=(
            "ifrs9_stage_v4",
            lambda x: float(pd.Series(x).str.startswith("Stage 3").mean()),
        ),
        actual_default_status_share=(
            "loan_status",
            lambda x: float(
                pd.Series(x).str.contains("Charged Off|Default", case=False, regex=True).mean()
            ),
        ),
    )
    summary["net_return_after_contractual_ecl_v4"] = (
        summary["realized_return_proxy_lgd45"] - summary["contractual_ecl_v4"]
    )
    quality = pd.DataFrame(
        [
            {
                "input": "performance_reference",
                "rows": int(len(performance)),
                "has_loan_status": bool("loan_status" in performance.columns),
                "has_actual_lgd": bool("lgd" in performance.columns),
                "has_installment": bool("installment" in performance.columns),
                "claim_scope": "contractual_proxy_using_available_status_lgd_installment_cif_macro",
            },
            {
                "input": "macro_scenarios",
                "rows": int(len(scenarios)),
                "has_loan_status": False,
                "has_actual_lgd": False,
                "has_installment": False,
                "claim_scope": ",".join(scenarios["macro_source"].astype(str).unique()),
            },
        ]
    )
    return loan_level, summary, pd.concat(stage_rows, ignore_index=True), quality


def _scenario_loss_matrix(pool: pd.DataFrame) -> tuple[list[tuple[str, float, float]], np.ndarray]:
    scenarios = [
        ("baseline_mid", 1.00, 0.45),
        ("baseline_high", 1.12, 0.48),
        ("adverse_mid", 1.40, 0.55),
        ("adverse_high", 1.65, 0.58),
        ("severe_mid", 1.90, 0.65),
        ("severe_high", 2.20, 0.70),
    ]
    pd_high = pool["pd_high_alpha01"].to_numpy(dtype=float)
    amount = pool["loan_amnt"].to_numpy(dtype=float)
    loss = np.vstack([np.clip(pd_high * mult, 0, 1) * lgd * amount for _, mult, lgd in scenarios])
    return scenarios, loss


def _solve_cvar_constraint(
    pool: pd.DataFrame,
    *,
    risk_tolerance: float,
    cvar_cap: float,
    return_floor: float,
    beta: float = 0.90,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    n = len(pool)
    scenarios, loss_matrix = _scenario_loss_matrix(pool)
    loan_amnt = pool["loan_amnt"].to_numpy(dtype=float)
    int_rate = pool["int_rate_decimal"].to_numpy(dtype=float)
    pd_point = pool["pd_point_alpha01"].to_numpy(dtype=float)
    pd_high = pool["pd_high_alpha01"].to_numpy(dtype=float)
    base_return_vec = loan_amnt * (int_rate - pd_point * DEFAULT_LGD)
    model = pyo.ConcreteModel("paper4_v4_cvar_constraint")
    model.I = pyo.RangeSet(0, n - 1)
    model.S = pyo.RangeSet(0, len(scenarios) - 1)
    model.x = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.eta = pyo.Var(domain=pyo.NonNegativeReals)
    model.u = pyo.Var(model.S, domain=pyo.NonNegativeReals)

    exposure = sum(model.x[i] * loan_amnt[i] for i in model.I)
    base_return = sum(model.x[i] * base_return_vec[i] for i in model.I)
    model.budget = pyo.Constraint(expr=exposure <= BUDGET)
    model.min_budget = pyo.Constraint(expr=exposure >= 0.80 * BUDGET)
    model.pd_cap = pyo.Constraint(
        expr=sum(model.x[i] * loan_amnt[i] * pd_high[i] for i in model.I)
        <= risk_tolerance * (exposure + 1e-6)
    )
    model.return_floor = pyo.Constraint(expr=base_return >= return_floor)

    def excess_rule(m, s):
        return m.u[s] >= sum(m.x[i] * loss_matrix[s, i] for i in m.I) - m.eta

    model.excess = pyo.Constraint(model.S, rule=excess_rule)
    cvar_expr = model.eta + (1 / ((1 - beta) * len(scenarios))) * sum(model.u[s] for s in model.S)
    model.cvar_cap = pyo.Constraint(expr=cvar_expr <= cvar_cap)
    model.obj = pyo.Objective(expr=base_return, sense=pyo.maximize)
    solver = Highs()
    solver.config.time_limit = 120
    t0 = time.perf_counter()
    policy_id = f"paper4_v4_cvar_rt{risk_tolerance:.4f}_cap{int(cvar_cap)}_floor{int(return_floor)}".replace(
        ".", "p"
    )
    try:
        results = solver.solve(model)
    except RuntimeError as exc:
        return (
            pd.DataFrame(),
            {
                "cvar_policy_id": policy_id,
                "risk_tolerance": risk_tolerance,
                "cvar_cap": cvar_cap,
                "return_floor": return_floor,
                "beta": beta,
                "candidate_pool_n": n,
                "solver_status": f"infeasible_or_no_solution: {str(exc).splitlines()[0]}",
                "elapsed_seconds": time.perf_counter() - t0,
                "n_funded": 0,
                "funded_exposure": 0.0,
                "objective_return": np.nan,
                "formal_cvar_loss": np.nan,
            },
            pd.DataFrame(),
        )
    elapsed = time.perf_counter() - t0
    allocation = np.array([float(pyo.value(model.x[i])) for i in model.I])
    mask = allocation > 1e-8
    funded = pool.loc[mask].copy()
    funded["cvar_policy_id"] = policy_id
    funded["loan_id"] = funded["loan_id"].astype(str)
    funded["allocation_fraction"] = allocation[mask]
    funded["funded_exposure"] = funded["allocation_fraction"] * funded["loan_amnt"]
    funded["realized_return_proxy_lgd45"] = (
        funded["funded_exposure"] * funded["int_rate_decimal"] * (1 - funded["y_true"])
        - funded["funded_exposure"] * DEFAULT_LGD * funded["y_true"]
    )
    scenario_rows = []
    for s_idx, (name, mult, lgd) in enumerate(scenarios):
        scenario_rows.append(
            {
                "cvar_policy_id": policy_id,
                "scenario": name,
                "pd_multiplier": mult,
                "lgd": lgd,
                "portfolio_loss": float(np.dot(allocation, loss_matrix[s_idx])),
                "cvar_excess_u": float(pyo.value(model.u[s_idx])),
            }
        )
    scenario_loss = pd.DataFrame(scenario_rows)
    formal_cvar = float(
        pyo.value(model.eta)
        + (1 / ((1 - beta) * len(scenarios))) * sum(float(pyo.value(model.u[s])) for s in model.S)
    )
    metrics = {
        "cvar_policy_id": policy_id,
        "risk_tolerance": risk_tolerance,
        "cvar_cap": cvar_cap,
        "return_floor": return_floor,
        "beta": beta,
        "candidate_pool_n": n,
        "solver_status": str(getattr(results, "termination_condition", "unknown")),
        "elapsed_seconds": elapsed,
        "n_funded": int(funded["loan_id"].nunique()),
        "funded_exposure": float(funded["funded_exposure"].sum()),
        "objective_return": float(pyo.value(model.obj)),
        "realized_return_proxy_lgd45": float(funded["realized_return_proxy_lgd45"].sum()),
        "formal_cvar_loss": formal_cvar,
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
    return funded[[col for col in keep if col in funded.columns]], metrics, scenario_loss


def build_cvar_oce_v4(
    pool: pd.DataFrame, local_search: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feasible = local_search[
        local_search["solver_status"].astype(str).str.contains("optimal", case=False)
    ].copy()
    best_return = (
        float(feasible["realized_return_proxy_lgd45"].max()) if not feasible.empty else 130_000.0
    )
    cvar_caps = [90_000.0, 120_000.0, 160_000.0, 210_000.0]
    floors = [0.50 * best_return, 0.70 * best_return, 0.85 * best_return]
    allocs: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    losses: list[pd.DataFrame] = []
    for rt in [0.1700, 0.1750, 0.1775]:
        for cap in cvar_caps:
            for floor in floors:
                funded, local_metrics, scenario_loss = _solve_cvar_constraint(
                    pool, risk_tolerance=rt, cvar_cap=cap, return_floor=floor
                )
                allocs.append(funded)
                metrics.append(local_metrics)
                if not scenario_loss.empty:
                    losses.append(scenario_loss)
    return (
        pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame(),
        pd.DataFrame(metrics),
        pd.concat(losses, ignore_index=True) if losses else pd.DataFrame(),
    )


def build_mdcp_v4(
    allocations: pd.DataFrame, online_source: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = [
        "original_grade",
        "period",
        "term",
        "state_top20",
        "income_band",
        "dti_band",
        "score_decile",
    ]
    work = allocations.copy()
    work["grade_period"] = work["original_grade"].astype(str) + "_" + work["period"].astype(str)
    work["grade_score"] = (
        work["original_grade"].astype(str) + "_d" + work["score_decile"].astype(str)
    )
    sources += ["grade_period", "grade_score"]
    rows = []
    pooling = []
    for policy_id, group in work.groupby("policy_id"):
        for source in sources:
            values = group[source].astype(str)
            stats = group.groupby(values).agg(
                n=("loan_id", "nunique"),
                coverage=("miscovered_alpha01", lambda x: 1.0 - float(pd.Series(x).mean())),
                exposure=("funded_exposure", "sum"),
            )
            small = int(stats["n"].lt(8).sum())
            defended = stats[stats["n"].ge(8)]
            if defended.empty and source in {"grade_period", "grade_score"}:
                parent = "original_grade"
                parent_stats = group.groupby(group[parent].astype(str)).agg(
                    n=("loan_id", "nunique"),
                    coverage=("miscovered_alpha01", lambda x: 1.0 - float(pd.Series(x).mean())),
                    exposure=("funded_exposure", "sum"),
                )
                defended = parent_stats[parent_stats["n"].ge(8)]
                pooling_rule = f"{source}->original_grade"
            else:
                pooling_rule = "native_min8"
            rows.append(
                {
                    "policy_id": policy_id,
                    "source_id": source,
                    "pooling_rule": pooling_rule,
                    "n_cells": int(len(stats)),
                    "n_defended_cells": int(len(defended)),
                    "n_small_cells": small,
                    "worst_source_coverage_v4": float(defended["coverage"].min())
                    if not defended.empty
                    else np.nan,
                    "mean_source_coverage_v4": float(defended["coverage"].mean())
                    if not defended.empty
                    else np.nan,
                    "defended_exposure": float(defended["exposure"].sum())
                    if not defended.empty
                    else 0.0,
                    "mdcp_gate_pass_80": bool(
                        not defended.empty and defended["coverage"].min() >= 0.80
                    ),
                }
            )
            pooling.append(
                {
                    "policy_id": policy_id,
                    "source_id": source,
                    "pooling_rule": pooling_rule,
                    "small_cell_share": float(stats["n"].lt(8).mean())
                    if not stats.empty
                    else np.nan,
                    "reason": "minimum-cell pooling prevents single-loan cells from controlling MDCP claims",
                }
            )
    mdcp = pd.DataFrame(rows)
    if not online_source.empty:
        best_method = (
            online_source.groupby("online_method_v4")["coverage_online_v4"]
            .min()
            .sort_values(ascending=False)
            .index[0]
        )
        online_best = online_source[online_source["online_method_v4"].eq(best_method)]
        online_worst = online_best.groupby("policy_id", as_index=False).agg(
            online_worst_source_coverage_v4=("coverage_online_v4", "min")
        )
        mdcp = mdcp.merge(online_worst, on="policy_id", how="left")
    return mdcp, pd.DataFrame(pooling)


def build_fairness_v4(
    base: pd.DataFrame, allocations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategy = pd.DataFrame(
        [
            {
                "item": "protected_attributes",
                "current_state": "race/ethnicity/sex/gender/age/date_of_birth unavailable",
                "decision": "no fair-lending legal claim",
                "next_step": "external validation only if valid protected attributes/proxies are approved",
            },
            {
                "item": "proxy_governance",
                "current_state": "grade, term, state, zip3, income and dti bands available",
                "decision": "use as concentration stress, not protected-class fairness",
                "next_step": "report max gaps and require caveat in selector",
            },
        ]
    )
    work = allocations.copy()
    work["zip3"] = work.get("zip_code", pd.Series("unknown", index=work.index)).astype(str).str[:3]
    universe = base.copy()
    universe["zip3"] = (
        universe.get("zip_code", pd.Series("unknown", index=universe.index)).astype(str).str[:3]
    )
    sources = ["original_grade", "term", "state_top20", "income_band", "dti_band", "zip3"]
    stress_rows = []
    for policy_id, group in work.groupby("policy_id"):
        for source in sources:
            funded = group[source].astype(str).value_counts(normalize=True)
            ref = universe[source].astype(str).value_counts(normalize=True)
            idx = sorted(set(funded.index) | set(ref.index))
            gap = (funded.reindex(idx, fill_value=0.0) - ref.reindex(idx, fill_value=0.0)).abs()
            stress_rows.append(
                {
                    "policy_id": policy_id,
                    "source_id": source,
                    "max_abs_gap_v4": float(gap.max()) if len(gap) else np.nan,
                    "mean_abs_gap_v4": float(gap.mean()) if len(gap) else np.nan,
                    "n_values": int(len(idx)),
                    "proxy_gate_pass_35": bool(gap.max() <= 0.35) if len(gap) else False,
                    "claim_scope": "proxy_governance_only_no_protected_attribute_claim",
                }
            )
    stress = pd.DataFrame(stress_rows)
    summary = stress.groupby("policy_id", as_index=False).agg(
        max_proxy_gap_v4=("max_abs_gap_v4", "max"),
        mean_proxy_gap_v4=("mean_abs_gap_v4", "mean"),
        proxy_sources_pass_35=("proxy_gate_pass_35", "mean"),
    )
    return strategy, stress, summary


def build_causal_v4(
    base: pd.DataFrame, performance: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    perf_cols = [col for col in ["loan_id", "loan_status", "lgd"] if col in performance.columns]
    work = base.merge(performance[perf_cols], on="loan_id", how="left", suffixes=("", "_perf"))
    if "loan_status" not in work.columns:
        status_candidates = [col for col in work.columns if col.startswith("loan_status")]
        work["loan_status"] = work[status_candidates[0]] if status_candidates else "unknown"
    if "lgd" not in work.columns:
        lgd_candidates = [col for col in work.columns if col.startswith("lgd")]
        work["lgd"] = work[lgd_candidates[0]] if lgd_candidates else DEFAULT_LGD
    work["treatment_high_rate_within_grade"] = work["int_rate_decimal"] > work.groupby(
        "original_grade"
    )["int_rate_decimal"].transform("median")
    work["outcome_default"] = work["y_true"].astype(float)
    work["outcome_prepay_proxy"] = (
        work["loan_status"]
        .fillna("")
        .astype(str)
        .str.contains("Fully Paid", case=False)
        .astype(float)
    )
    actual_lgd = pd.to_numeric(work["lgd"], errors="coerce").fillna(DEFAULT_LGD).clip(0, 1)
    work["outcome_loss_proxy"] = (
        work["y_true"].astype(float) * actual_lgd * work["loan_amnt"].astype(float)
    )
    work["outcome_net_return_proxy"] = work["loan_amnt"].astype(float) * work[
        "int_rate_decimal"
    ].astype(float) * (1 - work["y_true"].astype(float)) - work["loan_amnt"].astype(
        float
    ) * actual_lgd * work["y_true"].astype(float)
    covariates = ["annual_inc", "dti", "loan_amnt", "fico_score", "pd_point_alpha01"]
    x_num = (
        work[covariates]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(work[covariates].median(numeric_only=True))
    )
    x = pd.concat(
        [
            x_num,
            pd.get_dummies(work["original_grade"].astype(str), prefix="grade", drop_first=True),
            pd.get_dummies(work["term"].astype(str), prefix="term", drop_first=True),
            pd.get_dummies(work["period"].astype(str), prefix="period", drop_first=True),
        ],
        axis=1,
    )
    y = work["treatment_high_rate_within_grade"].astype(int)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    logit = LogisticRegression(max_iter=800, class_weight="balanced")
    logit.fit(x_scaled, y)
    ps = np.clip(logit.predict_proba(x_scaled)[:, 1], 0.03, 0.97)
    work["propensity_v4"] = ps
    trimmed = work[work["propensity_v4"].between(0.05, 0.95)].copy()
    treat = trimmed["treatment_high_rate_within_grade"].astype(bool)
    trimmed["ipw_att_v4"] = np.where(
        treat, 1.0, trimmed["propensity_v4"] / (1 - trimmed["propensity_v4"])
    )

    def smd(frame: pd.DataFrame, cov: str, weights: pd.Series | None = None) -> float:
        values = pd.to_numeric(frame[cov], errors="coerce").fillna(
            float(pd.to_numeric(frame[cov], errors="coerce").median())
        )
        t = frame["treatment_high_rate_within_grade"].astype(bool)
        if weights is None:
            mt, mc = values[t].mean(), values[~t].mean()
            vt, vc = values[t].var(), values[~t].var()
        else:
            w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
            mt = np.average(values[t], weights=w[t])
            mc = np.average(values[~t], weights=w[~t])
            vt = np.average((values[t] - mt) ** 2, weights=w[t])
            vc = np.average((values[~t] - mc) ** 2, weights=w[~t])
        return float(abs(mt - mc) / max(math.sqrt((vt + vc) / 2.0), 1e-12))

    balance = pd.DataFrame(
        [
            {
                "covariate": cov,
                "smd_unweighted": smd(work, cov),
                "smd_trimmed_ipw_att": smd(trimmed, cov, trimmed["ipw_att_v4"]),
                "balance_pass_0p10": smd(trimmed, cov, trimmed["ipw_att_v4"]) <= 0.10,
            }
            for cov in covariates
        ]
    )
    outcomes = []
    for outcome in [
        "outcome_default",
        "outcome_prepay_proxy",
        "outcome_loss_proxy",
        "outcome_net_return_proxy",
    ]:
        treated_mean = float(trimmed.loc[treat, outcome].mean())
        control_weighted = float(
            np.average(trimmed.loc[~treat, outcome], weights=trimmed.loc[~treat, "ipw_att_v4"])
        )
        outcomes.append(
            {
                "outcome": outcome,
                "treated_mean": treated_mean,
                "ipw_control_mean": control_weighted,
                "att_ipw": treated_mean - control_weighted,
                "claim_scope": "diagnostic_association_after_observed_adjustment",
            }
        )
    outcome_table = pd.DataFrame(outcomes)
    placebo = pd.DataFrame(
        [
            {
                "placebo": cov,
                "weighted_smd_after_trim_ipw": smd(trimmed, cov, trimmed["ipw_att_v4"]),
                "placebo_pass_0p10": smd(trimmed, cov, trimmed["ipw_att_v4"]) <= 0.10,
            }
            for cov in ["annual_inc", "dti", "loan_amnt", "fico_score"]
        ]
    )
    default_att = float(
        outcome_table.loc[outcome_table["outcome"].eq("outcome_default"), "att_ipw"].iloc[0]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "hidden_bias_shift": shift,
                "att_default_ipw": default_att,
                "lower": default_att - shift,
                "upper": default_att + shift,
                "sign_stable": bool(default_att - shift > 0 or default_att + shift < 0),
            }
            for shift in np.linspace(0, 0.06, 13)
        ]
    )
    # A small DR diagnostic for default only.  It is intentionally not policy promotion.
    rf_t = RandomForestClassifier(
        n_estimators=80, min_samples_leaf=200, random_state=RNG_SEED, n_jobs=-1
    )
    rf_c = RandomForestClassifier(
        n_estimators=80, min_samples_leaf=200, random_state=RNG_SEED + 1, n_jobs=-1
    )
    x_trim = x.loc[trimmed.index]
    rf_t.fit(x_trim[treat], trimmed.loc[treat, "outcome_default"])
    rf_c.fit(x_trim[~treat], trimmed.loc[~treat, "outcome_default"])
    cate_default = rf_t.predict_proba(x_trim)[:, 1] - rf_c.predict_proba(x_trim)[:, 1]
    dr_default_att = float(np.average(cate_default, weights=np.ones(len(cate_default))))
    dossier = pd.DataFrame(
        [
            {
                "treatment_id": "high_rate_within_grade",
                "n": int(len(work)),
                "n_trimmed_overlap": int(len(trimmed)),
                "trimmed_share": float(len(trimmed) / len(work)),
                "prevalence": float(y.mean()),
                "max_smd_trimmed_ipw": float(balance["smd_trimmed_ipw_att"].max()),
                "balance_all_pass_0p10": bool(balance["balance_pass_0p10"].all()),
                "placebo_pass_share": float(placebo["placebo_pass_0p10"].mean()),
                "att_default_ipw": default_att,
                "dr_default_att_diagnostic": dr_default_att,
                "sensitivity_sign_stable_6pp": bool(sensitivity["sign_stable"].all()),
                "cate_policy_value_allowed": False,
                "decision": "improved_dossier_still_not_policy_value",
            }
        ]
    )
    overlap = trimmed.groupby(["original_grade", "period"], as_index=False).agg(
        n=("loan_id", "nunique"),
        treatment_share=("treatment_high_rate_within_grade", "mean"),
        propensity_min=("propensity_v4", "min"),
        propensity_max=("propensity_v4", "max"),
    )
    return (
        dossier,
        balance,
        placebo,
        sensitivity,
        outcome_table.merge(
            pd.DataFrame({"dr_default_att_diagnostic": [dr_default_att]}), how="cross"
        ),
        overlap,
    )


def _build_strategy_policy(strategy: str, working_row: pd.Series) -> pd.Series:
    if strategy == "dynamic_return":
        return pd.Series(
            {
                "policy_id": "paper4_v4_dynamic_return",
                "risk_tolerance": float(working_row["risk_tolerance"]),
                "gamma": float(working_row["gamma"]),
                "policy_mode": "blended_uncertainty",
                "uncertainty_aversion": float(working_row["uncertainty_aversion"]),
            }
        )
    if strategy == "dynamic_ifrs9":
        return pd.Series(
            {
                "policy_id": "paper4_v4_dynamic_ifrs9",
                "risk_tolerance": max(0.165, float(working_row["risk_tolerance"]) - 0.0025),
                "gamma": min(0.55, float(working_row["gamma"]) + 0.025),
                "policy_mode": "blended_uncertainty",
                "uncertainty_aversion": min(
                    0.15, float(working_row["uncertainty_aversion"]) + 0.05
                ),
            }
        )
    return pd.Series(
        {
            "policy_id": "paper4_v4_dynamic_online_guarded",
            "risk_tolerance": max(0.165, float(working_row["risk_tolerance"]) - 0.005),
            "gamma": min(0.55, float(working_row["gamma"]) + 0.050),
            "policy_mode": "blended_uncertainty",
            "uncertainty_aversion": min(0.20, float(working_row["uncertainty_aversion"]) + 0.075),
        }
    )


def _month_dynamic_solve(
    loans: pd.DataFrame, policy: pd.Series, budget: float, deployment: float
) -> pd.DataFrame:
    if loans.empty or budget <= 500:
        return pd.DataFrame()
    local_budget = min(float(budget) * deployment, BUDGET)
    try:
        effective = _policy_effective_pd(loans, policy)
        sol = optimize_portfolio_allocation(
            loans=loans,
            pd_point=loans["pd_point_alpha01"].to_numpy(dtype=float),
            pd_low=loans["pd_low_alpha01"].to_numpy(dtype=float),
            pd_high=loans["pd_high_alpha01"].to_numpy(dtype=float),
            lgd=np.full(len(loans), DEFAULT_LGD),
            int_rates=loans["int_rate_decimal"].to_numpy(dtype=float),
            total_budget=local_budget,
            max_concentration=0.25,
            max_portfolio_pd=float(policy["risk_tolerance"]),
            robust=True,
            uncertainty_aversion=float(policy["uncertainty_aversion"]),
            min_budget_utilization=0.75,
            pd_constraint_override=effective,
            time_limit=90,
            threads=4,
            solver_backend="highs",
        )
    except RuntimeError:
        return pd.DataFrame()
    allocation = np.array([float(sol["allocation"].get(i, 0.0)) for i in range(len(loans))])
    mask = allocation > 1e-8
    funded = loans.loc[mask].copy()
    funded["allocation_fraction"] = allocation[mask]
    funded["funded_exposure"] = funded["allocation_fraction"] * funded["loan_amnt"]
    funded["solver_status"] = str(sol.get("solver_status", "unknown"))
    funded["loan_id"] = funded["id"].astype(str)
    funded["policy_id"] = str(policy["policy_id"])
    return funded


def build_sdam_v4_dynamic(
    base: pd.DataFrame,
    working_row: pd.Series,
    ifrs9_loan: pd.DataFrame,
    online_policy_month: pd.DataFrame,
    *,
    horizon_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = sorted(base["issue_month"].dropna().unique())[:horizon_months]
    ecl_lookup = (
        ifrs9_loan[ifrs9_loan["scenario"].eq("baseline")]
        .groupby("loan_id")["contractual_ecl_v4"]
        .mean()
        .to_dict()
    )
    states = []
    decisions = []
    for strategy in ["dynamic_return", "dynamic_ifrs9", "dynamic_online_guarded"]:
        policy = _build_strategy_policy(strategy, working_row)
        cash = BUDGET
        outstanding: list[dict[str, Any]] = []
        cum_interest = 0.0
        cum_losses = 0.0
        cum_ecl = 0.0
        for month_idx, month in enumerate(months, start=1):
            principal_in = 0.0
            interest_in = 0.0
            losses = 0.0
            survivors = []
            for item in outstanding:
                age = month_idx - item["start_idx"]
                term = item["term"]
                if age <= 0:
                    survivors.append(item)
                    continue
                balance = item["exposure"] * max(0.0, 1 - (age - 1) / term)
                default_month = min(12, term)
                loss = (
                    item["exposure"] * DEFAULT_LGD
                    if item["y_true"] >= 1 and age == default_month
                    else 0.0
                )
                if loss > 0:
                    losses += loss
                else:
                    principal_in += item["exposure"] / term
                    interest_in += balance * item["int_rate_decimal"] / 12
                    if age < term:
                        survivors.append(item)
            outstanding = survivors
            cash += principal_in + interest_in - losses
            cum_interest += interest_in
            cum_losses += losses
            available = base[base["issue_month"].eq(month)].copy()
            if strategy == "dynamic_ifrs9":
                available["ecl_month_proxy"] = (
                    available["loan_id"]
                    .map(ecl_lookup)
                    .fillna(available["pd_high_alpha01"] * DEFAULT_LGD * available["loan_amnt"])
                )
                available["int_rate_decimal"] = available["int_rate_decimal"] - available[
                    "ecl_month_proxy"
                ] / available["loan_amnt"].clip(lower=1)
            deployment = 0.35
            if strategy == "dynamic_online_guarded":
                prior = online_policy_month[
                    (online_policy_month["online_method_v4"].eq("source_aware_guarded"))
                    & (online_policy_month["month"].lt(month))
                ]
                if not prior.empty and prior["coverage_online_v4"].tail(3).mean() < 0.90:
                    deployment = 0.25
            funded = _month_dynamic_solve(available, policy, cash, deployment)
            deployed = float(funded["funded_exposure"].sum()) if not funded.empty else 0.0
            cash -= deployed
            if not funded.empty:
                funded["strategy"] = strategy
                funded["decision_month"] = month
                funded["month_idx"] = month_idx
                funded["ecl_at_decision_v4"] = (
                    funded["loan_id"]
                    .map(ecl_lookup)
                    .fillna(funded["pd_high_alpha01"] * DEFAULT_LGD * funded["funded_exposure"])
                )
                cum_ecl += float(funded["ecl_at_decision_v4"].sum())
                decisions.append(
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
                            "ecl_at_decision_v4",
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
            stage_mix = "unknown"
            if not funded.empty:
                risky = float((funded["pd_high_alpha01"] >= 0.30).mean())
                stage_mix = f"stage2_proxy_share={risky:.3f}"
            states.append(
                {
                    "strategy": strategy,
                    "policy_id": str(policy["policy_id"]),
                    "month": month,
                    "month_idx": month_idx,
                    "cash_budget_end": cash,
                    "principal_in": principal_in,
                    "interest_in": interest_in,
                    "realized_losses": losses,
                    "deployed_new": deployed,
                    "outstanding_exposure_end": sum(item["exposure"] for item in outstanding),
                    "active_loans_end": len(outstanding),
                    "cumulative_interest": cum_interest,
                    "cumulative_losses": cum_losses,
                    "cumulative_ecl_at_decision": cum_ecl,
                    "stage_mix_proxy": stage_mix,
                    "net_cash_result": cash
                    + sum(item["exposure"] for item in outstanding)
                    - BUDGET,
                    "status": "dynamic_loan_level_monthly_solver_v4",
                }
            )
    state = pd.DataFrame(states)
    decision = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()
    summary = state.groupby(["strategy", "policy_id"], as_index=False).agg(
        months=("month", "nunique"),
        final_cash_budget=("cash_budget_end", "last"),
        final_outstanding_exposure=("outstanding_exposure_end", "last"),
        cumulative_interest=("cumulative_interest", "last"),
        cumulative_losses=("cumulative_losses", "last"),
        cumulative_ecl_at_decision=("cumulative_ecl_at_decision", "last"),
        net_cash_result=("net_cash_result", "last"),
        total_deployed=("deployed_new", "sum"),
    )
    return state, decision, summary.sort_values("net_cash_result", ascending=False)


def build_common_sample_paths(
    allocations: pd.DataFrame,
    local_search: pd.DataFrame,
    *,
    n_paths: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top_ids = local_search["policy_id"].tolist()
    top_ids += [FROZEN_PAPER1_CHAMPION, GROSS_CHALLENGER]
    work = allocations[allocations["policy_id"].isin(set(top_ids))].copy()
    if FROZEN_PAPER1_CHAMPION not in set(work["policy_id"]):
        v3 = _load_v3_allocations()
        work = pd.concat([work, v3[v3["policy_id"].eq(FROZEN_PAPER1_CHAMPION)]], ignore_index=True)
    if GROSS_CHALLENGER not in set(work["policy_id"]):
        v3 = _load_v3_allocations()
        work = pd.concat([work, v3[v3["policy_id"].eq(GROSS_CHALLENGER)]], ignore_index=True)
    work["loan_id"] = work["loan_id"].astype(str)
    unique_loans = (
        work[["loan_id", "pd_high_alpha01"]].drop_duplicates("loan_id").set_index("loan_id")
    )
    unique_loan_ids = unique_loans.index.astype(str).to_numpy()
    work_loan_ids = work["loan_id"].to_numpy()
    scenarios = {"baseline": 1.0, "adverse": 1.35, "severe": 1.8}
    path_rows = []
    for scenario, mult in scenarios.items():
        for path_id in range(n_paths):
            shock_default = np.array(
                [
                    _hash_uniform(f"{loan_id}|{scenario}|{path_id}|default")
                    for loan_id in unique_loan_ids
                ]
            )
            shock_lgd = np.array(
                [
                    _hash_uniform(f"{loan_id}|{scenario}|{path_id}|lgd")
                    for loan_id in unique_loan_ids
                ]
            )
            pd_s = np.clip(unique_loans["pd_high_alpha01"].to_numpy(dtype=float) * mult, 0, 0.95)
            default_draw_unique = shock_default < pd_s
            lgd_draw = np.clip(
                DEFAULT_LGD * (0.75 + 0.70 * shock_lgd) * (0.90 + 0.15 * mult), 0.05, 0.95
            )
            default_draw = (
                pd.Series(default_draw_unique, index=unique_loan_ids)
                .reindex(work_loan_ids)
                .to_numpy()
            )
            lgd_draw_work = (
                pd.Series(lgd_draw, index=unique_loan_ids).reindex(work_loan_ids).to_numpy()
            )
            simulated_return = (
                work["funded_exposure"].to_numpy(dtype=float)
                * work["int_rate_decimal"].to_numpy(dtype=float)
                * (~default_draw)
                - work["funded_exposure"].to_numpy(dtype=float) * lgd_draw_work * default_draw
            )
            local = work[["policy_id", "funded_exposure"]].copy()
            local["simulated_return"] = simulated_return
            agg = local.groupby("policy_id", as_index=False).agg(
                simulated_return=("simulated_return", "sum"),
                funded_exposure=("funded_exposure", "sum"),
            )
            agg["scenario"] = scenario
            agg["path_id"] = path_id
            path_rows.append(agg)
    paths = pd.concat(path_rows, ignore_index=True)
    ref = paths[paths["policy_id"].eq(FROZEN_PAPER1_CHAMPION)][
        ["scenario", "path_id", "simulated_return"]
    ].rename(columns={"simulated_return": "paper1_reference_return"})
    pair = paths.merge(ref, on=["scenario", "path_id"], how="left")
    pair["diff_vs_paper1_frozen"] = pair["simulated_return"] - pair["paper1_reference_return"]
    ci = pair.groupby(["policy_id", "scenario"], as_index=False).agg(
        mean_diff_vs_paper1=("diff_vs_paper1_frozen", "mean"),
        p05_diff_vs_paper1=("diff_vs_paper1_frozen", lambda x: float(np.quantile(x, 0.05))),
        p50_diff_vs_paper1=("diff_vs_paper1_frozen", lambda x: float(np.quantile(x, 0.50))),
        p95_diff_vs_paper1=("diff_vs_paper1_frozen", lambda x: float(np.quantile(x, 0.95))),
        prob_beats_paper1=("diff_vs_paper1_frozen", lambda x: float(np.mean(np.asarray(x) > 0))),
        n_paths=("path_id", "nunique"),
    )
    scenario_replay = paths.groupby(["policy_id", "scenario"], as_index=False).agg(
        mean_simulated_return=("simulated_return", "mean"),
        p05_simulated_return=("simulated_return", lambda x: float(np.quantile(x, 0.05))),
        p95_simulated_return=("simulated_return", lambda x: float(np.quantile(x, 0.95))),
    )
    return paths, ci, scenario_replay


def build_regret_auditability_v4(
    local_search: pd.DataFrame,
    ifrs9_summary: pd.DataFrame,
    online_summary: pd.DataFrame,
    online_policy: pd.DataFrame,
    mdcp: pd.DataFrame,
    fairness_summary: pd.DataFrame,
    cvar_results: pd.DataFrame,
) -> pd.DataFrame:
    baseline = ifrs9_summary[ifrs9_summary["scenario"].eq("baseline")][
        ["policy_id", "net_return_after_contractual_ecl_v4", "contractual_ecl_v4"]
    ]
    best_method = str(online_summary.iloc[0]["online_method_v4"])
    online = (
        online_policy[online_policy["online_method_v4"].eq(best_method)]
        .groupby("policy_id", as_index=False)
        .agg(
            online_mean=("coverage_online_v4", "mean"),
            online_min=("coverage_online_v4", "min"),
            online_width=("avg_width_online_v4", "mean"),
        )
    )
    mdcp_sum = mdcp.groupby("policy_id", as_index=False).agg(
        mdcp_worst=("worst_source_coverage_v4", "min"),
        mdcp_pass_share=("mdcp_gate_pass_80", "mean"),
    )
    work = (
        local_search.merge(baseline, on="policy_id", how="left")
        .merge(online, on="policy_id", how="left")
        .merge(mdcp_sum, on="policy_id", how="left")
        .merge(fairness_summary, on="policy_id", how="left")
    )
    best_return = float(work["realized_return_proxy_lgd45"].max())
    work["regret_gross_v4"] = best_return - work["realized_return_proxy_lgd45"]
    best_net = float(work["net_return_after_contractual_ecl_v4"].max())
    work["regret_net_ifrs9_v4"] = best_net - work["net_return_after_contractual_ecl_v4"]
    work["auditability_score_v4"] = (
        0.20 * work["coverage_alpha01"].fillna(0).clip(0, 1)
        + 0.20 * work["online_min"].fillna(0).clip(0, 1)
        + 0.20 * work["mdcp_worst"].fillna(0).clip(0, 1)
        + 0.15 * (1 - work["max_proxy_gap_v4"].fillna(1).clip(0, 1))
        + 0.15 * work["proxy_sources_pass_35"].fillna(0).clip(0, 1)
        + 0.10 * 1.0
    )
    work["frontier_family"] = "CRPTO local v4"
    extra_rows = []
    spo = _safe_read_parquet(ROOT / "data" / "processed" / "crpto_vs_spo_stability_detail.parquet")
    if not spo.empty:
        extra_rows.append(
            {
                "policy_id": "spo_plus_periodic_comparator",
                "realized_return_proxy_lgd45": np.nan,
                "net_return_after_contractual_ecl_v4": np.nan,
                "regret_gross_v4": float(spo["spo_plus_mean_regret"].mean()),
                "regret_net_ifrs9_v4": np.nan,
                "auditability_score_v4": float(
                    0.35 * spo["coverage_90"].mean()
                    + 0.25 * spo["min_grade_coverage_90"].mean()
                    + 0.20 * (1 - min(spo["avg_width_90"].mean(), 1))
                    + 0.20 * 0.35
                ),
                "frontier_family": "SPO+ aggregate comparator",
            }
        )
    if not cvar_results.empty:
        for _, row in (
            cvar_results[
                cvar_results["solver_status"].astype(str).str.contains("optimal", case=False)
            ]
            .head(8)
            .iterrows()
        ):
            extra_rows.append(
                {
                    "policy_id": row["cvar_policy_id"],
                    "realized_return_proxy_lgd45": row.get(
                        "realized_return_proxy_lgd45", row.get("objective_return", np.nan)
                    ),
                    "net_return_after_contractual_ecl_v4": np.nan,
                    "regret_gross_v4": best_return
                    - float(row.get("realized_return_proxy_lgd45", row.get("objective_return", 0))),
                    "regret_net_ifrs9_v4": np.nan,
                    "auditability_score_v4": 0.70,
                    "frontier_family": "Formal CVaR constraint LP",
                }
            )
    keep = [
        "policy_id",
        "frontier_family",
        "realized_return_proxy_lgd45",
        "net_return_after_contractual_ecl_v4",
        "regret_gross_v4",
        "regret_net_ifrs9_v4",
        "auditability_score_v4",
        "online_min",
        "mdcp_worst",
        "max_proxy_gap_v4",
    ]
    frontier = work[[col for col in keep if col in work.columns]]
    if extra_rows:
        frontier = pd.concat([frontier, pd.DataFrame(extra_rows)], ignore_index=True, sort=False)
    frontier["pareto_candidate_v4"] = False
    for i, row in frontier.iterrows():
        dominated = (
            (frontier["regret_gross_v4"].fillna(np.inf) <= row["regret_gross_v4"])
            & (frontier["auditability_score_v4"].fillna(-np.inf) >= row["auditability_score_v4"])
            & (
                (frontier["regret_gross_v4"].fillna(np.inf) < row["regret_gross_v4"])
                | (frontier["auditability_score_v4"].fillna(-np.inf) > row["auditability_score_v4"])
            )
        ).any()
        frontier.loc[i, "pareto_candidate_v4"] = not bool(dominated)
    return frontier.sort_values(
        ["pareto_candidate_v4", "auditability_score_v4"], ascending=[False, False]
    )


def build_selector_v4(
    local_search: pd.DataFrame,
    ifrs9_summary: pd.DataFrame,
    online_summary: pd.DataFrame,
    online_policy: pd.DataFrame,
    mdcp: pd.DataFrame,
    fairness_summary: pd.DataFrame,
    cvar_results: pd.DataFrame,
    sample_ci: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    best_online = str(online_summary.iloc[0]["online_method_v4"])
    global_online_gate = bool(online_summary.iloc[0].get("gate_pass", False))
    online = (
        online_policy[online_policy["online_method_v4"].eq(best_online)]
        .groupby("policy_id", as_index=False)
        .agg(
            online_mean_v4=("coverage_online_v4", "mean"),
            online_min_v4=("coverage_online_v4", "min"),
            online_width_v4=("avg_width_online_v4", "mean"),
        )
    )
    baseline = ifrs9_summary[ifrs9_summary["scenario"].eq("baseline")][
        ["policy_id", "net_return_after_contractual_ecl_v4", "contractual_ecl_v4"]
    ]
    adverse = ifrs9_summary[ifrs9_summary["scenario"].eq("adverse")][
        ["policy_id", "net_return_after_contractual_ecl_v4"]
    ].rename(columns={"net_return_after_contractual_ecl_v4": "adverse_net_return_v4"})
    mdcp_sum = mdcp.groupby("policy_id", as_index=False).agg(
        mdcp_worst_v4=("worst_source_coverage_v4", "min"),
        mdcp_pass_share_v4=("mdcp_gate_pass_80", "mean"),
    )
    sample_base = sample_ci[sample_ci["scenario"].eq("baseline")][
        ["policy_id", "prob_beats_paper1", "p05_diff_vs_paper1"]
    ]
    work = (
        local_search.merge(baseline, on="policy_id", how="left")
        .merge(adverse, on="policy_id", how="left")
        .merge(online, on="policy_id", how="left")
        .merge(mdcp_sum, on="policy_id", how="left")
        .merge(fairness_summary, on="policy_id", how="left")
        .merge(sample_base, on="policy_id", how="left")
    )
    work["gate_return_positive"] = work["realized_return_proxy_lgd45"].gt(0)
    work["gate_ifrs9_baseline_positive"] = work["net_return_after_contractual_ecl_v4"].gt(0)
    work["gate_online_worst_cell"] = work["online_min_v4"].ge(0.80)
    work["gate_online_global_method"] = global_online_gate
    work["gate_mdcp"] = work["mdcp_worst_v4"].ge(0.80)
    work["gate_fairness_proxy"] = work["max_proxy_gap_v4"].le(0.35)
    work["gate_sample_path"] = work["prob_beats_paper1"].fillna(0).ge(0.50)
    work["selector_v4_score"] = (
        0.25 * _normalise(work["net_return_after_contractual_ecl_v4"], higher_is_better=True)
        + 0.20 * _normalise(work["realized_return_proxy_lgd45"], higher_is_better=True)
        + 0.15 * work["online_min_v4"].fillna(0).clip(0, 1)
        + 0.15 * work["mdcp_worst_v4"].fillna(0).clip(0, 1)
        + 0.10 * (1 - work["max_proxy_gap_v4"].fillna(1).clip(0, 1))
        + 0.10 * work["prob_beats_paper1"].fillna(0).clip(0, 1)
        + 0.05 * _normalise(work["adverse_net_return_v4"], higher_is_better=True)
    )
    gates = [
        "gate_return_positive",
        "gate_ifrs9_baseline_positive",
        "gate_online_worst_cell",
        "gate_mdcp",
        "gate_fairness_proxy",
    ]
    promotion_gates = gates + ["gate_online_global_method"]
    work["paper4_v4_decision"] = np.select(
        [
            work[promotion_gates].all(axis=1) & work["gate_sample_path"],
            work[gates].all(axis=1),
            work["gate_ifrs9_baseline_positive"] & work["gate_return_positive"],
        ],
        ["promote_to_paper4_working_champion", "review_candidate", "park_for_more_tests"],
        default="kill_or_rework",
    )
    work["selector_v4_rank"] = (
        work["selector_v4_score"].rank(ascending=False, method="first").astype(int)
    )
    work = work.sort_values("selector_v4_rank")
    thresholds = pd.DataFrame(
        [
            ("gross_return_positive", 0.0, "Sanity gate; not enough for promotion."),
            ("ifrs9_baseline_net_positive", 0.0, "Avoid promoting pure gross-return artifacts."),
            (
                "online_min_policy_month",
                0.80,
                "Worst policy-month floor after v4 guarded online CP.",
            ),
            (
                "online_global_method_gate",
                1.0,
                "No Paper 4 promotion while best online method still fails source-month gate.",
            ),
            (
                "mdcp_worst_source",
                0.80,
                "Worst defended source coverage floor with minimum-cell pooling.",
            ),
            (
                "fairness_proxy_gap",
                0.35,
                "Proxy governance only; not protected attribute fairness.",
            ),
            (
                "sample_path_prob_beats_paper1",
                0.50,
                "Paper 4 reference only; Paper Estrella remains frozen.",
            ),
        ],
        columns=["threshold_id", "value", "rationale"],
    )
    sensitivity_rows = []
    for online_floor in [0.75, 0.80, 0.85, 0.90]:
        for mdcp_floor in [0.75, 0.80, 0.85, 0.90]:
            for fair_gap in [0.30, 0.35, 0.40]:
                pass_mask = (
                    work["online_min_v4"].ge(online_floor)
                    & work["mdcp_worst_v4"].ge(mdcp_floor)
                    & work["max_proxy_gap_v4"].le(fair_gap)
                    & work["net_return_after_contractual_ecl_v4"].gt(0)
                )
                sensitivity_rows.append(
                    {
                        "online_floor": online_floor,
                        "mdcp_floor": mdcp_floor,
                        "fairness_gap": fair_gap,
                        "n_pass": int(pass_mask.sum()),
                        "best_pass_policy_id": str(work.loc[pass_mask].iloc[0]["policy_id"])
                        if pass_mask.any()
                        else "",
                    }
                )
    sensitivity = pd.DataFrame(sensitivity_rows)
    working = work[
        work["paper4_v4_decision"].isin(["promote_to_paper4_working_champion", "review_candidate"])
    ]
    if working.empty:
        working = work.head(1)
    champion = working.sort_values("selector_v4_score", ascending=False).iloc[0].to_dict()
    champion_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "paper4_working_champion_only",
        "paper1_frozen_reference_policy": FROZEN_PAPER1_CHAMPION,
        "does_not_modify_paper1": True,
        "policy_id": champion["policy_id"],
        "decision": champion["paper4_v4_decision"],
        "selector_v4_score": float(champion["selector_v4_score"]),
        "rationale": "Paper 4 internal working champion for continued experiments; not a thesis/journal champion.",
    }
    return thresholds, work, sensitivity, champion_payload


def build_claim_matrix_v4() -> pd.DataFrame:
    rows = [
        (
            "Challenger local search",
            "implemented_local_pool",
            "paper4_challenger_local_search.csv",
            "19ad-v4-challenger-online-mdcp.qmd",
            "100-policy exact LP over documented local candidate pool",
        ),
        (
            "Online conformal v4",
            "implemented_guarded",
            "paper4_online_conformal_v4_method_summary.csv",
            "19ad-v4-challenger-online-mdcp.qmd",
            "guarded method may trade width for worst-cell coverage",
        ),
        (
            "IFRS9 contractual proxy",
            "implemented_with_available_performance",
            "paper4_ifrs9_v4_contractual_policy_summary.csv",
            "19ae-v4-ifrs9-cvar-selector.qmd",
            "uses available loan_status/LGD/CIF, not full servicing panel",
        ),
        (
            "CVaR/OCE constraint",
            "implemented_candidate_pool_constraint_lp",
            "paper4_cvar_oce_v4_constraint_frontier.csv",
            "19ae-v4-ifrs9-cvar-selector.qmd",
            "candidate pool LP, not 276k full CVaR",
        ),
        (
            "Selector governance v4",
            "implemented_paper4_working_champion",
            "paper4_selector_v4_results.csv",
            "19ae-v4-ifrs9-cvar-selector.qmd",
            "Paper 4 working champion only",
        ),
        (
            "Dynamic SDAM v4",
            "implemented_dynamic_loan_level",
            "paper4_sdam_v4_dynamic_solver_summary.csv",
            "19af-v4-sdam-causal-fairness-regret.qmd",
            "state transition proxy",
        ),
        (
            "Causal high-rate v4",
            "implemented_dossier_only",
            "paper4_causal_high_rate_v4_dossier.csv",
            "19af-v4-sdam-causal-fairness-regret.qmd",
            "no CATE policy value promotion",
        ),
        (
            "Fairness proxy v4",
            "implemented_proxy_strategy",
            "paper4_fairness_v4_proxy_strategy.csv",
            "19af-v4-sdam-causal-fairness-regret.qmd",
            "no protected attributes",
        ),
        (
            "MDCP v4",
            "implemented_pooling",
            "paper4_mdcp_v4_source_coverage.csv",
            "19ad-v4-challenger-online-mdcp.qmd",
            "minimum-cell pooling caveat",
        ),
        (
            "Regret-auditability v4",
            "implemented_frontier",
            "paper4_regret_auditability_v4_frontier.csv",
            "19af-v4-sdam-causal-fairness-regret.qmd",
            "auditability score is formalized but still a governance score",
        ),
        (
            "Common sample paths",
            "implemented_pairwise_ci",
            "paper4_common_sample_path_pairwise_ci.csv",
            "19ag-v4-sample-paths-working-champion.qmd",
            "simulated scenario paths, not observed future performance",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["priority", "claim_status", "artifact", "quarto_page", "caveat"]
    )


def build_v4_figures(
    local_search: pd.DataFrame,
    online_summary: pd.DataFrame,
    cvar_results: pd.DataFrame,
    frontier: pd.DataFrame,
    sample_ci: pd.DataFrame,
) -> list[str]:
    """Write lightweight diagnostic figures for the v4 Quarto pages."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # Best gross return over uncertainty_aversion for each risk/gamma grid cell.
    heat = (
        local_search.groupby(["risk_tolerance", "gamma"], as_index=False)[
            "realized_return_proxy_lgd45"
        ]
        .max()
        .pivot(index="risk_tolerance", columns="gamma", values="realized_return_proxy_lgd45")
        .sort_index(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(heat.columns)), [f"{col:.3f}" for col in heat.columns])
    ax.set_yticks(range(len(heat.index)), [f"{idx:.4f}" for idx in heat.index])
    ax.set_xlabel("gamma")
    ax.set_ylabel("risk_tolerance")
    ax.set_title("Paper 4 v4 local search: best gross return")
    for row_idx, risk in enumerate(heat.index):
        for col_idx, gamma in enumerate(heat.columns):
            value = heat.loc[risk, gamma]
            ax.text(col_idx, row_idx, f"{value / 1000:.1f}k", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="gross return proxy")
    fig.tight_layout()
    path = FIGURE_DIR / "paper4_v4_local_search_return_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    methods = online_summary["online_method_v4"].astype(str)
    x = np.arange(len(methods))
    width = 0.38
    ax.bar(
        x - width / 2, online_summary["coverage_policy_month_min"], width, label="policy-month min"
    )
    ax.bar(
        x + width / 2, online_summary["coverage_source_month_min"], width, label="source-month min"
    )
    ax.axhline(0.80, color="#8b1e3f", linewidth=1.2, linestyle="--", label="gate 0.80")
    ax.set_xticks(x, methods, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("coverage")
    ax.set_title("Online conformal v4 worst-cell coverage")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = FIGURE_DIR / "paper4_v4_online_worst_cell_coverage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    cvar_ok = cvar_results[
        cvar_results["solver_status"].astype(str).str.contains("optimal", case=False, na=False)
    ].copy()
    if not cvar_ok.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.3))
        scatter = ax.scatter(
            cvar_ok["formal_cvar_loss"],
            cvar_ok["objective_return"],
            c=cvar_ok["cvar_cap"],
            cmap="plasma",
            s=36,
            alpha=0.85,
        )
        ax.set_xlabel("formal CVaR loss")
        ax.set_ylabel("objective return")
        ax.set_title("CVaR-constrained candidate-pool frontier")
        fig.colorbar(scatter, ax=ax, label="CVaR cap")
        fig.tight_layout()
        path = FIGURE_DIR / "paper4_v4_cvar_constraint_frontier.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path.name)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = np.where(frontier["pareto_candidate_v4"].astype(bool), "#1e5aa8", "#9aa5b1")
    sizes = np.where(frontier["pareto_candidate_v4"].astype(bool), 58, 24)
    ax.scatter(
        frontier["auditability_score_v4"],
        frontier["regret_gross_v4"],
        c=colors,
        s=sizes,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.set_xlabel("auditability score v4")
    ax.set_ylabel("gross regret v4")
    ax.set_title("Regret-auditability frontier v4")
    ax.invert_yaxis()
    fig.tight_layout()
    path = FIGURE_DIR / "paper4_v4_regret_auditability_frontier.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    baseline = sample_ci[
        sample_ci["scenario"].eq("baseline") & sample_ci["policy_id"].ne(FROZEN_PAPER1_CHAMPION)
    ].copy()
    baseline = baseline.sort_values("mean_diff_vs_paper1", ascending=False).head(12)
    if not baseline.empty:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        y = np.arange(len(baseline))
        lower = baseline["mean_diff_vs_paper1"] - baseline["p05_diff_vs_paper1"]
        upper = baseline["p95_diff_vs_paper1"] - baseline["mean_diff_vs_paper1"]
        ax.errorbar(
            baseline["mean_diff_vs_paper1"],
            y,
            xerr=[lower, upper],
            fmt="o",
            color="#1e5aa8",
            ecolor="#6b7280",
            capsize=3,
        )
        ax.axvline(0, color="#8b1e3f", linewidth=1.0, linestyle="--")
        ax.set_yticks(
            y, baseline["policy_id"].str.replace("paper4_v4_local_", "", regex=False), fontsize=7
        )
        ax.set_xlabel("baseline mean delta vs Paper Estrella")
        ax.set_title("Common sample paths: paired confidence screen")
        fig.tight_layout()
        path = FIGURE_DIR / "paper4_v4_common_sample_path_ci.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path.name)

    return written


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool-n", type=int, default=12_000)
    parser.add_argument("--sample-paths", type=int, default=300)
    parser.add_argument("--sdam-horizon-months", type=int, default=12)
    parser.add_argument("--force-local-search", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base = _prepare_base(_load_base_loan_frame())
    base["loan_id"] = base["id"].astype(str)
    v3_alloc = _load_v3_allocations()
    v3_eval = _load_v3_eval()
    performance = _load_performance_reference()

    local_search, local_alloc, pool, local_status = build_challenger_local_search(
        base,
        v3_alloc,
        v3_eval,
        candidate_pool_n=args.candidate_pool_n,
        force=args.force_local_search,
    )
    bootstrap = build_monthly_bootstrap(local_alloc, v3_alloc)
    _write_csv("paper4_challenger_local_bootstrap.csv", bootstrap)

    online_intervals, online_policy, online_source, online_summary = build_online_conformal_v4(
        base, local_alloc
    )
    _write_parquet("paper4_online_conformal_v4_intervals.parquet", online_intervals)
    _write_parquet("paper4_online_conformal_v4_policy_month.parquet", online_policy)
    _write_parquet("paper4_online_conformal_v4_source_month.parquet", online_source)
    _write_csv("paper4_online_conformal_v4_method_summary.csv", online_summary)

    ifrs9_loan, ifrs9_summary, ifrs9_stage, ifrs9_quality = build_ifrs9_v4_contractual(
        local_alloc, performance
    )
    _write_parquet("paper4_ifrs9_v4_contractual_loan_level.parquet", ifrs9_loan)
    _write_csv("paper4_ifrs9_v4_contractual_policy_summary.csv", ifrs9_summary)
    _write_csv("paper4_ifrs9_v4_stage_summary.csv", ifrs9_stage)
    _write_csv("paper4_ifrs9_v4_input_quality.csv", ifrs9_quality)

    cvar_alloc, cvar_results, cvar_losses = build_cvar_oce_v4(pool, local_search)
    _write_parquet("paper4_cvar_oce_v4_allocations.parquet", cvar_alloc)
    _write_csv("paper4_cvar_oce_v4_constraint_frontier.csv", cvar_results)
    _write_csv("paper4_cvar_oce_v4_scenario_losses.csv", cvar_losses)

    mdcp, pooling = build_mdcp_v4(local_alloc, online_source)
    _write_csv("paper4_mdcp_v4_source_coverage.csv", mdcp)
    _write_csv("paper4_mdcp_v4_pooling_diagnostics.csv", pooling)

    fair_strategy, fair_stress, fair_summary = build_fairness_v4(base, local_alloc)
    _write_csv("paper4_fairness_v4_proxy_strategy.csv", fair_strategy)
    _write_csv("paper4_fairness_v4_proxy_stress.csv", fair_stress)
    _write_csv("paper4_fairness_v4_proxy_summary.csv", fair_summary)

    sample_paths, sample_ci, scenario_replay = build_common_sample_paths(
        local_alloc, local_search, n_paths=args.sample_paths
    )
    _write_parquet("paper4_common_sample_paths_v4.parquet", sample_paths)
    _write_csv("paper4_common_sample_path_pairwise_ci.csv", sample_ci)
    _write_csv("paper4_common_sample_path_scenario_replay.csv", scenario_replay)

    thresholds, selector, sensitivity, working_champion = build_selector_v4(
        local_search,
        ifrs9_summary,
        online_summary,
        online_policy,
        mdcp,
        fair_summary,
        cvar_results,
        sample_ci,
    )
    _write_csv("paper4_selector_v4_governance_protocol.csv", thresholds)
    _write_csv("paper4_selector_v4_results.csv", selector)
    _write_csv("paper4_selector_v4_sensitivity.csv", sensitivity)
    _write_json("paper4_v4_working_champion.json", working_champion)

    working_policy = pd.Series(
        {
            "policy_id": working_champion["policy_id"],
            "risk_tolerance": float(
                selector.loc[
                    selector["policy_id"].eq(working_champion["policy_id"]), "risk_tolerance"
                ].iloc[0]
            ),
            "gamma": float(
                selector.loc[selector["policy_id"].eq(working_champion["policy_id"]), "gamma"].iloc[
                    0
                ]
            ),
            "uncertainty_aversion": float(
                selector.loc[
                    selector["policy_id"].eq(working_champion["policy_id"]), "uncertainty_aversion"
                ].iloc[0]
            ),
        }
    )
    sdam_state, sdam_decisions, sdam_summary = build_sdam_v4_dynamic(
        base,
        working_policy,
        ifrs9_loan,
        online_policy,
        horizon_months=args.sdam_horizon_months,
    )
    _write_parquet("paper4_sdam_v4_dynamic_solver_state.parquet", sdam_state)
    _write_parquet("paper4_sdam_v4_dynamic_solver_decisions.parquet", sdam_decisions)
    _write_csv("paper4_sdam_v4_dynamic_solver_summary.csv", sdam_summary)

    (
        causal_dossier,
        causal_balance,
        causal_placebo,
        causal_sensitivity,
        causal_outcomes,
        causal_overlap,
    ) = build_causal_v4(base, performance)
    _write_csv("paper4_causal_high_rate_v4_dossier.csv", causal_dossier)
    _write_csv("paper4_causal_high_rate_v4_balance.csv", causal_balance)
    _write_csv("paper4_causal_high_rate_v4_placebo.csv", causal_placebo)
    _write_csv("paper4_causal_high_rate_v4_sensitivity.csv", causal_sensitivity)
    _write_csv("paper4_causal_high_rate_v4_outcomes.csv", causal_outcomes)
    _write_csv("paper4_causal_high_rate_v4_overlap.csv", causal_overlap)

    frontier = build_regret_auditability_v4(
        local_search,
        ifrs9_summary,
        online_summary,
        online_policy,
        mdcp,
        fair_summary,
        cvar_results,
    )
    _write_csv("paper4_regret_auditability_v4_frontier.csv", frontier)

    generated_figures = build_v4_figures(
        local_search,
        online_summary,
        cvar_results,
        frontier,
        sample_ci,
    )

    claim_matrix = build_claim_matrix_v4()
    _write_csv("paper4_v4_claim_artifact_matrix.csv", claim_matrix)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v4_eleven_open_priorities",
        "mode": "paper4_working_champions_allowed_paper1_frozen",
        "paper1_frozen_reference_policy": FROZEN_PAPER1_CHAMPION,
        "paper1_artifacts_modified": False,
        "paper4_final_promotion_created": False,
        "paper4_working_champion_created": True,
        "priorities_completed": 11,
        "local_grid_policy_count": int(len(local_search)),
        "local_candidate_pool_n": int(len(pool)),
        "online_best_method": str(online_summary.iloc[0]["online_method_v4"]),
        "paper4_working_champion_policy_id": working_champion["policy_id"],
        "generated_artifacts": claim_matrix["artifact"].tolist(),
        "generated_figures": generated_figures,
    }
    _write_json("paper4_v4_open_priorities_status.json", status)
    _write_note(
        "paper4_v4_open_priorities_memo.qmd",
        """---
title: "Paper 4 v4 Open Priorities Memo"
format: html
---

# Paper 4 v4 Open Priorities Memo

V4 implements the eleven active Paper 4 priorities as a research layer. Paper
Estrella remains frozen. Paper 4 can name a working champion, but this is an
internal laboratory object used for iteration, sample paths and future
experiments.

The main new distinction is between `paper1_frozen_reference_policy` and
`paper4_working_champion_policy_id`.
""",
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
